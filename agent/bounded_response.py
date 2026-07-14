"""Bounded reads of HTTP error response bodies.

When a provider returns a non-OK status on a *streaming* request, Hermes reads
the response body to build a useful diagnostic error. A bare ``response.read()``
on a streaming httpx response is unbounded in two dangerous ways:

1. A server can declare (or stream) an arbitrarily large body, so the read can
   balloon memory.
2. A server can open the body and then stall forever (no ``Content-Length``,
   no further bytes), so the read hangs the agent indefinitely.

Both are realistic against a misbehaving proxy, a hijacked endpoint, or a
provider having a bad day. The diagnostic body is only ever shown to the user
truncated to a few hundred characters, so reading megabytes — or blocking
forever — buys nothing.

``read_streaming_error_body`` bounds the read to a byte cap and enforces a
hard wall-clock deadline, returning the decoded text snippet. Callers pass the
returned text into their existing error builders instead of touching
``response.text`` (which would be unbounded / would raise after a partial
stream read).

A subtlety the implementation must respect: ``httpx``'s ``iter_bytes()`` blocks
*inside* the C/socket read while waiting for the next chunk. A wall-clock check
placed only between yielded chunks cannot interrupt a server that opens the
body and then stalls mid-chunk — control never returns to Python until httpx's
own (often 30s+) read timeout fires. To guarantee a bounded stop regardless of
socket behavior, the read runs on a daemon worker thread and the caller waits
on it with a hard deadline; on timeout we close the response (which unblocks /
cancels the read) and return whatever partial bytes were collected.

Ported and adapted from openclaw/openclaw#95108 ("bound Anthropic error
streams"), generalized to cover Hermes's three streaming error-body sites
(native Gemini, Gemini Cloud Code, Antigravity Cloud Code).
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

# Defaults chosen to comfortably hold any real provider error envelope (Google
# RPC error JSON, Anthropic error JSON) while rejecting pathological bodies.
DEFAULT_ERROR_BODY_MAX_BYTES = 64 * 1024
# Hard wall-clock deadline for the whole bounded read. A streaming error body
# that does not finish within this window is abandoned and the connection is
# closed; we keep whatever partial bytes arrived.
DEFAULT_ERROR_BODY_TIMEOUT_S = 10.0


def read_streaming_error_body(
    response: httpx.Response,
    *,
    max_bytes: int = DEFAULT_ERROR_BODY_MAX_BYTES,
    timeout_s: float = DEFAULT_ERROR_BODY_TIMEOUT_S,
) -> str:
    """Read a non-OK streaming response body with a byte cap and a hard deadline.

    Returns the decoded body text (UTF-8, errors replaced), truncated to
    ``max_bytes``. Never raises: any transport error, stall, or oversize
    condition is swallowed and the best-effort partial text (or an empty
    string) is returned, because this runs on the error path and must not
    mask the original HTTP failure with a read error.

    The byte cap protects against huge bodies; the wall-clock deadline (enforced
    via a worker thread so it can interrupt a socket read that stalls mid-chunk)
    protects against bodies that open and then hang.
    """
    chunks: List[bytes] = []
    state = {"truncated": False}
    done = threading.Event()

    def _drain() -> None:
        total = 0
        try:
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                remaining = max_bytes - total
                if remaining <= 0:
                    state["truncated"] = True
                    break
                if len(chunk) > remaining:
                    chunks.append(chunk[:remaining])
                    total += remaining
                    state["truncated"] = True
                    break
                chunks.append(chunk)
                total += len(chunk)
        except Exception as exc:  # noqa: BLE001 - error path must not raise
            logger.debug("bounded error-body read failed: %s", exc)
        finally:
            done.set()

    worker = threading.Thread(
        target=_drain, name="bounded-error-body-read", daemon=True
    )
    worker.start()
    finished = done.wait(timeout=timeout_s)

    if not finished:
        logger.debug(
            "bounded error-body read: hard timeout after %.1fs (%d bytes so far)",
            timeout_s,
            sum(len(c) for c in chunks),
        )
        # Closing the response cancels the in-flight socket read, letting the
        # worker thread unwind. We do not join (it is a daemon and may be
        # blocked in C); the partial `chunks` collected so far are returned.
        _safe_close(response)
    else:
        _safe_close(response)

    if state["truncated"]:
        logger.debug(
            "bounded error-body read: capped at %d bytes (max=%d)",
            sum(len(c) for c in chunks),
            max_bytes,
        )
    return b"".join(chunks).decode("utf-8", errors="replace")


def _safe_close(response: httpx.Response) -> None:
    try:
        response.close()
    except Exception:  # noqa: BLE001
        logger.debug("Suppressed exception", exc_info=True)


def read_error_body_or_default(
    response: httpx.Response,
    *,
    max_bytes: int = DEFAULT_ERROR_BODY_MAX_BYTES,
    timeout_s: float = DEFAULT_ERROR_BODY_TIMEOUT_S,
) -> Optional[str]:
    """Like ``read_streaming_error_body`` but returns ``None`` on empty body.

    Convenience for callers that distinguish "no body" from "empty string".
    """
    text = read_streaming_error_body(
        response, max_bytes=max_bytes, timeout_s=timeout_s
    )
    return text or None
