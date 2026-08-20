"""Bounded reads of streamed HTTP response bodies.

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

``read_streaming_error_body`` returns a best-effort decoded diagnostic.
``read_streaming_json_response`` applies the same byte and deadline bounds to
successful identity-encoded JSON responses, but raises if the body is
incomplete so callers cannot accidentally accept truncated data.

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

import json
import logging
import threading
from typing import Any, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Defaults chosen to comfortably hold any real provider error envelope (Google
# RPC error JSON, Anthropic error JSON) while rejecting pathological bodies.
DEFAULT_ERROR_BODY_MAX_BYTES = 64 * 1024
# Hard wall-clock deadline for the whole bounded read. A streaming error body
# that does not finish within this window is abandoned and the connection is
# closed; we keep whatever partial bytes arrived.
DEFAULT_ERROR_BODY_TIMEOUT_S = 10.0


class HTTPXResponseBodyTooLarge(ValueError):
    """Raised when a streamed response exceeds its byte cap."""


class HTTPXUnsupportedContentEncoding(ValueError):
    """Raised when an identity-only response uses a content coding."""


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
    data, timed_out, truncated, error = _read_streaming_body(
        response,
        max_bytes=max_bytes,
        timeout_s=timeout_s,
        raw=False,
    )
    if error is not None:
        logger.debug("bounded error-body read failed: %s", error)
    if timed_out:
        logger.debug(
            "bounded error-body read: hard timeout after %.1fs (%d bytes so far)",
            timeout_s,
            len(data),
        )
    if truncated:
        logger.debug(
            "bounded error-body read: capped at %d bytes (max=%d)",
            len(data),
            max_bytes,
        )
    return data.decode("utf-8", errors="replace")


def read_streaming_json_response(
    response: httpx.Response,
    *,
    max_bytes: int,
    timeout_s: float = DEFAULT_ERROR_BODY_TIMEOUT_S,
) -> Any:
    """Read complete JSON through an identity-only raw stream within hard bounds.

    Callers must send ``Accept-Encoding: identity``. Rejecting a server that
    ignores that negotiation before calling ``iter_raw()`` prevents a small
    compressed chunk from expanding ahead of the byte cap.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")

    content_encoding = response.headers.get("content-encoding", "").strip().lower()
    if content_encoding not in {"", "identity"}:
        _safe_close(response)
        raise HTTPXUnsupportedContentEncoding(
            f"response used unsupported Content-Encoding: {content_encoding}"
        )

    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared_bytes = int(content_length)
        except ValueError:
            pass
        else:
            if declared_bytes > max_bytes:
                _safe_close(response)
                raise HTTPXResponseBodyTooLarge(
                    f"streaming JSON response exceeds {max_bytes} bytes"
                )

    data, timed_out, truncated, error = _read_streaming_body(
        response,
        max_bytes=max_bytes,
        timeout_s=timeout_s,
        raw=True,
    )
    if timed_out:
        raise TimeoutError(
            f"streaming JSON response exceeded {timeout_s:g}s deadline"
        )
    if truncated:
        raise HTTPXResponseBodyTooLarge(
            f"streaming JSON response exceeds {max_bytes} bytes"
        )
    if error is not None:
        raise error
    return json.loads(data)


def _read_streaming_body(
    response: httpx.Response,
    *,
    max_bytes: int,
    timeout_s: float,
    raw: bool,
) -> tuple[bytes, bool, bool, Optional[Exception]]:
    chunks: List[bytes] = []
    truncated = threading.Event()
    errors: List[Exception] = []
    done = threading.Event()

    def _drain() -> None:
        total = 0
        try:
            iterator = response.iter_raw() if raw else response.iter_bytes()
            for chunk in iterator:
                if not chunk:
                    continue
                remaining = max_bytes - total
                if remaining <= 0:
                    truncated.set()
                    break
                if len(chunk) > remaining:
                    chunks.append(chunk[:remaining])
                    truncated.set()
                    break
                chunks.append(chunk)
                total += len(chunk)
        except Exception as exc:  # noqa: BLE001 - surfaced by JSON reader
            errors.append(exc)
        finally:
            done.set()

    worker = threading.Thread(
        target=_drain, name="bounded-response-read", daemon=True
    )
    worker.start()
    finished = done.wait(timeout=timeout_s)

    # Closing cancels an in-flight socket read. Do not join: a daemon worker
    # may still be blocked in C, and callers must keep the hard deadline.
    _safe_close(response)
    data = b"".join(chunks)
    return data, not finished, truncated.is_set(), errors[0] if errors else None


def _safe_close(response: httpx.Response) -> None:
    try:
        response.close()
    except Exception:  # noqa: BLE001
        pass


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
