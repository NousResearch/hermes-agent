"""Bounded reads of HTTP error response bodies.

On a non-OK *streaming* response Hermes reads the body for a diagnostic (only ever shown truncated to
a few hundred chars). A bare ``response.read()`` is unbounded two ways: arbitrarily large body
(memory) or a body that stalls forever (hang). ``read_streaming_error_body`` caps bytes and enforces a
hard wall-clock deadline; callers use the returned text instead of ``response.text`` (unbounded /
raises after a partial stream read). ``httpx.iter_bytes()`` blocks *inside* the socket read, so the
read runs on a daemon thread; on timeout we ``shutdown()`` the socket (waking the read without releasing
the FD), give the reader a moment to unwind, then close the response and return the partial bytes. Used
by the streaming error-body sites: native Gemini, Gemini Cloud Code, Antigravity.
"""

from __future__ import annotations

import logging
import threading
from typing import List

import httpx

logger = logging.getLogger(__name__)

# Comfortably holds any real provider error envelope while rejecting pathological bodies.
DEFAULT_ERROR_BODY_MAX_BYTES = 64 * 1024
# Hard deadline for the whole read; past it the connection is closed and the partial bytes are kept.
DEFAULT_ERROR_BODY_TIMEOUT_S = 10.0
# After the deadline's socket shutdown, how long the reader gets to unwind before we close anyway.
_READER_UNWIND_GRACE_S = 2.0


def read_streaming_error_body(
    response: httpx.Response,
    *,
    max_bytes: int = DEFAULT_ERROR_BODY_MAX_BYTES,
    timeout_s: float = DEFAULT_ERROR_BODY_TIMEOUT_S,
) -> str:
    """Read a non-OK streaming body with a byte cap and a hard deadline.

    Returns UTF-8 text (errors replaced) truncated to ``max_bytes``. Never raises: transport errors,
    stalls and oversize bodies yield best-effort partial text (or ""), so a read error can't mask the
    original failure.
    """
    chunks: list[bytes] = []
    state = {"truncated": False}
    done = threading.Event()

    def _drain() -> None:
        total = 0
        try:
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                remaining = max_bytes - total
                if len(chunk) > remaining:
                    if remaining > 0:
                        chunks.append(chunk[:remaining])
                    state["truncated"] = True
                    break
                chunks.append(chunk)
                total += len(chunk)
        except Exception as exc:
            logger.debug("bounded error-body read failed: %s", exc)
        finally:
            done.set()

    threading.Thread(target=_drain, name="bounded-error-body-read", daemon=True).start()
    if not done.wait(timeout=timeout_s):
        logger.debug(
            "bounded error-body read: hard timeout after %.1fs (%d bytes so far)",
            timeout_s, sum(len(c) for c in chunks),
        )
        # Never close() under a read still blocked in the kernel: on Linux it does not wake the read,
        # and it releases an FD the reader's SSL BIO still caches (the TLS FD-recycle class, see
        # ``_shutdown_socket``). shutdown() is FD-safe from this thread and wakes the read.
        from agent.agent_runtime_helpers import _shutdown_socket, _socket_from_response

        sock = _socket_from_response(response)
        if sock is not None:
            _shutdown_socket(sock)
        done.wait(timeout=_READER_UNWIND_GRACE_S)
    # No join past the grace: the worker is a daemon and may be blocked in C.
    try:
        response.close()
    except Exception:
        pass

    if state["truncated"]:
        logger.debug(
            "bounded error-body read: capped at %d bytes (max=%d)", sum(len(c) for c in chunks), max_bytes,
        )
    return b"".join(chunks).decode("utf-8", errors="replace")
