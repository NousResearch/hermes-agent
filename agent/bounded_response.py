"""Bounded reads of HTTP error response bodies.

On a non-OK *streaming* response Hermes reads the body for a diagnostic (only ever shown truncated to
a few hundred chars). A bare ``response.read()`` is unbounded two ways: arbitrarily large body
(memory) or a body that stalls forever (hang). ``read_streaming_error_body`` caps bytes and enforces a
hard wall-clock deadline; callers use the returned text instead of ``response.text`` (unbounded /
raises after a partial stream read). ``httpx.iter_bytes()`` blocks *inside* the socket read, so the
read runs on a daemon thread; on timeout we close the response (unblocking the read) and return the
partial bytes. Used by the streaming error-body sites: native Gemini, Gemini Cloud Code, Antigravity.
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
                if len(chunk) > remaining:
                    if remaining > 0:
                        chunks.append(chunk[:remaining])
                    state["truncated"] = True
                    break
                chunks.append(chunk)
                total += len(chunk)
        except Exception as exc:  # noqa: BLE001 - error path must not raise
            logger.debug("bounded error-body read failed: %s", exc)
        finally:
            done.set()

    threading.Thread(target=_drain, name="bounded-error-body-read", daemon=True).start()
    if not done.wait(timeout=timeout_s):
        logger.debug(
            "bounded error-body read: hard timeout after %.1fs (%d bytes so far)",
            timeout_s, sum(len(c) for c in chunks),
        )
    # Closing cancels any in-flight socket read so the worker unwinds. No join (daemon, may be blocked in C).
    try:
        response.close()
    except Exception:  # noqa: BLE001
        pass

    if state["truncated"]:
        logger.debug(
            "bounded error-body read: capped at %d bytes (max=%d)", sum(len(c) for c in chunks), max_bytes,
        )
    return b"".join(chunks).decode("utf-8", errors="replace")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402

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
# ---- END PLUGIN-COMPAT ----
