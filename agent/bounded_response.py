"""Bounded reads of streamed HTTP response bodies.

On a non-OK *streaming* response Hermes reads the body for a diagnostic (only ever shown truncated to
a few hundred chars). A bare ``response.read()`` is unbounded two ways: arbitrarily large body
(memory) or a body that stalls forever (hang). ``read_streaming_error_body`` caps bytes and enforces a
hard wall-clock deadline; callers use the returned text instead of ``response.text`` (unbounded /
raises after a partial stream read). ``httpx.iter_bytes()`` blocks *inside* the socket read, so the
read runs on a daemon thread; on timeout we close the response (unblocking the read) and return the
partial bytes. Successful JSON reads use identity-only raw bytes and reject incomplete bodies.
Used by the streaming error-body sites: native Gemini, Gemini Cloud Code, Antigravity.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, List

import httpx

logger = logging.getLogger(__name__)

# Comfortably holds any real provider error envelope while rejecting pathological bodies.
DEFAULT_ERROR_BODY_MAX_BYTES = 64 * 1024
# Hard deadline for the whole read; past it the connection is closed and the partial bytes are kept.
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
    """Read a non-OK streaming body with a byte cap and a hard deadline.

    Returns UTF-8 text (errors replaced) truncated to ``max_bytes``. Never raises: transport errors,
    stalls and oversize bodies yield best-effort partial text (or ""), so a read error can't mask the
    original failure.
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
) -> tuple[bytes, bool, bool, Exception | None]:
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
