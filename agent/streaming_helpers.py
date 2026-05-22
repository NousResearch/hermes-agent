"""Streaming helpers extracted from AIAgent — standalone functions with `runner` param.

Each function takes ``runner`` (an :class:`~run_agent.AIAgent` instance) instead
of ``self``, enabling clean extraction without circular imports.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Imports that the extracted methods depend on ────────────────────────

def _get_strip_think_blocks():
    """Lazy import to break circular dependency on agent_runtime_helpers."""
    from agent.agent_runtime_helpers import strip_think_blocks
    return strip_think_blocks


def _get_sanitize_context():
    """Lazy import to break circular dependency on memory_manager."""
    from agent.memory_manager import sanitize_context as _sc
    return _sc


def _get_is_local_endpoint():
    """Lazy import to break circular dependency on model_metadata."""
    from agent.model_metadata import is_local_endpoint as _ile
    return _ile


def _get_provider_stale_timeout():
    """Lazy import to break circular dependency on hermes_cli.timeouts."""
    from hermes_cli.timeouts import get_provider_stale_timeout as _gpst
    return _gpst


# ── 1. stream_diag_init (was static) ───────────────────────────────────

def stream_diag_init() -> Dict[str, Any]:
    """Initialise a per-attempt stream diagnostics dict.

    Returns a dict with a ``"start_time"`` key set to the current monotonic
    clock (``time.monotonic()``) and ``"attempts"`` initialised to ``[]``.
    """
    import time as _time
    return {
        "start_time": _time.monotonic(),
        "attempts": [],
    }


# ── 2. stream_diag_capture_response ────────────────────────────────────

def stream_diag_capture_response(
    runner, diag: Dict[str, Any], http_response: Any
) -> None:
    """Capture diagnostic headers from an httpx response into *diag*."""
    # httpx has .headers / .status_code / .request etc.
    if http_response is None:
        return
    try:
        headers = getattr(http_response, "headers", None)
        if headers is not None:
            for key in ("cf-ray", "cf-cache-status", "x-openrouter-provider",
                        "x-openrouter-organization", "x-request-id",
                        "openai-organization", "x-amzn-requestid",
                        "x-amzn-trace-id", "x-amz-request-id",
                        "x-amz-id-2"):
                val = headers.get(key) or headers.get(key.lower()) or headers.get(key.replace("-", "_")) or headers.get(key.upper().replace("-", "_"))
                if val:
                    diag[key] = str(val)
            # Also grab all x- headers as a fallback
            for raw_key, val in headers.items():
                key = str(raw_key).lower()
                if key.startswith("x-") and key not in diag:
                    diag[key] = str(val)
        diag["http_status"] = getattr(http_response, "status_code", None)
    except Exception:
        logger.debug("stream_diag_capture_response: error reading response", exc_info=True)


# ── 3. is_provider_stream_parse_error ──────────────────────────────────

def is_provider_stream_parse_error(runner, error: BaseException) -> bool:
    """Return True for malformed provider streaming data from SDK parsers.

    Some Anthropic-compatible streaming providers can send a malformed
    event-stream frame.  The Anthropic SDK surfaces that as a plain
    ``ValueError`` such as ``expected ident at line 1 column 149``.  That
    is provider wire-format trouble, not local request validation, so it
    should follow the same retry path as a truncated JSON body.
    """
    if getattr(runner, "api_mode", None) != "anthropic_messages":
        return False
    if not isinstance(error, ValueError):
        return False
    if isinstance(error, (UnicodeEncodeError, json.JSONDecodeError)):
        return False
    message = str(error).strip().lower()
    return "expected ident at line" in message


# ── 4. log_stream_retry ────────────────────────────────────────────────

def log_stream_retry(
    runner,
    *,
    kind: str,
    error: BaseException,
    attempt: int,
    max_attempts: int,
    mid_tool_call: bool,
    diag: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a streaming retry event and surface a compact status line."""
    from agent.stream_diag import log_stream_retry as _lsr
    _lsr(runner, kind=kind, error=error, attempt=attempt,
         max_attempts=max_attempts, mid_tool_call=mid_tool_call, diag=diag)


# ── 5. emit_stream_drop ────────────────────────────────────────────────

def emit_stream_drop(
    runner,
    *,
    error: BaseException,
    attempt: int,
    max_attempts: int,
    mid_tool_call: bool,
    diag: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a stream drop event after exhausting all retries."""
    from agent.stream_diag import emit_stream_drop as _esd
    _esd(runner, error=error, attempt=attempt, max_attempts=max_attempts,
         mid_tool_call=mid_tool_call, diag=diag)


# ── 6. compute_non_stream_stale_timeout ────────────────────────────────

def compute_non_stream_stale_timeout(
    runner, messages: list[dict[str, Any]]
) -> float:
    """Compute the effective non-stream stale timeout for this request."""
    stale_base, uses_implicit_default = runner._resolved_api_call_stale_timeout_base()
    base_url = getattr(runner, "_base_url", None) or runner.base_url or ""
    if uses_implicit_default and base_url:
        is_local = _get_is_local_endpoint()
        if is_local(base_url):
            return float("inf")

    est_tokens = sum(len(str(v)) for v in messages) // 4
    if est_tokens > 100_000:
        return max(stale_base, 600.0)
    if est_tokens > 50_000:
        return max(stale_base, 450.0)
    return stale_base


# ── 7. reset_stream_delivery_tracking ──────────────────────────────────

def reset_stream_delivery_tracking(runner) -> None:
    """Reset tracking for text delivered during the current model response."""
    # Flush any benign partial-tag tail held by the think scrubber
    # first (#17924): an innocent '<' at the end of the stream that
    # turned out not to be a tag prefix should reach the UI.  Then
    # flush the context scrubber.  Order matters — the think
    # scrubber's output feeds into the context scrubber's state.
    think_scrubber = getattr(runner, "_stream_think_scrubber", None)
    if think_scrubber is not None:
        think_tail = think_scrubber.flush()
        if think_tail:
            # Route the tail through the context scrubber too so a
            # memory-context span straddling the final boundary is
            # still caught.
            ctx_scrubber = getattr(runner, "_stream_context_scrubber", None)
            if ctx_scrubber is not None:
                think_tail = ctx_scrubber.feed(think_tail)
            if think_tail:
                callbacks = [cb for cb in (runner.stream_delta_callback, getattr(runner, "_stream_callback", None)) if cb is not None]
                for cb in callbacks:
                    try:
                        cb(think_tail)
                    except Exception:
                        pass
                record_streamed_assistant_text(runner, think_tail)
    # Flush any benign partial-tag tail held by the context scrubber so it
    # reaches the UI before we clear state for the next model call.  If
    # the scrubber is mid-span, flush() drops the orphaned content.
    scrubber = getattr(runner, "_stream_context_scrubber", None)
    if scrubber is not None:
        tail = scrubber.flush()
        if tail:
            callbacks = [cb for cb in (runner.stream_delta_callback, getattr(runner, "_stream_callback", None)) if cb is not None]
            for cb in callbacks:
                try:
                    cb(tail)
                except Exception:
                    pass
            record_streamed_assistant_text(runner, tail)
    runner._current_streamed_assistant_text = ""


# ── 8. record_streamed_assistant_text ──────────────────────────────────

def record_streamed_assistant_text(runner, text: str) -> None:
    """Accumulate visible assistant text emitted through stream callbacks."""
    if isinstance(text, str) and text:
        runner._current_streamed_assistant_text = (
            getattr(runner, "_current_streamed_assistant_text", "") + text
        )


# ── Static helper: normalize_interim_visible_text ──────────────────────

def normalize_interim_visible_text(text: str) -> str:
    """Normalise visible text by collapsing whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


# ── 9. interim_content_was_streamed ────────────────────────────────────

def interim_content_was_streamed(runner, content: str) -> bool:
    """Return True if *content* is already reflected in streamed text."""
    strip_think_blocks = _get_strip_think_blocks()
    visible_content = normalize_interim_visible_text(
        strip_think_blocks(runner, content or "")
    )
    if not visible_content:
        return False
    streamed = normalize_interim_visible_text(
        strip_think_blocks(runner, getattr(runner, "_current_streamed_assistant_text", "") or "")
    )
    return bool(streamed) and streamed == visible_content


# ── 10. fire_stream_delta ──────────────────────────────────────────────

def fire_stream_delta(runner, text: str) -> None:
    """Fire all registered stream delta callbacks (display + TTS)."""
    # If a tool iteration set the break flag, prepend a single paragraph
    # break before the first real text delta.  This prevents the original
    # problem (text concatenation across tool boundaries) without stacking
    # blank lines when multiple tool iterations run back-to-back.
    if getattr(runner, "_stream_needs_break", False) and text and text.strip():
        runner._stream_needs_break = False
        text = "\n\n" + text
        prepended_break = True
    else:
        prepended_break = False
    if isinstance(text, str):
        # Suppress reasoning/thinking blocks via the stateful
        # scrubber (#17924).  Earlier versions ran _strip_think_blocks
        # per-delta here, which destroyed downstream state machines
        # when a tag was split across deltas (e.g. MiniMax-M2.7
        # sends '<think>' and its content as separate deltas —
        # regex case 2 erased the first delta, so the CLI/gateway
        # state machine never saw the open tag and leaked the
        # reasoning content as regular response text).
        think_scrubber = getattr(runner, "_stream_think_scrubber", None)
        if think_scrubber is not None:
            text = think_scrubber.feed(text or "")
        else:
            # Defensive: legacy callers without the scrubber attribute.
            strip_think_blocks = _get_strip_think_blocks()
            text = strip_think_blocks(runner, text or "")
        # Then feed through the stateful context scrubber so memory-context
        # spans split across chunks cannot leak to the UI (#5719).
        sanitize_context = _get_sanitize_context()
        scrubber = getattr(runner, "_stream_context_scrubber", None)
        if scrubber is not None:
            text = scrubber.feed(text)
        else:
            # Defensive: legacy callers without the scrubber attribute.
            text = sanitize_context(text)
        # Only strip leading newlines on the first delta — mid-stream "\n" is legitimate markdown.
        if not prepended_break and not getattr(
            runner, "_current_streamed_assistant_text", ""
        ):
            text = text.lstrip("\n")
    if not text:
        return
    callbacks = [cb for cb in (runner.stream_delta_callback, getattr(runner, "_stream_callback", None)) if cb is not None]
    delivered = False
    for cb in callbacks:
        try:
            cb(text)
            delivered = True
        except Exception:
            pass
    if delivered:
        record_streamed_assistant_text(runner, text)


# ── 11. has_stream_consumers ───────────────────────────────────────────

def has_stream_consumers(runner) -> bool:
    """Return True if any streaming consumer is registered."""
    return (
        runner.stream_delta_callback is not None
        or getattr(runner, "_stream_callback", None) is not None
    )


# ── 12. interruptible_streaming_api_call ───────────────────────────────

def interruptible_streaming_api_call(
    runner, api_kwargs: dict, *, on_first_delta: callable = None
):
    """Forwarder — see ``agent.chat_completion_helpers.interruptible_streaming_api_call``."""
    from agent.chat_completion_helpers import interruptible_streaming_api_call as _isc
    return _isc(runner, api_kwargs, on_first_delta=on_first_delta)
