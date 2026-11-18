"""Per-model-call events on the agent's structured progress feed.

API and TUI clients otherwise see usage only once a run completes. After every model API call
the turn emits ``usage.recorded`` (that call's tokens, cost, time-to-first-token, duration and
context-window fill), and when the retry loop backs off on a rate limit it emits
``model.rate_limited`` (the provider's ``Retry-After``, the wait chosen, the attempt).

Both ride ``agent.tool_progress_callback`` as ``(event, None, None, None, **payload)``: the same
structured channel ``reasoning.available`` / ``moa.*`` / ``subagent.*`` use, which every driver
dispatches by event name and ignores when unknown. ``status_callback`` is not used: messaging
adapters render its text into the chat. ``EVENT_FIELDS`` is the wire key set each surface
(``/v1/runs`` SSE, the TUI gateway, the local-viewer stream) forwards.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

USAGE_RECORDED = "usage.recorded"
MODEL_RATE_LIMITED = "model.rate_limited"

EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    USAGE_RECORDED: (
        "call_index", "api_request_id", "model", "provider", "input_tokens", "output_tokens",
        "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "cost_usd",
        "first_token_ms", "duration_ms", "used_tokens", "window_tokens",
    ),
    MODEL_RATE_LIMITED: (
        "api_request_id", "model", "provider", "attempt", "max_attempts", "retry_after_ms", "wait_ms",
    ),
}


def model_call_payload(event_type: str, fields: dict[str, Any]) -> dict[str, Any]:
    """The wire payload for one of these events: its declared keys, ``None`` dropped."""
    return {key: fields[key] for key in EVENT_FIELDS[event_type] if fields.get(key) is not None}


def _ms(seconds: Optional[float]) -> Optional[int]:
    # A provider's Retry-After is unbounded input ("1e400" parses to inf): unknown, never a crash.
    return None if seconds is None or not math.isfinite(seconds) else max(0, round(seconds * 1000))


def note_attempt_started(agent: Any) -> None:
    """Stamp the start of one provider attempt (the turn's ``api_start_time`` spans every retry)."""
    agent._last_api_attempt_started_at = time.time()


def _emit(agent: Any, event: str, payload: dict[str, Any]) -> None:
    callback = getattr(agent, "tool_progress_callback", None)
    if callback is None:
        return
    try:
        callback(event, None, None, None, **payload)
    except Exception:  # health: allow BLE001 -- a driver sink must never break the turn loop
        logger.debug("%s callback failed", event, exc_info=True)


def attempt_timing(agent: Any, api_duration: float) -> tuple[Optional[int], int]:
    """``(first_token_ms, duration_ms)`` for the attempt that produced the response just received.
    Call it before any accounting (a cold pricing lookup must not count as model time). First token
    is known only when the attempt streamed; duration falls back to the turn-loop's measure."""
    started = getattr(agent, "_last_api_attempt_started_at", None)
    first_chunk = getattr(agent, "_last_api_first_chunk_at", None)
    if not isinstance(started, (int, float)):
        return None, _ms(api_duration) or 0
    first_token_ms = _ms(first_chunk - started) if isinstance(first_chunk, (int, float)) else None
    return first_token_ms, _ms(time.time() - started) or 0


def emit_usage_recorded(
    agent: Any, usage: Any, *, used_tokens: int, cost_usd: Optional[float], timing: tuple[Optional[int], int],
) -> None:
    """``usage.recorded`` for the call just folded by ``record_response_usage``; ``usage`` is its
    canonical (MoA-folded) usage, ``used_tokens`` the context prompt now in the window, ``timing``
    the :func:`attempt_timing` taken when the response arrived."""
    first_token_ms, duration_ms = timing
    window = int(getattr(getattr(agent, "context_compressor", None), "context_length", 0) or 0)
    _emit(agent, USAGE_RECORDED, {
        "call_index": int(agent.session_api_calls),
        "api_request_id": getattr(agent, "_current_api_request_id", None),
        "model": agent.model or "", "provider": agent.provider or "",
        "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens, "cache_write_tokens": usage.cache_write_tokens,
        "reasoning_tokens": usage.reasoning_tokens, "cost_usd": cost_usd,
        "first_token_ms": first_token_ms, "duration_ms": duration_ms,
        "used_tokens": int(used_tokens), "window_tokens": window or None,
    })


def emit_rate_limited(
    agent: Any, *, retry_after_s: Optional[float], wait_s: float, attempt: int, max_attempts: int,
) -> None:
    """``model.rate_limited`` as the retry loop starts waiting: ``retry_after_ms`` is what the
    provider declared (``None`` when it named no cooldown), ``wait_ms`` the wait actually taken."""
    _emit(agent, MODEL_RATE_LIMITED, {
        "api_request_id": getattr(agent, "_current_api_request_id", None),
        "model": agent.model or "", "provider": agent.provider or "",
        "attempt": int(attempt), "max_attempts": int(max_attempts),
        "retry_after_ms": _ms(retry_after_s), "wait_ms": _ms(wait_s) or 0,
    })


__all__ = [
    "EVENT_FIELDS", "MODEL_RATE_LIMITED", "USAGE_RECORDED", "attempt_timing", "emit_rate_limited", "emit_usage_recorded",
    "model_call_payload", "note_attempt_started",
]
