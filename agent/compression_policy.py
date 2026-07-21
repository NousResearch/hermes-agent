"""Compression timing policy for safe turn-boundary compaction.

The built-in compressor decides *whether* a token count is above the normal
compression threshold.  This module decides *when* automatic compaction may
run:

* historical/default mode: compact immediately whenever the normal threshold
  is crossed;
* deferred mode: allow a completed tool-calling turn to finish after the soft
  threshold, compact at the turn boundary, and retain a higher emergency
  threshold for requests made while the turn is still running.

Keeping this policy separate from ``ContextCompressor`` lets plugin context
engines participate without changing their compression implementation.
"""

from __future__ import annotations

from typing import Any


_DEFAULT_EMERGENCY_THRESHOLD = 0.92


def _coerce_ratio(value: Any, default: float) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return default
    if not 0.0 < ratio <= 1.0:
        return default
    return ratio


def mid_turn_threshold_tokens(agent: Any) -> int:
    """Return the token threshold allowed to interrupt an active turn.

    In the default mode this is exactly the compressor's normal threshold. In
    deferred mode it is the configured emergency ratio of the effective input
    window, never lower than the normal threshold.
    """
    compressor = getattr(agent, "context_compressor", None)
    normal_threshold = int(getattr(compressor, "threshold_tokens", 0) or 0)
    if not getattr(agent, "compression_defer_until_turn_end", False):
        return normal_threshold

    context_length = int(getattr(compressor, "context_length", 0) or 0)
    if context_length <= 0:
        return normal_threshold

    max_tokens = int(getattr(compressor, "max_tokens", 0) or 0)
    effective_input_window = max(1, context_length - max_tokens)
    emergency_ratio = _coerce_ratio(
        getattr(agent, "compression_emergency_threshold", None),
        _DEFAULT_EMERGENCY_THRESHOLD,
    )
    emergency_threshold = int(effective_input_window * emergency_ratio)
    return max(normal_threshold, min(emergency_threshold, effective_input_window))


def should_compress_mid_turn(agent: Any, prompt_tokens: int) -> bool:
    """Return whether automatic compression may interrupt the active turn."""
    if not getattr(agent, "compression_enabled", True):
        return False
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return False
    tokens = max(0, int(prompt_tokens or 0))
    if tokens < mid_turn_threshold_tokens(agent):
        return False
    return bool(compressor.should_compress(tokens))


def should_compress_at_turn_end(agent: Any, prompt_tokens: int) -> bool:
    """Return whether a successfully completed turn should compact now.

    This hook is intentionally active only in deferred mode.  Existing Hermes
    installations therefore keep their historical preflight/mid-turn timing
    unless they explicitly opt in with ``compression.defer_until_turn_end``.
    """
    if not getattr(agent, "compression_enabled", True):
        return False
    if not getattr(agent, "compression_defer_until_turn_end", False):
        return False
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return False
    return bool(compressor.should_compress(max(0, int(prompt_tokens or 0))))
