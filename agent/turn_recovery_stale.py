"""One-shot context compaction after the stale-call circuit breaker trips."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent.chat_completion_stale import _reset_stale_streak, _stale_streak
from agent.conversation_compression import conversation_history_after_compression
from agent.model_metadata import estimate_request_tokens_rough
from agent.turn_retry_state import TurnRetryState

logger = logging.getLogger("agent.conversation_loop")


def compact_stale_context(
    agent: Any, api_error: Exception, retry: TurnRetryState, *, messages: list[dict[str, Any]],
    api_messages: Any, system_message: Any, conversation_history: Any,
    compression_attempts: int, max_compression_attempts: int, task_id: Any,
) -> Optional[tuple[list[dict[str, Any]], Any, Any]]:
    """Compact once above the threshold; a smaller request may survive slow prefill (#139209)."""
    if (
        type(api_error).__name__ != "StaleGiveupError"
        or not getattr(agent, "compression_enabled", True)
        or getattr(retry, "stale_compaction_attempted", False)
        or compression_attempts >= max_compression_attempts
    ):
        return None
    compressor = getattr(agent, "context_compressor", None)
    threshold = getattr(compressor, "threshold_tokens", 0)
    approx_tokens = estimate_request_tokens_rough(
        api_messages, tools=getattr(agent, "tools", None) or None,
    )
    if compressor is None or not approx_tokens > threshold > 0:
        return None

    retry.stale_compaction_attempted = True
    logger.warning(
        "%sStale giveup with large context (streak=%s, ~%s tokens): compacting before final attempt.",
        getattr(agent, "log_prefix", ""), _stale_streak(agent), f"{approx_tokens:,}",
    )
    messages, active_system_prompt = agent._compress_context(
        messages, system_message,
        approx_tokens=approx_tokens, task_id=task_id,
        trigger="stale", overflow_reason="stale_provider",
    )
    conversation_history = conversation_history_after_compression(agent, messages, conversation_history)
    _reset_stale_streak(agent)
    return messages, active_system_prompt, conversation_history
