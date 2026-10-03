"""Session budget guardrail phase for the decomposed conversation loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _guardrail_stop_result(agent: Any, message: str, *, messages: Any) -> dict[str, Any]:
    """Return the normal run_conversation envelope shape for non-CLI callers."""
    history = list(messages or [])
    history.append({"role": "assistant", "content": message})
    result: dict[str, Any] = {
        "final_response": message,
        "last_reasoning": None,
        "messages": history,
        "api_calls": int(getattr(agent, "api_call_count", 0) or 0),
        "completed": False,
        "turn_exit_reason": "session_budget_guardrail_stop",
        "failed": True,
        "partial": True,
        "interrupted": False,
        "error": message,
        "failure_reason": "session_budget_guardrail_stop",
        "model": getattr(agent, "model", None),
        "provider": getattr(agent, "provider", None),
        "base_url": getattr(agent, "base_url", None),
        "session_id": getattr(agent, "session_id", None),
    }
    for key in (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
        "reasoning_tokens", "prompt_tokens", "completion_tokens", "total_tokens",
        "estimated_cost_usd", "cost_status", "cost_source",
    ):
        result[key] = getattr(agent, f"session_{key}", 0)
    return result


@dataclass
class SessionBudgetGuardrailVerdict:
    """``action`` is ``fallthrough`` or ``return`` with a final turn result."""

    action: str
    result: Any = None


def run_session_budget_guardrail_gate(agent: Any, *, messages: Any, conversation_history: Any) -> SessionBudgetGuardrailVerdict:
    """Pause before the next provider call when the configured hard budget latch is armed.

    This runs before ``begin_iteration`` so a stopped/compacted turn does not consume an
    API-call counter or iteration budget slot.
    """
    try:
        from agent.session_budget_guardrail import enforce_before_provider_call
    except Exception:
        return SessionBudgetGuardrailVerdict(action="fallthrough")
    result = enforce_before_provider_call(agent, messages=messages, conversation_history=conversation_history)
    if result is not None:
        return SessionBudgetGuardrailVerdict(action="return", result=_guardrail_stop_result(agent, result, messages=messages))
    return SessionBudgetGuardrailVerdict(action="fallthrough")
