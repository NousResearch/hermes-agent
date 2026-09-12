"""Install the Codex daily quota guard into the turn pre-call gate."""

from __future__ import annotations

from typing import Any

from agent import turn_api_call
from agent.codex_quota_guard import check_daily_budget

_original_guard = turn_api_call.nous_rate_limit_guard


def _quota_guard(agent: Any, **kwargs: Any):
    if getattr(agent, "provider", None) == "openai-codex":
        message = check_daily_budget(
            base_url=getattr(agent, "base_url", "") or "",
            api_key=getattr(agent, "api_key", "") or "",
        )
        if message:
            messages = kwargs.get("messages")
            conversation_history = kwargs.get("conversation_history")
            persist = getattr(agent, "_persist_session", None)
            if callable(persist):
                persist(messages, conversation_history)
            return turn_api_call.NousRateGuardVerdict(
                action="return",
                active_system_prompt=kwargs.get("active_system_prompt"),
                retry_count=kwargs.get("retry_count"),
                compression_attempts=kwargs.get("compression_attempts"),
                result={
                    "final_response": message,
                    "messages": messages,
                    "api_calls": kwargs.get("api_call_count"),
                    "completed": False,
                    "failed": True,
                    "error": message,
                },
            )
    return _original_guard(agent, **kwargs)


turn_api_call.nous_rate_limit_guard = _quota_guard
