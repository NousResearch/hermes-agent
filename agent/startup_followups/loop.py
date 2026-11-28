"""Bridge a turn's first request to the startup follow-up handoff."""

from typing import Any

from .state import StartupFollowups


def stage_first_request(agent: Any, state: Any) -> StartupFollowups | None:
    startup = getattr(agent, "_gateway_startup_followups", None)
    if not isinstance(startup, StartupFollowups):
        return None
    if state.api_call_count != 1:
        return startup
    if (agent.api_mode not in ("chat_completions", "codex_responses", "anthropic_messages")
            or agent.provider == "moa" or not startup.can_persist(agent)):
        startup.finish()
        return startup
    threshold = getattr(agent.context_compressor, "threshold_tokens", 0)
    pressure = state.request_pressure_tokens
    allowance = max(0, int(threshold - pressure)) if (
        isinstance(threshold, (int, float)) and threshold > 0 and isinstance(pressure, (int, float))
    ) else None
    if startup.stage(state.messages, state.api_messages, state.current_turn_user_idx,
                     max_extra_tokens=allowance):
        if isinstance(state.request_pressure_tokens, (int, float)):
            state.request_pressure_tokens += startup.added_tokens
        if isinstance(state.approx_tokens, (int, float)):
            state.approx_tokens += startup.added_tokens
    return startup


def current_message(startup: StartupFollowups | None, fallback: Any) -> Any:
    return startup.current_message(fallback) if startup is not None else fallback
