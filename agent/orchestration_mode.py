"""Session-scoped orchestration mode helpers.

This module implements the Claude API Docs "orchestration mode" pattern in a
provider-safe way:

* turn the active session into high-effort mode by default;
* periodically remind the model to scout and fan out substantive work;
* keep the frozen top-level system prompt byte-stable for prompt caching;
* use true mid-conversation system messages only on native Claude models that
  support them, otherwise fall back to ephemeral user-message context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agent.orchestration_hooks import run_lifecycle_hooks
from agent.orchestration_trace import get_or_create_trace, record_agent_event


DEFAULT_REFRESH_TURNS = 10
ORCHESTRATION_EFFORT = "xhigh"

MODE_ENTER = (
    "Orchestration mode is on: optimize for the most exhaustive, correct answer "
    "rather than the fastest or cheapest one. For every substantive task, scout "
    "the work-list yourself and use delegate_task to fan out independent "
    "research, implementation, review, or verification work. Token cost is not "
    "the primary constraint. Work solo only on conversational, trivial, or "
    "single-tool turns."
)

MODE_REFRESH = (
    "Orchestration mode is still on. For substantive tasks, scout first, then "
    "use delegate_task for parallel or adversarial sub-work; verify child claims "
    "before finalizing."
)

MODE_EXIT = (
    "Orchestration mode is off. delegate_task returns to normal opt-in behavior: "
    "use it only when it is clearly warranted or explicitly requested."
)


@dataclass
class OrchestrationState:
    enabled: bool = False
    pending_notice: str = ""  # enter|refresh|exit
    turns_since_reminder: int = 0
    refresh_turns: int = DEFAULT_REFRESH_TURNS
    previous_reasoning_config: Optional[dict] = None
    reasoning_overridden: bool = False


def _state(agent: Any) -> OrchestrationState:
    state = getattr(agent, "_orchestration_state", None)
    if not isinstance(state, OrchestrationState):
        state = OrchestrationState()
        setattr(agent, "_orchestration_state", state)
    return state


def get_state(agent: Any) -> OrchestrationState:
    """Return the mutable orchestration state attached to *agent*."""

    return _state(agent)


def is_enabled(agent: Any) -> bool:
    return _state(agent).enabled


def set_enabled(agent: Any, enabled: bool, *, refresh_turns: Optional[int] = None) -> OrchestrationState:
    """Enable/disable orchestration mode on an AIAgent instance.

    Enabling stores the current reasoning config and raises effort to xhigh for
    this session. Disabling restores the prior reasoning config if this helper
    was the one that overrode it.
    """

    state = _state(agent)
    if refresh_turns is not None:
        try:
            state.refresh_turns = max(1, int(refresh_turns))
        except (TypeError, ValueError):
            state.refresh_turns = DEFAULT_REFRESH_TURNS

    if enabled and not state.enabled:
        state.enabled = True
        state.pending_notice = "enter"
        state.turns_since_reminder = 0
        state.previous_reasoning_config = getattr(agent, "reasoning_config", None)
        state.reasoning_overridden = True
        agent.reasoning_config = {"enabled": True, "effort": ORCHESTRATION_EFFORT}
        setattr(agent, "_orchestration_trace_enabled", True)
        trace = get_or_create_trace(agent)
        trace.record("mode_enabled", reasoning_effort=ORCHESTRATION_EFFORT)
        run_lifecycle_hooks(
            "orchestration_mode_enabled",
            {"run_id": trace.run_id, "session_id": getattr(agent, "session_id", "") or ""},
        )
    elif not enabled and state.enabled:
        state.enabled = False
        state.pending_notice = "exit"
        state.turns_since_reminder = 0
        if state.reasoning_overridden:
            agent.reasoning_config = state.previous_reasoning_config
        record_agent_event(agent, "mode_disabled")
        trace = getattr(agent, "_orchestration_trace", None)
        run_lifecycle_hooks(
            "orchestration_mode_disabled",
            {"run_id": getattr(trace, "run_id", ""), "session_id": getattr(agent, "session_id", "") or ""},
        )
        state.previous_reasoning_config = None
        state.reasoning_overridden = False
    return state


def reminder_for_next_turn(agent: Any) -> str:
    """Return the ephemeral reminder to inject for this API call, if any.

    The returned reminder is consumed: entry/exit notices fire once, refreshers
    fire every ``refresh_turns`` user turns while enabled.
    """

    state = _state(agent)
    notice = state.pending_notice
    state.pending_notice = ""

    if notice == "enter":
        state.turns_since_reminder = 0
        record_agent_event(agent, "reminder", notice="enter")
        run_lifecycle_hooks("orchestration_reminder", {"notice": "enter"})
        return MODE_ENTER
    if notice == "exit":
        record_agent_event(agent, "reminder", notice="exit")
        run_lifecycle_hooks("orchestration_reminder", {"notice": "exit"})
        return MODE_EXIT

    if not state.enabled:
        return ""

    state.turns_since_reminder += 1
    if state.turns_since_reminder >= max(1, state.refresh_turns):
        state.turns_since_reminder = 0
        record_agent_event(agent, "reminder", notice="refresh")
        run_lifecycle_hooks("orchestration_reminder", {"notice": "refresh"})
        return MODE_REFRESH
    return ""


def supports_mid_conversation_system_messages(agent: Any) -> bool:
    """Return True when true mid-conversation system reminders are safe.

    Per Anthropic's docs, this is currently a Claude Opus 4.8 feature. Keep the
    gate intentionally narrow; other providers receive the same reminder as an
    ephemeral addition to the user message instead of a role=system message.
    """

    api_mode = str(getattr(agent, "api_mode", "") or "").lower()
    provider = str(getattr(agent, "provider", "") or "").lower()
    model = str(getattr(agent, "model", "") or "").lower()
    base_url = str(getattr(agent, "base_url", "") or "").lower()

    if api_mode != "anthropic_messages":
        return False
    if provider not in {"anthropic", "claude", "custom:anthropic"} and "api.anthropic.com" not in base_url:
        return False
    return "opus" in model and ("4.8" in model or "4-8" in model or "4_8" in model)


def wrap_user_reminder(text: str) -> str:
    return f"<system-reminder orchestration_mode=\"on\">\n{text}\n</system-reminder>"
