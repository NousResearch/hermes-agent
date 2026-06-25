"""Engine-enforced autopilot for Hermes.

Public surface used by the conversation loop and agent init:

    is_autopilot_active(agent)                  -> bool
    reset_turn_state(agent)                     -> None
    maybe_continue(agent, messages, final, msg) -> Optional[str]   (directive or None)
    resolve_goal(agent, user_message)           -> str

All policy (the goal-completion quality gate, the no-progress safety, the
continuation injection) lives here; the conversation loop only calls
``maybe_continue`` at the no-tool-calls branch. The independent judge is the real
Hermes Council via its Hermes-native lane (see ``council_gate``).
"""

from __future__ import annotations

from agent.autopilot.driver import (
    is_autopilot_active,
    keep_budget_ahead,
    make_clarify_autoanswer,
    maybe_continue,
    reenter_after_abnormal_exit,
    reset_turn_state,
    resolve_goal,
)

__all__ = [
    "is_autopilot_active",
    "keep_budget_ahead",
    "make_clarify_autoanswer",
    "maybe_continue",
    "reenter_after_abnormal_exit",
    "reset_turn_state",
    "resolve_goal",
]
