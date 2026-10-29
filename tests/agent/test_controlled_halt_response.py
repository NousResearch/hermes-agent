"""Truthfulness of the user-facing message emitted when a tool guardrail halts a turn.

`RunAgent._toolguard_controlled_halt_response` is the reply shown to the user after
a hard halt, so it must describe the *actual* reason for the halt rather than a
one-size-fits-all "kept retrying without progress" line. In particular a
`loop_web_search_cap` halt is a per-turn budget limit, not evidence that search
is broken or that the agent stalled on a single call.
"""

from run_agent import AIAgent

from agent.tool_guardrails import ToolGuardrailDecision


def _reply(code: str, message: str, tool_name: str = "web_search", count: int = 20) -> str:
    decision = ToolGuardrailDecision(
        action="block", code=code, message=message, tool_name=tool_name, count=count
    )
    # The method only reads `decision`; `self` is never dereferenced, so an
    # uninitialised instance is enough (avoids AIAgent's heavy __init__).
    agent = AIAgent.__new__(AIAgent)
    return AIAgent._toolguard_controlled_halt_response(agent, decision)


def test_web_search_cap_reports_budget_limit_not_stalled_progress():
    reply = _reply(
        "loop_web_search_cap",
        "Blocked web_search: this turn has already made 20 web searches, the per-turn limit.",
    )
    assert "web_search limit" in reply
    assert "20 searches" in reply
    assert "budget limit" in reply
    assert "not evidence that search is broken" in reply
    assert "without making progress" not in reply


def test_other_halts_relay_the_actual_decision_message():
    message = "Blocked delegate_task: this turn has already spawned 8 subagents (limit 8)."
    reply = _reply("loop_subagent_cap", message, tool_name="delegate_task", count=8)
    assert "Task incomplete" in reply
    # The reply relays the real reason carried by the decision...
    assert message in reply
    # ...instead of the stale generic "kept retrying without progress" wording.
    assert "without making progress" not in reply
