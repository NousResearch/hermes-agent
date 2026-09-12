"""Tests for the futile-loop circuit breaker (AIAgent._note_tool_failure_repetition).

The background review fork runs unattended: when a tool kept failing with an
identical error (the curator's view→patch→refused loop against the skill
read-before-write guard), the model retried the same call until the
16-iteration budget was exhausted — every retry a full-context inference call
that kept the local GPU pinned for minutes. The breaker counts consecutive
identical failures PER TOOL and trips ``_repeated_tool_failure_tripped`` at
the opt-in limit; agent/turn_tool_round.py aborts the turn when it is set.

Keyed per tool on purpose: in the observed incident, successful ``skill_view``
calls were interleaved between the refused ``skill_manage`` calls, so any
"consecutive failing iterations" scheme would never have tripped.
"""

from run_agent import AIAgent


class _StubAgent:
    """Bare object carrying only the attributes the breaker reads/writes."""

    note = AIAgent._note_tool_failure_repetition

    def __init__(self, limit=3):
        if limit:
            self._repeated_tool_failure_limit = limit


ERR = '{"success": false, "error": "Refusing background curator patch"}'
OK = '{"success": true}'


class TestRepeatedToolFailureBreaker:
    def test_trips_after_limit_identical_failures(self):
        agent = _StubAgent(limit=3)
        agent.note("skill_manage", ERR, failed=True)
        agent.note("skill_manage", ERR, failed=True)
        assert getattr(agent, "_repeated_tool_failure_tripped", None) is None
        agent.note("skill_manage", ERR, failed=True)
        assert agent._repeated_tool_failure_tripped == ("skill_manage", 3)

    def test_interleaved_success_of_other_tool_does_not_reset(self):
        """The incident pattern: skill_view succeeds between skill_manage
        refusals. Only a success of the SAME tool shows progress."""
        agent = _StubAgent(limit=3)
        agent.note("skill_manage", ERR, failed=True)
        agent.note("skill_view", OK, failed=False)
        agent.note("skill_manage", ERR, failed=True)
        agent.note("skill_view", OK, failed=False)
        agent.note("skill_manage", ERR, failed=True)
        assert agent._repeated_tool_failure_tripped == ("skill_manage", 3)

    def test_success_of_same_tool_resets_counter(self):
        agent = _StubAgent(limit=3)
        agent.note("skill_manage", ERR, failed=True)
        agent.note("skill_manage", ERR, failed=True)
        agent.note("skill_manage", OK, failed=False)
        agent.note("skill_manage", ERR, failed=True)
        agent.note("skill_manage", ERR, failed=True)
        assert getattr(agent, "_repeated_tool_failure_tripped", None) is None

    def test_different_error_restarts_count(self):
        """A changing error message means the model IS making progress
        (e.g. fixing one validation complaint and hitting the next)."""
        agent = _StubAgent(limit=3)
        agent.note("skill_manage", "error: description too long", failed=True)
        agent.note("skill_manage", "error: description too long", failed=True)
        agent.note("skill_manage", "error: name invalid", failed=True)
        assert getattr(agent, "_repeated_tool_failure_tripped", None) is None
        agent.note("skill_manage", "error: name invalid", failed=True)
        agent.note("skill_manage", "error: name invalid", failed=True)
        assert agent._repeated_tool_failure_tripped == ("skill_manage", 3)

    def test_disabled_without_opt_in(self):
        agent = _StubAgent(limit=0)
        for _ in range(10):
            agent.note("skill_manage", ERR, failed=True)
        assert getattr(agent, "_repeated_tool_failure_tripped", None) is None
        assert getattr(agent, "_repeated_tool_failures", None) is None

    def test_non_string_result_is_stringified(self):
        agent = _StubAgent(limit=2)
        payload = [{"type": "text", "text": "err"}]
        agent.note("vision_tool", payload, failed=True)
        agent.note("vision_tool", payload, failed=True)
        assert agent._repeated_tool_failure_tripped == ("vision_tool", 2)
