"""Core compaction observers get truthful outcomes without changing legacy statuses."""

from types import SimpleNamespace
import pytest
from agent.compaction_lifecycle import _CompactionLifecycle


@pytest.mark.parametrize("committed,force,interrupted,expected", [
    (True, False, False, "completed"), (False, False, False, "failed"),
    (False, True, False, "cancelled"), (False, False, True, "cancelled")])
def test_terminal_observer_is_one_shot_and_preserves_legacy_status(committed, force, interrupted, expected):
    outcomes, statuses = [], []
    agent = SimpleNamespace(compaction_callback=outcomes.append,
                            status_callback=lambda *args: statuses.append(args), _interrupt_requested=interrupted)
    cycle = _CompactionLifecycle(agent, True)
    cycle.commit_status = "committed" if committed else "aborted"
    cycle.complete(force_terminal=force)
    cycle.complete(force_terminal=force)
    assert outcomes == [expected]
    assert bool(statuses) == (committed or force)
    if statuses:
        assert statuses[0][0] == "compacted"


def test_observer_failure_does_not_break_committed_compaction():
    statuses = []

    def broken(status):
        raise RuntimeError("observer unavailable")

    agent = SimpleNamespace(compaction_callback=broken, status_callback=lambda *args: statuses.append(args))
    cycle = _CompactionLifecycle(agent, True)
    cycle.commit_status = "committed"
    cycle.complete()
    assert statuses[0][0] == "compacted"


def test_late_completion_keeps_its_original_observer():
    original, next_turn = [], []
    agent = SimpleNamespace(compaction_callback=original.append, status_callback=None)
    cycle = _CompactionLifecycle(agent, False)
    agent.compaction_callback = next_turn.append
    cycle.commit_status = "committed"
    cycle.complete()
    assert original == ["completed"]
    assert next_turn == []


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
def test_real_codex_compaction_reports_outcome_without_changing_transcript(outcome):
    from agent.conversation_compression import compress_context
    from agent.transports.codex_app_server_session import TurnResult
    from tests.agent.test_codex_app_server_compaction import DummyAgent

    result = TurnResult(error="provider failed" if outcome == "failed" else None,
                        interrupted=outcome == "cancelled")
    agent = DummyAgent(result)
    observed = []
    agent.compaction_callback = observed.append
    messages = [{"role": "user", "content": "earlier context"}]
    returned, _ = compress_context(agent, messages, "system", approx_tokens=100000, force=True)
    assert returned is messages
    assert observed == [outcome]
