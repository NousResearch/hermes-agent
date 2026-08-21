"""Process-free fake-provider lifecycle certification.

This harness exercises the real GoalManager persistence and continuation lease
boundary without making a network/provider claim. The provider is deterministic;
all lifecycle decisions remain owned by GoalManager.
"""
from collections import deque

from hermes_cli import goals


def _manager(monkeypatch, tmp_path, session_id="fake-e2e"):
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    return goals.GoalManager(session_id, default_max_turns=4)


class FakeProvider:
    def __init__(self, outputs):
        self.outputs = deque(outputs)
        self.calls = 0

    def turn(self):
        self.calls += 1
        return self.outputs.popleft()


def test_fake_provider_budget_restart_enqueue_loss_race_and_completion(monkeypatch, tmp_path):
    provider = FakeProvider(["", "partial work", "final verified work"])
    mgr = _manager(monkeypatch, tmp_path)
    mgr.set("finish two bounded tasks", max_turns=4)

    # First provider boundary: empty output must checkpoint and request a
    # continuation without pretending that the goal is complete.
    first = mgr.evaluate_after_turn(
        provider.turn(),
        turn_outcome=goals.EXECUTION_FAILED,
        turn_metadata={"reason": "fake provider empty output"},
    )
    assert first["verdict"] == "continuation_required"
    assert mgr.state.continuation_pending is True
    assert mgr.state.turns_used == 1
    goal_id = mgr.state.goal_id

    # Restart/reload and owner race: exactly one owner may claim the pending
    # checkpoint. A queue loss after release leaves it recoverable.
    restarted = goals.GoalManager("fake-e2e")
    duplicate = goals.GoalManager("fake-e2e")
    assert restarted.state.goal_id == goal_id
    assert restarted.claim_continuation("owner-a") is True
    assert duplicate.claim_continuation("owner-b") is False
    assert restarted.release_continuation(queued=True) is True
    assert goals.GoalManager("fake-e2e").state.continuation_pending is True

    # Simulate enqueue loss/process death: a fresh owner can reclaim, then the
    # synthetic continuation starts without a user prompt.
    recovered = goals.GoalManager("fake-e2e")
    assert recovered.claim_continuation("owner-after-restart") is True
    assert recovered.release_continuation(queued=True) is True
    assert recovered.start_continuation() is True

    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *args, **kwargs: ("continue", "one task remains", False, None, False),
    )
    second = recovered.evaluate_after_turn(provider.turn(), user_initiated=False)
    assert second["should_continue"] is True
    assert recovered.state.turns_used == 2
    assert recovered.state.continuation_pending is True

    assert recovered.claim_continuation("owner-final") is True
    assert recovered.release_continuation(queued=True) is True
    assert recovered.start_continuation() is True

    # A model completion claim is not completion authority. Explicit evidence
    # is required, and after that transition no queued continuation is valid.
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *args, **kwargs: ("done", "model claims done", False, None, False),
    )
    waiting = recovered.evaluate_after_turn(provider.turn(), user_initiated=False)
    assert waiting["verdict"] == "waiting_for_authority"
    assert recovered.state.status == "active"
    assert recovered.confirm_completion("receipt=fake-provider-verified", source="test") is True
    assert recovered.state.outcome == goals.GOAL_COMPLETED
    assert recovered.start_continuation() is False
    assert provider.calls == 3
    assert goals.GoalManager("fake-e2e").state.goal_id == goal_id
    assert goals.GoalManager("fake-e2e").state.turns_used == 3


def test_fake_provider_hard_turn_budget_stops_without_hidden_reset(monkeypatch, tmp_path):
    mgr = _manager(monkeypatch, tmp_path, session_id="fake-budget")
    mgr.set("bounded budget", max_turns=1)
    decision = mgr.evaluate_after_turn(
        "partial",
        turn_outcome=goals.TURN_BUDGET_EXHAUSTED,
        turn_metadata={"reason": "fake provider reached the hard turn boundary"},
    )
    assert decision["should_continue"] is False
    assert mgr.state.status == "paused"
    assert mgr.state.outcome == goals.TURN_BUDGET_EXHAUSTED
    assert mgr.state.turns_used == 1
    assert mgr.state.continuation_pending is False
