from types import SimpleNamespace

from hermes_cli.kanban_block_action import build_block_action, reason_from_events


def task(**overrides):
    data = {
        "body": "", "block_kind": "transient", "assignee": "systems",
        "created_by": "creator", "last_failure_error": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_needs_input_is_an_explicit_matt_action():
    action = build_block_action(task(block_kind="needs_input"), reason="Choose A or B")
    assert action.disposition == "Matt action required"
    assert action.action_required is True
    assert action.owner == "Matt"
    assert action.action == "Choose A or B"
    assert action.auto_resume is False


def test_dependency_is_internal_and_auto_resumes():
    action = build_block_action(task(block_kind="dependency", assignee="worker"), reason="Waiting on parent")
    assert action.disposition == "Dependency"
    assert action.action_required is False
    assert action.owner == "worker"
    assert action.auto_resume is True
    assert "parent" in action.retry_condition.lower()


def test_capability_and_transient_are_not_labeled_as_human_actions():
    capability = build_block_action(task(block_kind="capability"), reason="Missing integration")
    transient = build_block_action(task(block_kind="transient"), reason="Worker timed out")
    assert capability.disposition == "Internal owner action"
    assert transient.disposition == "Stale/recovery"
    assert capability.action_required is transient.action_required is False


def test_latest_block_reason_wins():
    events = [
        SimpleNamespace(kind="blocked", payload={"reason": "old"}),
        SimpleNamespace(kind="status", payload={"status": "ready"}),
        SimpleNamespace(kind="blocked", payload='{"reason": "current"}'),
    ]
    assert reason_from_events(events) == "current"


def test_new_reasonless_block_does_not_reuse_repaired_history():
    events = [
        SimpleNamespace(kind="blocked", payload={"reason": "old approval"}),
        SimpleNamespace(kind="unblocked", payload={}),
        SimpleNamespace(kind="blocked", payload={}),
    ]
    assert reason_from_events(events) is None


def test_historical_failure_is_not_used_as_current_block_action():
    action = build_block_action(
        task(block_kind="needs_input", last_failure_error="old worker failure"),
        reason=None,
    )
    assert action.action == "Resolve the blocker recorded on the task."
