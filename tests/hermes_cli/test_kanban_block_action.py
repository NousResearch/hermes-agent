from types import SimpleNamespace

import pytest

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


def test_verified_body_contract_overrides_generated_contract():
    body = """Source context.

```kanban-block-action
{"verified": true, "disposition": "Internal owner action", "owner": "Release manager", "action": "Publish the signed release manifest"}
```
"""
    action = build_block_action(
        task(block_kind="needs_input", body=body),
        reason="Choose staging or production",
    )

    assert action.disposition == "Internal owner action"
    assert action.action_required is False
    assert action.owner == "Release manager"
    assert action.action == "Publish the signed release manifest"
    assert action.reply_format == "No reply required."
    assert "internal capability gap" in action.consequence_if_no_action
    assert action.next_action.startswith("Release manager will resolve")


@pytest.mark.parametrize(
    "body",
    [
        "Owner: Release manager\nAction: Publish the signed release manifest",
        '```kanban-block-action\n{"verified": false, "disposition": "Internal owner action", '
        '"owner": "Release manager", "action": "Publish"}\n```',
        '```kanban-block-action\n{"verified": true, "owner": "Release manager", '
        '"action": "Publish"}\n```',
        '```kanban-block-action\n{"verified": true, "disposition": "Invented", '
        '"owner": "Release manager", "action": "Publish"}\n```',
    ],
)
def test_unverified_or_partial_body_text_cannot_override_contract(body):
    action = build_block_action(
        task(block_kind="needs_input", body=body),
        reason="Choose staging or production",
    )

    assert action.disposition == "Matt action required"
    assert action.action_required is True
    assert action.owner == "Matt"
    assert action.action == "Choose staging or production"


@pytest.mark.parametrize("kind", [None, "", "unknown", "transient"])
@pytest.mark.parametrize("reason", [None, ""])
def test_unknown_and_missing_kind_or_reason_use_safe_recovery_fallback(kind, reason):
    action = build_block_action(task(block_kind=kind), reason=reason)

    assert action.disposition == "Stale/recovery"
    assert action.action_required is False
    assert action.owner == "systems"
    assert action.action == "Resolve the blocker recorded on the task."
    assert action.reply_format == "No reply required."
    assert action.retry_condition
