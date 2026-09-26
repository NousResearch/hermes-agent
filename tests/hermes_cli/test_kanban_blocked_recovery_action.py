"""Stale-block diagnostics expose the existing operator recovery control."""
import pytest
from hermes_cli.kanban_diagnostics import compute_task_diagnostics


def blocked_diagnostics(*, status="blocked", events=None, hours=24):
    return [d.to_dict() for d in compute_task_diagnostics(
        {"id": "t_stale", "status": status},
        events if events is not None else [{"kind": "blocked", "created_at": 100}],
        [], now=172900, config={"blocked_stale_hours": hours},
    ) if d.kind == "stuck_in_blocked"]


def test_stale_block_offers_unblock_without_removing_comment():
    [diagnostic] = blocked_diagnostics()
    actions = {a["kind"]: a for a in diagnostic["actions"]}
    assert "unblock" in actions
    assert "comment" in actions
    assert actions["comment"]["suggested"]  # Inspect/answer before releasing a hold.
    assert not actions["unblock"]["suggested"]


@pytest.mark.parametrize("kwargs", [
    {"status": "ready"}, {"events": []}, {"hours": 49},
    {"events": [{"kind": "blocked", "created_at": 100},
                {"kind": "commented", "created_at": 101}]},
    {"events": [{"kind": "blocked", "created_at": 100},
                {"kind": "unblocked", "created_at": 101}]},
])
def test_no_recovery_action_when_stale_signal_is_absent(kwargs):
    assert blocked_diagnostics(**kwargs) == []
