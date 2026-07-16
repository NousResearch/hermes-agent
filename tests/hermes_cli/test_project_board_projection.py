"""Behavior tests for explicit active and historical project projections."""
from datetime import datetime, timedelta, timezone

from hermes_cli.project_board_projection import (
    build_project_projection,
    historical_project_projection,
)


NOW = datetime(2026, 7, 16, 12, tzinfo=timezone.utc)


def _task(task_id, status, title=None, **extra):
    return {"id": task_id, "title": title or task_id, "status": status, **extra}


def _member(task_id, root="root", kind="required", generation=1):
    return {"task_id": task_id, "root_task_id": root, "membership_kind": kind, "generation": generation}


def test_active_projection_is_explicit_and_keeps_required_running_repairs_checker_and_blockers():
    tasks = [
        _task("root", "running"),
        _task("required", "todo"),
        _task("worker", "running"),
        _task("repair", "ready"),
        _task("checker", "review"),
        _task("blocked", "blocked"),
        _task("done-checker", "done"),
        _task("superseded-repair", "ready", superseded=True),
        _task("placeholder", "archived", run_count=0),
        _task("unrelated", "running", title="root required worker"),
    ]
    members = [
        _member("required"), _member("worker", kind="support"), _member("repair", kind="repair"),
        _member("checker", kind="checker"), _member("blocked"), _member("done-checker", kind="checker"),
        _member("superseded-repair", kind="repair"), _member("placeholder", kind="support"),
    ]
    projection = build_project_projection(tasks, memberships=members, now=NOW)
    assert projection.task_ids == ("blocked", "checker", "repair", "required", "root", "worker")
    assert "unrelated" not in projection.task_ids
    excluded = {(item["task_id"], item["reason"]) for item in projection.excluded}
    assert ("done-checker", "completed_checker") in excluded
    assert ("superseded-repair", "current_repair") in excluded
    assert any(item["task_id"] == "placeholder" and item["reason"] == "archived_zero_run_placeholder" for item in projection.excluded)


def test_historical_projection_retains_terminal_work_but_still_requires_membership():
    tasks = [_task("root", "done"), _task("old-worker", "archived", run_count=2), _task("not-owned", "done")]
    projection = historical_project_projection(tasks, memberships=[_member("old-worker")], now=NOW)
    assert projection.task_ids == ("old-worker", "root")
    assert "not-owned" not in projection.task_ids
    assert all(item.reason == "historical_explicit_membership" for item in projection.entries)


def test_old_terminal_root_is_excluded_only_when_policy_is_configured():
    tasks = [_task("root", "done"), _task("old", "done")]
    finalizations = [{"root_task_id": "root", "finalized_at": int((NOW - timedelta(days=4)).timestamp())}]
    members = [_member("old")]
    active = build_project_projection(
        tasks, memberships=members, finalizations=finalizations, now=NOW,
        active_root_max_age=timedelta(days=3),
    )
    assert active.task_ids == ()
    assert active.excluded == ({"root_task_id": "root", "reason": "terminal_root_expired"},)
    without_policy = build_project_projection(tasks, memberships=members, finalizations=finalizations, now=NOW)
    assert without_policy.task_ids == ("root",)


def test_naive_now_and_invalid_membership_are_rejected_fail_closed():
    try:
        build_project_projection([], now=datetime(2026, 7, 16, 12))
    except ValueError as exc:
        assert "timezone-aware" in str(exc)
    else:
        raise AssertionError("naive now must be refused")
    projection = build_project_projection([_task("root", "running")], memberships=[{"task_id": "x"}])
    assert projection.task_ids == ()
