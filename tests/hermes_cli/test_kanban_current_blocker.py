"""Offline evidence and human-output contracts for the current blocking episode."""
from pathlib import Path

import pytest

from hermes_cli import kanban_diagnostics as kd


TASK_ID = "t_demo00"


def task(**fields):
    return {"id": TASK_ID, "status": "blocked", "consecutive_failures": 0, **fields}


def event(kind="blocked", reason="needs approval", **fields):
    return {"id": 1, "task_id": TASK_ID, "run_id": None, "kind": kind,
            "payload": {"reason": reason}, "created_at": 100, **fields}


def current(events, runs=None, task_fields=None):
    assert Path(kd.__file__).resolve().parents[1] == Path(__file__).resolve().parents[2]
    return [d for d in kd.compute_task_diagnostics(
        task(**(task_fields or {})), events, runs or [], now=101
    ) if d.kind == "current_blocker"]


@pytest.mark.parametrize("with_kind", [False, True])
def test_current_blocker_is_visible_in_human_cli(capsys, with_kind):
    from hermes_cli.kanban import _print_diagnostics
    _print_diagnostics(current([event()]), "", with_kind=with_kind)
    line = capsys.readouterr().out.strip().splitlines()[0]
    assert "Blocked: needs approval" in line


@pytest.mark.parametrize("status", ["todo", "ready", "running", "done", "archived",
                                   "review", "scheduled", "triage", None, "", 7])
def test_current_blocker_requires_blocked_status(status):
    assert current([event()], task_fields={"status": status}) == []
    missing = task()
    del missing["status"]
    assert not [d for d in kd.compute_task_diagnostics(missing, [event()], [], now=101)
                if d.kind == "current_blocker"]


@pytest.mark.parametrize("events", [[], [event(reason="")]])
def test_current_blocker_never_uses_failure_fields_or_runs(events):
    runs = [{"id": 7, "task_id": TASK_ID, "summary": "run summary",
             "error": "run error", "outcome": "blocked"}]
    assert current(events, runs, {"last_failure_error": "failure error",
                                 "last_spawn_error": "legacy error"}) == []


def test_current_blocker_exposes_ordinary_reason():
    diags = current([event()])
    assert len(diags) == 1
    assert diags[0].to_dict() == {
        "kind": "current_blocker", "severity": "warning",
        "title": "Blocked: needs approval", "detail": "needs approval",
        "actions": [{"kind": "comment", "label": "Review and discuss the blocker",
                     "payload": {}, "suggested": True}],
        "first_seen_at": 100, "last_seen_at": 100, "count": 1, "run_id": None,
        "data": {"source_event_kind": "blocked", "source_event_id": 1},
    }
