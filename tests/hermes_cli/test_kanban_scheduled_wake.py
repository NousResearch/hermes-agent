"""Scheduled follow-ups must not wait forever for a manual unblock."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


NOW = 1791748200


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kbd, "_live_worker_procs", {})
    monkeypatch.setattr(kb.time, "time", lambda: NOW)
    with kbc.connect_closing() as conn:
        yield conn


def tick(conn, **kwargs):
    # Exercise the real tick, including promotion, without starting any worker.
    return kbd.dispatch_once(conn, max_spawn=0, **kwargs)


def park(conn, reason, **kwargs):
    task_id = kb.create_task(conn, title="follow-up", assignee="ops", **kwargs)
    assert kb.schedule_task(conn, task_id, reason=reason)
    return task_id


@pytest.mark.parametrize("deadline", [
    "2026-10-11T22:19:00+11:00.",
    "2026-10-11T11:19:00Z",
    "2026-10-12T06:50:00+11:00",
])
def test_past_or_exact_deadline_wakes_next_tick(board, deadline):
    task_id = park(board, f"SCHEDULED_UNTIL={deadline} ready to recheck")
    tick(board)
    assert kb.get_task(board, task_id).status == "ready"
    events = kb.list_events(board, task_id)
    wakes = [e for e in events if e.kind == "scheduled_woke"]
    assert len(wakes) == 1
    assert wakes[0].payload["trigger"] == "deadline"
    tick(board)
    assert len([e for e in kb.list_events(board, task_id) if e.kind == "scheduled_woke"]) == 1


def test_future_deadline_stays_even_with_wake_condition(board):
    task_id = park(board, "SCHEDULED_UNTIL=2026-10-13T00:00:00+11:00 WAKE_ON=install window")
    tick(board)
    assert kb.get_task(board, task_id).status == "scheduled"


@pytest.mark.parametrize("reason", [
    "SCHEDULED_UNTIL=not-a-date",
    "SCHEDULED_UNTIL=2026-10-12T06:50:00",  # ambiguous timezone
    "SCHEDULED_UNTIL=2026-99-12T06:50:00Z WAKE_ON=window",
    "SCHEDULED_UNTIL=",
    "SCHEDULED_UNTIL=2026-10-13T00:00:00Z SCHEDULED_UNTIL=2026-10-14T00:00:00Z",
    "WAKE_ON=",
    "run next week",
    None,
])
def test_invalid_or_missing_marker_is_surfaced_not_parked(board, reason):
    task_id = park(board, reason)
    tick(board)
    task = kb.get_task(board, task_id)
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"
    assert "Invalid scheduled wake" in task.last_failure_error
    assert any(e.kind == "blocked" and e.payload["source"] == "scheduled_wake"
               for e in kb.list_events(board, task_id))
    tick(board)
    assert kb.get_task(board, task_id).status == "blocked"


def test_condition_recheck_due_at_thirty_minutes_and_reschedule_resets_clock(board, monkeypatch):
    task_id = park(board, "WAKE_ON=supervisor confirms the window")
    monkeypatch.setattr(kb.time, "time", lambda: NOW + 1799)
    tick(board)
    assert kb.get_task(board, task_id).status == "scheduled"
    monkeypatch.setattr(kb.time, "time", lambda: NOW + 1800)
    tick(board)
    assert kb.get_task(board, task_id).status == "ready"
    assert [e for e in kb.list_events(board, task_id) if e.kind == "scheduled_woke"][0].payload["trigger"] == "condition_recheck"
    assert "recheck" in kb.list_comments(board, task_id)[-1].body.lower()
    assert kb.schedule_task(board, task_id, reason="WAKE_ON=still awaiting confirmation")
    tick(board)
    assert kb.get_task(board, task_id).status == "scheduled"
    monkeypatch.setattr(kb.time, "time", lambda: NOW + 3600)
    tick(board)
    assert kb.get_task(board, task_id).status == "ready"


def test_wake_preserves_parent_gate_and_does_not_release_children(board):
    parent = kb.create_task(board, title="prerequisite", assignee="ops")
    task_id = park(board, "SCHEDULED_UNTIL=2026-10-11T22:19:00+11:00", parents=[parent])
    child = kb.create_task(board, title="dependent", assignee="ops", parents=[task_id])
    tick(board)
    assert kb.get_task(board, task_id).status == "todo"
    assert kb.get_task(board, child).status == "todo"
    assert kb.complete_task(board, parent, summary="fixture prerequisite finished")
    tick(board)
    assert kb.get_task(board, task_id).status == "ready"
    assert kb.get_task(board, child).status == "todo"


def test_latest_schedule_replaces_old_deadline_and_preserves_retry_budget(board):
    task_id = park(board, "SCHEDULED_UNTIL=2026-10-11T22:19:00+11:00")
    assert kb.unblock_task(board, task_id)
    assert kb.schedule_task(board, task_id, reason="SCHEDULED_UNTIL=2026-10-13T00:00:00Z")
    board.execute("UPDATE tasks SET consecutive_failures = 1 WHERE id = ?", (task_id,))
    tick(board)
    assert kb.get_task(board, task_id).status == "scheduled"
    assert kb.unblock_task(board, task_id)
    assert kb.schedule_task(board, task_id, reason="SCHEDULED_UNTIL=2026-10-11T22:19:00+11:00")
    board.execute("UPDATE tasks SET consecutive_failures = 1 WHERE id = ?", (task_id,))
    tick(board)
    assert kb.get_task(board, task_id).consecutive_failures == 1


def test_dry_run_does_not_wake_or_surface_invalid_schedule(board):
    due = park(board, "SCHEDULED_UNTIL=2026-10-11T22:19:00+11:00")
    invalid = park(board, "SCHEDULED_UNTIL=bad")
    tick(board, dry_run=True)
    assert kb.get_task(board, due).status == "scheduled"
    assert kb.get_task(board, invalid).status == "scheduled"


def test_missing_event_is_surfaced_without_stalling_due_sibling(board):
    legacy = kb.create_task(board, title="legacy park", assignee="ops")
    board.execute("UPDATE tasks SET status = 'scheduled' WHERE id = ?", (legacy,))
    due = park(board, "SCHEDULED_UNTIL=2026-10-11T22:19:00+11:00")
    tick(board)
    assert kb.get_task(board, legacy).status == "blocked"
    assert "missing scheduled event" in kb.get_task(board, legacy).last_failure_error
    assert kb.get_task(board, due).status == "ready"


def test_due_card_dispatches_in_same_tick_without_bypassing_profile_cap(board, monkeypatch):
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _: True)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "normal")
    task_id = park(board, "SCHEDULED_UNTIL=2026-10-11T22:19:00+11:00")
    spawned = []
    result = kbd.dispatch_once(board, max_spawn=1, max_in_progress_per_profile=1,
                               spawn_fn=lambda task, workspace: spawned.append(task.id))
    assert spawned == [task_id]
    assert result.spawned[0][0] == task_id
    assert kb.get_task(board, task_id).status == "running"
    second = park(board, "SCHEDULED_UNTIL=2026-10-11T22:19:00+11:00")
    result = kbd.dispatch_once(board, max_spawn=2, max_in_progress_per_profile=1,
                               spawn_fn=lambda task, workspace: spawned.append(task.id))
    assert spawned == [task_id]
    assert kb.get_task(board, second).status == "ready"
    assert result.skipped_per_profile_capped[0][0] == second
