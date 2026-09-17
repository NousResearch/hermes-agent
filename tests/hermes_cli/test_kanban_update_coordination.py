"""Update hand-off coordination for the kanban dispatcher."""

from __future__ import annotations

import os
import time


def test_live_update_pauses_dispatch_and_quiesce_reclaims_without_failure(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch
    from hermes_cli import kanban_update_coordination as coordination

    with kbc.connect_closing() as conn:
        running = kb.create_task(conn, title="in flight", assignee="dev")
        queued = kb.create_task(conn, title="queued", assignee="dev")
        for ended_at in (1, 2):
            conn.execute(
                "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at, metadata) "
                "VALUES (?, 'crashed', 'crashed', ?, ?, ?)",
                (running, ended_at, ended_at, '{"protocol_violation": true}'),
            )
        assert kb.claim_task(conn, running, claimer=f"{kb._host_prefix()}update-test") is not None
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_started_at = NULL, consecutive_failures = 2 "
            "WHERE id = ?",
            (999_999_999, running),
        )
        conn.commit()

        monkeypatch.setattr(coordination, "update_dispatch_paused", lambda: True)
        spawned = []
        result = dispatch.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: spawned.append(args))

        assert result.skipped_update is True
        assert spawned == []
        assert kb.get_task(conn, queued).status == "ready"

    monkeypatch.setattr(
        dispatch,
        "_terminate_reclaimed_worker",
        lambda *_args, **_kwargs: {
            "host_local": True,
            "termination_attempted": True,
            "terminated": True,
            "tree_termination_attempted": True,
            "tree_signal_succeeded": True,
            "tree_terminated": True,
        },
    )
    outcome = coordination.quiesce_all_workers()

    assert outcome == {
        "ok": True,
        "reclaimed": [{"board": "default", "task_id": running}],
        "failed": [],
    }
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, running)
        assert task.status == "ready"
        assert task.consecutive_failures == 2
        assert task.worker_pid is None
        assert dispatch._protocol_violation_streak(conn, running) == 2


def test_unknown_update_marker_state_keeps_dispatch_paused(monkeypatch):
    from hermes_cli import kanban_update_coordination as coordination
    from hermes_cli import update_lock

    monkeypatch.setattr(update_lock, "read_update_marker_state", lambda: "unknown")

    assert coordination.update_dispatch_paused() is True


def test_stale_partial_update_marker_allows_dispatch(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    marker = home / ".hermes-update-in-progress"
    marker.write_text("12345", encoding="utf-8")

    from hermes_cli import kanban_update_coordination as coordination
    from hermes_cli.update_lock import UPDATE_MARKER_MAX_AGE_SECONDS

    stale_time = time.time() - UPDATE_MARKER_MAX_AGE_SECONDS - 60
    os.utime(marker, (stale_time, stale_time))

    assert coordination.update_dispatch_paused() is False
    assert marker.exists(), "authorization is logical; only the update owner removes the marker"


def test_quiesce_retains_claim_when_worker_tree_survives(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch
    from hermes_cli import kanban_update_coordination as coordination

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="surviving worker", assignee="dev")
        claim = f"{kb._host_prefix()}update-test"
        assert kb.claim_task(conn, task_id, claimer=claim) is not None
        conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (12345, task_id))
        conn.commit()

    monkeypatch.setattr(coordination, "update_dispatch_paused", lambda: True)
    monkeypatch.setattr(
        dispatch,
        "_terminate_reclaimed_worker",
        lambda *_args, **_kwargs: {
            "host_local": True,
            "termination_attempted": True,
            "terminated": False,
            "tree_termination_attempted": True,
            "tree_terminated": False,
        },
    )

    outcome = coordination.quiesce_all_workers()

    assert outcome["ok"] is False
    assert outcome["failed"][0]["task_id"] == task_id
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.claim_lock == claim


def test_quiesce_retains_claim_when_tree_kill_has_no_receipt(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch
    from hermes_cli import kanban_update_coordination as coordination

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="unproven tree", assignee="dev")
        claim = f"{kb._host_prefix()}update-test"
        assert kb.claim_task(conn, task_id, claimer=claim) is not None
        conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (12345, task_id))
        conn.commit()

    monkeypatch.setattr(coordination, "update_dispatch_paused", lambda: True)
    monkeypatch.setattr(
        dispatch,
        "_terminate_reclaimed_worker",
        lambda *_args, **_kwargs: {
            "host_local": True,
            "termination_attempted": True,
            "terminated": True,
            "tree_termination_attempted": True,
            "tree_signal_succeeded": False,
            "tree_terminated": False,
        },
    )

    assert coordination.quiesce_all_workers()["ok"] is False
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.claim_lock == claim


def test_quiesce_retains_claim_without_worker_pid(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_update_coordination as coordination

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="unpublished pid", assignee="dev")
        claim = f"{kb._host_prefix()}update-test"
        assert kb.claim_task(conn, task_id, claimer=claim) is not None

    monkeypatch.setattr(coordination, "update_dispatch_paused", lambda: True)

    assert coordination.quiesce_all_workers()["ok"] is False
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.claim_lock == claim


def test_quiesce_leaves_foreign_host_claims_owned(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_update_coordination as coordination

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="remote worker", assignee="dev")
        assert kb.claim_task(conn, task_id, claimer="FOREIGN-HOST:12345") is not None

    monkeypatch.setattr(coordination, "update_dispatch_paused", lambda: True)
    assert coordination.quiesce_all_workers() == {"ok": True, "reclaimed": [], "failed": []}

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.claim_lock == "FOREIGN-HOST:12345"


def test_quiesce_accepts_task_completion_during_reclaim(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_update_coordination as coordination

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="finishing worker", assignee="dev")
        assert kb.claim_task(conn, task_id, claimer=f"{kb._host_prefix()}update-test") is not None

    def complete_instead_of_quiesce(conn, finishing_task_id):
        conn.execute(
            "UPDATE tasks SET status = 'done', claim_lock = NULL, claim_expires = NULL "
            "WHERE id = ?",
            (finishing_task_id,),
        )
        conn.commit()
        return False

    monkeypatch.setattr(coordination, "update_dispatch_paused", lambda: True)
    monkeypatch.setattr(coordination, "_quiesce_task_for_update", complete_instead_of_quiesce)

    assert coordination.quiesce_all_workers() == {"ok": True, "reclaimed": [], "failed": []}
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "done"
