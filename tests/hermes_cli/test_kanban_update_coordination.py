"""Update hand-off coordination for the kanban dispatcher."""

from __future__ import annotations


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
        assert kb.claim_task(conn, running, claimer=f"{kb._host_prefix()}update-test") is not None
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_started_at = NULL WHERE id = ?",
            (999_999_999, running),
        )
        conn.commit()

        monkeypatch.setattr(coordination, "update_dispatch_paused", lambda: True)
        spawned = []
        result = dispatch.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: spawned.append(args))

        assert result.skipped_update is True
        assert spawned == []
        assert kb.get_task(conn, queued).status == "ready"

    outcome = coordination.quiesce_all_workers()

    assert outcome == {
        "ok": True,
        "reclaimed": [{"board": "default", "task_id": running}],
        "failed": [],
    }
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, running)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert task.worker_pid is None


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

    def complete_instead_of_reclaim(conn, finishing_task_id, **_kwargs):
        conn.execute(
            "UPDATE tasks SET status = 'done', claim_lock = NULL, claim_expires = NULL "
            "WHERE id = ?",
            (finishing_task_id,),
        )
        conn.commit()
        return False

    monkeypatch.setattr(coordination, "update_dispatch_paused", lambda: True)
    monkeypatch.setattr(kb, "reclaim_task", complete_instead_of_reclaim)

    assert coordination.quiesce_all_workers() == {"ok": True, "reclaimed": [], "failed": []}
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "done"
