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
