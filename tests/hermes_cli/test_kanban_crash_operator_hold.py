
from hermes_cli import kanban_db_connect, kanban_db_dispatch
"""A genuine worker crash is held for operator disposition, not re-run."""


from hermes_cli import kanban_db as kb


# Upstream removed the direct-Claude worker launcher; native worker argv is
# covered by test_delivery_review_reconciliation.py.


def test_nonzero_worker_crash_is_blocked_and_not_dispatchable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "crash.db"))
    kanban_db_connect.init_db()
    with kanban_db_connect.connect_closing() as conn:
        task_id = kb.create_task(conn, title="unsafe retry", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer=f"{kb._claimer_id().split(':', 1)[0]}:test")
        assert claimed is not None
        kanban_db_dispatch._set_worker_pid(conn, task_id, 424242)
        conn.execute("UPDATE tasks SET started_at = 1 WHERE id = ?", (task_id,))
        conn.commit()

        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        monkeypatch.setattr(kanban_db_dispatch, "_classify_worker_exit", lambda _pid: ("nonzero_exit", 17))
        monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)

        assert kanban_db_dispatch.detect_crashed_workers(conn) == [task_id]
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"
        assert task.claim_lock is None
        assert task.worker_pid is None
        crash = [event for event in kb.list_events(conn, task_id) if event.kind == "crashed"][-1]
        assert crash.payload["operator_held"] is True
        assert crash.payload["retry_status"] == "ready"

        dispatched = kanban_db_dispatch.dispatch_once(conn, max_spawn=1, spawn_fn=lambda *_args: 999)
        assert dispatched.spawned == []


def test_rate_limit_exit_keeps_intentional_cooldown_requeue(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "rate-limit.db"))
    kanban_db_connect.init_db()
    with kanban_db_connect.connect_closing() as conn:
        task_id = kb.create_task(conn, title="quota wait", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer=f"{kb._claimer_id().split(':', 1)[0]}:test")
        assert claimed is not None
        kanban_db_dispatch._set_worker_pid(conn, task_id, 424243)
        conn.execute("UPDATE tasks SET started_at = 1 WHERE id = ?", (task_id,))
        conn.commit()

        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        monkeypatch.setattr(
            kanban_db_dispatch,
            "_classify_worker_exit",
            lambda _pid: ("rate_limited", kb.KANBAN_RATE_LIMIT_EXIT_CODE),
        )
        monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)

        assert kanban_db_dispatch.detect_crashed_workers(conn) == []
        assert kb.get_task(conn, task_id).status == "ready"
