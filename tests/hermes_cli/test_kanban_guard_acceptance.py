"""Regression for #121651: durable, independently held dispatch guards."""

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def test_rejected_completion_keeps_auth_hold_and_coalesces_ticks(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _: True)
    kbc.init_db()

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="held", assignee="default", workspace_kind="scratch",
            completion_contract="https://github.com/example/widgets/pull/1",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", ("401 auth failed", tid))
        spawned = []

        def spawn(task, workspace, board=None):
            spawned.append(task.id)
            return None

        for _ in range(2):
            kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1, reconcile_orphans=False)
        assert spawned == []
        assert kb.get_task(conn, tid).guard_reason == "blocker_auth"
        assert kb.get_task(conn, tid).guard_count == 2
        assert conn.execute(
            "SELECT count(*) FROM task_events WHERE task_id=? AND kind='respawn_guarded'", (tid,),
        ).fetchone()[0] == 1

        assert not kb.complete_task(
            conn, tid, summary="rejected", metadata={
                "published_pr": "https://github.com/other/repo/pull/2",
            },
        )
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.last_failure_error == "401 auth failed"
        assert task.acceptance_rejected
        assert kbd.check_respawn_guard(conn, tid) == "blocker_auth"
        kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1, reconcile_orphans=False)
        assert spawned == []
        assert kb.get_task(conn, tid).guard_count == 3
        assert conn.execute(
            "SELECT count(*) FROM task_events WHERE task_id=? AND kind='respawn_guarded'", (tid,),
        ).fetchone()[0] == 1

        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error=NULL WHERE id=?", (tid,))
        assert kbd.check_respawn_guard(conn, tid) == "acceptance_rejected"
        kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1, reconcile_orphans=False)
        assert spawned == []
        assert kb.get_task(conn, tid).guard_count == 1
        assert conn.execute(
            "SELECT count(*) FROM task_events WHERE task_id=? AND kind='respawn_guarded'", (tid,),
        ).fetchone()[0] == 2

        assert kb.block_task(conn, tid, reason="operator review")
        assert kb.unblock_task(conn, tid)
        assert not kb.get_task(conn, tid).acceptance_rejected
        assert kbd.check_respawn_guard(conn, tid) is None
        kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1, reconcile_orphans=False)
        assert kb.get_task(conn, tid).guard_reason is None
        assert kb.get_task(conn, tid).guard_count == 0
        assert conn.execute(
            "SELECT count(*) FROM task_events WHERE task_id=? AND kind='respawn_guard_cleared'", (tid,),
        ).fetchone()[0] == 1
        assert spawned == [tid]
