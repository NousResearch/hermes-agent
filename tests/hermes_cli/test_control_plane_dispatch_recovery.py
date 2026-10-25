"""Control-plane isolated dispatch and recovery gates; never touches the live board."""
from unittest.mock import patch
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch


def test_dispatch_and_recover_needs_input_without_resurrection(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    kb.init_db()
    spawned = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 424242

    with kbc.connect_closing() as conn:
        first = kb.create_task(conn, title="dependency-owner", assignee="worker")
        second = kb.create_task(conn, title="dependent", assignee="worker")
        kb.link_tasks(conn, parent_id=first, child_id=second)
        kb.recompute_ready(conn)
        with patch("hermes_cli.profiles.profile_exists", return_value=True):
            tick = dispatch.dispatch_once(conn, spawn_fn=spawn, max_in_progress=2)
        assert first in spawned and second not in spawned
        assert any(row[0] == first for row in tick.spawned)
        assert kb.get_task(conn, first).status == "running"
        run = kb.get_task(conn, first).current_run_id
        assert kb.complete_task(conn, first, result="ok", expected_run_id=run)
        assert kb.get_task(conn, second).status == "ready"
        claimed = kb.claim_task(conn, second, claimer="worker")
        assert claimed and claimed.status == "running"
        assert kb.block_task(conn, second, reason="needs decision", kind="needs_input")
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, second).status == "blocked"
        assert kb.unblock_task(conn, second)
        asserted = kb.claim_task(conn, second, claimer="worker")
        assert asserted and asserted.status == "running"
        assert kb.complete_task(conn, second, result="done", expected_run_id=asserted.current_run_id)
        assert kb.get_task(conn, second).status == "done"
