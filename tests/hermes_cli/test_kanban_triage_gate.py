"""Regression tests for the supported triage human-gate repair."""

from pathlib import Path

from hermes_cli import kanban_db as kb


def _home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="default").resolve()))
    kb.init_db()


def test_triage_gate_preserves_payload_history_and_records_rollback(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="approval", body="keep this", assignee="elon")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='triage', block_recurrences=3 WHERE id=?",
                (tid,),
            )
        before = kb.get_task(conn, tid)
        ok, error = kb.route_triage_to_human_gate(
            conn, tid, actor="operator", reason="SoLo approval required"
        )
        after = kb.get_task(conn, tid)
        assert (ok, error) == (True, None)
        assert after.status == "blocked"
        assert after.block_kind == "needs_input"
        assert after.title == before.title
        assert after.body == before.body
        assert after.assignee == before.assignee == "elon"
        assert after.block_recurrences == before.block_recurrences == 3
        event = [e for e in kb.list_events(conn, tid) if e.kind == "triage_human_gate"][-1]
        assert event.payload["rollback"] == {
            "status": "triage",
            "block_kind": None,
            "expected_status": "blocked",
        }
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"


def test_triage_gate_is_compare_and_set_and_dry_run_is_non_mutating(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="approval")
        ok, error = kb.route_triage_to_human_gate(
            conn, tid, actor="operator", reason="approval", dry_run=True
        )
        assert (ok, error) == (False, f"task {tid} is 'ready'; expected 'triage'")
        assert kb.get_task(conn, tid).status == "ready"
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
        ok, error = kb.route_triage_to_human_gate(
            conn, tid, actor="operator", reason="approval", dry_run=True
        )
        assert (ok, error) == (True, None)
        assert kb.get_task(conn, tid).status == "triage"
