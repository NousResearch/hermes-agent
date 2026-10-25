"""A coordinator may close a triaged block without dispatching another attempt."""
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc


def test_triaged_card_can_be_completed_without_unblock_or_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Resolved during triage", triage=True)
        assert kb.get_task(conn, tid).status == "triage"
        assert kb.complete_task(conn, tid, result="Prerequisite resolved independently")
        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert task.result == "Prerequisite resolved independently"
        assert [ev.kind for ev in kb.list_events(conn, tid)].count("completed") == 1
        assert not kb.complete_task(conn, tid, result="Duplicate")
