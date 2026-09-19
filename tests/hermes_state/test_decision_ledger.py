from hermes_state import SessionDB


def test_decision_ledger_is_fifo_capped_and_survives_reopen(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("session", source="test")
    for index in range(52):
        db.append_decision_ledger_entry("session", "correction", f"choice {index}", turn_id="turn")
    assert db.get_decision_ledger_entries("session") == [{"kind": "correction", "text": f"choice {index}", "turn_id": "turn"} for index in range(2, 52)]
    db.close()
    reopened = SessionDB(path)
    reopened.create_session("child", source="test", parent_session_id="session")
    reopened.copy_decision_ledger_entries("session", "child")
    assert reopened.get_decision_ledger_entries("child") == reopened.get_decision_ledger_entries("session")
    reopened.close()
