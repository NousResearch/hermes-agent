import ast
import inspect

from hermes_state import SessionDB


def test_restore_ids_reactivates_only_inactive_ids_in_same_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="cli")
        db.create_session("s2", source="cli")
        active = db.append_message("s1", "user", "active")
        inactive = db.append_message("s1", "assistant", "inactive")
        other_session = db.append_message("s2", "user", "other")
        db._conn.execute("UPDATE messages SET active = 0 WHERE id IN (?, ?)", (inactive, other_session))

        assert db.restore_ids("s1", [active, inactive, other_session, 99999]) == 1
        rows1 = {m["id"]: m["active"] for m in db.get_messages("s1", include_inactive=True)}
        rows2 = {m["id"]: m["active"] for m in db.get_messages("s2", include_inactive=True)}
        assert rows1[active] == 1
        assert rows1[inactive] == 1
        assert rows2[other_session] == 0
        assert db.restore_ids("s1", [inactive]) == 0
    finally:
        db.close()


def test_restore_ids_empty_list_is_noop(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="cli")
        assert db.restore_ids("s1", []) == 0
    finally:
        db.close()


def test_restore_ids_uses_bounded_parameterized_in_clause():
    import hermes_state

    source = inspect.getsource(hermes_state.SessionDB.restore_ids)
    positive = "conn.execute(f'UPDATE x WHERE id IN ({placeholders})', (session_id, *bounded_ids))"
    assert ast.parse(positive)
    assert "bounded_ids = [int(i) for i in ids]" in source
    assert "id IN ({placeholders})" in source
    assert "(session_id, *bounded_ids)" in source
