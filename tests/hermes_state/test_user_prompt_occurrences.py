"""User occurrence folding is pre-page, user-only, and does not edit canonical rows."""
import json

import pytest

from hermes_state import SessionDB
from agent.context_compressor import MODEL_ONLY_DISPLAY_METADATA_KEY
from hermes_state_timeline import get_session_messages_around, get_session_timeline


def _seed(db):
    db.create_session("prompt", source="desktop")
    uid = "authored-turn"
    messages = [
        {"role": "user", "content": "first", "message_uid": uid},
        {"role": "tool", "content": "FULL" * 4000, "message_uid": "result"},
        {"role": "assistant", "content": "interim", "message_uid": uid},
        {"role": "user", "content": "second", "message_uid": uid},
        {"role": "tool", "content": "stub", "message_uid": "result"},
        {"role": "user", "content": "final", "message_uid": uid},
        {"role": "user", "content": "final", "message_uid": "fresh-turn"},
        {"role": "user", "content": "notice", "display_kind": "notification", "message_uid": uid},
        {"role": "user", "content": "hidden", "display_kind": "hidden", "message_uid": uid},
        {"role": "user", "content": "model-only", "display_metadata": {MODEL_ONLY_DISPLAY_METADATA_KEY: True}, "message_uid": uid},
    ]
    ids = db.append_messages_batch("prompt", [{**msg, "timestamp": 1700000000.0} for msg in messages])
    # Three generations, with archived tool payloads retained alongside the active stub.
    db._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET active = 0, compacted = 1 WHERE session_id = 'prompt' "
        "AND content IN ('first', 'second') OR (session_id = 'prompt' AND role = 'tool' AND content <> 'stub')"))
    return ids


@pytest.mark.parametrize("read_only", [False, True])
def test_prompt_occurrence_is_folded_before_all_display_pages(tmp_path, read_only):
    db = SessionDB(tmp_path / "state.db")
    _seed(db)
    raw = db.get_messages("prompt", include_inactive=True)
    if read_only:
        # Reproduce an already-populated index from the old content-key writer.
        db._execute_write(lambda conn: (conn.execute(
            "UPDATE messages SET display_order = id, display_identity = CAST(id AS BLOB) WHERE role = 'user'"),
            conn.execute("UPDATE messages SET display_order = (SELECT MIN(id) FROM messages WHERE content = 'final') WHERE content = 'final'"),
            conn.execute("DELETE FROM state_meta WHERE key = 'display_user_occurrence_version'")))
        db.close()
        db = SessionDB(tmp_path / "state.db", read_only=True)
    try:
        whole = db.get_messages("prompt", include_compacted=True)
        uid = next(m["message_uid"] for m in raw if m["content"] == "first")
        authored = [m for m in whole if m["role"] == "user" and not m["display_kind"]]
        assert [(m["message_uid"], m["content"]) for m in authored] == [(uid, "final"), ("fresh-turn", "final")]
        assert whole[0]["content"] == "final"
        pages = [m for offset in range(len(whole)) for m in db.get_messages(
            "prompt", include_compacted=True, limit=1, offset=offset)]
        assert [m["id"] for m in pages] == [m["id"] for m in whole]
        assert db.display_message_count("prompt") == len(whole)
        assert db.get_messages("prompt", include_compacted=True, latest=True, limit=3) == whole[-3:]
        assert [(m["id"], m["content"]) for m in whole if m["role"] == "tool"] == [
            (m["id"], m["content"]) for m in raw if m["role"] == "tool"]
        assert [(m["id"], m["content"]) for m in whole if m["role"] == "assistant"] == [
            (m["id"], m["content"]) for m in raw if m["role"] == "assistant"]
        assert db.get_messages("prompt", include_inactive=True) == raw
        timeline = get_session_timeline(db, "prompt")
        assert [m["row_id"] for m in timeline["entries"]] == [m["id"] for m in authored]
        around = get_session_messages_around(db, "prompt", authored[0]["id"], limit=3)
        assert around is not None
        assert around["pagination"]["offset"] == 0
        assert around["messages"] == whole[:3]
        assert not any(m["content"] == "model-only" for m in whole)
        assert any(m["display_kind"] == "notification" for m in whole)
        json.dumps(whole)
    finally:
        db.close()


def test_user_index_upgrade_is_cursor_bounded_and_changes_no_canonical_rows(tmp_path, monkeypatch):
    import hermes_state_schema as schema
    db = SessionDB(tmp_path / "state.db")
    _seed(db)
    before = db.get_messages("prompt", include_inactive=True)
    controls = db._read_all("SELECT id, display_order, display_identity FROM messages WHERE role <> 'user'")
    # Already indexed legacy users, including a wrongly merged identical fresh UID.
    db._execute_write(lambda conn: (conn.execute(
        "UPDATE messages SET display_order = id, display_identity = CAST(id AS BLOB) WHERE role = 'user'"),
        conn.execute("UPDATE messages SET display_order = (SELECT MIN(id) FROM messages WHERE content = 'final') "
                     "WHERE content = 'final'"),
        conn.execute("DELETE FROM state_meta WHERE key = 'display_user_occurrence_version'")))
    monkeypatch.setattr(schema, "_MESSAGE_UID_BACKFILL_CHUNK", 2)
    monkeypatch.setattr(schema, "_MESSAGE_UID_BACKFILL_BUDGET_S", 0)
    cursors = []
    while db.get_meta("display_user_occurrence_version") is None:
        db._execute_write(lambda conn: db._advance_user_display_index(conn.cursor()))
        progress = db.get_meta("display_user_occurrence_cursor")
        if progress:
            cursors.append(int(progress))
        db.close()
        db = SessionDB(tmp_path / "state.db")
    assert cursors == [2, 4, 6, 8, 10]
    assert db.get_messages("prompt", include_inactive=True) == before
    assert [tuple(r) for r in controls] == [tuple(r) for r in db._read_all(
        "SELECT id, display_order, display_identity FROM messages WHERE role <> 'user'")]
    assert [m["content"] for m in db.get_messages("prompt", include_compacted=True)
            if m["role"] == 'user' and not m["display_kind"]] == ['final', 'final']
    db.close()


def _settled_page_vm_steps(tmp_path):
    steps = []
    for size in (2000, 20000):
        db = SessionDB(tmp_path / f"state-{size}.db")
        db.create_session("large", source="desktop")
        db.append_messages_batch("large", [{"role": "user", "content": f"turn-{i}",
            "message_uid": f"turn-{i}"} for i in range(size)])
        calls = []
        with db._read_ctx() as conn:
            conn.set_progress_handler(lambda: calls.append(1) or 0, 100)
            rows = db._display_rows_from_conn(conn, "large", latest=True, limit=120)
            conn.set_progress_handler(None, 0)
        assert len(rows) == 120
        steps.append(len(calls) * 100)
        db.close()
    return steps


def test_settled_prompt_display_page_vm_work_is_bounded(tmp_path):
    steps = _settled_page_vm_steps(tmp_path)
    assert steps[1] < steps[0] * 3, steps


def test_old_readonly_non_user_index_groups_are_preserved(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("control", source="desktop")
    db.append_messages_batch("control", [{"role": "tool", "content": "same full output",
        "timestamp": 1700000000.0, "message_uid": uid} for uid in ('tool-one', 'tool-two')])
    db._execute_write(lambda conn: (conn.execute("UPDATE messages SET display_order = id"),
        conn.execute("DELETE FROM state_meta WHERE key = 'display_user_occurrence_version'")))
    expected = db._read_all("SELECT id FROM messages ORDER BY id")
    db.close()
    db = SessionDB(tmp_path / "state.db", read_only=True)
    assert [m['id'] for m in db.get_messages('control', include_compacted=True)] == [r[0] for r in expected]
    assert db.display_message_count('control') == 2
    db.close()
