"""Regression for #111996 / PR #112191: rematerialized copies must not re-INSERT.

Adopting an already-ACTIVE logical identity does not grow the table. Identity is
the durable row, else role+timestamp+content plus stored tool identity
(json_extract of tool_calls[].id for assistants; tool_call_id for tool rows).
A reused provider tool id on a later timestamp still inserts. Missing timestamps
are not same-batch collapsed. Repair flush through _db_flush_row updates the
survivor, archives the dropped row, and recomputes active counters.
"""

from types import SimpleNamespace

from agent.agent_runtime_helpers import repair_message_sequence
from agent.context_compressor import _fresh_compaction_message_copy
from agent.session_persistence import _db_flush_row
from hermes_state import SessionDB

SESSION_ID = "s111996"
CALL_ID = "call_1b2db7a0565e4f92b0911631"
TS = 1_700_000_000.0


def _db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id=SESSION_ID, source="cli")
    return db


def _count(db, *, active_only=False):
    sql = "SELECT COUNT(*) FROM messages WHERE session_id = ?"
    if active_only:
        sql += " AND active = 1"
    return db._read_one(sql, (SESSION_ID,))[0]


def _unmarked_copies(rows):
    return [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in rows]


def _tool_turn(base_ts=TS, content="output"):
    return [
        {"role": "user", "content": "run it", "timestamp": base_ts},
        {"role": "assistant", "content": "", "timestamp": base_ts + 1,
         "finish_reason": "tool_calls", "tool_calls": [{"id": CALL_ID, "type": "function",
         "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": CALL_ID, "content": content, "timestamp": base_ts + 2},
    ]


def test_production_shaped_assistant_is_adopted(tmp_path):
    db = _db(tmp_path)
    try:
        db.append_messages_batch(SESSION_ID, _tool_turn())
        db.append_messages_batch(SESSION_ID, _unmarked_copies(_tool_turn()))
        assert _count(db) == 3
    finally:
        db.close()


def test_reused_provider_tool_id_different_turn_is_inserted(tmp_path):
    db = _db(tmp_path)
    try:
        db.append_messages_batch(SESSION_ID, _tool_turn(TS))
        db.append_messages_batch(SESSION_ID, _tool_turn(TS + 10, "second output"))
        assert _count(db) == 6
        assert {r[0] for r in db._read_all("SELECT content FROM messages WHERE role = 'tool' AND session_id = ?", (SESSION_ID,))} == {"output", "second output"}
    finally:
        db.close()


def test_missing_timestamps_are_not_same_batch_collapsed(tmp_path):
    db = _db(tmp_path)
    try:
        db.append_messages_batch(SESSION_ID, [{"role": "user", "content": "yes"}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "yes"}])
        assert _count(db) == 3
    finally:
        db.close()


def test_same_timestamp_different_tool_ids_are_not_collapsed(tmp_path):
    db = _db(tmp_path)
    try:
        db.append_messages_batch(SESSION_ID, [
            {"role": "tool", "tool_call_id": "call_a", "content": "ok", "timestamp": TS},
            {"role": "tool", "tool_call_id": "call_b", "content": "ok", "timestamp": TS},
        ])
        assert _count(db) == 2
    finally:
        db.close()


def test_rematerialized_unmarked_transcript_is_adopted(tmp_path):
    db = _db(tmp_path)
    try:
        db.append_messages_batch(SESSION_ID, _tool_turn())
        db.append_messages_batch(SESSION_ID, _unmarked_copies(_tool_turn()))
        assert _count(db) == 3
    finally:
        db.close()


def test_repeated_repack_does_not_grow(tmp_path):
    db = _db(tmp_path)
    try:
        db.append_messages_batch(SESSION_ID, _tool_turn())
        for _ in range(3):
            db.append_messages_batch(SESSION_ID, _unmarked_copies(_tool_turn()))
        assert _count(db) == 3
    finally:
        db.close()


def test_fresh_compaction_copy_flush_does_not_grow(tmp_path):
    db = _db(tmp_path)
    try:
        db.append_messages_batch(SESSION_ID, _tool_turn())
        db.append_messages_batch(SESSION_ID, [_fresh_compaction_message_copy(row) for row in _tool_turn()])
        assert _count(db) == 3
    finally:
        db.close()


def test_repaired_assistant_projected_through_db_flush_updates_and_counts(tmp_path):
    db = _db(tmp_path)
    try:
        original = [{"role": "assistant", "content": "junk", "timestamp": TS},
                    {"role": "assistant", "content": "", "timestamp": TS + 1, "tool_calls": [{"id": CALL_ID, "type": "function"}]},
                    {"role": "tool", "tool_call_id": CALL_ID, "content": "output", "timestamp": TS + 2}]
        db.append_messages_batch(SESSION_ID, original)
        live = [{k: v for k, v in row.items()} for row in original]
        repair_message_sequence(None, live)
        agent = SimpleNamespace(_persist_user_message_override=None)
        projected = [_db_flush_row(agent, msg, False) for msg in live]
        db.append_messages_batch(SESSION_ID, projected)
        active = db._read_all("SELECT id, role, active, content, tool_calls FROM messages WHERE session_id = ? ORDER BY id", (SESSION_ID,))
        assert sum(r[2] for r in active) == 2
        assert any(r[2] == 0 for r in active)
        assert any(r[2] == 1 and r[1] == "assistant" and r[4] for r in active)
        assert db.get_session(SESSION_ID)["message_count"] == 2
        assert db.get_session(SESSION_ID)["tool_call_count"] == 1
    finally:
        db.close()


def test_new_timestamp_same_text_is_new_turn(tmp_path):
    db = _db(tmp_path)
    try:
        db.append_messages_batch(SESSION_ID, [{"role": "user", "content": "same", "timestamp": TS}])
        db.append_messages_batch(SESSION_ID, [{"role": "user", "content": "same", "timestamp": TS + 10}])
        assert _count(db) == 2
    finally:
        db.close()


def test_archive_and_compact_still_republishes_active_generation(tmp_path):
    db = _db(tmp_path)
    try:
        db.append_messages_batch(SESSION_ID, _tool_turn())
        db.archive_and_compact(SESSION_ID, [{"role": "user", "content": "summary", "timestamp": TS + 50}, {"role": "assistant", "content": "ok", "timestamp": TS + 51}])
        assert _count(db, active_only=True) == 2
    finally:
        db.close()
