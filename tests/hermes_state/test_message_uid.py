"""``messages.message_uid``: a durable per-message id that survives every host copy path.

The physical ``id`` is re-issued by every copy (in-place compaction generation, rotation child,
concurrent-tail clone, ``replace_messages``), and ``_row_id`` is opt-in on restore. A consumer that
keys on messages across restarts and boundaries (a context engine plugin) therefore had nothing
stable to key on and re-identified rows by content and timestamp. ``message_uid`` is minted once at
the row's first insert, stamped on the caller's dict, restored unconditionally, copied by every
clone, kept by every re-insert of the same dict and left alone by row-addressed rewrites.
"""

from __future__ import annotations

import copy
import re

import pytest

from hermes_state import SessionDB

UID_RE = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture()
def db(tmp_path):
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


def _rows(db, sid, *, active_only=True, include_session=False):
    clause = " AND active = 1" if active_only else ""
    cols = "id, session_id, role, content, message_uid, active, compacted"
    return [dict(r) for r in db._conn.execute(
        f"SELECT {cols} FROM messages WHERE session_id = ?{clause} ORDER BY id", (sid,)).fetchall()]


def _seed(db, sid, n=4):
    db.create_session(sid, "cli", model="m")
    for i in range(n):
        db.append_message(session_id=sid, role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
    return _rows(db, sid)


class TestMintAndRestore:
    def test_fresh_store_carries_the_column_at_schema_v31(self, db):
        cols = {r[1] for r in db._conn.execute("PRAGMA table_info(messages)").fetchall()}
        assert "message_uid" in cols
        assert db._conn.execute("SELECT version FROM schema_version").fetchone()[0] >= 31

    def test_insert_mints_a_uid_and_stamps_the_callers_dict(self, db):
        db.create_session("s", "cli")
        msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        db.append_messages_batch("s", msgs)
        stored = _rows(db, "s")
        assert [r["message_uid"] for r in stored] == [m["message_uid"] for m in msgs]
        assert all(UID_RE.match(r["message_uid"]) for r in stored)
        assert len({r["message_uid"] for r in stored}) == 2

    def test_uid_is_per_occurrence_not_derived_from_content_or_time(self, db):
        db.create_session("s", "cli")
        twins = [{"role": "user", "content": "ok", "timestamp": 1_700_000_000.5},
                 {"role": "user", "content": "ok", "timestamp": 1_700_000_000.5}]
        db.append_messages_batch("s", twins)
        assert twins[0]["message_uid"] != twins[1]["message_uid"]

    def test_restore_carries_the_uid_without_include_row_ids(self, db):
        stored = _seed(db, "s")
        # The ACP / gateway / compression-adoption shape: no ``include_row_ids``.
        restored = db.get_messages_as_conversation("s", repair_alternation=True)
        assert "_row_id" not in restored[0]
        assert [m["message_uid"] for m in restored] == [r["message_uid"] for r in stored]
        # And every other projection (display resume, lineage) agrees.
        model_history, display_history = db.get_resume_conversations("s")
        assert [m["message_uid"] for m in model_history] == [r["message_uid"] for r in stored]
        assert [m["message_uid"] for m in display_history] == [r["message_uid"] for r in stored]

    def test_an_explicit_uid_on_the_dict_is_written_not_replaced(self, db):
        db.create_session("s", "cli")
        msg = {"role": "user", "content": "q", "message_uid": "0" * 32}
        db.append_messages_batch("s", [msg])
        assert _rows(db, "s")[0]["message_uid"] == "0" * 32

    def test_legacy_rows_are_backfilled_once_on_open(self, tmp_path):
        path = tmp_path / "legacy.db"
        first = SessionDB(db_path=path)
        try:
            _seed(first, "s")
            # Rows written by a build that predates the column: NULL uid, schema behind v31.
            first._conn.execute("UPDATE messages SET message_uid = NULL")
            first._conn.execute("UPDATE schema_version SET version = 30")
            first._conn.commit()
        finally:
            first.close()
        second = SessionDB(db_path=path)
        try:
            uids = [r["message_uid"] for r in _rows(second, "s")]
            assert all(u and UID_RE.match(u) for u in uids)
            assert len(set(uids)) == len(uids)
            assert second._conn.execute("SELECT version FROM schema_version").fetchone()[0] >= 31
            # Idempotent: a second open keeps the backfilled values.
            second.close()
            third = SessionDB(db_path=path)
            try:
                assert [r["message_uid"] for r in _rows(third, "s")] == uids
            finally:
                third.close()
        finally:
            second.close()


class TestCopyPathsKeepTheUid:
    def test_in_place_compaction_keeps_uids_on_the_new_generation_and_the_tail_clone(self, db):
        stored = _seed(db, "s", n=4)
        watermark = stored[-1]["id"]
        # Concurrent appends after the compressor captured its watermark.
        db.append_message(session_id="s", role="user", content="late user")
        db.append_message(session_id="s", role="assistant", content="late assistant")
        late = _rows(db, "s")[4:]
        restored = db.get_messages_as_conversation("s")
        # The compressor's output: a summary the ENGINE already identified plus marker-swept COPIES of the
        # kept tail (an engine may pre-mint its own rows' uids; the host writes them as given).
        compacted = [{"role": "user", "content": "[CONTEXT COMPACTION] summary", "message_uid": "e" * 32},
                     copy.copy(restored[2]), copy.copy(restored[3])]
        db.archive_and_compact("s", compacted, watermark=watermark)
        active = _rows(db, "s")
        assert [r["content"] for r in active] == [
            "[CONTEXT COMPACTION] summary", "msg 2", "msg 3", "late user", "late assistant"]
        # New generation: the copies keep the archived originals' uids; the clone keeps the late rows' uids.
        assert [r["message_uid"] for r in active[1:3]] == [stored[2]["message_uid"], stored[3]["message_uid"]]
        assert [r["message_uid"] for r in active[3:]] == [r["message_uid"] for r in late]
        assert active[0]["message_uid"] == compacted[0]["message_uid"] == "e" * 32
        # The physical ids DID change (that is the whole point), the archived originals keep theirs too.
        assert {r["id"] for r in active}.isdisjoint({r["id"] for r in stored} | {r["id"] for r in late})
        archived = [r for r in _rows(db, "s", active_only=False) if not r["active"]]
        assert [r["message_uid"] for r in archived] == [r["message_uid"] for r in stored + late]

    def test_rotation_child_keeps_uids_on_handoff_copies_and_the_foreign_tail_clone(self, db):
        stored = _seed(db, "parent", n=2)
        watermark = stored[-1]["id"]
        db.append_message(session_id="parent", role="user", content="foreign append")
        foreign = _rows(db, "parent")[2]
        restored = db.get_messages_as_conversation("parent")
        handoff = [{"role": "user", "content": "[CONTEXT COMPACTION] summary", "message_uid": "e" * 32},
                   copy.copy(restored[1])]
        db.publish_compression_child(
            parent_session_id="parent", child_session_id="child", source="cli", messages=handoff,
            require_compression_lease=False, watermark=watermark, watermark_ceiling=foreign["id"])
        child = _rows(db, "child")
        assert [r["content"] for r in child] == ["[CONTEXT COMPACTION] summary", "msg 1", "foreign append"]
        assert child[1]["message_uid"] == stored[1]["message_uid"]
        assert child[2]["message_uid"] == foreign["message_uid"]
        assert child[0]["message_uid"] == "e" * 32
        # A restore of the child (what the resumed agent sees) carries the same uids.
        assert [m["message_uid"] for m in db.get_messages_as_conversation("child")] == [
            r["message_uid"] for r in child]

    def test_replace_messages_keeps_uids_on_the_kept_prefix_and_the_reissued_rows(self, db):
        stored = _seed(db, "s", n=4)
        restored = db.get_messages_as_conversation("s")
        # ACP non-owning persist / gateway rewrite: DELETE every active row and re-INSERT the history.
        db.replace_messages("s", restored + [{"role": "user", "content": "new"}], active_only=True)
        reissued = _rows(db, "s")
        assert [r["message_uid"] for r in reissued[:4]] == [r["message_uid"] for r in stored]
        assert reissued[4]["message_uid"] and reissued[4]["message_uid"] not in {r["message_uid"] for r in stored}
        assert {r["id"] for r in reissued[:4]}.isdisjoint({r["id"] for r in stored})
        # Rewind-style replace: the matched live prefix keeps its rows AND stamps their uids on the dicts.
        prefix = [{"role": m["role"], "content": m["content"]} for m in restored[:2]]
        db.replace_messages("s", prefix + [{"role": "user", "content": "edited"}], archive_dropped=True)
        assert [m["message_uid"] for m in prefix] == [r["message_uid"] for r in reissued[:2]]
        assert [r["message_uid"] for r in _rows(db, "s")[:2]] == [r["message_uid"] for r in reissued[:2]]

    def test_row_addressed_rewrite_keeps_the_uid(self, db):
        db.create_session("s", "cli")
        msg = {"role": "user", "content": "api variant of the prompt"}
        db.append_messages_batch("s", [msg])
        before = _rows(db, "s")[0]
        # The persist override / sanitizer rewrite: same dict, same row id and digest, new content.
        msg["content"] = "clean prompt"
        db.append_messages_batch("s", [msg])
        after = _rows(db, "s")
        assert len(after) == 1 and after[0]["id"] == before["id"]
        assert after[0]["content"] == "clean prompt"
        assert after[0]["message_uid"] == before["message_uid"] == msg["message_uid"]

    def test_rewrite_adopts_the_stored_uid_onto_a_dict_that_lacks_one(self, db):
        db.create_session("s", "cli")
        msg = {"role": "user", "content": "prompt"}
        db.append_messages_batch("s", [msg])
        stored_uid = msg.pop("message_uid")
        msg["content"] = "prompt (edited)"
        db.append_messages_batch("s", [msg])
        assert msg["message_uid"] == stored_uid
        assert [r["message_uid"] for r in _rows(db, "s")] == [stored_uid]

    def test_export_import_round_trip_keeps_the_uid(self, db, tmp_path):
        stored = _seed(db, "s", n=3)
        payload = db.export_session("s")
        assert [m["message_uid"] for m in payload["messages"]] == [r["message_uid"] for r in stored]
        other = SessionDB(db_path=tmp_path / "other.db")
        try:
            assert other.import_sessions([payload])["ok"]
            assert [r["message_uid"] for r in _rows(other, "s")] == [r["message_uid"] for r in stored]
        finally:
            other.close()


def _absorbed(db, sid):
    return [r[0] for r in db._conn.execute(
        "SELECT absorbed_message_uids FROM messages WHERE session_id = ? AND active = 1 ORDER BY id", (sid,))]


class TestPersistedMergeWitness:
    """``_absorbed_message_uids`` (which rows a consecutive-user merge folded into the survivor) rides on the
    survivor's row, so a consumer that restarts after the merge still knows the composite's constituents."""

    def test_flushed_survivor_persists_the_absorbed_uids_and_restore_returns_them(self, db):
        from agent.agent_runtime_helpers import _merge_consecutive_users

        db.create_session("s", "cli")
        dangling = {"role": "user", "content": "first, never answered"}
        db.append_messages_batch("s", [dangling])
        prompt = {"role": "user", "content": "second"}
        db.append_messages_batch("s", [prompt])
        _merge_consecutive_users([dangling, prompt])
        assert dangling["_absorbed_message_uids"] == [prompt["message_uid"]]
        # The survivor re-flushes as a row-addressed rewrite of its own row (same id, same uid).
        db.append_messages_batch("s", [dangling])
        rows = _rows(db, "s")
        assert [r["content"] for r in rows] == ["first, never answered\n\nsecond", "second"]
        assert rows[0]["message_uid"] == dangling["message_uid"]
        assert _absorbed(db, "s") == ['["%s"]' % prompt["message_uid"], None]
        restored = db.get_messages_as_conversation("s")
        assert restored[0]["_absorbed_message_uids"] == [prompt["message_uid"]]
        assert "_absorbed_message_uids" not in restored[1]

    def test_a_survivor_inserted_as_a_fresh_row_carries_the_witness(self, db):
        db.create_session("s", "cli")
        composite = {"role": "user", "content": "a\n\nb", "message_uid": "a" * 32,
                     "_absorbed_message_uids": ["b" * 32, "c" * 32]}
        db.append_messages_batch("s", [composite])
        assert _absorbed(db, "s") == ['["%s", "%s"]' % ("b" * 32, "c" * 32)]
        assert db.get_messages_as_conversation("s")[0]["_absorbed_message_uids"] == ["b" * 32, "c" * 32]

    def test_compaction_copy_and_export_import_keep_the_witness(self, db, tmp_path):
        db.create_session("s", "cli")
        composite = {"role": "user", "content": "a\n\nb", "message_uid": "a" * 32,
                     "_absorbed_message_uids": ["b" * 32]}
        db.append_messages_batch("s", [composite, {"role": "assistant", "content": "ok"}])
        restored = db.get_messages_as_conversation("s")
        db.archive_and_compact("s", [{"role": "user", "content": "[CONTEXT COMPACTION] s"},
                                     copy.copy(restored[0]), copy.copy(restored[1])])
        assert _absorbed(db, "s") == [None, '["%s"]' % ("b" * 32), None]
        payload = db.export_session("s")
        other = SessionDB(db_path=tmp_path / "other.db")
        try:
            assert other.import_sessions([payload])["ok"]
            assert other.get_messages_as_conversation("s")[1]["_absorbed_message_uids"] == ["b" * 32]
        finally:
            other.close()
