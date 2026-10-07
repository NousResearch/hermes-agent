"""``SessionDB.resolve_message_ref``: an exact ``m:<hex>`` pointer back to a message's original row.

Every copy of a logical message (in-place compaction generation, carried tail, rotation child) keeps
the ``message_uid`` of the row it copies, so the EARLIEST row carrying a uid is the original, in its
original conversation position, whatever its ``active``/``compacted`` flags now say. A ref is a hex
prefix of that uid; the lookup is an index range scan, never ``LIKE`` and never a table scan.
"""

from __future__ import annotations

import contextlib
import copy
import sqlite3
import threading
import time

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION


@pytest.fixture()
def db(tmp_path):
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


def _rows(db, sid, *, active_only=False):
    clause = " AND active = 1" if active_only else ""
    return [dict(r) for r in db._conn.execute(
        "SELECT id, session_id, content, message_uid, active, compacted FROM messages "
        f"WHERE session_id = ?{clause} ORDER BY id", (sid,)).fetchall()]


def _seed(db, sid, n=4):
    db.create_session(sid, "cli", model="m")
    for i in range(n):
        db.append_message(session_id=sid, role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
    return _rows(db, sid)


def _set_uid(db, row_id, uid):
    db._conn.execute("UPDATE messages SET message_uid = ? WHERE id = ?", (uid, row_id))
    db._conn.commit()


class TestPrefixResolve:
    def test_short_and_full_refs_with_and_without_the_m_prefix(self, db):
        rows = _seed(db, "s")
        target = rows[2]
        uid = target["message_uid"]
        expected = {"id": target["id"], "session_id": "s", "message_uid": uid}
        for ref in (f"m:{uid[:12]}", uid[:12], f"m:{uid}", uid, f"M:{uid[:16].upper()}", f"  m:{uid[:12]}  "):
            assert db.resolve_message_ref(ref) == expected, ref

    def test_a_prefix_of_all_f_digits_still_bounds_its_range(self, db):
        rows = _seed(db, "s", n=2)
        _set_uid(db, rows[0]["id"], "f" * 32)
        assert db.resolve_message_ref("m:" + "f" * 12)["id"] == rows[0]["id"]

    def test_not_found_is_none(self, db):
        _seed(db, "s", n=2)
        assert db.resolve_message_ref("m:" + "0" * 12) is None


class TestEarliestRowWins:
    def test_in_place_compaction_resolves_to_the_archived_original_not_the_carried_copy(self, db):
        stored = _seed(db, "s", n=4)
        watermark = stored[-1]["id"]
        db.append_message(session_id="s", role="user", content="late user")
        late = _rows(db, "s")[4]
        restored = db.get_messages_as_conversation("s")
        compacted = [{"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                     copy.copy(restored[2]), copy.copy(restored[3])]
        db.archive_and_compact("s", compacted, watermark=watermark)
        live = _rows(db, "s", active_only=True)
        copied = next(r for r in live if r["content"] == "msg 2")
        assert copied["message_uid"] == stored[2]["message_uid"] and copied["id"] != stored[2]["id"]

        resolved = db.resolve_message_ref("m:" + stored[2]["message_uid"][:12])
        assert resolved == {"id": stored[2]["id"], "session_id": "s", "message_uid": stored[2]["message_uid"]}
        # The concurrent late append was cloned into the new generation; its original is hidden, still first.
        assert db.resolve_message_ref(late["message_uid"][:12])["id"] == late["id"]

    def test_rotation_clone_resolves_to_the_parent_original(self, db):
        stored = _seed(db, "parent", n=2)
        restored = db.get_messages_as_conversation("parent")
        handoff = [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}, copy.copy(restored[1])]
        db.publish_compression_child(
            parent_session_id="parent", child_session_id="child", source="cli", messages=handoff,
            require_compression_lease=False, watermark=stored[-1]["id"])
        child_copy = _rows(db, "child")[1]
        assert child_copy["message_uid"] == stored[1]["message_uid"]
        resolved = db.resolve_message_ref("m:" + stored[1]["message_uid"][:12])
        assert resolved["id"] == stored[1]["id"] and resolved["session_id"] == "parent"


class TestAmbiguityAndInvalidInput:
    def test_two_uids_sharing_a_prefix_are_ambiguous_and_a_longer_ref_disambiguates(self, db):
        rows = _seed(db, "s", n=2)
        _set_uid(db, rows[0]["id"], "abcdef012345" + "0" * 20)
        _set_uid(db, rows[1]["id"], "abcdef012345" + "1" * 20)
        with pytest.raises(LookupError):
            db.resolve_message_ref("m:abcdef012345")
        assert db.resolve_message_ref("m:abcdef0123451")["id"] == rows[1]["id"]
        assert db.resolve_message_ref("abcdef0123450")["id"] == rows[0]["id"]

    def test_copies_of_one_uid_are_not_ambiguous(self, db):
        stored = _seed(db, "s", n=4)
        restored = db.get_messages_as_conversation("s")
        db.archive_and_compact("s", [{"role": "user", "content": "[CONTEXT COMPACTION] s"},
                                     copy.copy(restored[3])], watermark=stored[-1]["id"])
        assert db.resolve_message_ref("m:" + stored[3]["message_uid"][:12])["id"] == stored[3]["id"]

    @pytest.mark.parametrize("ref", [
        "", "m:", "m:abc", "m:" + "a" * 11, "m:" + "a" * 33, "m:xyz0123456789", "r:abcdef012345",
        "m:abcdef01234g", "m: abcdef012345", "m:abcdef012345%", "m:abcdef_12345", None, 12345,
    ])
    def test_invalid_refs_raise_value_error(self, db, ref):
        with pytest.raises(ValueError):
            db.resolve_message_ref(ref)


class TestIndex:
    def test_both_lookups_use_the_message_uid_index(self, db):
        from hermes_state_search import _MESSAGE_REF_CANDIDATES_SQL, _MESSAGE_REF_ROW_SQL
        for sql, params in ((_MESSAGE_REF_CANDIDATES_SQL, ("abc", "abd")), (_MESSAGE_REF_ROW_SQL, ("abc",))):
            plan = " ".join(str(r[-1]) for r in db._conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall())
            assert "SEARCH messages USING" in plan and "idx_messages_message_uid" in plan, plan
            assert "SCAN" not in plan, plan

    def test_an_existing_store_gains_the_index_on_open(self, tmp_path):
        path = tmp_path / "state.db"
        SessionDB(db_path=path).close()
        conn = sqlite3.connect(path)
        conn.execute("DROP INDEX IF EXISTS idx_messages_message_uid")
        conn.commit()
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        conn.close()
        reopened = SessionDB(db_path=path)
        try:
            assert reopened._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'idx_messages_message_uid'"
            ).fetchone() is not None
        finally:
            reopened.close()

    def test_a_settled_open_with_the_index_does_not_wait_on_a_sibling_write_lock(self, tmp_path):
        path = tmp_path / "state.db"
        SessionDB(db_path=path).close()
        SessionDB(db_path=path).close()
        holder = sqlite3.connect(path, timeout=60)
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("UPDATE state_meta SET value = value WHERE key = 'nonexistent'")
        release = threading.Timer(4.0, holder.rollback)
        release.start()
        try:
            started = time.perf_counter()
            SessionDB(db_path=path).close()
            elapsed = time.perf_counter() - started
        finally:
            release.cancel()
            with contextlib.suppress(sqlite3.Error):
                holder.rollback()
            holder.close()
        assert elapsed < 2.0, f"open blocked on the write lock for {elapsed:.3f}s"
