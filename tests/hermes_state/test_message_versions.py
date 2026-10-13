"""Copy provenance must survive later compaction without hiding a newer authored version."""

import copy
import sqlite3

import pytest

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX, _SUMMARY_END_MARKER
from agent.message_metadata import record_absorbed_message, without_persistence_fields
from agent.transcript_repair import sync_flushed_message_markers
from hermes_state import SessionDB

VERSION = "message_version_uid"


def unaddressed(message):
    row = copy.deepcopy(message)
    for key in ("_row_id", "_db_row_snapshot", "_canonical_row", "_db_persisted"):
        row.pop(key, None)
    return row



def version_cursors(db, session_id):
    return [row[0] for row in db._conn.execute(
        "SELECT v.id FROM messages m JOIN message_versions v ON v.session_id = m.session_id "
        "AND v.message_uid = m.message_uid AND v.version_uid = m.message_version_uid "
        "WHERE m.session_id = ? AND m.active = 1 ORDER BY m.id", (session_id,))]


def test_edit_compaction_versions_survive_reopen_and_concurrent_clones(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        db.create_session("s", "webui")
        original = {"role": "user", "content": "Refactor the parser", "timestamp": 1000}
        db.append_messages_batch("s", [original])
        edited = {**unaddressed(original), "content": "Only fix the parser error", "timestamp": 2000}
        db.append_messages_batch("s", [edited])
        assert original.get(VERSION) != edited.get(VERSION), "An ordinary edit needs a distinct durable version"
        assert original["message_uid"] == edited["message_uid"]
        watermark = edited["_row_id"]
        tail = {"role": "assistant", "content": "Concurrent progress", "timestamp": 2001}
        db.append_messages_batch("s", [tail])
        compressor = ContextCompressor.__new__(ContextCompressor)
        compressor.quiet_mode = True
        carrier = {"role": "assistant", "content": SUMMARY_PREFIX + "\nSummary\n" + _SUMMARY_END_MARKER}
        compacted = compressor._reappend_inflight_user_task([carrier], unaddressed(edited))
        replay = compacted[-1]
        assert replay.get(VERSION) == edited[VERSION]
        db.archive_and_compact("s", compacted, watermark=watermark)
        assert replay[VERSION] == edited[VERSION]
        expected = [carrier[VERSION], edited[VERSION], tail[VERSION]]
        saved = db.get_messages("s", include_inactive=True)
        assert all(row[VERSION] for row in saved)
        assert [row[VERSION] for row in saved if row["active"]] == expected
        # Archiving both earlier user versions must never erase their distinction.
        assert saved[0]["compacted"] == saved[1]["compacted"] == 1
        assert saved[0][VERSION] != saved[1][VERSION]
        replacement = db.get_messages_as_conversation("s", include_row_ids=True)
        cursors = version_cursors(db, "s")
        db.replace_messages("s", replacement)
        assert [row[VERSION] for row in db.get_messages("s")] == expected
        assert version_cursors(db, "s") == cursors, "Physical replacement must not reset the version cursor"
        prefix = [{"role": row["role"], "content": row["content"]} for row in replacement]
        db.replace_messages("s", prefix, archive_dropped=True)
        assert [row[VERSION] for row in prefix] == expected
        db.publish_compression_child(
            parent_session_id="s", child_session_id="child", source="webui", messages=prefix,
            require_compression_lease=False,
        )
        assert [row[VERSION] for row in db.get_messages("child")] == expected
        assert min(version_cursors(db, "child")) > max(cursors), "A version's arrival is scoped to its session"
    finally:
        db.close()
    reopened = SessionDB(path)
    try:
        assert [row[VERSION] for row in reopened.get_messages_as_conversation("child")] == expected
        exported = reopened.export_session("child")
    finally:
        reopened.close()
    imported = SessionDB(tmp_path / "imported.db")
    try:
        assert imported.import_sessions([exported])["ok"]
        assert [row[VERSION] for row in imported.get_messages_as_conversation("child")] == expected
    finally:
        imported.close()


def test_rewrites_merges_and_failed_transactions_do_not_reuse_a_committed_version(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", "webui")
        message = {"role": "user", "content": "Original request"}
        db.append_messages_batch("s", [message])
        before = message.get(VERSION)
        before_cursor = version_cursors(db, "s")
        db.set_user_message_content("s", message["_row_id"], "Edited request")
        restored = db.get_messages_as_conversation("s")[0]
        assert before != restored.get(VERSION), "An in-place content rewrite needs a new durable version"
        assert version_cursors(db, "s")[0] > before_cursor[0]
        assert restored["message_uid"] == message["message_uid"]
        assert VERSION not in without_persistence_fields(restored)
        # The stale writer adopts the stored winner's version; an owned rewrite then advances it.
        message["content"] = "Stale writer's request"
        db.append_messages_batch("s", [message])
        sync_flushed_message_markers([message], [message])
        assert message[VERSION] == restored[VERSION]
        assert message["content"] == "Edited request"
        message["content"] = "Owned rewrite"
        db.append_messages_batch("s", [message])
        assert message[VERSION] != restored[VERSION]
        owned_version = message[VERSION]
        db._conn.execute("UPDATE messages SET observed = 1 WHERE id = ?", (message["_row_id"],))
        db._conn.commit()
        assert db.get_messages("s")[0][VERSION] == owned_version
        merged = {**unaddressed(message), "content": "Owned rewrite and another instruction"}
        record_absorbed_message(merged, {"message_uid": "other-message"})
        assert VERSION not in merged, "A merge must not claim its first constituent's old version"
        db.append_messages_batch("s", [merged])
        assert merged[VERSION] != owned_version
        failed = {"role": "user", "content": "Rolled back request"}

        def fail(conn):
            db._insert_message_rows(conn, "s", [failed])
            assert failed.get(VERSION)
            raise RuntimeError("synthetic rollback")

        committed_versions = db._conn.execute("SELECT COUNT(*) FROM message_versions").fetchone()[0]
        with pytest.raises(RuntimeError, match="synthetic rollback"):
            db._execute_transcript_write(fail, [failed])
        assert VERSION not in failed
        assert db._conn.execute("SELECT COUNT(*) FROM message_versions").fetchone()[0] == committed_versions
        assert [row["content"] for row in db.get_messages("s")] == ["Owned rewrite", "Owned rewrite and another instruction"]
    finally:
        db.close()

    # Reopen the same rows as a pre-contract database: unknown lineage gets independent versions.
    with sqlite3.connect(tmp_path / "state.db") as legacy:
        dependent = legacy.execute("SELECT type, name FROM sqlite_master WHERE type IN ('trigger', 'index') AND sql LIKE '%message_version_uid%'").fetchall()
        for kind, name in dependent:
            legacy.execute(f'DROP {kind} "{name}"')
        legacy.execute("DROP TABLE message_versions")
        legacy.execute("ALTER TABLE messages DROP COLUMN message_version_uid")
        legacy.execute("UPDATE schema_version SET version = 31")
        legacy.execute("DELETE FROM state_meta WHERE key LIKE 'message_version_uid_backfill%'")
    legacy_reader = SessionDB(tmp_path / "state.db", read_only=True)
    try:
        assert all(row.get(VERSION) is None for row in legacy_reader.get_messages_as_conversation("s"))
        assert VERSION not in [row[1] for row in legacy_reader._conn.execute("PRAGMA table_info(messages)")]
    finally:
        legacy_reader.close()
    migrated = SessionDB(tmp_path / "state.db")
    try:
        versions = [row[VERSION] for row in migrated.get_messages("s")]
        assert all(versions) and versions[0] != versions[1]
        assert len(version_cursors(migrated, "s")) == len(versions)
    finally:
        migrated.close()
    reopened = SessionDB(tmp_path / "state.db")
    try:
        assert [row[VERSION] for row in reopened.get_messages("s")] == versions
    finally:
        reopened.close()
