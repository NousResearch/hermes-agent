"""Transactional persistence contracts for composite compaction carriers."""

from __future__ import annotations

import pytest

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from hermes_state import (
    CompressionSessionClosedError,
    SessionCompressionInProgressError,
    SessionDB,
    SessionTranscriptChangedError,
)


def _carrier(ask: str = "REAL ASK") -> str:
    return (
        f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n"
        f"{_SUMMARY_END_MARKER}\n\n{ask}"
    )


@pytest.fixture()
def db(tmp_path):
    state = SessionDB(db_path=tmp_path / "state.db")
    yield state
    state.close()


def _session_counts(db: SessionDB, session_id: str) -> tuple[int, int, int]:
    row = db._conn.execute(
        "SELECT message_count, tool_call_count, rewind_count "
        "FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return row["message_count"], row["tool_call_count"], row["rewind_count"]


def _row_state(db: SessionDB, session_id: str) -> list[tuple]:
    return [
        tuple(row)
        for row in db._conn.execute(
            "SELECT id, role, content, active, display_kind "
            "FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    ]


def test_composite_rewind_archives_tail_and_inserts_its_hidden_scaffold(db):
    sid = "carrier-rewind"
    db.create_session(sid, source="tui")
    db.append_message(sid, "user", "older ask")
    db.append_message(
        sid,
        "assistant",
        None,
        tool_calls=[{"id": "call-1", "function": {"name": "terminal"}}],
    )
    db.append_message(sid, "tool", "ok", tool_call_id="call-1")
    target_id = db.append_message(sid, "user", _carrier())
    db.append_message(sid, "assistant", "failed")
    _, revision = db.get_active_conversation_snapshot(sid)

    result = db.rewind_to_message(
        sid,
        target_id,
        preserve_compaction_handoff=True,
        expected_active_revision=revision,
    )

    assert result["rewound_count"] == 2
    assert result["replacement_message_id"] == result["new_head_id"]
    active = db.get_messages_as_conversation(sid, include_row_ids=True)
    assert len(active) == 4
    assert active[-1]["_row_id"] == result["replacement_message_id"]
    assert active[-1]["display_kind"] == "hidden"
    assert SUMMARY_PREFIX in active[-1]["content"]
    assert "REAL ASK" not in active[-1]["content"]
    archived = db._conn.execute(
        "SELECT active FROM messages WHERE id IN (?, ?) ORDER BY id",
        (target_id, target_id + 1),
    ).fetchall()
    assert [row[0] for row in archived] == [0, 0]
    assert _session_counts(db, sid) == (4, 1, 1)


def test_default_rewind_return_shape_and_active_counters_remain_compatible(db):
    sid = "default-rewind"
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", "first")
    db.append_message(sid, "assistant", "answer")
    target_id = db.append_message(sid, "user", "second")
    db.append_message(
        sid,
        "assistant",
        None,
        tool_calls=[{"id": "call-2", "function": {"name": "terminal"}}],
    )

    result = db.rewind_to_message(sid, target_id)

    assert set(result) == {"rewound_count", "target_message", "new_head_id"}
    assert result["rewound_count"] == 2
    assert _session_counts(db, sid) == (2, 0, 1)


def test_guarded_composite_rewind_rejects_append_without_inserting_scaffold(db):
    sid = "guarded-rewind-append"
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", "first")
    db.append_message(sid, "assistant", "answer")
    target_id = db.append_message(sid, "user", _carrier())
    db.append_message(sid, "assistant", "failed")
    snapshot, revision = db.get_active_conversation_snapshot(
        sid, repair_alternation=True, include_row_ids=True
    )
    assert snapshot[-2]["_row_id"] == target_id

    # Deterministic validation -> write race: a sibling writer commits after
    # the snapshot but before rewind_to_message begins its write transaction.
    sibling = SessionDB(db_path=db.db_path)
    sibling.append_message(sid, "assistant", "concurrent append")
    sibling.close()
    before_rows = _row_state(db, sid)
    before_counts = _session_counts(db, sid)

    with pytest.raises(
        SessionTranscriptChangedError,
        match="session history changed before the rewind could be persisted",
    ):
        db.rewind_to_message(
            sid,
            target_id,
            preserve_compaction_handoff=True,
            expected_active_revision=revision,
        )

    assert _row_state(db, sid) == before_rows
    assert _session_counts(db, sid) == before_counts


def test_guarded_rewind_revision_detects_same_head_and_count_row_update(db):
    sid = "guarded-rewind-in-place"
    db.create_session(sid, source="cli")
    db.append_message(sid, "user", "first")
    db.append_message(sid, "assistant", "answer")
    target_id = db.append_message(sid, "user", "second")
    db.append_message(sid, "assistant", "failed")
    _, revision = db.get_active_conversation_snapshot(sid)

    # api_content is updated in place: MAX(id) and COUNT(*) are unchanged, so
    # a head+count guard would accept this stale snapshot.
    sibling = SessionDB(db_path=db.db_path)
    assert sibling.set_latest_user_api_content(sid, "second", "injected second") == 1
    sibling.close()
    before_rows = _row_state(db, sid)
    before_counts = _session_counts(db, sid)

    with pytest.raises(SessionTranscriptChangedError):
        db.rewind_to_message(
            sid,
            target_id,
            expected_active_revision=revision,
        )

    assert _row_state(db, sid) == before_rows
    assert _session_counts(db, sid) == before_counts


def test_guarded_snapshot_and_rewind_support_blob_content(db):
    sid = "guarded-rewind-blob"
    db.create_session(sid, source="cli")
    target_id = db.append_message(sid, "user", b"\x00\xffraw")
    db.append_message(sid, "assistant", "failed")

    snapshot, revision = db.get_active_conversation_snapshot(
        sid, include_row_ids=True
    )
    assert snapshot[0]["content"] == b"\x00\xffraw"

    result = db.rewind_to_message(
        sid,
        target_id,
        expected_active_revision=revision,
    )

    assert result["rewound_count"] == 2
    assert db.get_messages_as_conversation(sid) == []


def test_rewind_guard_rejects_foreign_live_compression_without_any_change(db):
    sid = "locked-rewind"
    db.create_session(sid, source="tui")
    target_id = db.append_message(sid, "user", _carrier())
    db.append_message(sid, "assistant", "failed")
    assert db.try_acquire_compression_lock(sid, "foreign-writer", ttl_seconds=60)
    before_rows = _row_state(db, sid)
    before_counts = _session_counts(db, sid)

    with pytest.raises(SessionCompressionInProgressError):
        db.rewind_to_message(
            sid, target_id, preserve_compaction_handoff=True
        )

    assert _row_state(db, sid) == before_rows
    assert _session_counts(db, sid) == before_counts


def test_rewind_guard_rejects_compression_ended_parent_without_any_change(db):
    sid = "closed-rewind"
    db.create_session(sid, source="tui")
    target_id = db.append_message(sid, "user", _carrier())
    db.append_message(sid, "assistant", "failed")
    db.end_session(sid, "compression")
    before_rows = _row_state(db, sid)
    before_counts = _session_counts(db, sid)

    with pytest.raises(CompressionSessionClosedError):
        db.rewind_to_message(
            sid, target_id, preserve_compaction_handoff=True
        )

    assert _row_state(db, sid) == before_rows
    assert _session_counts(db, sid) == before_counts


def test_multirow_insert_preserves_serialized_display_metadata(db):
    sid = "serialized-display"
    db.create_session(sid, source="tui")

    db.replace_messages(
        sid,
        [
            {
                "role": "assistant",
                "content": "visible",
                "display_metadata": '{"reactions":[{"emoji":"👍"}]}',
            }
        ],
    )

    restored = db.get_messages_as_conversation(sid)
    assert restored[0]["display_metadata"] == {
        "reactions": [{"emoji": "👍"}]
    }


def test_multirow_insert_applies_summary_visibility_policy(db):
    sid = "multirow-summary-policy"
    db.create_session(sid, source="tui")
    pure = _carrier().rsplit("\n\nREAL ASK", 1)[0]

    db.replace_messages(
        sid,
        [
            {"role": "user", "content": pure},
            {"role": "user", "content": _carrier()},
            {"role": "assistant", "content": pure},
        ],
    )

    rows = db._conn.execute(
        "SELECT content, display_kind FROM messages "
        "WHERE session_id = ? ORDER BY id",
        (sid,),
    ).fetchall()
    assert rows[0]["display_kind"] == "hidden"
    assert rows[1]["display_kind"] is None
    assert rows[2]["display_kind"] == "hidden"


def test_cold_merged_assistant_stays_visible_across_multirow_rewrite(db):
    sid = "cold-merged-assistant"
    db.create_session(sid, source="tui")
    merged = (
        "visible assistant tail\n\n"
        f"{_MERGED_SUMMARY_DELIMITER}\n"
        f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n"
        f"{_SUMMARY_END_MARKER}"
    )
    db.append_message(sid, "assistant", merged)
    cold = db.get_messages_as_conversation(sid)
    assert cold[0].get("display_kind") is None

    db.replace_messages(sid, cold)

    restored = db.get_messages_as_conversation(sid)
    assert restored[0]["content"].startswith("visible assistant tail")
    assert restored[0].get("display_kind") is None
