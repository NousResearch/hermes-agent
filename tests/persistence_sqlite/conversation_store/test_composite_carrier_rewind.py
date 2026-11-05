"""Transactional persistence contracts for composite compaction carriers."""

from __future__ import annotations

import os

import pytest

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _MERGED_SUMMARY_DELIMITER,
    _SUMMARY_END_MARKER,
)
from persistence.sqlite._holder import StoreHolder
from hermes_state_errors import (
    CompressionSessionClosedError, SessionCompressionInProgressError, SessionTurnLeaseLostError,
)
from persistence.sqlite.conversation._store.messages import (
    append_message,
    get_ancestor_display_prefix,
    get_messages_as_conversation,
    get_resume_conversations,
    replace_messages,
    rewind_to_message,
    set_message_reaction,
)
from persistence.sqlite.conversation._store.sessions import create_session, end_session


def _carrier(ask: str = "REAL ASK") -> str:
    return (
        f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n"
        f"{_SUMMARY_END_MARKER}\n\n{ask}"
    )


@pytest.fixture()
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture()
def db(store):
    return store.holder()


def _session_counts(db: StoreHolder, session_id: str) -> tuple[int, int, int]:
    row = db._conn.execute(
        "SELECT message_count, tool_call_count, rewind_count "
        "FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return row["message_count"], row["tool_call_count"], row["rewind_count"]


def _row_state(db: StoreHolder, session_id: str) -> list[tuple]:
    return [
        tuple(row)
        for row in db._conn.execute(
            "SELECT id, role, content, active, display_kind "
            "FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    ]


def _active_ids(db: StoreHolder, session_id: str) -> list[int]:
    return [
        int(message["_row_id"])
        for message in get_messages_as_conversation(
            db,
            session_id, include_row_ids=True
        )
    ]


def test_composite_rewind_archives_tail_and_inserts_its_hidden_scaffold(db):
    sid = "carrier-rewind"
    create_session(db, sid, source="tui")
    append_message(db, sid, "user", "older ask")
    append_message(
        db,
        sid,
        "assistant",
        None,
        tool_calls=[{"id": "call-1", "function": {"name": "terminal"}}],
    )
    append_message(db, sid, "tool", "ok", tool_call_id="call-1")
    target_id = append_message(db, sid, "user", _carrier())
    append_message(db, sid, "assistant", "failed")
    expected_active_ids = _active_ids(db, sid)

    result = rewind_to_message(
        db,
        sid,
        target_id,
        preserve_compaction_handoff=True,
        expected_active_ids=expected_active_ids,
        expected_target_content="REAL ASK",
    )

    assert result["rewound_count"] == 2
    assert result["replacement_message_id"] == result["new_head_id"]
    active = get_messages_as_conversation(db, sid, include_row_ids=True)
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


def test_lineage_display_prefers_tip_carrier_over_replayed_parent_ask(db):
    parent = "carrier-parent"
    child = "carrier-child"
    create_session(db, parent, source="tui")
    append_message(db, parent, "user", "REAL ASK")
    end_session(db, parent, "compression")
    create_session(db, child, source="tui", parent_session_id=parent)
    carrier_id = append_message(db, child, "user", _carrier())

    model_history, display_history = get_resume_conversations(db, child)

    from agent.context_compressor import user_originated_turn_view

    visible_users = [
        user_originated_turn_view(message)
        for message in display_history
        if user_originated_turn_view(message) is not None
    ]
    assert [message["content"] for message in visible_users] == ["REAL ASK"]
    assert display_history[-1]["_row_id"] == carrier_id
    assert model_history[-1]["_row_id"] == carrier_id
    assert get_ancestor_display_prefix(db, child) == []


def test_lineage_display_dedupes_multimodal_ask_in_tip_carrier(db):
    parent = "media-carrier-parent"
    child = "media-carrier-child"
    ask = [
        {"type": "text", "text": "inspect this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]
    carrier = [
        {"type": "text", "text": f"{_carrier('')}\n"},
        *ask,
    ]
    create_session(db, parent, source="tui")
    append_message(db, parent, "user", ask)
    end_session(db, parent, "compression")
    create_session(db, child, source="tui", parent_session_id=parent)
    carrier_id = append_message(db, child, "user", carrier)

    _, display_history = get_resume_conversations(db, child)

    from agent.context_compressor import user_originated_turn_view

    visible_users = [
        user_originated_turn_view(message)
        for message in display_history
        if user_originated_turn_view(message) is not None
    ]
    assert [message["content"] for message in visible_users] == [ask]
    assert display_history[-1]["_row_id"] == carrier_id


def test_default_rewind_return_shape_and_active_counters_remain_compatible(db):
    sid = "default-rewind"
    create_session(db, sid, source="cli")
    append_message(db, sid, "user", "first")
    append_message(db, sid, "assistant", "answer")
    target_id = append_message(db, sid, "user", "second")
    append_message(
        db,
        sid,
        "assistant",
        None,
        tool_calls=[{"id": "call-2", "function": {"name": "terminal"}}],
    )

    result = rewind_to_message(db, sid, target_id)

    assert set(result) == {"rewound_count", "target_message", "new_head_id"}
    assert result["rewound_count"] == 2
    assert _session_counts(db, sid) == (2, 0, 1)


def test_guarded_composite_rewind_rejects_append_without_inserting_scaffold(db):
    sid = "guarded-rewind-append"
    create_session(db, sid, source="cli")
    append_message(db, sid, "user", "first")
    append_message(db, sid, "assistant", "answer")
    target_id = append_message(db, sid, "user", _carrier())
    append_message(db, sid, "assistant", "failed")
    snapshot = get_messages_as_conversation(db, sid, include_row_ids=True)
    expected_active_ids = [int(message["_row_id"]) for message in snapshot]
    assert snapshot[-2]["_row_id"] == target_id

    # Deterministic validation -> write race: a sibling writer commits after
    # the snapshot but before rewind_to_message begins its write transaction.
    sibling = StoreHolder(db_path=db.db_path)
    append_message(sibling, sid, "assistant", "concurrent append")
    sibling.close()
    before_rows = _row_state(db, sid)
    before_counts = _session_counts(db, sid)

    with pytest.raises(RuntimeError, match="active transcript changed"):
        rewind_to_message(
            db,
            sid,
            target_id,
            preserve_compaction_handoff=True,
            expected_active_ids=expected_active_ids,
            expected_target_content=_carrier(),
        )

    assert _row_state(db, sid) == before_rows
    assert _session_counts(db, sid) == before_counts

def test_guarded_rewind_rejects_selected_target_content_change(db):
    sid = "guarded-rewind-in-place"
    create_session(db, sid, source="cli")
    append_message(db, sid, "user", "first")
    append_message(db, sid, "assistant", "answer")
    target_id = append_message(db, sid, "user", "second")
    append_message(db, sid, "assistant", "failed")
    expected_active_ids = _active_ids(db, sid)

    sibling = StoreHolder(db_path=db.db_path)
    sibling._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("changed second", target_id),
        )
    )
    sibling.close()
    before_rows = _row_state(db, sid)
    before_counts = _session_counts(db, sid)

    with pytest.raises(RuntimeError, match="rewind target changed"):
        rewind_to_message(
            db,
            sid,
            target_id,
            expected_active_ids=expected_active_ids,
            expected_target_content="second",
        )

    assert _row_state(db, sid) == before_rows
    assert _session_counts(db, sid) == before_counts


def test_guarded_rewind_ignores_reaction_metadata_change(db):
    sid = "guarded-rewind-reaction"
    create_session(db, sid, source="cli")
    target_id = append_message(db, sid, "user", "second")
    append_message(db, sid, "assistant", "failed")
    expected_active_ids = _active_ids(db, sid)

    assert set_message_reaction(db, sid, target_id + 1, "👍", author="user")

    result = rewind_to_message(
        db,
        sid,
        target_id,
        expected_active_ids=expected_active_ids,
        expected_target_content="second",
    )

    assert result["rewound_count"] == 2
    assert get_messages_as_conversation(db, sid) == []


def test_rewind_guard_rejects_foreign_live_compression_without_any_change(store, db):
    sid = "locked-rewind"
    create_session(db, sid, source="tui")
    target_id = append_message(db, sid, "user", _carrier())
    append_message(db, sid, "assistant", "failed")
    assert store.coordination.try_acquire_compression_lock(session_id=sid, holder="foreign-writer", ttl_seconds=60)
    before_rows = _row_state(db, sid)
    before_counts = _session_counts(db, sid)

    with pytest.raises(SessionCompressionInProgressError):
        rewind_to_message(
            db,
            sid, target_id, preserve_compaction_handoff=True
        )

    assert _row_state(db, sid) == before_rows
    assert _session_counts(db, sid) == before_counts


def test_rewind_guard_rejects_foreign_turn_lease_without_any_change(store, db):
    sid = "leased-rewind"
    create_session(db, sid, source="tui")
    target_id = append_message(db, sid, "user", _carrier())
    expected_active_ids = _active_ids(db, sid)
    holder = f"pid={os.getpid()}:turn=active"
    assert store.coordination.try_acquire_turn_lease(session_id=sid, holder=holder, ttl_seconds=60)
    before_rows = _row_state(db, sid)
    before_counts = _session_counts(db, sid)

    with pytest.raises(SessionTurnLeaseLostError, match="active turn lease"):
        rewind_to_message(
            db,
            sid,
            target_id,
            preserve_compaction_handoff=True,
            expected_active_ids=expected_active_ids,
            expected_target_content="REAL ASK",
        )

    assert _row_state(db, sid) == before_rows
    assert _session_counts(db, sid) == before_counts

    store.coordination.release_turn_lease(session_id=sid, holder=holder)
    result = rewind_to_message(
        db,
        sid,
        target_id,
        preserve_compaction_handoff=True,
        expected_active_ids=expected_active_ids,
        expected_target_content="REAL ASK",
    )
    assert result["rewound_count"] == 1


def test_guarded_replace_rejects_foreign_turn_lease_without_any_change(store, db):
    sid = "leased-replace"
    create_session(db, sid, source="tui")
    append_message(db, sid, "user", "old ask")
    holder = f"pid={os.getpid()}:turn=active"
    assert store.coordination.try_acquire_turn_lease(session_id=sid, holder=holder, ttl_seconds=60)
    before_rows = _row_state(db, sid)
    before_counts = _session_counts(db, sid)

    with pytest.raises(SessionTurnLeaseLostError, match="active turn lease"):
        replace_messages(
            db,
            sid,
            [{"role": "user", "content": "replacement"}],
            active_only=True,
            archive_dropped=True,
            reject_active_turn_lease=True,
        )

    assert _row_state(db, sid) == before_rows
    assert _session_counts(db, sid) == before_counts

    store.coordination.release_turn_lease(session_id=sid, holder=holder)
    replace_messages(
        db,
        sid,
        [{"role": "user", "content": "replacement"}],
        active_only=True,
        archive_dropped=True,
        reject_active_turn_lease=True,
    )
    assert [m[2] for m in _row_state(db, sid) if m[3] == 1] == ["replacement"]


def test_guarded_replace_rejects_foreign_live_compression_without_any_change(store, db):
    sid = "compression-locked-replace"
    create_session(db, sid, source="tui")
    append_message(db, sid, "user", "old ask")
    assert store.coordination.try_acquire_compression_lock(session_id=sid, holder="foreign-writer", ttl_seconds=60)
    before_rows = _row_state(db, sid)
    before_counts = _session_counts(db, sid)

    with pytest.raises(SessionCompressionInProgressError):
        replace_messages(
            db,
            sid,
            [{"role": "user", "content": "replacement"}],
            active_only=True,
            archive_dropped=True,
            reject_active_turn_lease=True,
        )

    assert _row_state(db, sid) == before_rows
    assert _session_counts(db, sid) == before_counts


def test_rewind_guard_rejects_compression_ended_parent_without_any_change(db):
    sid = "closed-rewind"
    create_session(db, sid, source="tui")
    target_id = append_message(db, sid, "user", _carrier())
    append_message(db, sid, "assistant", "failed")
    end_session(db, sid, "compression")
    before_rows = _row_state(db, sid)
    before_counts = _session_counts(db, sid)

    with pytest.raises(CompressionSessionClosedError):
        rewind_to_message(
            db,
            sid, target_id, preserve_compaction_handoff=True
        )

    assert _row_state(db, sid) == before_rows
    assert _session_counts(db, sid) == before_counts
