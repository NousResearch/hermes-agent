from copy import deepcopy

import pytest

from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor
from conversation_store import ConversationConflictError
from hermes_state import SessionDB
from tests.plugins.conversation_store_mutation_fixture import MutationStore


@pytest.fixture
def db_store(tmp_path):
    store = MutationStore()
    db = SessionDB(db_path=tmp_path / "shadow.db", conversation_store=store)
    db.create_session("s1", source="cli")
    try:
        yield db, store
    finally:
        db.close()


def _append(db, messages):
    rows = [deepcopy(message) for message in messages]
    db.append_messages_batch("s1", rows)
    return rows


def _local_message_rows(db):
    return int(db._read_one("SELECT COUNT(*) FROM messages")[0])


def test_stale_in_place_compaction_rejects_after_concurrent_append(db_store):
    db, store = db_store
    _append(db, [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ])
    revision, active_ids = db.conversation_compaction_fence("s1")

    store.append_messages(
        "s1", [{"role": "user", "content": "concurrent"}],
        expected_revision=store.get_revision("s1"),
    )
    before = deepcopy(store.messages["s1"])

    with pytest.raises(ConversationConflictError):
        db.archive_and_compact(
            "s1", [{"role": "assistant", "content": "stale summary"}],
            expected_revision=revision, expected_active_ids=active_ids,
        )

    assert store.messages["s1"] == before
    assert [row["content"] for row in store.messages["s1"] if row.get("active", True)] == [
        "one", "two", "concurrent",
    ]
    assert _local_message_rows(db) == 0


def test_successful_in_place_compaction_stays_external_and_rebinds_ids(db_store):
    db, store = db_store
    original = _append(db, [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "keep me"},
    ])
    revision, active_ids = db.conversation_compaction_fence("s1")
    compacted = [
        {"role": "assistant", "content": "summary"},
        {"role": "user", "content": "keep me"},
    ]

    count = db.archive_and_compact(
        "s1", compacted, tail_count=1,
        model_config_patch={"_test_compaction": 1},
        expected_revision=revision, expected_active_ids=active_ids,
    )

    assert count == 2
    active = [row for row in store.messages["s1"] if row.get("active", True)]
    assert [row["content"] for row in active] == ["summary", "keep me"]
    assert all(isinstance(row.get("_row_id"), int) for row in compacted)
    assert {row["_row_id"] for row in compacted}.isdisjoint(
        {row["_row_id"] for row in original}
    )
    carried_original = next(row for row in store.messages["s1"] if row["_row_id"] == original[-1]["_row_id"])
    assert carried_original["active"] is False
    assert carried_original["compacted"] is False
    assert store.conversations["s1"]["model_config"]["_test_compaction"] == 1
    assert _local_message_rows(db) == 0
    local = db._read_one(
        "SELECT message_count FROM sessions WHERE id = ?", ("s1",)
    )
    assert local["message_count"] == 2


def test_provider_compaction_failure_never_falls_back_to_sqlite(db_store):
    db, store = db_store
    _append(db, [{"role": "user", "content": "one"}])
    revision, active_ids = db.conversation_compaction_fence("s1")
    before = deepcopy(store.messages["s1"])

    def fail(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    store.publish_compaction = fail
    with pytest.raises(RuntimeError, match="provider unavailable"):
        db.archive_and_compact(
            "s1", [{"role": "assistant", "content": "summary"}],
            expected_revision=revision, expected_active_ids=active_ids,
        )

    assert store.messages["s1"] == before
    assert _local_message_rows(db) == 0


def test_rotated_compaction_atomically_publishes_pending_parent_rows_and_child(db_store):
    db, store = db_store
    _append(db, [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ])
    revision, active_ids = db.conversation_compaction_fence("s1")
    pending = [{"role": "user", "content": "current turn"}]
    child_messages = [{"role": "assistant", "content": "handoff"}]

    db.publish_compression_child(
        parent_session_id="s1", child_session_id="s2", source="cli",
        messages=child_messages, model="fake", model_config={"x": 1},
        require_compression_lease=False,
        expected_revision=revision, expected_active_ids=active_ids,
        pending_parent_messages=pending,
    )

    assert store.conversations["s1"]["end_reason"] == "compression"
    assert store.conversations["s2"]["parent_session_id"] == "s1"
    assert [row["content"] for row in store.messages["s1"]] == [
        "one", "two", "current turn",
    ]
    assert [row["content"] for row in store.messages["s2"]] == ["handoff"]
    assert isinstance(pending[0]["_row_id"], int)
    assert isinstance(child_messages[0]["_row_id"], int)
    assert _local_message_rows(db) == 0
    parent = db._read_one("SELECT end_reason FROM sessions WHERE id = ?", ("s1",))
    child = db._read_one("SELECT parent_session_id, message_count FROM sessions WHERE id = ?", ("s2",))
    assert parent["end_reason"] == "compression"
    assert child["parent_session_id"] == "s1"
    assert child["message_count"] == 1


def test_stale_rotated_compaction_leaves_parent_current_and_no_child(db_store):
    db, store = db_store
    _append(db, [{"role": "user", "content": "one"}])
    revision, active_ids = db.conversation_compaction_fence("s1")
    store.append_messages(
        "s1", [{"role": "assistant", "content": "concurrent"}],
        expected_revision=store.get_revision("s1"),
    )

    with pytest.raises(ConversationConflictError):
        db.publish_compression_child(
            parent_session_id="s1", child_session_id="s2", source="cli",
            messages=[{"role": "assistant", "content": "handoff"}],
            require_compression_lease=False,
            expected_revision=revision, expected_active_ids=active_ids,
        )

    assert "s2" not in store.conversations
    assert store.conversations["s1"].get("end_reason") != "compression"
    assert db._read_one("SELECT id FROM sessions WHERE id = ?", ("s2",)) is None
    assert [row["content"] for row in store.messages["s1"] if row.get("active", True)] == [
        "one", "concurrent",
    ]
    assert _local_message_rows(db) == 0


def _micro_compressor(db, summary="ROLLING SUMMARY"):
    compressor = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=1,
        protect_last_n=2,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    compressor._micro_compact_enabled = True
    compressor._micro_summarize_one = lambda _text: summary
    compressor.bind_session_state(db, "s1")
    return compressor


def _micro_conversation(exchanges=6):
    messages = [{"role": "system", "content": "system prompt"}]
    for index in range(exchanges):
        messages.extend([
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "content": f"answer {index} " + "z" * 400},
        ])
    return messages


def _active_contents(store, conversation_id="s1"):
    return [
        row.get("content") for row in store.messages[conversation_id]
        if row.get("active", True)
    ]


def test_external_micro_compaction_conflict_rolls_back_local_rewrite(db_store):
    db, store = db_store
    messages = _micro_conversation()
    db.append_messages_batch("s1", messages)
    compressor = _micro_compressor(db)
    before = deepcopy(messages)
    summary_before = compressor._micro_compact_rolling_summary
    cursor_before = compressor._micro_compact_cursor
    injected = False

    def summarize(_text):
        nonlocal injected
        if not injected:
            injected = True
            store.append_messages(
                "s1", [{"role": "user", "content": "concurrent"}],
                expected_revision=store.get_revision("s1"),
            )
        return "STALE SUMMARY"

    compressor._micro_summarize_one = summarize
    result = compressor._micro_compact(messages)

    assert result == before
    assert compressor._micro_compact_rolling_summary == summary_before
    assert compressor._micro_compact_cursor == cursor_before
    assert "concurrent" in _active_contents(store)
    assert not any(
        isinstance(content, str) and "STALE SUMMARY" in content
        for content in _active_contents(store)
    )
    assert _local_message_rows(db) == 0


def test_external_micro_defrag_conflict_restores_marker_and_summary(db_store):
    db, store = db_store
    messages = _micro_conversation(exchanges=8)
    db.append_messages_batch("s1", messages)
    compressor = _micro_compressor(db)

    first = compressor._micro_compact(messages)
    assert any(message.get(COMPRESSED_SUMMARY_METADATA_KEY) for message in first)
    before = deepcopy(first)
    old_summary = "x" * 40_000
    compressor._micro_compact_rolling_summary = old_summary
    cursor_before = compressor._micro_compact_cursor
    injected = False

    def summarize(_text):
        nonlocal injected
        if not injected:
            injected = True
            store.append_messages(
                "s1", [{"role": "user", "content": "concurrent defrag"}],
                expected_revision=store.get_revision("s1"),
            )
        return "FRESH DEFRAGGED SUMMARY"

    compressor._micro_summarize_one = summarize
    result = compressor._micro_compact(first)

    assert result == before
    assert compressor._micro_compact_rolling_summary == old_summary
    assert compressor._micro_compact_cursor == cursor_before
    assert "concurrent defrag" in _active_contents(store)
    assert not any(
        isinstance(content, str) and "FRESH DEFRAGGED SUMMARY" in content
        for content in _active_contents(store)
    )
    assert _local_message_rows(db) == 0
