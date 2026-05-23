"""PR6 full transcript reconciliation tests for DAG context."""

import pytest

from agent.context_dag_reconcile import reconcile_full_transcript
from agent.context_dag_store import ContextDAGStore
from hermes_state import SessionDB


def _db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", "test")
    return db


def _msg(role, content, **extra):
    payload = {"role": role, "content": content}
    payload.update(extra)
    return payload


def test_full_transcript_reconcile_imports_missing_messages_idempotently(tmp_path):
    db = _db(tmp_path)
    store = ContextDAGStore(db)
    transcript = [_msg("user", "one"), _msg("assistant", "two")]

    first = reconcile_full_transcript(store, "s1", transcript)
    second = reconcile_full_transcript(store, "s1", transcript)

    assert first.inserted == 2
    assert first.duplicates_skipped == 0
    assert first.checkpoint_advanced is True
    assert second.inserted == 0
    assert second.duplicates_skipped == 2
    assert second.checkpoint_advanced is True
    rows = db.get_messages("s1")
    assert [r["content"] for r in rows] == ["one", "two"]
    checkpoint = store.read_checkpoint("s1")
    assert checkpoint is not None
    assert checkpoint.last_ingested_message_id == rows[-1]["id"]
    assert checkpoint.anchor_hash == store.deterministic_message_hash(transcript[-1])
    assert checkpoint.metadata["transcript_message_count"] == 2


def test_reconcile_does_not_advance_checkpoint_when_ingest_fails(tmp_path, monkeypatch):
    db = _db(tmp_path)
    store = ContextDAGStore(db)
    transcript = [_msg("user", "one"), _msg("assistant", "two")]
    original = db.append_message
    calls = {"count": 0}

    def flaky_add_message(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("boom")
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "append_message", flaky_add_message)

    with pytest.raises(RuntimeError):
        reconcile_full_transcript(store, "s1", transcript)

    assert store.read_checkpoint("s1") is None
    assert [r["content"] for r in db.get_messages("s1")] == ["one"]


def test_reconcile_preserves_repeated_transcript_message_occurrences(tmp_path):
    db = _db(tmp_path)
    store = ContextDAGStore(db)
    transcript = [_msg("user", "same"), _msg("user", "same")]

    first = reconcile_full_transcript(store, "s1", transcript)
    second = reconcile_full_transcript(store, "s1", transcript)

    assert first.inserted == 2
    assert first.duplicates_skipped == 0
    assert first.warnings == []
    assert second.inserted == 0
    assert second.duplicates_skipped == 2
    assert second.matched == 2
    assert [r["content"] for r in db.get_messages("s1")] == ["same", "same"]


def test_reconcile_consumes_existing_duplicate_occurrences_before_inserting_missing_ones(tmp_path):
    db = _db(tmp_path)
    store = ContextDAGStore(db)
    db.append_message("s1", "user", "same")
    transcript = [_msg("user", "same"), _msg("user", "same")]

    result = reconcile_full_transcript(store, "s1", transcript)

    assert result.inserted == 1
    assert result.duplicates_skipped == 1
    assert result.matched == 1
    assert [r["content"] for r in db.get_messages("s1")] == ["same", "same"]


def test_reconcile_reasoning_details_is_idempotent_across_db_json_serialization(tmp_path):
    db = _db(tmp_path)
    store = ContextDAGStore(db)
    transcript = [
        _msg(
            "assistant",
            "answer",
            reasoning_details=[{"type": "summary", "text": "thinking"}],
        )
    ]

    first = reconcile_full_transcript(store, "s1", transcript)
    second = reconcile_full_transcript(store, "s1", transcript)

    assert first.inserted == 1
    assert second.inserted == 0
    assert second.duplicates_skipped == 1
    rows = db.get_messages("s1")
    assert len(rows) == 1
    assert isinstance(rows[0]["reasoning_details"], str)


def test_reconcile_records_drift_for_edited_or_out_of_order_messages_without_rewrite(tmp_path):
    db = _db(tmp_path)
    store = ContextDAGStore(db)
    db.append_message("s1", "user", "original-a")
    db.append_message("s1", "assistant", "original-b")
    transcript = [_msg("assistant", "original-b"), _msg("user", "edited-a")]

    result = reconcile_full_transcript(store, "s1", transcript)

    assert result.inserted == 1
    assert any(w["code"] == "out_of_order_message" for w in result.warnings)
    assert any(w["code"] == "content_drift" for w in result.warnings)
    # Existing raw rows are not destructively rewritten; edited transcript is mirrored as an additive row.
    assert [r["content"] for r in db.get_messages("s1")] == ["original-a", "original-b", "edited-a"]
    checkpoint = store.read_checkpoint("s1")
    assert checkpoint.metadata["warnings"]


def test_reconcile_missing_anchor_warns_and_does_not_advance_checkpoint(tmp_path):
    db = _db(tmp_path)
    store = ContextDAGStore(db)
    first = reconcile_full_transcript(store, "s1", [_msg("user", "anchor")])
    assert first.checkpoint_advanced is True
    db.replace_messages("s1", [_msg("user", "rewritten elsewhere")])

    result = reconcile_full_transcript(store, "s1", [_msg("user", "new transcript")])

    assert result.checkpoint_advanced is False
    assert any(w["code"] == "anchor_missing" for w in result.warnings)
    checkpoint = store.read_checkpoint("s1")
    assert checkpoint.anchor_hash == ContextDAGStore.deterministic_message_hash(_msg("user", "anchor"))
