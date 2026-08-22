"""Durable accounting for successful context compression boundaries."""

import pytest

from hermes_state import SessionCompressionInProgressError, SessionDB


SUMMARY = [
    {"role": "user", "content": "compressed summary"},
    {"role": "assistant", "content": "continuation"},
]


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield store
    finally:
        store.close()


def _seed_session(db: SessionDB, session_id: str) -> None:
    db.create_session(session_id, source="test")
    db.append_message(session_id, "user", "original question")
    db.append_message(session_id, "assistant", "original answer")


def test_in_place_commit_increments_successful_compression_atomically(db: SessionDB) -> None:
    _seed_session(db, "session")

    db.archive_and_compact("session", SUMMARY)
    db.archive_and_compact("session", SUMMARY)

    assert db.get_session("session")["successful_compression_count"] == 2


def test_failed_in_place_commit_does_not_increment_successful_compression(
    db: SessionDB,
) -> None:
    _seed_session(db, "session")
    assert db.try_acquire_compression_lock("session", "winner") is True

    with pytest.raises(SessionCompressionInProgressError):
        db.archive_and_compact(
            "session",
            SUMMARY,
            lock_holder="loser",
        )

    assert db.get_session("session")["successful_compression_count"] == 0


def test_rotated_commit_counts_on_compression_parent(db: SessionDB) -> None:
    _seed_session(db, "parent")
    assert db.try_acquire_compression_lock("parent", "winner") is True

    db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="child",
        source="test",
        messages=SUMMARY,
        compression_lock_holder="winner",
    )

    assert db.get_session("parent")["successful_compression_count"] == 1
    assert db.get_session("child")["successful_compression_count"] == 0


def test_failed_rotated_commit_does_not_increment_successful_compression(
    db: SessionDB,
    monkeypatch,
) -> None:
    _seed_session(db, "parent")
    assert db.try_acquire_compression_lock("parent", "winner") is True

    def fail_handoff(*_args, **_kwargs):
        raise RuntimeError("handoff failed")

    monkeypatch.setattr(db, "_insert_message_rows", fail_handoff)
    with pytest.raises(RuntimeError, match="handoff failed"):
        db.publish_compression_child(
            parent_session_id="parent",
            child_session_id="child",
            source="test",
            messages=SUMMARY,
            compression_lock_holder="winner",
        )

    assert db.get_session("parent")["successful_compression_count"] == 0
    assert db.get_session("child") is None


def test_lineage_totals_sum_each_compression_segment_once(db: SessionDB) -> None:
    _seed_session(db, "parent")
    db.update_token_counts(
        "parent",
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=30,
        cache_write_tokens=4,
        reasoning_tokens=5,
        api_call_count=2,
    )
    assert db.try_acquire_compression_lock("parent", "winner") is True
    db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="child",
        source="test",
        messages=SUMMARY,
        compression_lock_holder="winner",
    )
    db.update_token_counts(
        "child",
        input_tokens=50,
        output_tokens=10,
        cache_read_tokens=7,
        cache_write_tokens=1,
        reasoning_tokens=3,
        api_call_count=1,
    )
    db.archive_and_compact("child", SUMMARY)

    assert db.get_compression_lineage_totals("child") == {
        "successful_compression_count": 2,
        "api_call_count": 3,
        "input_tokens": 150,
        "output_tokens": 30,
        "cache_read_tokens": 37,
        "cache_write_tokens": 5,
        "reasoning_tokens": 8,
        "total_tokens": 222,
    }


def test_missing_session_has_no_durable_totals(db: SessionDB) -> None:
    assert db.get_compression_lineage_totals("missing") is None


def test_reset_child_starts_a_fresh_accounting_scope(db: SessionDB) -> None:
    _seed_session(db, "old")
    db.update_token_counts("old", input_tokens=100, api_call_count=2)
    db.end_session("old", "session_reset")
    db.create_session(
        "fresh",
        source="test",
        parent_session_id="old",
        model_config={"_reset_from": "old"},
    )
    db.update_token_counts("fresh", input_tokens=7, api_call_count=1)

    totals = db.get_compression_lineage_totals("fresh")

    assert totals["api_call_count"] == 1
    assert totals["input_tokens"] == 7
    assert totals["successful_compression_count"] == 0


def test_explicit_branch_does_not_inherit_compression_parent_totals(
    db: SessionDB,
) -> None:
    _seed_session(db, "parent")
    db.update_token_counts("parent", input_tokens=100, api_call_count=2)
    db.end_session("parent", "compression")
    db.create_session(
        "branch",
        source="test",
        parent_session_id="parent",
        model_config={"_branched_from": "parent"},
    )
    db.update_token_counts("branch", input_tokens=9, api_call_count=1)

    totals = db.get_compression_lineage_totals("branch")

    assert totals["api_call_count"] == 1
    assert totals["input_tokens"] == 9
