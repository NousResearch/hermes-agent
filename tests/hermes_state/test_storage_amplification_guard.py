"""Fail-closed guards for per-session physical-row amplification."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

import hermes_state
from hermes_state import (
    SessionDB,
    SessionStorageAmplificationError,
    classify_persistence_error,
)


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    yield store
    store.close()


def _message_count(store: SessionDB, session_id: str, *, include_inactive: bool) -> int:
    return len(
        store.get_messages(
            session_id,
            include_inactive=include_inactive,
            include_compacted=True,
        )
    )


def test_single_append_stops_before_crossing_storage_limit(db, monkeypatch):
    sid = "append-cap"
    db.create_session(sid, "test")
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 3)

    for i in range(3):
        db.append_message(sid, "user", f"m-{i}")

    with pytest.raises(SessionStorageAmplificationError) as exc_info:
        db.append_message(sid, "assistant", "blocked")

    assert exc_info.value.stored_rows == 3
    assert exc_info.value.attempted_rows == 1
    assert exc_info.value.limit == 3
    assert "write was stopped before persistence" in str(exc_info.value)
    assert _message_count(db, sid, include_inactive=True) == 3


def test_delegation_delivery_stops_before_crossing_storage_limit(db, monkeypatch):
    sid = "delivery-cap"
    db.create_session(sid, "test")
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 1)
    db.append_message(sid, "user", "baseline")

    with pytest.raises(SessionStorageAmplificationError):
        db.append_delegation_delivery(
            sid,
            "blocked",
            {"delegation_id": "delegation-1", "delivery_notice": "done"},
        )

    assert _message_count(db, sid, include_inactive=True) == 1


def test_sparse_id_span_falls_back_to_exact_count(db, monkeypatch):
    sid = "sparse-cap"
    noise = "noise"
    db.create_session(sid, "test")
    db.create_session(noise, "test")
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 0)
    db.append_message(sid, "user", "first")
    db.append_messages_batch(
        noise,
        [{"role": "assistant", "content": f"noise-{i}"} for i in range(10)],
    )
    db.append_message(sid, "assistant", "second")

    # The target owns only two rows, but their global ids span the ten rows
    # written by another session. The conservative id bound must fall back to
    # an exact bounded count instead of rejecting a safe third row.
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 3)
    db.append_message(sid, "user", "third")
    with pytest.raises(SessionStorageAmplificationError):
        db.append_message(sid, "assistant", "blocked")
    assert _message_count(db, sid, include_inactive=True) == 3


def test_parallel_handles_do_not_both_cross_the_storage_limit(db, monkeypatch):
    sid = "concurrent-cap"
    db.create_session(sid, "test")
    other = SessionDB(db.db_path)
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 1)

    def write(store):
        try:
            store.append_message(sid, "user", "synthetic message")
            return "saved"
        except SessionStorageAmplificationError:
            return "blocked"

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(write, (db, other)))
    finally:
        other.close()
    assert sorted(outcomes) == ["blocked", "saved"]
    assert _message_count(db, sid, include_inactive=True) == 1


def test_batch_append_is_atomic_when_projected_rows_exceed_limit(db, monkeypatch):
    sid = "batch-cap"
    db.create_session(sid, "test")
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 4)
    db.append_messages_batch(
        sid,
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
    )

    with pytest.raises(SessionStorageAmplificationError):
        db.append_messages_batch(
            sid,
            [
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
                {"role": "user", "content": "e"},
            ],
        )

    assert _message_count(db, sid, include_inactive=True) == 2


def test_compaction_refuses_new_generation_and_rolls_back(db, monkeypatch):
    sid = "compact-cap"
    db.create_session(sid, "test")
    db.append_messages_batch(
        sid,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ],
    )
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 4)

    with pytest.raises(SessionStorageAmplificationError):
        db.archive_and_compact(
            sid,
            [
                {"role": "assistant", "content": "summary"},
                {"role": "user", "content": "tail"},
            ],
        )

    assert _message_count(db, sid, include_inactive=True) == 3
    assert _message_count(db, sid, include_inactive=False) == 3


def test_named_coverage_clones_cannot_bypass_storage_limit(db, monkeypatch):
    sid = "named-coverage-cap"
    db.create_session(sid, "test")
    db.append_messages_batch(sid, [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "unseen"},
    ])
    with db._read_ctx() as conn:
        ids = [row[0] for row in conn.execute(
            "SELECT id FROM messages WHERE session_id = ? AND active = 1 ORDER BY id", (sid,)
        )]
    assert len(ids) == 3
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 4)

    # One summary row would fit, but the uncovered row must also be cloned.
    with pytest.raises(SessionStorageAmplificationError):
        db.archive_and_compact(
            sid, [{"role": "assistant", "content": "summary"}], covered_ids=ids[:2]
        )
    assert _message_count(db, sid, include_inactive=True) == 3
    assert _message_count(db, sid, include_inactive=False) == 3


def test_tip_only_resume_still_blocks_large_physical_archive(db, monkeypatch):
    sid = "resume-cap"
    db.create_session(sid, "test")
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 0)
    db.append_messages_batch(
        sid,
        [{"role": "user", "content": f"m-{i}"} for i in range(5)],
    )
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 4)

    with pytest.raises(SessionStorageAmplificationError):
        db.assert_resume_safe(sid, max_messages=100, tip_only=True)


def test_full_replace_can_recover_an_already_amplified_session(db, monkeypatch):
    sid = "recovery"
    db.create_session(sid, "test")
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 0)
    db.append_messages_batch(
        sid,
        [{"role": "user", "content": f"old-{i}"} for i in range(6)],
    )
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 3)

    db.replace_messages(
        sid,
        [
            {"role": "user", "content": "kept"},
            {"role": "assistant", "content": "answer"},
        ],
    )

    assert _message_count(db, sid, include_inactive=True) == 2
    assert db.assert_storage_safe(sid) == 2


def test_default_storage_ceiling_is_enabled():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["sessions"]["max_storage_messages"] == 500_000


def test_amplification_error_has_a_dedicated_persistence_cause() -> None:
    error = SessionStorageAmplificationError("s", 500_001, 1, 500_000)
    assert classify_persistence_error(error) == "amplification"
    assert classify_persistence_error(str(error)) == "amplification"


def test_compression_child_tail_cannot_bypass_the_ceiling(
    db: SessionDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = "parent"
    child = "child"
    db.create_session(parent, source="test")
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 0)
    db.append_message(parent, role="user", content="baseline")
    watermark = db.get_active_message_watermark(parent)
    db.append_message(parent, role="assistant", content="concurrent-1")
    db.append_message(parent, role="user", content="concurrent-2")

    # Initial child handoff (1 row) fits; handoff + raw tail clone (3 rows)
    # does not. The whole publication transaction must roll back.
    monkeypatch.setattr(hermes_state, "resolved_max_storage_messages", lambda: 2)
    with pytest.raises(SessionStorageAmplificationError):
        db.publish_compression_child(
            parent_session_id=parent,
            child_session_id=child,
            source="test",
            messages=[{"role": "assistant", "content": "summary"}],
            require_compression_lease=False,
            watermark=watermark,
        )

    assert db.get_session(child) is None
    parent_row = db.get_session(parent)
    assert parent_row is not None
    assert parent_row["ended_at"] is None
    assert _message_count(db, parent, include_inactive=True) == 3
