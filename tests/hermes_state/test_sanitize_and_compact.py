"""Durable sanitation commit invariants."""

from pathlib import Path

import pytest

from hermes_state import SessionCompressionInProgressError, SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    handle = SessionDB(tmp_path / "state.db")
    handle.create_session("sess1", source="test")
    return handle


def test_sanitation_purges_rewind_inactive_rows_and_fts(db: SessionDB) -> None:
    for content in ("keep", "reply", "secret-token-alpha", "secret-token-beta"):
        db.append_message("sess1", role="user", content=content)
    secret_row = db.get_messages("sess1")[2]["id"]
    db.rewind_to_message("sess1", secret_row)
    watermark = db.get_active_message_watermark("sess1")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")

    db.sanitize_and_compact(
        "sess1",
        [{"role": "user", "content": "clean"}],
        watermark=watermark,
        lock_holder="sanitizer",
    )

    assert [row["content"] for row in db.get_messages("sess1", include_inactive=True)] == [
        "clean"
    ]
    assert db.search_messages("secret-token", include_inactive=True) == []


def test_sanitation_refuses_a_vanished_session(db: SessionDB) -> None:
    db.append_message("sess1", role="user", content="original")
    watermark = db.get_active_message_watermark("sess1")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")
    assert db.delete_session("sess1")

    with pytest.raises(ValueError, match="Session not found"):
        db.sanitize_and_compact(
            "sess1",
            [{"role": "user", "content": "orphan"}],
            watermark=watermark,
            lock_holder="sanitizer",
        )

    assert db.message_count("sess1") == 0


def test_sanitation_transaction_failure_restores_rows_and_fts(
    db: SessionDB, monkeypatch
) -> None:
    db.append_message("sess1", role="user", content="secret-token")
    watermark = db.get_active_message_watermark("sess1")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")
    monkeypatch.setattr(
        db,
        "_insert_message_rows",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("injected")),
    )

    with pytest.raises(RuntimeError, match="injected"):
        db.sanitize_and_compact(
            "sess1",
            [{"role": "user", "content": "clean"}],
            watermark=watermark,
            lock_holder="sanitizer",
        )

    assert [row["content"] for row in db.get_messages("sess1")] == [
        "secret-token"
    ]
    assert len(db.search_messages("secret-token", include_inactive=True)) == 1


def test_sanitation_lost_lease_cannot_mutate_transcript(db: SessionDB) -> None:
    db.append_message("sess1", role="user", content="original")
    watermark = db.get_active_message_watermark("sess1")

    with pytest.raises(SessionCompressionInProgressError):
        db.sanitize_and_compact(
            "sess1",
            [{"role": "user", "content": "clean"}],
            watermark=watermark,
            lock_holder="not-held",
        )

    assert [row["content"] for row in db.get_messages("sess1")] == ["original"]
