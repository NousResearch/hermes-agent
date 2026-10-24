"""Recover prior history before accepted pending work, preserving admission order."""

import sqlite3
from unittest.mock import patch

import pytest

from gateway.shutdown_flush import (
    flush_overflow_to_file,
    flush_pending_to_file,
    recover_pending_spool,
    spool_dropped_transcript_message,
)
from hermes_state import SessionDB
from tests.gateway.test_drained_event_conservation import (
    KEY,
    _make_runner,
    _overflow,
    _text_event,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("transfer_buffer", [False, True])
async def test_shutdown_replays_admission_order_across_queue_owners(
    monkeypatch, tmp_path, transfer_buffer,
):
    runner, adapter = _make_runner(monkeypatch)
    a, b, c = (_text_event(text, text) for text in ("A pending", "B overflow", "C debounce"))
    a.preserve_prompt_pins = True
    assert runner._queue_or_replace_pending_event(KEY, a) is True
    assert runner._queue_or_replace_pending_event(KEY, b) is True
    await adapter._queue_text_debounce(KEY, c)
    if transfer_buffer:
        assert await adapter._flush_text_debounce_now(KEY) is True
    else:
        runner._BUSY_QUEUE_MAX_PENDING = 2
        assert await adapter._flush_text_debounce_now(KEY) is False
        assert adapter._text_debounce_store()[KEY].event is c
    # Teardown visits adapter owners A/C before the runner-owned B tail.
    await adapter.cancel_background_tasks()
    assert flush_overflow_to_file({KEY: list(_overflow(runner))}) == (2 if transfer_buffer else 1)

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.ensure_session("recovered", "gateway")
        recovered, held = recover_pending_spool(
            db, session_resolver=lambda key, not_after=None: ("recovered", db),
        )
        assert (recovered, held) == (3, set())
        assert [row["content"] for row in db.get_messages("recovered")] == [a.text, b.text, c.text]
    finally:
        db.close()


@pytest.mark.parametrize("history_spooled_first", [False, True])
def test_prior_history_replays_before_unexecuted_queue(monkeypatch, tmp_path, history_spooled_first):
    runner, adapter = _make_runner(monkeypatch)
    history = {"role": "assistant", "content": "H history"}
    if history_spooled_first:
        assert spool_dropped_transcript_message("recovered", history)
    a = _text_event("A pending", "A")
    assert runner._queue_or_replace_pending_event(KEY, a) is True
    if not history_spooled_first:
        assert spool_dropped_transcript_message("recovered", history)
    assert flush_pending_to_file(adapter._pending_messages) == 1
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.ensure_session("recovered", "gateway")
        assert recover_pending_spool(
            db, session_resolver=lambda key, not_after=None: ("recovered", db),
        ) == (2, set())
        assert [(row["role"], row["content"]) for row in db.get_messages("recovered")] == [
            ("assistant", "H history"), ("user", "A pending"),
        ]
    finally:
        db.close()


@pytest.mark.parametrize("failed_content", ["H history", "A pending"])
@pytest.mark.parametrize("retry_mode", ["boot", "live", "stalled_live", "route_failure", "route_missing", "spool_unreadable"])
def test_failed_mixed_recovery_retries_before_later_and_live_rows(
    monkeypatch, tmp_path, failed_content, retry_mode,
):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    runner, adapter = _make_runner(monkeypatch)
    for text in ("A pending", "B overflow"):
        assert runner._queue_or_replace_pending_event(KEY, _text_event(text, text)) is True
    assert spool_dropped_transcript_message("recovered", {"role": "assistant", "content": "H history"})
    assert flush_pending_to_file(adapter._pending_messages) == 1
    assert flush_overflow_to_file({KEY: list(_overflow(runner))}) == 1
    assert spool_dropped_transcript_message("other", {"role": "user", "content": "other session"})
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        for sid in ("recovered", "other"):
            db.ensure_session(sid, "gateway")
        def resolver(key, not_after=None):
            return "recovered", db

        append = db.append_message

        def fail_one(**kwargs):
            if kwargs["content"] == failed_content:
                raise sqlite3.OperationalError("controlled database outage")
            return append(**kwargs)

        monkeypatch.setattr(db, "append_message", fail_one)
        expected_before_retry = [] if failed_content == "H history" else ["H history"]
        assert recover_pending_spool(db, session_resolver=resolver) == (
            1 + len(expected_before_retry), {"recovered"},
        )
        assert [row["content"] for row in db.get_messages("recovered")] == expected_before_retry
        assert [row["content"] for row in db.get_messages("other")] == ["other session"]
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
        store._loaded = True
        store._db = db
        monkeypatch.setattr(store, "resolve_session_id_for_key", resolver)
        store.mark_spooled_drop_sessions({"recovered"})
        stalled_rows = []
        if retry_mode == "stalled_live":
            store._MAX_PENDING_PER_SESSION = 1
            stalled_rows = [f"stalled live {idx}" for idx in range(3)]
            for text in stalled_rows:
                store.append_to_transcript("recovered", {"role": "user", "content": text})
            assert [row["content"] for row in db.get_messages("recovered")] == expected_before_retry
        monkeypatch.setattr(db, "append_message", append)
        if retry_mode in {"route_failure", "route_missing", "spool_unreadable"}:
            from pathlib import Path

            with monkeypatch.context() as blocked:
                def fail_route(*args, **kwargs):
                    raise RuntimeError("controlled routing failure")

                read_text = Path.read_text

                def fail_spool_read(path, *args, **kwargs):
                    if path.parent.name == "pending_messages":
                        raise OSError("controlled spool read failure")
                    return read_text(path, *args, **kwargs)

                if retry_mode == "route_failure":
                    blocked.setattr(store, "resolve_session_id_for_key", fail_route)
                elif retry_mode == "route_missing":
                    blocked.setattr(store, "resolve_session_id_for_key", lambda *a, **k: None)
                else:
                    blocked.setattr(Path, "read_text", fail_spool_read)
                store.append_to_transcript("recovered", {"role": "user", "content": "held live"})
                # The scan resolves retained routes before replaying any row, so an unavailable
                # route holds the entire remaining batch, including its older history.
                assert [row["content"] for row in db.get_messages("recovered")] == expected_before_retry
            stalled_rows = ["held live"]
        if retry_mode == "boot":
            assert recover_pending_spool(db, session_resolver=resolver) == (
                3 - len(expected_before_retry), set(),
            )
        store.append_to_transcript("recovered", {"role": "user", "content": "live"})
        assert [row["content"] for row in db.get_messages("recovered")] == [
            "H history", "A pending", "B overflow", *stalled_rows, "live",
        ]
    finally:
        db.close()
