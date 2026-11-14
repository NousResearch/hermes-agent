"""A turn whose reply is saved but not yet delivered is recovered once, not answered twice."""

import asyncio
import os
import sqlite3
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import delivery_ledger as ledger
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionStore
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

SESSION_KEY = "agent:main:telegram:dm:chat"


def pending_session(runner, session_key=SESSION_KEY, chat_id="chat"):
    entry = SessionEntry(
        session_key=session_key,
        session_id="sid-" + chat_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=make_restart_source(chat_id=chat_id),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_interrupted",
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries[session_key] = entry
    return entry


def save_reply(session_key=SESSION_KEY, *, oid="owed", state="pending"):
    ledger.record_obligation(
        obligation_id=oid, session_key=session_key, platform="telegram", chat_id="chat",
        thread_id=None, content="the answer",
    )
    if state == "attempting":
        ledger.mark_attempting(oid)
    elif state == "failed":
        ledger.mark_failed(oid, "send_path_degraded")
    elif state == "flood":
        ledger.mark_failed(oid, "flood_control:60")


def read(oid="owed"):
    with sqlite3.connect(ledger._db_path()) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM delivery_obligations WHERE obligation_id=?", (oid,)).fetchone())


def resume_runner():
    runner, adapter = make_restart_runner()
    adapter.handle_message = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "attempting", "failed", "flood"])
async def test_auto_resume_leaves_a_session_whose_reply_is_still_owed(state):
    runner, adapter = resume_runner()
    answered = pending_session(runner)
    save_reply(state=state)

    assert runner._schedule_resume_pending_sessions(platform=Platform.TELEGRAM) == 0
    await asyncio.sleep(0)

    adapter.handle_message.assert_not_awaited()
    assert answered.resume_pending


@pytest.mark.asyncio
async def test_auto_resume_still_runs_for_a_session_with_no_saved_reply():
    runner, adapter = resume_runner()
    pending_session(runner)
    save_reply("agent:main:telegram:dm:other")

    assert runner._schedule_resume_pending_sessions(platform=Platform.TELEGRAM) == 1
    await asyncio.sleep(0)

    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_resume_runs_again_once_the_saved_reply_is_delivered():
    runner, _ = resume_runner()
    pending_session(runner)
    save_reply()
    assert runner._schedule_resume_pending_sessions(platform=Platform.TELEGRAM) == 0

    ledger.mark_delivered("owed")

    assert runner._schedule_resume_pending_sessions(platform=Platform.TELEGRAM) == 1


def test_owed_reply_probe_creates_nothing_and_refuses_to_wait_on_a_writer():
    assert ledger.pending_resume_sessions() == set()
    assert not ledger._db_path().exists()

    save_reply()
    assert ledger.pending_resume_sessions() == {SESSION_KEY}

    ledger._DB_LOCK.acquire()
    try:
        with pytest.raises(RuntimeError, match="busy"):
            ledger.pending_resume_sessions()
    finally:
        ledger._DB_LOCK.release()


@pytest.mark.asyncio
async def test_auto_resume_is_deferred_when_the_ledger_cannot_be_read():
    runner, adapter = resume_runner()
    answered = pending_session(runner)
    save_reply()

    ledger._DB_LOCK.acquire()
    try:
        assert runner._schedule_resume_pending_sessions(platform=Platform.TELEGRAM) == 0
    finally:
        ledger._DB_LOCK.release()
    await asyncio.sleep(0)

    adapter.handle_message.assert_not_awaited()
    assert answered.resume_pending


def test_a_failed_save_leaves_the_resume_flag_set_and_a_later_clear_survives_reload(tmp_path, monkeypatch):
    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
    monkeypatch.setattr(store, "_routing_db_method", lambda name: None)
    runner = Mock(session_store=store)
    with store._lock:
        store._ensure_loaded_locked()
    entry = pending_session(runner)
    store._save()

    save = store._save_sessions_json
    monkeypatch.setattr(store, "_save_sessions_json", Mock(side_effect=OSError("disk unavailable")))
    with pytest.raises(OSError, match="disk unavailable"):
        store.clear_resume_pending(entry.session_key)
    assert entry.resume_pending
    assert entry.resume_reason == "restart_interrupted"

    monkeypatch.setattr(store, "_save_sessions_json", save)
    assert store.clear_resume_pending(entry.session_key)
    assert not entry.resume_pending

    loaded = SessionStore(sessions_dir=store.sessions_dir, config=config)
    monkeypatch.setattr(loaded, "_routing_db_method", lambda name: None)
    with loaded._lock:
        loaded._ensure_loaded_locked()
    assert not loaded._entries[entry.session_key].resume_pending


def orphan_and_expire_flood_wait():
    """The row's owner died and the platform's wait has passed: the next boot claims it."""
    with sqlite3.connect(ledger._db_path()) as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=NULL, owner_started_at=NULL, adapter_profile=NULL, "
            "updated_at=updated_at-120"
        )


def test_a_boot_claimed_flood_row_carries_the_refusal_it_was_waiting_on():
    save_reply(state="flood")
    orphan_and_expire_flood_wait()

    claimed = ledger.sweep_recoverable(deliverable_targets={("telegram", None)})

    assert len(claimed) == 1 and not claimed[0].get("adopted")
    assert claimed[0]["last_error"] == "flood_control:60"


@pytest.mark.asyncio
async def test_a_boot_claimed_flood_row_keeps_its_wait_when_no_adapter_can_send_it():
    runner, _ = resume_runner()
    save_reply(state="flood")
    orphan_and_expire_flood_wait()
    (row,) = ledger.sweep_recoverable(deliverable_targets={("telegram", None)})
    assert read()["state"] == "attempting" and read()["owner_pid"] == os.getpid()

    # The adapter went away between the claim and the dispatch.
    runner.adapters = {}
    assert await runner._obligation_adapter(row) is None

    released = read()
    assert (released["state"], released["attempts"], released["last_error"]) == ("failed", 0, "flood_control:60")
    assert ledger.pending_retries()[0]["not_before"] > released["updated_at"]
