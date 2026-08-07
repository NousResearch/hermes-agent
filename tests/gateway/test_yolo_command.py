"""Tests for gateway /yolo session scoping."""

import os
import types

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from tools.approval import disable_session_yolo, is_session_yolo_enabled


@pytest.fixture(autouse=True)
def _clean_yolo_state(monkeypatch):
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    disable_session_yolo("agent:main:telegram:dm:chat-a")
    disable_session_yolo("agent:main:telegram:dm:chat-b")
    yield
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    disable_session_yolo("agent:main:telegram:dm:chat-a")
    disable_session_yolo("agent:main:telegram:dm:chat-b")


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    return runner


def _make_event(chat_id: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id=f"user-{chat_id}",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(text="/yolo", source=source)


@pytest.mark.asyncio
async def test_yolo_command_toggles_only_current_session(monkeypatch):
    runner = _make_runner()

    event_a = _make_event("chat-a")
    session_a = runner._session_key_for_source(event_a.source)
    session_b = runner._session_key_for_source(_make_event("chat-b").source)

    result_on = await runner._handle_yolo_command(event_a)

    assert "ON" in result_on
    assert is_session_yolo_enabled(session_a) is True
    assert is_session_yolo_enabled(session_b) is False
    assert os.environ.get("HERMES_YOLO_MODE") is None

    result_off = await runner._handle_yolo_command(event_a)

    assert "OFF" in result_off
    assert is_session_yolo_enabled(session_a) is False
    assert os.environ.get("HERMES_YOLO_MODE") is None


@pytest.mark.asyncio
async def test_yolo_command_persists_flag_to_session_row():
    """/yolo must persist yolo_mode to the session row so a gateway restart
    can re-hydrate it — otherwise the bypass reverts to approvals.mode's
    default ("smart") on the next process."""
    runner = _make_runner()
    calls = []

    class _Store:
        def peek_session_id(self, session_key):
            return "sess-1"

    class _Db:
        async def set_session_yolo(self, session_id, enabled):
            calls.append((session_id, enabled))

    runner.session_store = _Store()
    runner._session_db = _Db()

    await runner._handle_yolo_command(_make_event("chat-a"))
    assert calls == [("sess-1", True)]

    await runner._handle_yolo_command(_make_event("chat-a"))
    assert calls == [("sess-1", True), ("sess-1", False)]


@pytest.mark.asyncio
async def test_hydrate_yolo_flag_restores_persisted_bypass(tmp_path):
    """First message in a fresh gateway process must re-apply a persisted
    /yolo bypass from the session row."""
    from hermes_state import AsyncSessionDB, SessionDB
    from tools.approval import disable_session_yolo, is_session_yolo_enabled

    session_key = "agent:main:telegram:dm:chat-a"
    disable_session_yolo(session_key)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(
            session_id="sess-1",
            source="telegram",
            model="m",
            model_config={"yolo_mode": True},
        )
        runner = _make_runner()
        runner._session_db = AsyncSessionDB(db)
        entry = types.SimpleNamespace(session_id="sess-1")

        await runner._hydrate_yolo_flag(entry, session_key)
        assert is_session_yolo_enabled(session_key) is True
    finally:
        disable_session_yolo(session_key)
        db.close()


@pytest.mark.asyncio
async def test_hydrate_yolo_flag_noop_when_flag_absent(tmp_path):
    """Sessions that never toggled yolo must stay disarmed after restart."""
    from hermes_state import AsyncSessionDB, SessionDB
    from tools.approval import disable_session_yolo, is_session_yolo_enabled

    session_key = "agent:main:telegram:dm:chat-b"
    disable_session_yolo(session_key)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="sess-2", source="telegram", model="m")
        runner = _make_runner()
        runner._session_db = AsyncSessionDB(db)
        entry = types.SimpleNamespace(session_id="sess-2")

        await runner._hydrate_yolo_flag(entry, session_key)
        assert is_session_yolo_enabled(session_key) is False
    finally:
        disable_session_yolo(session_key)
        db.close()
