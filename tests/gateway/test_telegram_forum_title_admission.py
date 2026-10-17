"""Forum-title updates remain metadata while a real session is busy or admission is denied."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from hermes_state import AsyncSessionDB, SessionDB
from plugins.platforms.telegram.adapter import TelegramAdapter


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="-100123", chat_type="group",
        user_id="7", user_name="tester", thread_id="42",
    )


def _update(name, *, created=False, is_bot=False):
    topic = SimpleNamespace(name=name)
    return SimpleNamespace(
        update_id=99,
        effective_message=SimpleNamespace(
            from_user=SimpleNamespace(id=7, is_bot=is_bot),
            forum_topic_created=topic if created else None,
            forum_topic_edited=None if created else topic,
        ),
    )


def _adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._build_message_event = MagicMock(
        side_effect=lambda *_args, **_kwargs: MessageEvent(
            text="", source=_source(), message_id="99"
        )
    )
    adapter._is_user_authorized_from_message = MagicMock(return_value=True)
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
async def test_create_then_rename_updates_real_session_without_turn_or_activity(tmp_path, busy):
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    entry = store.get_or_create_session(_source())
    original_activity = entry.updated_at
    db = store._db
    assert isinstance(db, SessionDB)
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)
    runner._hm_admit_event = AsyncMock(side_effect=lambda event: (event, event.source, False))
    runner._hm_estop_gate = MagicMock(side_effect=AssertionError("agent turn path reached"))
    adapter = _adapter()
    adapter.set_message_handler(runner._handle_message)
    pending = MessageEvent(text="pending human input", source=_source())
    if busy:
        adapter._active_sessions[entry.session_key] = asyncio.Event()
        task = asyncio.current_task()
        assert task is not None
        adapter._session_tasks[entry.session_key] = task
        adapter._pending_messages[entry.session_key] = pending
    try:
        for index, name in enumerate(("Hermes: updates", "Hermes: releases")):
            await adapter._handle_forum_topic_name_update(
                _update(name, created=index == 0), SimpleNamespace()
            )
            assert db.get_session_title(entry.session_id) == name
            assert store.get_or_create_session(_source(), touch_activity=False).session_id == entry.session_id
            assert entry.updated_at == original_activity
        assert runner._hm_admit_event.await_count == 2
        runner._hm_estop_gate.assert_not_called()
        if busy:
            assert adapter._pending_messages[entry.session_key] is pending
            assert not adapter._active_sessions[entry.session_key].is_set()
        else:
            assert not adapter._pending_messages
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authorized,is_bot,name,rejected_route",
    [(False, False, "Title", False), (True, True, "Title", False),
     (True, False, "  ", False), (True, False, "Title", True)],
)
async def test_unauthorized_bot_blank_and_rejected_route_updates_are_ignored(
    authorized, is_bot, name, rejected_route
):
    adapter = _adapter()
    adapter._is_user_authorized_from_message = MagicMock(return_value=authorized)
    if rejected_route:
        source = _source()
        source.profile_route_rejected = True
        adapter._build_message_event = MagicMock(return_value=MessageEvent(text="", source=source))
    handler = AsyncMock()
    adapter.set_message_handler(handler)

    await adapter._handle_forum_topic_name_update(_update(name, is_bot=is_bot), SimpleNamespace())

    handler.assert_not_awaited()
    assert not adapter._active_sessions
    assert not adapter._pending_messages


@pytest.mark.asyncio
async def test_gateway_admission_denial_never_persists_a_title():
    runner = object.__new__(GatewayRunner)
    runner._hm_admit_event = AsyncMock(return_value=None)
    persist_name = AsyncMock()
    runner._handle_telegram_forum_topic_name = persist_name
    adapter = _adapter()
    adapter.set_message_handler(runner._handle_message)

    await adapter._handle_forum_topic_name_update(_update("Denied title"), SimpleNamespace())

    runner._hm_admit_event.assert_awaited_once()
    persist_name.assert_not_awaited()
