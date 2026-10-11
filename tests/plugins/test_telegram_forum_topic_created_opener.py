"""A newly created Telegram DM topic's opening message must land in that topic (Refs #136567).

Telegram usually stamps the topic's opening message with the new ``message_thread_id``, but it
sometimes delivers that opener with NO thread id at all. The ``forum_topic_created`` service
message that precedes it is then the only update carrying the new topic id. Core registered no
handler for service messages, so the adapter never learned the topic and the opener reached the
runner as a thread-less lobby-shaped DM event - gateway lobby recovery then pinned it to the most
recently bound topic and the new topic stayed empty.

Transport is a stand-in; registration, gating, batching and PTB dispatch are real.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("telegram")
from telegram import Update
from telegram.request import BaseRequest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class NoNetwork(BaseRequest):
    def __init__(self, bot_id=111):
        self.bot_id = bot_id

    @property
    def read_timeout(self):
        return 1

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        assert endpoint == "getMe", "No Telegram polls, sends or downloads allowed"
        return 200, json.dumps({"ok": True, "result": {
            "id": self.bot_id, "is_bot": True, "first_name": "Offline", "username": "offline_bot",
        }}).encode()


def _chat(chat, group, *, is_forum=False):
    payload = {"id": chat, "type": "supergroup" if group else "private"}
    if group:
        payload["is_forum"] = is_forum
    return payload


def topic_created(bot, *, uid=10, chat=42, user=88, thread=40798, message_id=40799, name="Prova 3", group=False):
    """``forum_topic_created`` service message: carries the NEW topic's thread id, never text."""
    return Update.de_json({"update_id": uid, "message": {
        "message_id": message_id, "date": 1800000000, "chat": _chat(chat, group, is_forum=group),
        "from": {"id": user, "is_bot": False, "first_name": "Human"},
        "message_thread_id": thread, "is_topic_message": True,
        "forum_topic_created": {"name": name, "icon_color": 7322096},
    }}, bot)


def opener(bot, *, uid=11, chat=42, user=88, message_id=40800, text="Prova 3", thread=None,
           is_topic=None, group=False):
    """The topic's opening user message. ``thread=None`` is the shape Telegram regressed to."""
    message = {
        "message_id": message_id, "date": 1800000001, "chat": _chat(chat, group, is_forum=group),
        "from": {"id": user, "is_bot": False, "first_name": "Human"}, "text": text,
    }
    if thread is not None:
        message["message_thread_id"] = thread
    if is_topic is not None:
        message["is_topic_message"] = is_topic
    return Update.de_json({"update_id": uid, "message": message}, bot)


@asynccontextmanager
async def connected(monkeypatch, *, bot_id=111):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=f"{bot_id}:offline-test", extra={}))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "88")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "")
    monkeypatch.setattr(adapter, "_build_ptb_requests", AsyncMock(return_value=(NoNetwork(bot_id), NoNetwork(bot_id))))
    monkeypatch.setattr(adapter, "_start_polling_mode", AsyncMock())
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", lambda: None)
    monkeypatch.setattr(adapter, "_restart_task_attr", lambda name, coroutine: coroutine.close())
    monkeypatch.setattr(adapter, "_set_status_indicator", AsyncMock())
    delivered = []
    adapter._message_handler = AsyncMock()
    monkeypatch.setattr(adapter, "_start_session_processing", lambda event, session_key: delivered.append(event) or True)
    assert await adapter.connect(is_reconnect=False)
    try:
        yield adapter, adapter._app, delivered
    finally:
        await adapter.disconnect()
        store = getattr(adapter, "_session_store", None)
        if store is not None:
            store.close_all_db_handles()


async def _deliver(app, adapter, *updates):
    for item in updates:
        await app.process_update(item)
    await asyncio.gather(*adapter._pending_text_batch_tasks.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("is_topic", [None, True])
async def test_new_topic_opener_lands_in_the_new_topic(monkeypatch, is_topic):
    """The service message's topic id is adopted by the opening message that follows it."""
    async with connected(monkeypatch) as (adapter, app, delivered):
        await _deliver(app, adapter, topic_created(app.bot), opener(app.bot, is_topic=is_topic))

    assert [event.text for event in delivered] == ["Prova 3"], "the service message must not reach the agent"
    assert delivered[0].source.thread_id == "40798"


@pytest.mark.asyncio
async def test_threadless_message_without_topic_creation_stays_lobby(monkeypatch):
    """No preceding service message: an ordinary thread-less DM keeps its lobby shape."""
    async with connected(monkeypatch) as (adapter, app, delivered):
        await _deliver(app, adapter, opener(app.bot))

    assert delivered[0].source.thread_id is None


@pytest.mark.asyncio
async def test_non_adjacent_threadless_message_is_not_adopted(monkeypatch):
    """Only the message immediately after the creation may claim the new topic."""
    async with connected(monkeypatch) as (adapter, app, delivered):
        await _deliver(app, adapter, topic_created(app.bot), opener(app.bot, message_id=40805))

    assert delivered[0].source.thread_id is None


@pytest.mark.asyncio
async def test_pending_topic_creation_expires(monkeypatch):
    """A stale creation record must not claim a much later thread-less message."""
    async with connected(monkeypatch) as (adapter, app, delivered):
        monkeypatch.setattr(TelegramAdapter, "_NEW_TOPIC_OPENER_TTL_SECONDS", -1.0)
        await _deliver(app, adapter, topic_created(app.bot), opener(app.bot))

    assert delivered[0].source.thread_id is None


@pytest.mark.asyncio
async def test_only_the_opener_consumes_the_creation(monkeypatch):
    """One-shot: a second thread-less message in the same chat is not re-pinned."""
    async with connected(monkeypatch) as (adapter, app, delivered):
        await _deliver(
            app, adapter,
            topic_created(app.bot),
            opener(app.bot),
            opener(app.bot, uid=12, message_id=40801, text="second"),
        )

    assert [event.source.thread_id for event in delivered] == ["40798", None]


@pytest.mark.asyncio
async def test_forum_group_general_message_is_not_adopted(monkeypatch):
    """In a forum group a thread-less message IS General; a creation must not re-route it."""
    async with connected(monkeypatch) as (adapter, app, delivered):
        await _deliver(
            app, adapter,
            topic_created(app.bot, thread=99, group=True),
            opener(app.bot, group=True),
        )

    assert delivered[0].source.thread_id == "1"


@pytest.mark.asyncio
async def test_explicit_thread_on_the_opener_wins(monkeypatch):
    """Telegram's healthy shape (opener carries its thread id) is untouched."""
    async with connected(monkeypatch) as (adapter, app, delivered):
        await _deliver(app, adapter, topic_created(app.bot), opener(app.bot, thread=40798, is_topic=True))

    assert delivered[0].source.thread_id == "40798"
