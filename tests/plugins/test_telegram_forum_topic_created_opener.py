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
from gateway.platforms.event import MessageType
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


def topic_created(bot, *, uid=10, chat=42, user=88, thread=40798, message_id=40799, name="Prova 3", group=False, is_bot=False):
    """``forum_topic_created`` service message: carries the NEW topic's thread id, never text.

    ``is_bot=True`` models a topic Hermes itself created via ``createForumTopic`` (the service
    message is attributed to the bot), as opposed to a topic the chat user created."""
    return Update.de_json({"update_id": uid, "message": {
        "message_id": message_id, "date": 1800000000, "chat": _chat(chat, group, is_forum=group),
        "from": {"id": user, "is_bot": is_bot, "first_name": "Bot" if is_bot else "Human"},
        "message_thread_id": thread, "is_topic_message": True,
        "forum_topic_created": {"name": name, "icon_color": 7322096},
    }}, bot)


def opener(bot, *, uid=11, chat=42, user=88, message_id=40800, text="Prova 3", thread=None,
           is_topic=None, group=False, is_bot=False, reply_to_message_id=None, reply_to_user=None):
    """The topic's opening user message. ``thread=None`` is the shape Telegram regressed to.

    ``is_bot=True`` models an opening message from a bot other than the service message's
    author (a same-sender bot is covered by ``test_bot_created_topic_other_bots_opener_is_not_adopted``).
    ``reply_to_message_id``/``reply_to_user`` model an opener that is a Telegram reply to the
    creation service message — the shape the field captures associate with a labeled opener.
    """
    message = {
        "message_id": message_id, "date": 1800000001, "chat": _chat(chat, group, is_forum=group),
        "from": {"id": user, "is_bot": is_bot, "first_name": "Bot" if is_bot else "Human"}, "text": text,
    }
    if thread is not None:
        message["message_thread_id"] = thread
    if is_topic is not None:
        message["is_topic_message"] = is_topic
    if reply_to_message_id is not None:
        message["reply_to_message"] = {
            "message_id": reply_to_message_id, "date": 1800000000,
            "chat": _chat(chat, group, is_forum=group),
            "from": {"id": reply_to_user if reply_to_user is not None else user,
                     "is_bot": reply_to_user is not None,
                     "first_name": "Creation"},
        }
    return Update.de_json({"update_id": uid, "message": message}, bot)


@asynccontextmanager
async def connected(monkeypatch, *, bot_id=111):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=f"{bot_id}:offline-test", extra={}))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "88")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "")
    # Clean any stale update receipt so the admission controller doesn't drop our updates.
    receipt = adapter._update_receipt_dir / f"telegram_update_receipts_{bot_id}.json"
    if receipt.exists():
        receipt.unlink()
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
        if receipt.exists():
            receipt.unlink()


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


@pytest.mark.asyncio
async def test_captured_updates_through_build_message_event(monkeypatch):
    """The reviewer's captured shape, driven at the event-building seam (Refs #136567).

    Service message with ``forum_topic_created`` followed by an adjacent thread-less opener:
    the event the runner sees must carry the created topic as its thread id.
    """
    async with connected(monkeypatch) as (adapter, app, delivered):
        await adapter._handle_forum_topic_created(topic_created(app.bot), None)
        event = adapter._build_message_event(opener(app.bot).message, MessageType.TEXT, update_id=11)

    assert event.source.thread_id == "40798"


@pytest.mark.asyncio
async def test_different_sender_opener_is_not_adopted(monkeypatch):
    """A creation recorded for one sender must not claim another sender's thread-less message."""
    async with connected(monkeypatch) as (adapter, app, delivered):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "88,99")
        await _deliver(
            app, adapter,
            topic_created(app.bot, message_id=40799, user=88),
            opener(app.bot, message_id=40800, user=99),
        )

    assert delivered[0].source.thread_id is None


@pytest.mark.asyncio
async def test_bot_created_topic_opener_is_not_adopted(monkeypatch):
    """A topic Hermes itself created (service message attributed to the bot) is recorded under
    the bot's sender key, so the human's thread-less opener is a different key: routing unchanged."""
    async with connected(monkeypatch) as (adapter, app, delivered):
        await _deliver(
            app, adapter,
            topic_created(app.bot, is_bot=True, user=111),
            opener(app.bot),
        )

    assert delivered[0].source.thread_id is None


@pytest.mark.asyncio
async def test_bot_created_topic_other_bots_opener_is_not_adopted(monkeypatch):
    """A second bot's thread-less message must not claim a topic Hermes created."""
    async with connected(monkeypatch) as (adapter, app, delivered):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111,222")
        await _deliver(
            app, adapter,
            topic_created(app.bot, is_bot=True, user=111),
            opener(app.bot, user=222, is_bot=True),
        )

    assert delivered[0].source.thread_id is None


@pytest.mark.asyncio
async def test_pending_created_topics_stays_bounded(monkeypatch):
    """Distinct-key growth is bounded by prune-on-record, not by the lookup path (#136567).

    Jarvis4Sleaf's finding: records whose opener arrives already threaded (or that simply never
    got claimed) are never revisited by the lookup path, so without a prune at record time the
    map grows for the process lifetime. Three phases: (1) the leak — 10 distinct keys all
    survive with the real TTL; (2) with a negative TTL every record is expired, so the next
    record's prune evicts all of them; (3) a fresh record still claims its opener.
    """
    async with connected(monkeypatch) as (adapter, app, delivered):
        # Every delivered user must be allow-listed, or the message is blocked before it is
        # even built and never reaches the topic-routing path we are measuring.
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "88," + ",".join(str(300 + i) for i in range(10)))

        # Phase 1 — the leak: with the real TTL, no record is ever revisited by the lookup
        # path, so all 10 distinct (chat, sender) keys survive for the process lifetime.
        for i in range(10):
            # Distinct keys; half get the healthy already-threaded opener (returns early at
            # the adopt guard, never popped), half get nothing.
            if i % 2 == 0:
                await _deliver(
                    app, adapter,
                    topic_created(app.bot, uid=100 + i, chat=200 + i, user=300 + i,
                                  thread=40000 + i, message_id=40001 + i),
                    opener(app.bot, uid=500 + i, chat=200 + i, user=300 + i,
                           message_id=40002 + i, thread=40000 + i, is_topic=True),
                )
            else:
                await _deliver(
                    app, adapter,
                    topic_created(app.bot, uid=100 + i, chat=200 + i, user=300 + i,
                                  thread=40000 + i, message_id=40001 + i),
                )

        pending = adapter._pending_created_topics_store()
        assert len(pending) == 10, "leak repro: every record survived; the map is unbounded"

        # Phase 2 — prune-on-record: a negative TTL makes every delivered record immediately
        # expired, so the next record's prune evicts all of them at record time.
        monkeypatch.setattr(TelegramAdapter, "_NEW_TOPIC_OPENER_TTL_SECONDS", -1.0)
        adapter._pending_created_topics.clear()
        for i in range(10):
            await _deliver(
                app, adapter,
                topic_created(app.bot, uid=1000 + i, chat=200 + i, user=300 + i,
                              thread=40000 + i, message_id=40001 + i),
            )
        pending = adapter._pending_created_topics_store()
        assert len(pending) == 0, "prune-on-record must evict every expired record"

        # Phase 3 — the fresh record still claims its opener: pruning never breaks the
        # routing contract.
        monkeypatch.setattr(TelegramAdapter, "_NEW_TOPIC_OPENER_TTL_SECONDS", 120.0)
        delivered.clear()
        await _deliver(app, adapter, topic_created(app.bot, uid=999, chat=42, user=88,
                                                   thread=50000, message_id=50001))
        pending = adapter._pending_created_topics_store()
        assert len(pending) == 1, "only the fresh record remains"
        assert pending[("42", "88")][0] == "50000"
        await _deliver(app, adapter, opener(app.bot, uid=2000, message_id=50002, text="fresh"))
        assert delivered[-1].source.thread_id == "50000"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_id, opener_id, update_service, update_opener, topic_id, opener_thread, is_reply",
    [
        # itsjustmichele's verbatim captures, Telegram for Android, receiving side, between
        # 2026-10-13 13:45 and 2026-10-14 12:55. Texts redacted; ids, thread ids and update ids
        # verbatim. Both shapes of the same defect, same device, same chat, same day.
        (9406, 9407, 990665605, 990665606, "494984", None, False),      # regressed: thread-less opener
        (9499, 9500, 990665619, 990665620, "495077", "495077", True),   # healthy: labeled opener
        (9538, 9539, 990665626, 990665627, "495116", None, False),      # regressed
        (9604, 9605, 990665630, 990665631, "495182", "495182", True),   # healthy
    ],
)
async def test_reported_captures_are_routed_by_the_adoption_gate(
        monkeypatch, service_id, opener_id, update_service, update_opener, topic_id,
        opener_thread, is_reply):
    """Every capture the reporter supplied must land in the topic it was created for.

    A regression matrix, not a single case. ``opener_thread`` is what the opener actually
    carried: the two regressed rows arrive with no thread id at all and must be *adopted* into
    the topic the preceding service message announced; the two healthy rows already carry it and
    must keep it. ``update_id`` is the reporter's own, so both axes they named (message id
    adjacency, update id adjacency) are exercised.
    """
    async with connected(monkeypatch) as (adapter, app, delivered):
        await _deliver(
            app, adapter,
            topic_created(app.bot, uid=update_service, chat=42, user=88,
                          thread=topic_id, message_id=service_id),
            opener(app.bot, uid=update_opener, chat=42, user=88, message_id=opener_id,
                   thread=opener_thread, is_topic=opener_thread is not None,
                   reply_to_message_id=service_id if is_reply else None),
        )

    assert [event.text for event in delivered] == ["Prova 3"], "the service message must not reach the agent"
    assert delivered[0].source.thread_id == topic_id


@pytest.mark.asyncio
@pytest.mark.parametrize("is_reply", [True, False])
async def test_reply_to_the_creation_does_not_change_routing(monkeypatch, is_reply):
    """``reply_to_message`` must not alter adoption: it correlates with the two shapes in the
    field, but it is a consequence of the client race, not a routing input (#136567).

    Both directions are pinned deliberately: a reply must not create a record where there is
    none, and its absence must not stop one being claimed. The quoted parent is the creation
    service message *and a bot*, exercising the fixture's bot-reply branch, so the shape is
    built for real rather than left unparameterised.
    """
    async with connected(monkeypatch) as (adapter, app, delivered):
        await _deliver(
            app, adapter,
            topic_created(app.bot),
            opener(app.bot, reply_to_message_id=40799 if is_reply else None,
                   reply_to_user=111 if is_reply else None),
        )

    assert delivered[0].source.thread_id == "40798", "adjacency governs adoption, not the reply flag"
