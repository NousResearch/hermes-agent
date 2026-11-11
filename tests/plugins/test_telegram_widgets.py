"""Telegram HTML widgets (Bot API 10.4) for plugins.

Three pieces, all on the real python-telegram-bot and without a network: the ``widget_query`` update type
the adapter asks Telegram for, the ``gateway_platform_event`` envelope a tap in a widget becomes, and the
Bot API calls behind the ``send_widget`` / ``edit_widget_data`` / ``answer_widget_query`` platform actions.
PTB 22.8 models Bot API 10.0, so the update field arrives in ``Update.api_kwargs`` and the methods go
through ``Bot.do_api_request``; the tests pin both.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("telegram")  # the messaging extra: the ordinary test slices run without it

from telegram import Update
from telegram.request import BaseRequest

from gateway.config import Platform, PlatformConfig
from gateway.session import build_session_key
from plugins.platforms.telegram import telegram_widgets
from plugins.platforms.telegram.adapter import TelegramAdapter

USER = 88
CHAT = 88  # a private chat: its id is the user's


def widget_query(**overrides):
    """A ``widget_query`` as Telegram sends it for a tap in a widget message; ``None`` drops a field."""
    query = {
        "id": "q1", "from": {"id": USER, "is_bot": False, "first_name": "Human", "username": "human"},
        "chat": {"id": CHAT, "type": "private"}, "message_id": 472, "chat_instance": "ci-1",
        "type": {"type": "data"}, "data": "refresh",
    }
    query.update(overrides)
    return {key: value for key, value in query.items() if value is not None}


def update_of(query, uid=7):
    return Update.de_json({"update_id": uid, "widget_query": query}, None)


def bare_adapter() -> TelegramAdapter:
    """A TelegramAdapter without the heavy ``__init__``: normalization and source extraction read only
    ``name``, ``config`` and ``gateway_runner``."""
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = SimpleNamespace(extra={"allow_from": ["*"]})
    adapter.gateway_runner = None
    return adapter


class StubRequest(BaseRequest):
    """No network: records every Bot API call, answers getMe, hands out queued getUpdates results."""

    def __init__(self, updates=(), replies=None):
        self.calls: list[tuple[str, dict]] = []
        self._updates = list(updates)
        self._replies = dict(replies or {})

    @property
    def read_timeout(self):
        return None

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, **_):
        endpoint = url.rsplit("/", 1)[-1]
        params = dict(request_data.parameters) if request_data is not None else {}
        self.calls.append((endpoint, params))
        if endpoint in self._replies:
            status, envelope = self._replies[endpoint]
            return status, json.dumps(envelope).encode()
        if endpoint == "getMe":
            result = {"id": 1, "is_bot": True, "first_name": "bot", "username": "bot"}
        elif endpoint == "getUpdates":
            result = self._updates.pop(0) if self._updates else []
        elif endpoint == "sendWidget":
            result = {"message_id": 501, "date": 0, "chat": {"id": CHAT, "type": "private"}}
        else:
            result = True
        return 200, json.dumps({"ok": True, "result": result}).encode()

    def sent(self, endpoint):
        return [params for name, params in self.calls if name == endpoint]


@asynccontextmanager
async def connected(monkeypatch, tmp_path, stub: StubRequest):
    """A TelegramAdapter connected through the real PTB Application and the engine's own
    ``TelegramApplication``; only transport and the model-work boundary are replaced."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="111:offline-test", extra={}))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", str(USER))
    monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "")
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)  # a plugin subscribed
    monkeypatch.setattr(adapter, "_build_ptb_requests", AsyncMock(return_value=(stub, stub)))
    monkeypatch.setattr(adapter, "_start_polling_mode", AsyncMock())  # the tests start polling themselves
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", lambda: None)
    monkeypatch.setattr(adapter, "_restart_task_attr", lambda name, coroutine: coroutine.close())
    monkeypatch.setattr(adapter, "_set_status_indicator", AsyncMock())
    adapter._update_receipt_dir = tmp_path  # receipts of handled updates never go to the real Hermes home
    adapter._message_handler = AsyncMock()
    assert await adapter.connect()
    try:
        yield adapter
    finally:
        await adapter.disconnect()


# ---------------------------------------------------------------------------
# The update type Telegram is asked for
# ---------------------------------------------------------------------------


class TestAllowedUpdates:
    def test_ptbs_types_plus_widget_query(self):
        allowed = telegram_widgets.allowed_updates(Update)
        assert allowed[:-1] == list(Update.ALL_TYPES)
        assert allowed[-1] == "widget_query"
        assert "widget_query" not in Update.ALL_TYPES  # PTB's own list is left alone

    def test_a_ptb_that_lists_the_type_itself_is_not_asked_twice(self):
        knows = SimpleNamespace(ALL_TYPES=("message", "widget_query"))
        assert telegram_widgets.allowed_updates(knows) == ["message", "widget_query"]

    def test_a_tuple_stand_in_works(self):
        assert telegram_widgets.allowed_updates(SimpleNamespace(ALL_TYPES=())) == ["widget_query"]


# ---------------------------------------------------------------------------
# The query out of an update, on a PTB that does not model it
# ---------------------------------------------------------------------------


class TestWidgetQueryOf:
    def test_ptb_without_the_type_keeps_it_in_api_kwargs(self):
        assert telegram_widgets.widget_query_of(update_of(widget_query())) == widget_query()

    def test_a_ptb_that_models_the_type_is_read_through_to_dict(self):
        native = SimpleNamespace(to_dict=lambda: widget_query())
        update = SimpleNamespace(api_kwargs={}, widget_query=native)
        assert telegram_widgets.widget_query_of(update) == widget_query()

    def test_other_updates_and_mocks_are_none(self):
        message = Update.de_json({"update_id": 1, "message": {
            "message_id": 9, "date": 0, "text": "hi", "chat": {"id": CHAT, "type": "private"}}}, None)
        assert telegram_widgets.widget_query_of(message) is None
        assert telegram_widgets.widget_query_of(MagicMock()) is None  # a mock never looks like a query


# ---------------------------------------------------------------------------
# The platform event and its authorizable source
# ---------------------------------------------------------------------------


class TestWidgetQueryEvent:
    def test_envelope(self):
        # A widget this adapter did not send (or sent before a restart): no topic is known.
        assert bare_adapter()._normalize_platform_event(update_of(widget_query())) == {
            "platform": "telegram",
            "event_type": "widget_query",
            "payload": {"query_id": "q1", "chat_id": "88", "message_id": "472", "data": "refresh", "thread_id": None,
                        "type": {"type": "data"}, "widget_id": None},
        }

    def test_ids_are_text_and_bounded_and_json_safe(self):
        event = bare_adapter()._normalize_platform_event(update_of(widget_query(
            id="q" * 300, message_id=int("9" * 300), data="d" * 300, widget_id="w" * 300)))
        payload = event["payload"]
        assert (len(payload["query_id"]), len(payload["message_id"]), len(payload["data"]),
                len(payload["widget_id"])) == (128, 128, 128, 128)
        json.dumps(event)

    def test_a_wallet_query_from_a_shared_widget_keeps_its_type_and_widget(self):
        """A shared widget's sign-data query: the plugin gets Telegram's ``WidgetQueryType`` whole (the wallet and
        the signed data it needs to verify) and the shared widget's id, as a copy of the update's own."""
        kind = {"type": "sign_data", "sign_data": '{"signature":"c2ln","timestamp":1}',
                "wallet": {"address": "0:abc", "amount": "1000", "public_key": "ab01", "state_init": "te6c"}}
        update = update_of(widget_query(type=kind, widget_id="shared-1"))
        payload = bare_adapter()._normalize_platform_event(update)["payload"]
        assert (payload["type"], payload["widget_id"]) == (kind, "shared-1")
        payload["type"]["wallet"]["address"] = "changed"
        assert telegram_widgets.widget_query_of(update)["type"]["wallet"]["address"] == "0:abc"

    @pytest.mark.parametrize("kind", [None, "data", {"kind": "data"}, {"type": 5}, ["data"]])
    def test_a_query_type_of_another_shape_is_none_and_the_tap_still_arrives(self, kind):
        payload = bare_adapter()._normalize_platform_event(update_of(widget_query(type=kind)))["payload"]
        assert (payload["type"], payload["query_id"], payload["data"]) == (None, "q1", "refresh")

    @pytest.mark.parametrize("broken", [
        {"id": None}, {"data": None}, {"data": 5}, {"message_id": None}, {"chat": None},
    ])
    def test_a_malformed_query_is_no_event(self, broken):
        assert bare_adapter()._normalize_platform_event(update_of(widget_query(**broken))) is None

    def test_source_is_the_sender_in_the_widgets_chat(self):
        source = bare_adapter()._source_for_platform_event_auth(update_of(widget_query()))
        assert (source.user_id, source.user_name, source.chat_id, source.chat_type, source.message_id) == (
            "88", "human", "88", "dm", "472")
        assert source.thread_id is None

    def test_a_group_chat_keeps_its_type(self):
        query = widget_query(chat={"id": -100777, "type": "supergroup"})
        source = bare_adapter()._source_for_platform_event_auth(update_of(query))
        assert (source.chat_id, source.chat_type) == ("-100777", "group")

    def test_a_query_without_a_sender_fails_closed(self):
        with pytest.raises(ValueError):
            bare_adapter()._source_for_platform_event_auth(update_of(widget_query(**{"from": None})))

    def test_a_tap_fires_through_the_boundary_with_its_source(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
        adapter = bare_adapter()
        seen = []

        async def observe(event, source):
            seen.append((event, source))

        adapter.set_platform_event_handler(observe)
        asyncio.run(adapter._on_platform_update(update_of(widget_query()), context=MagicMock()))
        assert len(seen) == 1
        event, source = seen[0]
        assert event["event_type"] == "widget_query"
        assert (source.user_id, source.chat_id) == ("88", "88")

    def test_an_inline_message_widget_has_no_chat_to_authorize_and_is_dropped(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
        adapter = bare_adapter()
        seen = []

        async def observe(event, source):
            seen.append((event, source))

        adapter.set_platform_event_handler(observe)
        inline = widget_query(chat=None, message_id=None, inline_message_id="inline-1")
        asyncio.run(adapter._on_platform_update(update_of(inline), context=MagicMock()))
        assert seen == []


# ---------------------------------------------------------------------------
# As the gateway runs it: PTB's Updater, the engine's admission, the group-99 observer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_tap_polled_from_telegram_reaches_the_platform_event_boundary(monkeypatch, tmp_path):
    """PTB's own ``Updater`` polls with the adapter's update types, the engine's ``TelegramApplication``
    admits the update, and the observer hands the normalized tap and its source to the gateway boundary."""
    stub = StubRequest(updates=[[{"update_id": 7, "widget_query": widget_query()}]])
    seen = []

    async def observe(event, source):
        seen.append((event, source))

    async with connected(monkeypatch, tmp_path, stub) as adapter:
        adapter.set_platform_event_handler(observe)
        await adapter._start_polling_once(
            adapter._app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)
        deadline = time.monotonic() + 10
        while not seen and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        claimed = set(adapter._seen_update_ids)

    polled = stub.sent("getUpdates")
    assert polled, "the updater never asked Telegram for updates"
    assert "widget_query" in polled[0]["allowed_updates"]
    assert "message" in polled[0]["allowed_updates"]  # the stock types stay
    assert len(seen) == 1
    event, source = seen[0]
    assert event == {
        "platform": "telegram", "event_type": "widget_query",
        "payload": {"query_id": "q1", "chat_id": "88", "message_id": "472", "data": "refresh", "thread_id": None,
                    "type": {"type": "data"}, "widget_id": None},
    }
    assert (source.user_id, source.chat_id, source.chat_type, source.message_id) == ("88", "88", "dm", "472")
    assert "1:7" in claimed  # admitted and completed once


@pytest.mark.asyncio
async def test_the_webhook_subscription_asks_for_widget_queries_too(monkeypatch, tmp_path):
    stub = StubRequest()
    started = {}

    async def start_webhook(*_args, **kwargs):
        started.update(kwargs)

    async with connected(monkeypatch, tmp_path, stub) as adapter:
        monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "s3cret")
        # PTB's Updater has __slots__: patch the class, not the instance.
        monkeypatch.setattr(type(adapter._app.updater), "start_webhook", start_webhook)
        await adapter._start_webhook_mode("https://example.test/telegram", is_reconnect=True)

    assert "widget_query" in started["allowed_updates"]
    assert "message" in started["allowed_updates"]


# ---------------------------------------------------------------------------
# The topic a widget was sent into: Telegram's widget_query names none
# ---------------------------------------------------------------------------


def sent_into_topic(message_id=501, thread=7):
    """``sendWidget``'s reply for a widget sent into a topic of a private chat with the bot."""
    return 200, {"ok": True, "result": {
        "message_id": message_id, "date": 0, "chat": {"id": CHAT, "type": "private"},
        "message_thread_id": thread, "is_topic_message": True}}


async def tap_after_send(monkeypatch, tmp_path, stub, **send):
    """Send a widget, then poll a tap on it the way the gateway does; returns what reached the boundary."""
    stub._updates.append([{"update_id": 9, "widget_query": widget_query(message_id=501)}])
    seen = []

    async def observe(event, source):
        seen.append((event, source))

    async with connected(monkeypatch, tmp_path, stub) as adapter:
        adapter.set_platform_event_handler(observe)
        await adapter.send_widget("88", "<b>hi</b>", **send)
        await adapter._start_polling_once(
            adapter._app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)
        deadline = time.monotonic() + 10
        while not seen and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
    assert len(seen) == 1
    return seen[0]


class TestWidgetTopic:
    @pytest.mark.asyncio
    async def test_a_tap_in_a_private_chat_topic_names_the_topic(self, monkeypatch, tmp_path):
        """The widget went into topic 7 of the private chat. Its tap names the topic in the payload and in the
        authorized source, so a plugin's origin and the source both key the topic's session, not the root DM."""
        stub = StubRequest(replies={"sendWidget": sent_into_topic()})
        event, source = await tap_after_send(monkeypatch, tmp_path, stub, thread_id="7")
        assert event["payload"] == {
            "query_id": "q1", "chat_id": "88", "message_id": "501", "data": "refresh", "thread_id": "7",
            "type": {"type": "data"}, "widget_id": None}
        assert source.thread_id == "7"
        assert build_session_key(source).endswith(":dm:88:7")

    @pytest.mark.asyncio
    async def test_a_widget_in_the_chat_itself_has_no_topic(self, monkeypatch, tmp_path):
        event, source = await tap_after_send(monkeypatch, tmp_path, StubRequest())
        assert event["payload"]["thread_id"] is None
        assert source.thread_id is None
        assert build_session_key(source).endswith(":dm:88")

    def test_the_topic_is_the_one_telegram_reports(self):
        threads = telegram_widgets.WidgetThreads()
        threads.remember(sent_into_topic(thread=7)[1]["result"])
        assert threads.thread_of(widget_query(message_id=501)) == "7"
        assert threads.thread_of(widget_query(message_id=502)) is None
        assert threads.thread_of(widget_query(message_id=501, chat={"id": 99, "type": "private"})) is None

    @pytest.mark.parametrize("reply", [
        None, True, "501", {"message_id": 501},
        {"message_id": 501, "chat": {"id": CHAT}, "message_thread_id": 7},  # not a topic message
        {"message_id": 501, "message_thread_id": 7, "is_topic_message": True},  # no chat
        {"message_id": 501, "chat": {"id": CHAT}, "message_thread_id": True, "is_topic_message": True},
    ])
    def test_a_reply_without_a_topic_is_not_remembered(self, reply):
        threads = telegram_widgets.WidgetThreads()
        threads.remember(reply)
        assert threads.thread_of(widget_query(message_id=501)) is None

    def test_the_memory_is_bounded_and_forgets_the_oldest(self):
        threads = telegram_widgets.WidgetThreads(limit=2)
        for message_id in (501, 502, 503):
            threads.remember(sent_into_topic(message_id=message_id)[1]["result"])
        assert [threads.thread_of(widget_query(message_id=m)) for m in (501, 502, 503)] == [None, "7", "7"]


# ---------------------------------------------------------------------------
# The Bot API calls behind the platform actions
# ---------------------------------------------------------------------------


class TestBotApiCalls:
    @pytest.mark.asyncio
    async def test_send_widget_sends_html_and_data_and_returns_the_message(self, monkeypatch, tmp_path):
        stub = StubRequest()
        async with connected(monkeypatch, tmp_path, stub) as adapter:
            sent = await adapter.send_widget(
                "88", "<b>hi</b>", chat_data='{"n":1}', public_data="p", caption="Dashboard",
                allow_queries=True, disable_notification=True)
        assert sent["message_id"] == 501
        assert stub.sent("sendWidget") == [{
            "chat_id": 88, "html": "<b>hi</b>", "chat_data": '{"n":1}', "public_data": "p",
            "caption": "Dashboard", "allow_queries": True, "disable_notification": True}]

    @pytest.mark.asyncio
    async def test_send_widget_leaves_out_what_is_not_given(self, monkeypatch, tmp_path):
        stub = StubRequest()
        async with connected(monkeypatch, tmp_path, stub) as adapter:
            await adapter.send_widget("88", "<b>hi</b>", thread_id="5")
        (params,) = stub.sent("sendWidget")
        assert params == {"chat_id": 88, "html": "<b>hi</b>", "allow_queries": True,
                          "disable_notification": False, "message_thread_id": 5}

    @pytest.mark.asyncio
    async def test_send_widget_passes_the_widget_size(self, monkeypatch, tmp_path):
        stub = StubRequest()
        async with connected(monkeypatch, tmp_path, stub) as adapter:
            await adapter.send_widget("88", "<b>hi</b>", width=320, height=240)
        (params,) = stub.sent("sendWidget")
        assert (params["width"], params["height"]) == (320, 240)

    @pytest.mark.asyncio
    async def test_edit_widget_data_rewrites_only_the_packet_given(self, monkeypatch, tmp_path):
        stub = StubRequest()
        async with connected(monkeypatch, tmp_path, stub) as adapter:
            assert await adapter.edit_widget_data("88", "472", chat_data='{"n":2}') is True
            assert await adapter.edit_widget_data("88", "472", public_data="") is True  # "" removes it
        assert stub.sent("editMessageWidgetData") == [
            {"chat_id": 88, "message_id": 472, "chat_data": '{"n":2}'},
            {"chat_id": 88, "message_id": 472, "public_data": ""},
        ]

    @pytest.mark.asyncio
    async def test_an_unchanged_packet_is_a_no_op_not_a_failure(self, monkeypatch, tmp_path):
        stub = StubRequest(replies={"editMessageWidgetData": (
            400, {"ok": False, "error_code": 400, "description": "Bad Request: message is not modified"})})
        async with connected(monkeypatch, tmp_path, stub) as adapter:
            assert await adapter.edit_widget_data("88", "472", chat_data="{}") is False

    @pytest.mark.asyncio
    async def test_other_telegram_errors_propagate(self, monkeypatch, tmp_path):
        from telegram.error import BadRequest

        stub = StubRequest(replies={"editMessageWidgetData": (
            400, {"ok": False, "error_code": 400, "description": "Bad Request: message to edit not found"})})
        async with connected(monkeypatch, tmp_path, stub) as adapter:
            with pytest.raises(BadRequest):
                await adapter.edit_widget_data("88", "472", chat_data="{}")

    @pytest.mark.asyncio
    async def test_answer_widget_query_carries_the_reply(self, monkeypatch, tmp_path):
        stub = StubRequest()
        async with connected(monkeypatch, tmp_path, stub) as adapter:
            await adapter.answer_widget_query("q1", '{"ok":1}')
            await adapter.answer_widget_query("q2")
        assert stub.sent("answerWidgetQuery") == [
            {"widget_query_id": "q1", "data": '{"ok":1}'},
            {"widget_query_id": "q2"},
        ]
