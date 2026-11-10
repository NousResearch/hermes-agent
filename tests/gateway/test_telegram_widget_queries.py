"""Bot API 10.4 widget taps on the Telegram adapter (opt-in ``platforms.telegram.extra.widgets``).

Driven through the adapter's real entry points: the polling/webhook starters, ``_register_handlers`` and
the widget update callback. Updates are shaped the way python-telegram-bot 22.8's ``Update.de_json``
shapes an update kind it does not model (raw payload in a read-only ``api_kwargs``, every ``effective_*``
None); tests/plugins/platforms/telegram/test_telegram_widgets_ptb.py pins that shape against the real
library. Telegram requires every widget query to be answered once within 15 s, failures included,
otherwise the widget shows TIMEOUT.
"""

import asyncio
import functools
import json
import logging
from contextvars import ContextVar
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest, NetworkError

from gateway.config import Platform, PlatformConfig, load_gateway_config
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram import telegram_widgets as widgets
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_widgets import WidgetQuery

_AUTH_ENV = (
    "TELEGRAM_ALLOW_BOTS", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_ALLOW_ALL_USERS", "TELEGRAM_GROUP_ALLOWED_USERS",
    "TELEGRAM_GROUP_ALLOWED_CHATS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS")
# Stand-in for PTB's ``Update.ALL_TYPES`` (the conftest mock leaves it empty).
_ALL_TYPES = ["message", "edited_message", "callback_query", "inline_query", "message_reaction"]
_DROP = object()


@pytest.fixture(autouse=True)
def _no_allowlist(monkeypatch):
    for var in _AUTH_ENV:
        monkeypatch.delenv(var, raising=False)


def _adapter(**extra) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:widget-test", extra=extra))
    adapter._bot = MagicMock()
    adapter._bot.do_api_request = AsyncMock(return_value=True)
    return adapter


def _payload(**fields) -> dict:
    payload = {
        "id": "wq-1", "from": {"id": 42, "is_bot": False, "first_name": "Ann"}, "chat": {"id": 42, "type": "private"},
        "message_id": 7, "chat_instance": "-5512", "type": {"type": "data"}, "data": "approve:abc"}
    payload.update(fields)
    return {key: value for key, value in payload.items() if value is not _DROP}


def _shared_payload(**fields) -> dict:
    payload = {"id": "swq-1", "from": {"id": 42, "is_bot": False, "first_name": "Ann"},
               "type": {"type": "data"}, "data": "approve:abc", "widget_id": "w-9"}
    payload.update(fields)
    return {key: value for key, value in payload.items() if value is not _DROP}


def _update(kind: str, payload: dict, update_id: int = 1001):
    """``Update.de_json`` of a raw update JSON (as Telegram sends it) for a kind PTB 22.8 does not model."""
    data = json.loads(json.dumps({"update_id": update_id, kind: payload}))
    return SimpleNamespace(
        update_id=data.pop("update_id"), api_kwargs=MappingProxyType(data),
        effective_user=None, effective_chat=None, effective_message=None)


def _claim(adapter):
    """The admission claim ``TelegramApplication.process_update`` holds for the update being handled."""
    claim = SimpleNamespace(accepted=False, failed=False)
    adapter._update_admission = ContextVar("claim", default=claim)
    return claim


def _api_calls(adapter) -> list:
    return [(c.args[0], c.kwargs["api_kwargs"]) for c in adapter._bot.do_api_request.await_args_list]


def _answers(adapter) -> list:
    calls = _api_calls(adapter)
    assert {method for method, _ in calls} <= {"answerWidgetQuery"}, calls
    return [kwargs for _, kwargs in calls]


def _error(answer: dict):
    return json.loads(answer["data"]).get("error")


async def _tap(adapter, update) -> None:
    await adapter._handle_widget_update(update, SimpleNamespace())


# --- allowed_updates: polling and webhook --------------------------------------------------------------


async def _polling_allowed_updates(adapter):
    app = MagicMock()
    app.updater.start_polling = AsyncMock()
    await adapter._start_polling_once(app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)
    return app.updater.start_polling.await_args.kwargs["allowed_updates"]


async def _webhook_allowed_updates(adapter, monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "s3cret")
    adapter._app = MagicMock()
    adapter._app.updater.start_webhook = AsyncMock()
    await adapter._start_webhook_mode("https://example.test/telegram", is_reconnect=False)
    return adapter._app.updater.start_webhook.await_args.kwargs["allowed_updates"]


@pytest.fixture
def all_types(monkeypatch):
    update_cls = SimpleNamespace(ALL_TYPES=list(_ALL_TYPES))
    monkeypatch.setattr(tg_adapter, "Update", update_cls)
    return update_cls


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [{}, {"widgets": False}, {"widgets": "off"}])
async def test_flag_off_requests_exactly_update_all_types(all_types, monkeypatch, extra):
    assert await _polling_allowed_updates(_adapter(**extra)) is all_types.ALL_TYPES
    assert await _webhook_allowed_updates(_adapter(**extra), monkeypatch) is all_types.ALL_TYPES
    assert all_types.ALL_TYPES == _ALL_TYPES


@pytest.mark.asyncio
async def test_flag_on_requests_both_widget_update_types_once(all_types, monkeypatch):
    expected = [*_ALL_TYPES, "widget_query", "shared_widget_query"]

    assert await _polling_allowed_updates(_adapter(widgets=True)) == expected
    assert await _webhook_allowed_updates(_adapter(widgets="true"), monkeypatch) == expected
    assert all_types.ALL_TYPES == _ALL_TYPES  # PTB's process-wide list is never extended in place


@pytest.mark.asyncio
async def test_flag_on_does_not_duplicate_types_a_newer_sdk_already_lists(all_types):
    all_types.ALL_TYPES = ["message", "widget_query", "callback_query"]

    allowed = await _polling_allowed_updates(_adapter(widgets=True))

    assert allowed == ["message", "widget_query", "callback_query", "shared_widget_query"]


# --- handler registration ------------------------------------------------------------------------------


class _Handler:
    def __init__(self, *args, block=True):
        self.callback, self.block = args[-1], block


class _MessageHandler(_Handler):
    pass


class _CallbackQueryHandler(_Handler):
    pass


class _InlineQueryHandler(_Handler):
    pass


class _TypeHandler(_Handler):
    def check_update(self, update) -> bool:
        return True  # PTB's TypeHandler(Update) takes every update


class _PtbApp:
    """PTB 22.8 ``Application.add_handler`` table semantics, including the dict swap on a new group."""

    def __init__(self):
        self.handlers: dict = {}

    def add_handler(self, handler, group=0):
        if group not in self.handlers:
            self.handlers[group] = []
            self.handlers = dict(sorted(self.handlers.items()))
        self.handlers[group].append(handler)


def _registered(monkeypatch, **extra):
    for name, cls in (("TelegramMessageHandler", _MessageHandler), ("CallbackQueryHandler", _CallbackQueryHandler),
                      ("InlineQueryHandler", _InlineQueryHandler), ("TypeHandler", _TypeHandler)):
        monkeypatch.setattr(tg_adapter, name, cls)
    adapter, app = _adapter(**extra), _PtbApp()
    adapter._register_handlers(app)
    return adapter, app


def _table(adapter, app) -> list:
    """(group, callback, recorded as core) for every handler, in dispatch order."""
    core = getattr(adapter, "_core_handler_ids", set())
    return [(group, h.callback.__name__, id(h) in core) for group, hs in app.handlers.items() for h in hs]


def _widget_handlers(adapter, app) -> list:
    return [h for hs in app.handlers.values() for h in hs if h.callback == adapter._handle_widget_update]


def test_flag_off_registers_no_widget_handler(monkeypatch):
    adapter, app = _registered(monkeypatch)

    assert _widget_handlers(adapter, app) == []
    assert "_handle_widget_update" not in {callback for _, callback, _ in _table(adapter, app)}


def test_flag_on_adds_one_core_handler_after_the_group_zero_core_handlers(monkeypatch):
    off_adapter, off_app = _registered(monkeypatch)
    on_adapter, on_app = _registered(monkeypatch, widgets=True)
    on = _table(on_adapter, on_app)
    widget_rows = [row for row in on if row[1] == "_handle_widget_update"]

    assert widget_rows == [(0, "_handle_widget_update", True)]
    # Every core handler keeps its group, its order and its #87770 core-id membership.
    assert [row for row in on if row not in widget_rows] == _table(off_adapter, off_app)
    assert [row for row in on if row[0] == 0][-1] == widget_rows[0]
    # Non-blocking: PTB runs it as a task, so a slow handler never holds a shared update slot.
    [widget_handler] = _widget_handlers(on_adapter, on_app)
    assert widget_handler.block is False


def _first_class(kind: str, payload: dict):
    """An update from a python-telegram-bot that models widgets: ``to_dict()`` gives the wire shape."""
    return SimpleNamespace(update_id=1001, api_kwargs=MappingProxyType({}), **{kind: SimpleNamespace(to_dict=lambda: payload)})


def test_widget_handler_matches_only_widget_query_updates(monkeypatch):
    adapter, app = _registered(monkeypatch, widgets=True)
    [handler] = _widget_handlers(adapter, app)
    message = SimpleNamespace(update_id=5, message=SimpleNamespace(text="hi"), api_kwargs=MappingProxyType({}))
    callback = SimpleNamespace(update_id=6, callback_query=SimpleNamespace(data="ea:x"), api_kwargs=MappingProxyType({}))

    assert handler.check_update(_update("widget_query", _payload()))
    assert handler.check_update(_update("shared_widget_query", _shared_payload()))
    assert handler.check_update(_first_class("widget_query", _payload()))
    for other in (message, callback, MagicMock(), "a raw string update", None):
        assert not handler.check_update(other)


@pytest.mark.asyncio
async def test_a_first_class_sdk_widget_query_is_handled_like_the_raw_payload(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _adapter(widgets=True)
    handler = AsyncMock(return_value="ok")
    adapter.register_widget_query_handler("approve:", handler)

    await _tap(adapter, _first_class("widget_query", _payload()))

    assert handler.await_args.args[0].data == "approve:abc"
    assert _answers(adapter) == [{"widget_query_id": "wq-1", "data": "ok"}]


# --- dispatch, answering and admission -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_authorized_query_reaches_the_longest_matching_prefix_and_is_answered_once(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _adapter(widgets=True)
    claim = _claim(adapter)
    seen = []

    async def approve(query):
        seen.append(("long", query))
        return '{"ok":true}'

    async def short(query):
        seen.append(("short", query))
        return "short"

    # The longer prefix is registered first, so "longest wins" and "newest wins" disagree.
    adapter.register_widget_query_handler("approve:", approve)
    adapter.register_widget_query_handler("app", short)

    await _tap(adapter, _update("widget_query", _payload()))

    assert seen == [("long", WidgetQuery(
        kind="widget_query", query_id="wq-1", user_id="42", chat_id="42", chat_type="private", message_id="7",
        inline_message_id=None, widget_id=None, data="approve:abc", type="data"))]
    assert _answers(adapter) == [{"widget_query_id": "wq-1", "data": '{"ok":true}'}]
    assert claim.accepted and not claim.failed


@pytest.mark.asyncio
async def test_handler_may_answer_without_data(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _adapter(widgets=True)
    handler = AsyncMock(return_value=None)
    adapter.register_widget_query_handler("approve:", handler)

    await _tap(adapter, _update("widget_query", _payload()))

    handler.assert_awaited_once()
    assert _answers(adapter) == [{"widget_query_id": "wq-1"}]  # no data: the widget gets null


@pytest.mark.asyncio
async def test_unauthorized_query_is_refused_and_never_dispatched(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")
    adapter = _adapter(widgets=True)
    claim = _claim(adapter)
    handler = AsyncMock(return_value="secret")
    adapter.register_widget_query_handler("approve:", handler)

    await _tap(adapter, _update("widget_query", _payload()))

    handler.assert_not_called()
    [answer] = _answers(adapter)
    assert answer["widget_query_id"] == "wq-1" and _error(answer) == "unauthorized"
    assert claim.accepted and not claim.failed


@pytest.mark.asyncio
async def test_no_allowlist_fails_closed():
    adapter = _adapter(widgets=True)
    adapter._message_handler = None  # no runner: the gate's env-only path decides
    handler = AsyncMock(return_value="secret")
    adapter.register_widget_query_handler("approve:", handler)

    await _tap(adapter, _update("widget_query", _payload()))

    handler.assert_not_called()
    assert [_error(answer) for answer in _answers(adapter)] == ["unauthorized"]


def _recording_check(adapter, decision=True):
    seen = []

    def check(user_id, chat_type, chat_id, **kwargs):
        seen.append((user_id, chat_type, chat_id, kwargs))
        return decision

    adapter.set_authorization_check(check)
    return seen


@pytest.mark.asyncio
async def test_query_from_a_chat_is_authorized_against_that_chat():
    adapter = _adapter(widgets=True)
    seen = _recording_check(adapter)
    adapter.register_widget_query_handler("approve:", AsyncMock(return_value="ok"))

    await _tap(adapter, _update("widget_query", _payload(chat={"id": -100777, "type": "supergroup"})))

    assert seen == [("42", "group", "-100777", {})]
    assert _answers(adapter) == [{"widget_query_id": "wq-1", "data": "ok"}]


@pytest.mark.asyncio
async def test_shared_widget_query_is_judged_as_the_senders_private_chat():
    adapter = _adapter(widgets=True)
    seen = _recording_check(adapter)
    handler = AsyncMock(return_value="ok")
    adapter.register_widget_query_handler("approve:", handler)

    await _tap(adapter, _update("shared_widget_query", _shared_payload()))

    assert seen == [("42", "dm", "42", {})]
    handler.assert_awaited_once_with(WidgetQuery(
        kind="shared_widget_query", query_id="swq-1", user_id="42", chat_id=None, chat_type=None, message_id=None,
        inline_message_id=None, widget_id="w-9", data="approve:abc", type="data"))
    assert _answers(adapter) == [{"widget_query_id": "swq-1", "data": "ok"}]


@pytest.mark.asyncio
async def test_inline_message_query_never_gets_a_group_scope():
    adapter = _adapter(widgets=True)
    seen = _recording_check(adapter)
    adapter.register_widget_query_handler("approve:", AsyncMock(return_value="ok"))
    payload = _payload(chat=_DROP, message_id=_DROP, inline_message_id="AgAAAH", chat_type="supergroup")

    await _tap(adapter, _update("widget_query", payload))

    assert seen == [("42", "dm", "42", {})]


@pytest.mark.asyncio
@pytest.mark.parametrize("sender", [_DROP, {}, {"id": None}, {"id": ""}, {"id": True}])
async def test_query_without_a_sender_fails_closed(sender):
    adapter = _adapter(widgets=True)
    seen = _recording_check(adapter)
    handler = AsyncMock(return_value="ok")
    adapter.register_widget_query_handler("approve:", handler)

    await _tap(adapter, _update("shared_widget_query", _shared_payload(**{"from": sender})))

    assert seen == []
    handler.assert_not_called()
    assert [_error(answer) for answer in _answers(adapter)] == ["unauthorized"]


@pytest.mark.asyncio
async def test_gate_exception_refuses():
    adapter = _adapter(widgets=True)
    adapter._is_callback_user_authorized = MagicMock(side_effect=RuntimeError("auth backend down"))
    handler = AsyncMock(return_value="ok")
    adapter.register_widget_query_handler("approve:", handler)

    await _tap(adapter, _update("widget_query", _payload()))

    handler.assert_not_called()
    assert [_error(answer) for answer in _answers(adapter)] == ["unauthorized"]


async def _never_returns(query):
    await asyncio.Event().wait()


async def _times_out_on_its_own(query):
    raise TimeoutError("the handler's own upstream call timed out")


async def _awaits_a_cancelled_future(query):
    future = asyncio.get_running_loop().create_future()
    future.cancel()  # e.g. work the handler waits on was cancelled elsewhere; this dispatch was not
    await future


@pytest.mark.asyncio
@pytest.mark.parametrize(("register", "error"), [
    (None, "unknown_query"),
    (AsyncMock(side_effect=RuntimeError("handler bug")), "failed"),
    (AsyncMock(return_value={"not": "text"}), "failed"),
    (_times_out_on_its_own, "failed"),
    (_awaits_a_cancelled_future, "failed"),
    (_never_returns, "timeout"),
], ids=["unknown-prefix", "raises", "non-text-result", "own-timeout-error", "stray-cancellation", "deadline"])
async def test_failures_are_answered_once_and_never_raise(monkeypatch, register, error):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    monkeypatch.setattr(widgets, "HANDLER_TIMEOUT_SECONDS", 0.05)
    adapter = _adapter(widgets=True)
    claim = _claim(adapter)
    if register is not None:
        adapter.register_widget_query_handler("approve:", register)
    adapter.register_widget_query_handler("other:", AsyncMock(return_value="wrong handler"))

    await _tap(adapter, _update("widget_query", _payload()))

    [answer] = _answers(adapter)
    assert answer["widget_query_id"] == "wq-1" and _error(answer) == error
    assert claim.accepted and not claim.failed


def test_handler_and_answer_budgets_fit_telegrams_15_second_window():
    assert widgets.HANDLER_TIMEOUT_SECONDS + widgets.ANSWER_TIMEOUT_SECONDS < 15


@pytest.mark.asyncio
async def test_a_hung_answer_request_is_abandoned_not_waited_on(monkeypatch):
    # The claim stays pinned until the dispatch returns, and Application.stop() waits for it.
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    monkeypatch.setattr(widgets, "ANSWER_TIMEOUT_SECONDS", 0.05)
    adapter = _adapter(widgets=True)
    claim = _claim(adapter)

    async def never_answers(method, api_kwargs=None, **kwargs):
        await asyncio.Event().wait()

    adapter._bot.do_api_request = AsyncMock(side_effect=never_answers)
    adapter.register_widget_query_handler("approve:", AsyncMock(return_value="ok"))

    await asyncio.wait_for(_tap(adapter, _update("widget_query", _payload())), timeout=10)

    assert adapter._bot.do_api_request.await_count == 1
    assert claim.accepted and not claim.failed


@pytest.mark.asyncio
async def test_cancelling_the_dispatch_itself_is_never_swallowed(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _adapter(widgets=True)
    entered = asyncio.Event()

    async def parked(query):
        entered.set()
        await asyncio.Event().wait()

    adapter.register_widget_query_handler("approve:", parked)
    dispatch = asyncio.create_task(_tap(adapter, _update("widget_query", _payload())))
    await asyncio.wait_for(entered.wait(), timeout=10)
    dispatch.cancel()  # e.g. the gateway shutting down mid-dispatch

    with pytest.raises(asyncio.CancelledError):
        await dispatch
    adapter._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_warnings_name_the_prefix_never_the_untrusted_data(monkeypatch, caplog):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    monkeypatch.setattr(widgets, "HANDLER_TIMEOUT_SECONDS", 0.05)
    adapter = _adapter(widgets=True)
    data = "approve:tok-5c1d9e"

    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.telegram.adapter"):
        adapter.register_widget_query_handler("approve:", AsyncMock(side_effect=RuntimeError("handler bug")))
        await _tap(adapter, _update("widget_query", _payload(id="wq-1", data=data)))
        adapter.register_widget_query_handler("approve:", _never_returns)
        await _tap(adapter, _update("widget_query", _payload(id="wq-2", data=data)))

    warnings = [logging.Formatter().format(r) for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("'approve:'" in text and "wq-1" in text for text in warnings)
    assert any("'approve:'" in text and "wq-2" in text for text in warnings)
    assert not any("tok-5c1d9e" in text for text in warnings)


@pytest.mark.asyncio
async def test_a_query_past_the_running_handler_slots_is_answered_busy(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    monkeypatch.setattr(widgets, "WIDGET_HANDLER_SLOTS", 1)
    adapter = _adapter(widgets=True)
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(query):
        entered.set()
        await release.wait()
        return "done"

    adapter.register_widget_query_handler("approve:", slow)
    first = asyncio.create_task(_tap(adapter, _update("widget_query", _payload(id="wq-1"))))
    await asyncio.wait_for(entered.wait(), timeout=10)
    await _tap(adapter, _update("widget_query", _payload(id="wq-2")))
    release.set()
    await asyncio.wait_for(first, timeout=10)
    await _tap(adapter, _update("widget_query", _payload(id="wq-3")))  # the slot is free again

    answers = {answer["widget_query_id"]: answer["data"] for answer in _answers(adapter)}
    assert answers == {"wq-2": '{"error":"busy"}', "wq-1": "done", "wq-3": "done"}


@pytest.mark.asyncio
@pytest.mark.parametrize("query_type", [
    {"type": "sign_data", "wallet": {"address": "0:abc", "amount": "0"}, "sign_data": "{}"},
    {"type": "sent_transaction", "wallet": {"address": "0:abc", "amount": "1"}, "msg_hash": "h"},
    {"type": "a_future_kind"},
], ids=["sign_data", "sent_transaction", "unknown"])
async def test_wallet_and_unknown_query_types_are_refused_without_dispatch(monkeypatch, query_type):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _adapter(widgets=True)
    handler = AsyncMock(return_value="ok")
    adapter.register_widget_query_handler("approve:", handler)

    await _tap(adapter, _update("widget_query", _payload(type=query_type)))

    handler.assert_not_called()
    assert [_error(answer) for answer in _answers(adapter)] == ["unsupported"]


@pytest.mark.asyncio
async def test_query_without_a_type_is_a_plain_data_request(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _adapter(widgets=True)
    handler = AsyncMock(return_value="ok")
    adapter.register_widget_query_handler("approve:", handler)

    await _tap(adapter, _update("widget_query", _payload(type=_DROP)))

    assert handler.await_args.args[0].type == "data"
    assert _answers(adapter) == [{"widget_query_id": "wq-1", "data": "ok"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    NetworkError("connection reset"),
    BadRequest("Query is too old and response timeout expired or query id is invalid"),
], ids=["network", "expired"])
async def test_a_failed_answer_is_neither_raised_nor_retried(monkeypatch, failure):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _adapter(widgets=True)
    adapter._bot.do_api_request = AsyncMock(side_effect=failure)
    adapter.register_widget_query_handler("approve:", AsyncMock(return_value="ok"))

    await _tap(adapter, _update("widget_query", _payload()))

    assert adapter._bot.do_api_request.await_count == 1


@pytest.mark.asyncio
async def test_long_answer_is_cut_to_1024_utf8_bytes_on_a_character_boundary(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _adapter(widgets=True)
    adapter.register_widget_query_handler("approve:", AsyncMock(return_value="a" + "😀" * 300))  # 1201 bytes

    await _tap(adapter, _update("widget_query", _payload()))

    [answer] = _answers(adapter)
    assert answer["data"] == "a" + "😀" * 255  # the 256th emoji would end at byte 1025
    assert len(answer["data"].encode("utf-8")) <= 1024


@pytest.mark.parametrize(("text", "limit", "expected"), [
    ("é" * 600, 1024, "é" * 512),
    ("x" + "€" * 400, 1024, "x" + "€" * 341),
    ("fits", 1024, "fits"),
    ("", 1024, ""),
])
def test_truncate_utf8_never_splits_a_character(text, limit, expected):
    cut = widgets.truncate_utf8(text, limit)

    assert cut == expected
    assert len(cut.encode("utf-8")) <= limit


@pytest.mark.asyncio
async def test_non_widget_updates_pass_through_without_side_effects(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _adapter(widgets=True)
    claim = _claim(adapter)
    handler = AsyncMock(return_value="ok")
    adapter.register_widget_query_handler("approve:", handler)
    message = SimpleNamespace(update_id=5, message=SimpleNamespace(text="approve:abc"), api_kwargs=MappingProxyType({}))
    callback = SimpleNamespace(
        update_id=6, callback_query=SimpleNamespace(data="approve:abc"), api_kwargs=MappingProxyType({}))

    for update in (message, callback, MagicMock()):
        await _tap(adapter, update)

    handler.assert_not_called()
    adapter._bot.do_api_request.assert_not_awaited()
    assert claim == SimpleNamespace(accepted=False, failed=False)


@pytest.mark.asyncio
async def test_query_without_an_id_cannot_be_answered_and_is_not_dispatched(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _adapter(widgets=True)
    claim = _claim(adapter)
    handler = AsyncMock(return_value="ok")
    adapter.register_widget_query_handler("approve:", handler)

    await _tap(adapter, _update("widget_query", _payload(id=_DROP)))

    handler.assert_not_called()
    adapter._bot.do_api_request.assert_not_awaited()
    assert claim == SimpleNamespace(accepted=False, failed=False)


@pytest.mark.asyncio
async def test_caught_preparation_error_marks_the_update_failed(monkeypatch):
    def broken(update):
        raise RuntimeError("payload parser bug")

    monkeypatch.setattr(widgets, "widget_query_payload", broken)
    adapter = _adapter(widgets=True)
    claim = _claim(adapter)

    await _tap(adapter, _update("widget_query", _payload()))

    adapter._bot.do_api_request.assert_not_awaited()
    assert claim == SimpleNamespace(accepted=False, failed=True)


def test_registry_rejects_unusable_registrations():
    adapter = _adapter(widgets=True)

    with pytest.raises(ValueError):
        adapter.register_widget_query_handler("", AsyncMock())
    with pytest.raises(ValueError):
        adapter.register_widget_query_handler(None, AsyncMock())
    with pytest.raises(TypeError):
        adapter.register_widget_query_handler("approve:", "not callable")


async def _async_handler(query, suffix=""):
    return "ok" + suffix


class _AsyncCallable:
    async def __call__(self, query):
        return "ok"


def test_registry_takes_only_handlers_that_cannot_block_the_event_loop():
    adapter = _adapter(widgets=True)

    for blocking in (lambda query: "ok", MagicMock(return_value="ok"), print):
        with pytest.raises(TypeError):
            adapter.register_widget_query_handler("approve:", blocking)
    for awaitable in (_async_handler, functools.partial(_async_handler, suffix="!"), AsyncMock(), _AsyncCallable()):
        adapter.register_widget_query_handler("approve:", awaitable)


@pytest.mark.asyncio
async def test_registering_a_prefix_again_replaces_its_handler(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    adapter = _adapter(widgets=True)
    old, new = AsyncMock(return_value="old"), AsyncMock(return_value="new")
    adapter.register_widget_query_handler("approve:", old)
    adapter.register_widget_query_handler("approve:", new)

    await _tap(adapter, _update("widget_query", _payload()))

    old.assert_not_called()
    assert _answers(adapter) == [{"widget_query_id": "wq-1", "data": "new"}]


# --- config surface ------------------------------------------------------------------------------------


@pytest.mark.parametrize("yaml_text", [
    "gateway:\n  platforms:\n    telegram:\n      extra:\n        widgets: true\n",
    "platforms:\n  telegram:\n    extra:\n      widgets: true\n",
    "telegram:\n  extra:\n    widgets: true\n",
], ids=["gateway.platforms", "platforms", "top-level telegram"])
def test_widgets_flag_reaches_the_adapter_through_the_gateway_config_loader(tmp_path, monkeypatch, yaml_text):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    adapter = TelegramAdapter(load_gateway_config().platforms[Platform.TELEGRAM])

    assert adapter._allowed_update_types(["message"]) == ["message", "widget_query", "shared_widget_query"]
