"""Bot API 10.4 widget taps through real python-telegram-bot parsing and dispatch.

python-telegram-bot 22.8 models no widget types. These pin that ``Update.de_json`` hands the adapter
the raw payload in ``api_kwargs``, that a real ``TelegramApplication`` routes it to the opt-in handler,
answers it once and receipts it, and that with ``extra.widgets`` off nothing changes. The raw updates are
the examples from Telegram's widget guide (core.telegram.org/bots/widgets).
"""

import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("telegram")
from telegram import Update
from telegram.request import BaseRequest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.waiting import async_wait_until

# Linux CI slices have no python-telegram-bot (importorskip skips there); the macOS and Windows PR
# lanes install the ``telegram`` extra and import only platforms()-marked files.
pytestmark = pytest.mark.platforms("any")

_BOT_ID = 111
_USER = 111111111

WIDGET_QUERY = """{
  "update_id": 100000001,
  "widget_query": {
    "id": "8123456789012345678",
    "from": { "id": 111111111, "is_bot": false, "first_name": "Alice", "language_code": "en" },
    "chat": { "id": 111111111, "type": "private", "first_name": "Alice" },
    "message_id": 4242,
    "chat_instance": "901234567890123456",
    "widget_id": "7770011223344556677",
    "data": "refresh"
  }
}"""

WALLET_OUTCOME = """{
  "update_id": 100000004,
  "widget_query": {
    "id": "8123456789012340002",
    "from": { "id": 111111111, "is_bot": false, "first_name": "Alice" },
    "chat": { "id": 111111111, "type": "private", "first_name": "Alice" },
    "message_id": 4243,
    "chat_instance": "901234567890123456",
    "type": { "type": "sent_transaction", "wallet": { "address": "UQ0", "amount": "1250000000" }, "msg_hash": "h" },
    "data": "paid:42"
  }
}"""

SHARED_WIDGET_QUERY = """{
  "update_id": 100000002,
  "shared_widget_query": {
    "id": "8123456789012340001",
    "from": { "id": 111111111, "is_bot": false, "first_name": "Alice" },
    "type": { "type": "data" },
    "data": "refresh",
    "widget_id": "7770011223344556677"
  }
}"""


class RecordingBotApi(BaseRequest):
    """Offline Bot API: ``getMe`` answers; every other call is recorded as the parameters PTB puts on the
    wire and answered with ``results[endpoint]`` (``[]`` for getUpdates, else ``true``)."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.results: dict[str, object] = {"getUpdates": []}

    @property
    def read_timeout(self):
        return 1

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        if endpoint == "getMe":
            return 200, json.dumps({"ok": True, "result": {
                "id": _BOT_ID, "is_bot": True, "first_name": "Offline", "username": "offline_bot"}}).encode()
        self.calls.append((endpoint, dict(request_data.json_parameters) if request_data else {}))
        return 200, json.dumps({"ok": True, "result": self.results.get(endpoint, True)}).encode()

    def answers(self) -> list[dict]:
        return [params for endpoint, params in self.calls if endpoint == "answerWidgetQuery"]


@asynccontextmanager
async def connected(monkeypatch, **extra):
    """The real ``connect()`` (TelegramApplication, admission, handler registration) with no network."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=f"{_BOT_ID}:offline-test", extra=extra))
    api = RecordingBotApi()
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", str(_USER))
    monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "")
    monkeypatch.setattr(adapter, "_build_ptb_requests", AsyncMock(return_value=(api, api)))
    monkeypatch.setattr(adapter, "_start_polling_mode", AsyncMock())
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", lambda: None)
    monkeypatch.setattr(adapter, "_restart_task_attr", lambda name, coroutine: coroutine.close())
    monkeypatch.setattr(adapter, "_set_status_indicator", AsyncMock())
    adapter._message_handler = AsyncMock()
    assert await adapter.connect()
    try:
        yield adapter, adapter._app, api
    finally:
        await adapter.disconnect()
        store = getattr(adapter, "_session_store", None)
        if store is not None:
            store.close_all_db_handles()


def _update(raw: str, bot) -> Update:
    return Update.de_json(json.loads(raw), bot)


def _receipted(adapter, app, raw: str) -> bool:
    return f"{app.bot.id}:{json.loads(raw)['update_id']}" in adapter._seen_update_ids


async def _process(adapter, app, raw: str) -> None:
    """Dispatch like the fetcher, then wait out the non-blocking widget task that holds the update's claim."""
    await app.process_update(_update(raw, app.bot))
    await async_wait_until(lambda: not adapter._inflight_update_ids, what="the update's claim released")


def _table(adapter, app) -> list:
    """(group, handler class, callback, recorded as core) per handler, in dispatch order."""
    core = adapter._core_handler_ids
    return [(group, type(h).__name__, h.callback.__name__, id(h) in core) for group, hs in app.handlers.items() for h in hs]


def test_ptb_keeps_a_widget_query_raw_in_api_kwargs():
    update = Update.de_json(json.loads(WIDGET_QUERY), None)

    assert dict(update.api_kwargs["widget_query"])["data"] == "refresh"
    assert update.effective_user is None and update.effective_chat is None and update.effective_message is None


@pytest.mark.asyncio
async def test_widget_query_is_dispatched_answered_once_and_receipted(monkeypatch):
    async with connected(monkeypatch, widgets=True) as (adapter, app, api):
        seen = []

        async def refresh(query):
            seen.append(query)
            return '{"p":5.47}'

        adapter.register_widget_query_handler("refresh", refresh)
        await _process(adapter, app, WIDGET_QUERY)
        receipted = _receipted(adapter, app, WIDGET_QUERY)
        await _process(adapter, app, WIDGET_QUERY)  # Telegram redelivering an update it saw no ack for

    [query] = seen
    assert (query.kind, query.user_id, query.chat_id, query.chat_type, query.message_id, query.widget_id, query.type) == (
        "widget_query", "111111111", "111111111", "private", "4242", "7770011223344556677", "data")
    assert api.answers() == [{"widget_query_id": "8123456789012345678", "data": '{"p":5.47}'}]
    assert receipted


@pytest.mark.asyncio
async def test_shared_widget_query_is_dispatched_and_answered(monkeypatch):
    async with connected(monkeypatch, widgets=True) as (adapter, app, api):
        handler = AsyncMock(return_value="ok")
        adapter.register_widget_query_handler("refresh", handler)
        await _process(adapter, app, SHARED_WIDGET_QUERY)

    query = handler.await_args.args[0]
    assert (query.kind, query.chat_id, query.widget_id) == ("shared_widget_query", None, "7770011223344556677")
    assert api.answers() == [{"widget_query_id": "8123456789012340001", "data": "ok"}]


@pytest.mark.asyncio
async def test_unauthorized_sender_is_refused_once_and_receipted(monkeypatch):
    async with connected(monkeypatch, widgets=True) as (adapter, app, api):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "222")
        handler = AsyncMock(return_value="secret")
        adapter.register_widget_query_handler("refresh", handler)
        await _process(adapter, app, WIDGET_QUERY)
        receipted = _receipted(adapter, app, WIDGET_QUERY)

    handler.assert_not_called()
    [answer] = api.answers()
    assert json.loads(answer["data"]) == {"error": "unauthorized"}
    assert receipted


@pytest.mark.asyncio
async def test_wallet_outcome_is_not_routed_by_its_data_label(monkeypatch):
    async with connected(monkeypatch, widgets=True) as (adapter, app, api):
        handler = AsyncMock(return_value="ok")
        adapter.register_widget_query_handler("paid:", handler)
        await _process(adapter, app, WALLET_OUTCOME)

    handler.assert_not_called()
    assert [json.loads(answer["data"]) for answer in api.answers()] == [{"error": "unsupported"}]


@pytest.mark.asyncio
async def test_flag_off_leaves_widget_queries_as_today_unanswered_and_unreceipted(monkeypatch):
    async with connected(monkeypatch) as (adapter, app, api):
        await _process(adapter, app, WIDGET_QUERY)
        receipted = _receipted(adapter, app, WIDGET_QUERY)

    assert api.answers() == []
    assert not receipted


@pytest.mark.asyncio
async def test_flag_on_only_appends_one_core_widget_handler_to_group_zero(monkeypatch):
    async with connected(monkeypatch) as (off_adapter, off_app, _):
        off = _table(off_adapter, off_app)
    async with connected(monkeypatch, widgets=True) as (on_adapter, on_app, _):
        on = _table(on_adapter, on_app)
        [widget_handler] = [h for h in on_app.handlers[0] if h.callback.__name__ == "_handle_widget_update"]
        message = Update.de_json({"update_id": 7, "message": {
            "message_id": 1, "date": 1800000000, "chat": {"id": _USER, "type": "private"},
            "from": {"id": _USER, "is_bot": False, "first_name": "Alice"}, "text": "refresh"}}, on_app.bot)
        callback = Update.de_json({"update_id": 8, "callback_query": {
            "id": "cq", "from": {"id": _USER, "is_bot": False, "first_name": "Alice"}, "chat_instance": "ci",
            "data": "refresh"}}, on_app.bot)
        matches = [bool(widget_handler.check_update(u)) for u in (
            _update(WIDGET_QUERY, on_app.bot), _update(SHARED_WIDGET_QUERY, on_app.bot), message, callback)]

    widget_rows = [row for row in on if row[2] == "_handle_widget_update"]
    assert len(widget_rows) == 1 and widget_rows[0][0] == 0 and widget_rows[0][3] is True
    assert [row for row in on if row[0] == 0][-1] == widget_rows[0]
    assert [row for row in on if row not in widget_rows] == off
    assert matches == [True, True, False, False]


@pytest.mark.asyncio
async def test_allowed_update_types_reach_get_updates_with_both_widget_types_once(monkeypatch):
    before = list(Update.ALL_TYPES)
    async with connected(monkeypatch, widgets=True) as (adapter, app, api):
        await app.bot.get_updates(allowed_updates=adapter._allowed_update_types(Update.ALL_TYPES))

    [(_, params)] = [call for call in api.calls if call[0] == "getUpdates"]
    assert json.loads(params["allowed_updates"]) == [*before, "widget_query", "shared_widget_query"]
    assert Update.ALL_TYPES == before


@pytest.mark.asyncio
async def test_a_slow_widget_handler_does_not_hold_the_only_update_slot(monkeypatch):
    entered, release, runs = asyncio.Event(), asyncio.Event(), []
    async with connected(monkeypatch, widgets=True, max_concurrent_updates=1) as (adapter, app, api):

        async def slow(query):
            runs.append(query.query_id)
            entered.set()
            await release.wait()
            return "done"

        adapter.register_widget_query_handler("refresh", slow)
        app.update_queue.put_nowait(_update(WIDGET_QUERY, app.bot))
        await asyncio.wait_for(entered.wait(), timeout=10)
        app.update_queue.put_nowait(_update(WIDGET_QUERY, app.bot))  # a duplicate while the first is in flight
        app.update_queue.put_nowait(Update.de_json({"update_id": 9, "callback_query": {
            "id": "cq-1", "from": {"id": _USER, "is_bot": False, "first_name": "Alice"}, "chat_instance": "ci",
            "data": "zz:stale-build-button"}}, app.bot))
        # With the widget handler parked, the next update still gets the processor's only slot.
        await async_wait_until(lambda: any(endpoint == "answerCallbackQuery" for endpoint, _ in api.calls),
                               what="the callback query answered while the widget handler runs")
        assert api.answers() == []
        release.set()
        await async_wait_until(api.answers, what="the widget query answer")
        await async_wait_until(lambda: not adapter._inflight_update_ids, what="the widget claim released")
        receipted = _receipted(adapter, app, WIDGET_QUERY)

    assert api.answers() == [{"widget_query_id": "8123456789012345678", "data": "done"}]
    assert runs == ["8123456789012345678"]
    assert receipted


@pytest.mark.asyncio
async def test_outbound_requests_reach_the_wire_with_omit_and_clear_kept_apart(monkeypatch):
    async with connected(monkeypatch, widgets=True) as (adapter, _app, api):
        api.results["sendWidget"] = {
            "message_id": 77, "date": 1800000000, "chat": {"id": _USER, "type": "private"},
            "widget": {"id": "7770011223344556677", "html": "<b>x</b>"}}
        sent = await adapter.send_widget(str(_USER), "<b>x</b>", reply_to="9", chat_data="", is_shared=True)
        edited = await adapter.edit_widget_data(str(_USER), sent.message_id, public_data="")

    wire = {endpoint: params for endpoint, params in api.calls if endpoint in ("sendWidget", "editMessageWidgetData")}
    assert wire["sendWidget"] == {
        "chat_id": str(_USER), "html": "<b>x</b>", "chat_data": "", "is_shared": "true",
        "reply_parameters": '{"message_id": 9}', "disable_notification": "true"}
    assert (sent.success, sent.message_id, sent.raw_response) == (True, "77", {"widget_id": "7770011223344556677"})
    assert wire["editMessageWidgetData"] == {"chat_id": str(_USER), "message_id": "77", "public_data": ""}
    assert edited.success


def test_flag_off_requests_ptbs_own_update_types_list():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=f"{_BOT_ID}:offline-test", extra={}))

    assert adapter._allowed_update_types(Update.ALL_TYPES) is Update.ALL_TYPES


_ROOT = Path(__file__).resolve().parents[4]
# The gateway's first Telegram run on a fresh install: python-telegram-bot is missing while plugins load,
# then the registry's lazy installer makes it importable in-process. Production loads the adapter module
# twice (hermes_plugins.platforms__telegram.adapter and plugins.platforms.telegram.adapter) and the
# installer rebinds the PTB names only in the copy whose class is live.
_FIRST_USE_PROBE = r"""
import asyncio, importlib.abc, json, os, sys
from unittest.mock import AsyncMock, MagicMock
home, widgets = sys.argv[1], sys.argv[2]
os.environ["HERMES_HOME"] = home
os.environ["TELEGRAM_WEBHOOK_SECRET"] = "s3cret"
with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as fh:
    fh.write("platforms:\n  telegram:\n    enabled: true\n    token: '123456:offline'\n"
             "    extra:\n      widgets: " + widgets + "\n")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "telegram" or name.startswith("telegram."):
            raise ModuleNotFoundError("No module named " + repr(name), name=name)
blocker = Block()
sys.meta_path.insert(0, blocker)
from hermes_cli.plugins import discover_plugins
discover_plugins()
from gateway.config import Platform, load_gateway_config
from gateway.platform_registry import platform_registry
platform_registry.get("telegram")
sys.meta_path.remove(blocker)  # the on-demand install has landed
import pm
pm.ensure_import = lambda name: None
adapter = platform_registry.create_adapter("telegram", load_gateway_config().platforms[Platform.TELEGRAM])
import telegram
from telegram.ext import ApplicationBuilder
async def starts():
    app = MagicMock()
    app.updater.start_polling = AsyncMock()
    await adapter._start_polling_once(app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)
    adapter._app = MagicMock()
    adapter._app.updater.start_webhook = AsyncMock()
    await adapter._start_webhook_mode("https://example.test/telegram", is_reconnect=False)
    return (app.updater.start_polling.await_args.kwargs["allowed_updates"],
            adapter._app.updater.start_webhook.await_args.kwargs["allowed_updates"])
polling, webhook = asyncio.run(starts())
app = ApplicationBuilder().token("123456:offline").build()
adapter._register_handlers(app)
print("RESULT " + json.dumps({
    "live_module": type(adapter).__module__,
    "polling_is_all_types": polling is telegram.Update.ALL_TYPES,
    "webhook_is_all_types": webhook is telegram.Update.ALL_TYPES,
    "polling": [str(t) for t in polling],
    "widget_handler_groups": [group for group, handlers in app.handlers.items() for h in handlers
                              if getattr(h.callback, "__name__", "") == "_handle_widget_update"]}))
"""


@pytest.mark.parametrize("widgets", ["false", "true"])
def test_first_use_lazy_install_still_polls_and_registers_handlers(tmp_path, widgets):
    run = subprocess.run([sys.executable, "-c", _FIRST_USE_PROBE, str(tmp_path), widgets], cwd=_ROOT,
                         capture_output=True, text=True, timeout=240, stdin=subprocess.DEVNULL, check=False)
    line = next((ln for ln in run.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, run.stdout[-2000:] + run.stderr[-6000:]
    result = json.loads(line[len("RESULT "):])

    assert result["live_module"].startswith("hermes_plugins."), "not the production plugin-loading path"
    if widgets == "false":
        assert result["polling_is_all_types"] and result["webhook_is_all_types"]
        assert result["widget_handler_groups"] == []
    else:
        assert result["polling"][-2:] == ["widget_query", "shared_widget_query"]
        assert result["widget_handler_groups"] == [0]
