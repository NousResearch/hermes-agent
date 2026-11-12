"""Bot API 10.4 Telegram HTML widgets: transport validation, payloads, and failure semantics.

Widgets are production-private-only, so every send is checked before the network call. The
HTML is passed through untouched (the Telegram QuickJS sandbox owns execution/sanitizing),
and a failure never degrades to text: that would silently drop the interactive UI.
"""

import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from plugins.platforms.telegram import telegram_widgets as tw
from plugins.platforms.telegram.telegram_widgets import TelegramWidgetsMixin

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[2]

_REAL_PTB_PROBE = r"""
import json, sys
sys.path.insert(0, r"{repo}")
from plugins.platforms.telegram.telegram_widgets import TelegramWidgetsMixin
from telegram.request._requestparameter import RequestParameter
from telegram.request._requestdata import RequestData

class Host(TelegramWidgetsMixin):
    def _metadata_thread_id(self, metadata):
        return (metadata or {{}}).get("thread_id")
    def _thread_kwargs_for_send(self, chat_id, thread_id, metadata=None, reply_to_message_id=None, reply_to_mode=None):
        return {{"message_thread_id": int(thread_id)}} if thread_id else {{}}
    def _notification_kwargs(self, metadata):
        return {{}}

payload = Host()._widget_send_payload(
    "123",
    {{"html": "<h3>Order</h3>", "caption": "Order", "width": 320, "height": 180,
      "public_data": "{{}}", "chat_data": "x", "allow_queries": True, "is_shared": True}},
    {{"thread_id": "7"}},
)
data = RequestData(parameters=[RequestParameter.from_input(k, v) for k, v in payload.items()])
print("___RESULT___")
print(json.dumps(data.json_parameters))
"""


def _probe_real_ptb():
    proc = subprocess.run(
        [sys.executable, "-c", _REAL_PTB_PROBE.format(repo=str(_REPO))],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        pytest.skip(f"real python-telegram-bot unavailable: {proc.stderr.strip()[-300:]}")
    marker = proc.stdout.rfind("___RESULT___")
    return json.loads(proc.stdout[marker + len("___RESULT___"):].strip())


class RetryAfter(Exception):
    def __init__(self, seconds):
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


class BadRequest(Exception):
    """PTB-faithful class name: the adapter's classifier is type/name based."""


class Host(TelegramWidgetsMixin):
    def __init__(self, result=None, side_effect=None):
        self.name = "Telegram"
        self._bot = SimpleNamespace(
            do_api_request=AsyncMock(return_value=result, side_effect=side_effect))
        self.flood = {}

    def _metadata_thread_id(self, metadata):
        return (metadata or {}).get("thread_id")

    def _thread_kwargs_for_send(self, chat_id, thread_id, metadata=None,
                                reply_to_message_id=None, reply_to_mode=None):
        return {"message_thread_id": int(thread_id)} if thread_id else {}

    def _notification_kwargs(self, metadata):
        return {} if (metadata or {}).get("notify") else {"disable_notification": True}

    def _record_send_flood_cooldown(self, chat_id, wait):
        self.flood[str(chat_id)] = wait


def _call(host):
    args, kwargs = host._bot.do_api_request.call_args
    return args[0], kwargs


FULL_SPEC = {
    "html": "<h3>Poll</h3><button>Yes</button>",
    "caption": "Daily poll",
    "width": 320,
    "height": 180,
    "public_data": '{"v":1}',
    "chat_data": '{"picked":null}',
    "allow_queries": True,
    "is_shared": True,
    "show_caption_above_media": True,
}


# --- payload + response ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_widget_payload_and_shared_widget_id():
    message = {
        "message_id": 777,
        "widget": {"id": "shared-1", "html": FULL_SPEC["html"], "public_data": '{"v":1}',
                   "chat_data": '{"picked":null}'},
    }
    host = Host(result=message)
    result = await host.send_widget(
        "123", FULL_SPEC, {"thread_id": "12", "chat_type": "private", "notify": True})

    assert isinstance(result, SendResult) and result.success
    assert result.message_id == "777"
    assert result.raw_response["widget"]["id"] == "shared-1"
    assert result.raw_response["widget"]["public_data"] == '{"v":1}'
    assert result.raw_response["widget"]["chat_data"] == '{"picked":null}'
    assert result.raw_response["message"]["widget"]["id"] == "shared-1"
    assert result.raw_response["widget_id"] == "shared-1"

    method, kwargs = _call(host)
    payload = kwargs["api_kwargs"]
    assert method == "sendWidget"
    assert payload["chat_id"] == "123"
    assert payload["message_thread_id"] == 12
    assert "disable_notification" not in payload  # notify=True preserved
    assert payload["html"] == FULL_SPEC["html"]  # never sanitized locally
    assert payload["caption"] == "Daily poll"
    assert payload["width"] == 320 and payload["height"] == 180
    assert payload["public_data"] == '{"v":1}'
    assert payload["chat_data"] == '{"picked":null}'
    assert payload["allow_queries"] is True
    assert payload["is_shared"] is True
    assert payload["show_caption_above_media"] is True


@pytest.mark.asyncio
async def test_send_widget_silent_by_default_and_never_text_fallback():
    host = Host(result={"message_id": 1})
    result = await host.send_widget("123", {"html": "<b>x</b>", "caption": "x"})

    payload = _call(host)[1]["api_kwargs"]
    assert payload["disable_notification"] is True
    assert result.success


@pytest.mark.asyncio
async def test_send_shared_widget_payload():
    host = Host(result={"message_id": 9, "widget": {"id": "w-9"}})
    result = await host.send_shared_widget(
        "123", {"widget_id": "w-9", "caption": "Again", "chat_data": "", "show_caption_above_media": False})

    method, kwargs = _call(host)
    assert method == "sendSharedWidget"
    assert kwargs["api_kwargs"] == {
        "chat_id": "123", "widget_id": "w-9", "caption": "Again", "chat_data": "",
        "show_caption_above_media": False, "disable_notification": True,
    }
    assert result.raw_response["widget_id"] == "w-9"


# --- documented bounds -------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("html", ["", "<p>\x00</p>", "<p>" + "a" * 32768 + "</p>"], ids=["empty", "nul", "oversize"])
async def test_html_size_and_nul_rejected(html):
    host = Host(result={"message_id": 1})
    result = await host.send_widget("123", {"html": html, "caption": "c"})
    assert not result.success and result.raw_response["validation_error"]
    host._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_html_compressed_limit_is_enforced():
    # Hex is ~2 chars per random byte, so 9000 random bytes compress well past 4096.
    import random
    html = random.Random(0).randbytes(9000).hex()
    assert len(html.encode()) < tw._WIDGET_MAX_HTML_BYTES
    assert len(tw.zlib.compress(html.encode(), 9)) > tw._WIDGET_MAX_COMPRESSED_BYTES
    host = Host(result={"message_id": 1})
    result = await host.send_widget("123", {"html": html, "caption": "c"})
    assert not result.success and result.raw_response["validation_error"]


@pytest.mark.asyncio
async def test_data_limits_are_utf8_bytes_not_characters():
    host = Host(result={"message_id": 1})
    ok = await host.send_widget(
        "123", {"html": "<p>x</p>", "caption": "c", "public_data": "я" * 512, "chat_data": ""})
    assert ok.success
    bad = await host.send_widget(
        "123", {"html": "<p>x</p>", "caption": "c", "chat_data": "я" * 513})
    assert not bad.success and bad.raw_response["validation_error"]


@pytest.mark.asyncio
async def test_caption_required_and_utf16_bound():
    host = Host(result={"message_id": 1})
    missing = await host.send_widget("123", {"html": "<p>x</p>"})
    assert not missing.success and missing.raw_response["validation_error"]
    blank = await host.send_widget("123", {"html": "<p>x</p>", "caption": "   "})
    assert not blank.success
    exact = await host.send_widget("123", {"html": "<p>x</p>", "caption": "😀" * 512})
    assert exact.success
    over = await host.send_widget("123", {"html": "<p>x</p>", "caption": "😀" * 513})
    assert not over.success and over.raw_response["validation_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("width", True), ("width", 199), ("width", 601), ("height", False), ("height", 49), ("height", 601),
    ("width", "320"), ("allow_queries", 1), ("is_shared", "yes"),
])
async def test_dimension_and_flag_types(field, value):
    host = Host(result={"message_id": 1})
    spec = {"html": "<p>x</p>", "caption": "c", field: value}
    result = await host.send_widget("123", spec)
    assert not result.success and result.raw_response["validation_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_time", [4, 3601, True, 5.0, "30"])
async def test_cache_time_bounds(cache_time):
    host = Host(result=True)
    result = await host.edit_shared_widget_data("w1", "{}", cache_time=cache_time)
    assert not result.success and result.raw_response["validation_error"]
    host._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_and_paid_or_wallet_keys_rejected():
    host = Host(result={"message_id": 1})
    for extra in ({"allow_paid_broadcast": True}, {"parse_mode": "HTML"},
                  {"wallet": "x"}, {"protect_content": True}):
        result = await host.send_widget(
            "123", {"html": "<p>x</p>", "caption": "c", **extra})
        assert not result.success and "unsupported key" in result.error
    host._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id,metadata", [
    ("-100123", None), ("0", None), ("@channel", None), ("123", {"chat_type": "group"}),
    ("123", {"chat_type": "supergroup"}), ("123", {"chat_type": "channel"}),
])
async def test_private_chat_only(chat_id, metadata):
    host = Host(result={"message_id": 1})
    result = await host.send_widget(chat_id, {"html": "<p>x</p>", "caption": "c"}, metadata)
    assert not result.success and "private" in result.error
    host._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_dm_chat_type_is_accepted():
    host = Host(result={"message_id": 1})
    result = await host.send_widget(
        "123", {"html": "<p>x</p>", "caption": "c"}, {"chat_type": "dm"})
    assert result.success


# --- edits -------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_widget_data_missing_vs_empty_preserved():
    host = Host(result=True)
    result = await host.edit_widget_data("123", 55, {"chat_data": ""})
    method, kwargs = _call(host)
    assert method == "editMessageWidgetData"
    assert kwargs["api_kwargs"] == {"chat_id": "123", "message_id": 55, "chat_data": ""}
    assert "public_data" not in kwargs["api_kwargs"]  # missing = unchanged
    assert result.success and result.message_id == "55"


@pytest.mark.asyncio
async def test_edit_widget_data_rejects_empty_or_unknown_changes():
    host = Host(result=True)
    empty = await host.edit_widget_data("123", 55, {})
    unknown = await host.edit_widget_data("123", 55, {"html": "x"})
    assert not empty.success and not unknown.success
    host._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_widget_data_message_not_modified_is_success():
    host = Host(side_effect=BadRequest("Bad Request: message is not modified"))
    result = await host.edit_widget_data("123", 55, {"public_data": "{}"})
    assert result.success and result.message_id == "55"


@pytest.mark.asyncio
async def test_edit_shared_widget_data_payload_and_cache_time():
    host = Host(result=True)
    result = await host.edit_shared_widget_data("w1", "", cache_time=30)
    method, kwargs = _call(host)
    assert method == "editSharedWidgetData"
    assert kwargs["api_kwargs"] == {"widget_id": "w1", "public_data": "", "cache_time": 30}
    assert result.success
    host2 = Host(result=True)
    await host2.edit_shared_widget_data("w1", "{}")
    assert "cache_time" not in _call(host2)[1]["api_kwargs"]


@pytest.mark.asyncio
async def test_edit_shared_widget_data_empty_public_data_clears():
    host = Host(result=True)
    result = await host.edit_shared_widget_data("w1", "")
    assert result.success
    assert _call(host)[1]["api_kwargs"]["public_data"] == ""


# --- widget queries ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_answer_widget_query_payload_and_bool():
    host = Host(result=True)
    assert await host.answer_widget_query("q1", "ok") is True
    method, kwargs = _call(host)
    assert method == "answerWidgetQuery"
    assert kwargs["api_kwargs"] == {"widget_query_id": "q1", "data": "ok"}
    host2 = Host(result=False)
    assert await host2.answer_widget_query("q2") is False
    assert _call(host2)[1]["api_kwargs"] == {"widget_query_id": "q2"}


@pytest.mark.asyncio
async def test_answer_widget_query_rejects_bad_input_and_reports_failure():
    host = Host(result=True)
    assert await host.answer_widget_query("", "x") is False
    assert await host.answer_widget_query("q", "x" * 1025) is False
    host._bot.do_api_request.assert_not_awaited()
    failing = Host(side_effect=BadRequest("Bad Request: query is too old"))
    assert await failing.answer_widget_query("q") is False


# --- error semantics ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_permanent_refusal_is_descriptive_and_not_retryable():
    host = Host(side_effect=BadRequest("Bad Request: INPUT_MEDIA_WIDGET_HTML_INVALID"))
    result = await host.send_widget("123", {"html": "<p>x</p>", "caption": "c"})
    assert not result.success
    assert result.retryable is False
    assert "sendWidget failed" in result.error and "INPUT_MEDIA_WIDGET_HTML_INVALID" in result.error
    assert result.raw_response["error_kind"] == "refused"


@pytest.mark.asyncio
async def test_timeout_is_ambiguous_and_never_retried():
    host = Host(side_effect=Exception("Timed out"))
    result = await host.send_widget("123", {"html": "<p>x</p>", "caption": "c"})
    assert not result.success
    assert result.retryable is False
    assert result.raw_response["delivery_uncertain"] is True


@pytest.mark.asyncio
async def test_flood_returns_canonical_result_and_arms_window():
    host = Host(side_effect=RetryAfter(97))
    result = await host.send_widget("123", {"html": "<p>x</p>", "caption": "c"})
    assert not result.success
    assert result.error == "flood_control:97.0"
    assert result.retry_after == 97.0
    assert host.flood == {"123": 97}


@pytest.mark.asyncio
async def test_shared_widget_refusal_does_not_fall_back_to_text():
    host = Host(side_effect=BadRequest("Forbidden: bot was blocked by the user"))
    result = await host.send_shared_widget(
        "123", {"widget_id": "w1", "caption": "c"})
    assert not result.success and result.retryable is False
    assert not hasattr(host._bot, "send_message")


# --- real PTB serialization --------------------------------------------------------------

def test_real_ptb_request_data_serializes_widget_payload():
    params = _probe_real_ptb()
    assert json.loads(params["chat_id"]) == 123
    assert json.loads(params["message_thread_id"]) == 7
    assert params["html"] == "<h3>Order</h3>"
    assert params["caption"] == "Order"
    assert json.loads(params["width"]) == 320
    assert json.loads(params["height"]) == 180
    assert params["public_data"] == "{}"
    assert params["chat_data"] == "x"
    assert json.loads(params["allow_queries"]) is True
    assert json.loads(params["is_shared"]) is True
