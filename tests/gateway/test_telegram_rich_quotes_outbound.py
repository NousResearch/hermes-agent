"""Bot API 10.4 outgoing rich quotes: ReplyParameters.quote_rich_texts validation and the
single ``sendRichMessage`` transport.

Pure builder tests assert exact structure/formatting preservation, the official limits (32
quoted texts, 1024 rendered UTF-16 code units) and rejection of types Telegram would silently
replace. Transport tests assert one request, no quote-less fallback, and ambiguous failures
that are never retried. A subprocess-level test runs the REAL PTB ``RequestData`` pipeline over
the captured payload — the repo conftest installs a MagicMock ``telegram`` in-process.
"""

import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.telegram import telegram_rich_quotes as rq

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[2]

_REAL_PTB_PROBE = r'''
import asyncio, json, sys
sys.path.insert(0, r"__REPO__")
from types import SimpleNamespace
from unittest.mock import AsyncMock
from plugins.platforms.telegram.telegram_rich_quotes import (
    build_rich_reply_parameters, send_rich_reply)

class Host:
    def __init__(self):
        self._bot = SimpleNamespace(do_api_request=AsyncMock(return_value={"message_id": 555}))
        self._reply_to_mode = "on"
        self._disable_link_previews = False

    def _rich_message_payload(self, content, *, skip_entity_detection=False, metadata=None):
        return {"markdown": content}

    def _metadata_thread_id(self, metadata):
        return (metadata or {}).get("thread_id")

    def _thread_kwargs_for_send(self, chat_id, thread_id, metadata=None,
                                reply_to_message_id=None, reply_to_mode=None):
        return {"message_thread_id": int(thread_id)} if thread_id else {}

    def _notification_kwargs(self, metadata):
        return {"disable_notification": True} if (metadata or {}).get("notify") is False else {}

    def _send_flood_cooldown_remaining(self, chat_id):
        return None

    async def _record_rich_sent(self, chat_id, message_id, content):
        return None

async def main():
    host = Host()
    params = build_rich_reply_parameters(
        99, [[{"type": "bold", "text": "quoted"}], ["second"]], 3)
    result = await send_rich_reply(
        host, -100123, "answer **here**", params, {"thread_id": "8", "notify": False})
    payload = host._bot.do_api_request.call_args.kwargs["api_kwargs"]
    from telegram.request._requestparameter import RequestParameter
    from telegram.request._requestdata import RequestData
    data = RequestData(parameters=[RequestParameter.from_input(k, v) for k, v in payload.items()])
    print("___RESULT___")
    print(json.dumps({"json": data.json_parameters, "ok": result.success,
                      "message_id": result.message_id}))

asyncio.run(main())
'''


def _probe_real_ptb():
    """Run the serialization probe in a clean interpreter (no conftest telegram mock)."""
    code = _REAL_PTB_PROBE.replace("__REPO__", str(_REPO))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        pytest.skip(f"real python-telegram-bot unavailable: {proc.stderr.strip()[-300:]}")
    marker = proc.stdout.rfind("___RESULT___")
    return json.loads(proc.stdout[marker + len("___RESULT___"):].strip())


class Host:
    """Adapter surface ``send_rich_reply`` composes, without a live Telegram connection."""

    def __init__(self, cooldown=None):
        self._bot = SimpleNamespace(do_api_request=AsyncMock(return_value={"message_id": 777}))
        self._reply_to_mode = "on"
        self._disable_link_previews = False
        self._rich_send_disabled = False
        self._record_rich_sent = AsyncMock()
        self._cooldown = cooldown
        self.flood_notes = []

    def _rich_message_payload(self, content, *, skip_entity_detection=False, metadata=None):
        return {"markdown": content}

    def _metadata_thread_id(self, metadata):
        return (metadata or {}).get("thread_id")

    def _thread_kwargs_for_send(self, chat_id, thread_id, metadata=None,
                                reply_to_message_id=None, reply_to_mode=None):
        return {"message_thread_id": int(thread_id)} if thread_id else {}

    def _notification_kwargs(self, metadata):
        return {"disable_notification": True} if (metadata or {}).get("notify") is False else {}

    def _send_flood_cooldown_remaining(self, chat_id):
        return self._cooldown

    def _record_send_flood_cooldown(self, chat_id, wait):
        self.flood_notes.append((chat_id, float(wait)))

    def _is_rich_fallback_error(self, exc):
        text = str(exc).lower()
        return "bad request" in text or "unsupported" in text or "method not found" in text

    def _is_rich_capability_error(self, exc):
        return "method not found" in str(exc).lower()


def _params(message_id=99, quote=None, position=None):
    quote = [["quoted"]] if quote is None else quote
    return rq.build_rich_reply_parameters(message_id, quote, position)


# --- builder: exact structure and formatting ---------------------------------------------

def test_build_preserves_nested_formatting_and_allowed_types():
    quote = [
        ["Intro ", [{"type": "bold", "text": "bold"}], " and ",
         {"type": "italic", "text": ["nest", {"type": "underline", "text": "ed"}]}],
        {"type": "date_time", "text": "2026-11-11",
         "unix_time": 1762819200, "date_time_format": "r"},
        {"type": "custom_emoji", "custom_emoji_id": "5368324170671202286",
         "alternative_text": "👍"},
        {"type": "mathematical_expression", "expression": "x^2"},
    ]
    params = rq.build_rich_reply_parameters(42, quote, 0)

    assert params == {"message_id": 42, "quote_rich_texts": quote, "quote_position": 0}
    # A validated copy: the caller can keep using/mutating its own structure.
    assert params["quote_rich_texts"] is not quote
    assert params["quote_rich_texts"][0] is not quote[0]
    assert params["quote_rich_texts"][0][1][0] == {"type": "bold", "text": "bold"}
    assert rq.rich_quote_plain_text(quote) == "Intro bold and nested\n2026-11-11\n👍\nx^2"


def test_quote_position_is_omitted_when_not_supplied():
    assert "quote_position" not in rq.build_rich_reply_parameters(1, ["x"])


def test_message_id_accepts_a_numeric_string_and_position_zero():
    params = rq.build_rich_reply_parameters("42", ["x"], "0")
    assert params["message_id"] == 42 and params["quote_position"] == 0


def test_rich_quote_error_is_a_value_error():
    assert issubclass(rq.RichQuoteError, ValueError)


def test_multi_block_quote_joins_segments_with_line_feeds():
    quote = [["Item A"], ["Item B: ", [{"type": "bold", "text": "rotate"}], " keys"]]
    assert rq.rich_quote_plain_text(quote) == "Item A\nItem B: rotate keys"
    assert rq.build_rich_reply_parameters(1, quote)["quote_rich_texts"] == quote


def test_thirty_two_quoted_texts_is_the_limit():
    quote = ["x"] * rq.MAX_QUOTE_TEXTS
    assert len(rq.build_rich_reply_parameters(1, quote)["quote_rich_texts"]) == 32
    with pytest.raises(rq.RichQuoteError):
        rq.build_rich_reply_parameters(1, ["x"] * (rq.MAX_QUOTE_TEXTS + 1))


def test_rendered_length_limit_accepts_1024_and_rejects_1025():
    assert rq.build_rich_reply_parameters(1, ["a" * 1024])
    with pytest.raises(rq.RichQuoteError, match="1024"):
        rq.build_rich_reply_parameters(1, ["a" * 1025])


def test_rendered_length_is_total_across_segments():
    assert rq.build_rich_reply_parameters(1, ["a" * 512, "b" * 512])
    with pytest.raises(rq.RichQuoteError, match="1024"):
        rq.build_rich_reply_parameters(1, ["a" * 512, "b" * 513])


def test_rendered_length_uses_utf16_code_units():
    # Telegram measures text positions in UTF-16 code units; an emoji is two units.
    assert rq.build_rich_reply_parameters(1, ["😀" * 512])
    with pytest.raises(rq.RichQuoteError, match="1024"):
        rq.build_rich_reply_parameters(1, ["😀" * 513])


@pytest.mark.parametrize("quote", [
    None,
    "plain text is not a top-level array",
    {"type": "bold", "text": "x"},
    [],
    [""],
    ["   "],
    [{"type": "code", "text": "x"}],
    [{"type": "url", "url": "https://example.com"}],
    [{"text": "missing type"}],
    [{"type": "bold"}],
    [{"type": "bold", "text": 1}],
    [{"type": "bold", "text": "x", "url": "https://example.com"}],
    [{"type": "date_time", "text": "t", "unix_time": 0}],
    [{"type": "date_time", "text": "t", "unix_time": True, "date_time_format": "r"}],
    [{"type": "date_time", "text": "t", "unix_time": 0, "date_time_format": ""}],
    [{"type": "custom_emoji", "custom_emoji_id": "abc", "alternative_text": "x"}],
    [{"type": "custom_emoji", "custom_emoji_id": "1", "alternative_text": ""}],
    [{"type": "mathematical_expression"}],
    [{"type": "mathematical_expression", "expression": ""}],
    [{"type": "mathematical_expression", "expression": "x", "extra": 1}],
    [None],
    [["x", 5]],
    [["x", []]],
])
def test_rejects_malformed_or_unquotable_nodes(quote):
    with pytest.raises(rq.RichQuoteError):
        rq.build_rich_reply_parameters(1, quote)


@pytest.mark.parametrize("message_id", [0, -1, True, False, 1.5, None, "", "abc", "-1"])
def test_rejects_invalid_message_ids(message_id):
    with pytest.raises(rq.RichQuoteError):
        rq.build_rich_reply_parameters(message_id, ["x"])


@pytest.mark.parametrize("position", [-1, True, 1.5, "abc"])
def test_rejects_invalid_quote_positions(position):
    with pytest.raises(rq.RichQuoteError):
        rq.build_rich_reply_parameters(1, ["x"], position)


def test_rejects_excessive_recursion_and_node_counts():
    node = "x"
    for _ in range(rq._MAX_QUOTE_DEPTH + 8):
        node = {"type": "bold", "text": node}
    with pytest.raises(rq.RichQuoteError, match="deep"):
        rq.build_rich_reply_parameters(1, [node])

    cycle = []
    cycle.append(cycle)
    with pytest.raises(rq.RichQuoteError, match="deep"):
        rq.build_rich_reply_parameters(1, [cycle])

    many = [[{"type": "bold", "text": "x"}] * 2100]
    with pytest.raises(rq.RichQuoteError, match="nodes"):
        rq.build_rich_reply_parameters(1, many)


# --- transport: one sendRichMessage, no quote-less fallback -------------------------------

@pytest.mark.asyncio
async def test_send_rich_reply_builds_one_request_with_quote_and_routing():
    host = Host()
    quote = [[{"type": "bold", "text": "quoted "}, "text"]]
    params = rq.build_rich_reply_parameters(99, quote, 0)

    result = await rq.send_rich_reply(
        host, "-100123", "answer", params, {"thread_id": "8", "notify": False})

    assert result.success and result.message_id == "777"
    assert result.raw_response["message"] == {"message_id": 777}
    host._bot.do_api_request.assert_awaited_once()
    call = host._bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessage"
    payload = call.kwargs["api_kwargs"]
    assert payload["chat_id"] == -100123
    assert payload["rich_message"] == {"markdown": "answer"}
    assert payload["reply_parameters"] == params
    assert payload["reply_parameters"]["quote_rich_texts"] == quote
    assert payload["message_thread_id"] == 8
    assert payload["disable_notification"] is True
    assert "reply_to_message_id" not in payload
    host._record_rich_sent.assert_awaited_once_with(-100123, 777, "answer")


@pytest.mark.asyncio
async def test_send_rich_reply_filters_none_thread_id_and_keeps_direct_topic():
    host = Host()
    host._thread_kwargs_for_send = lambda *a, **k: {
        "message_thread_id": None, "direct_messages_topic_id": 55}

    await rq.send_rich_reply(host, "42", "x", _params())

    payload = host._bot.do_api_request.call_args.kwargs["api_kwargs"]
    assert "message_thread_id" not in payload
    assert payload["direct_messages_topic_id"] == 55


@pytest.mark.asyncio
async def test_send_rich_reply_disables_link_previews_when_configured():
    host = Host()
    host._disable_link_previews = True
    await rq.send_rich_reply(host, "42", "x", _params())
    assert host._bot.do_api_request.call_args.kwargs["api_kwargs"]["link_preview_options"] == {
        "is_disabled": True}


@pytest.mark.asyncio
async def test_send_rich_reply_exposes_message_and_widget_details():
    host = Host()
    host._bot.do_api_request = AsyncMock(
        return_value={"message_id": 5, "widget": {"id": "w1"}, "rich_message": {"blocks": []}})

    result = await rq.send_rich_reply(host, "42", "x", _params())

    assert result.success and result.message_id == "5"
    assert result.raw_response["widget"] == {"id": "w1"}
    assert result.raw_response["rich_message"] == {"blocks": []}
    assert result.raw_response["message"]["message_id"] == 5


@pytest.mark.asyncio
async def test_permanent_rejection_never_retries_or_changes_format():
    host = Host()
    host._bot.do_api_request = AsyncMock(
        side_effect=RuntimeError("Bad Request: quote_rich_texts not found in the message"))
    params = _params()

    result = await rq.send_rich_reply(host, "42", "answer", params)

    assert not result.success and not result.retryable
    assert result.raw_response["rejected"] is True
    host._bot.do_api_request.assert_awaited_once()
    # The one and only request still carried the original quote and rich body.
    payload = host._bot.do_api_request.call_args.kwargs["api_kwargs"]
    assert payload["reply_parameters"] == params
    assert payload["rich_message"] == {"markdown": "answer"}


@pytest.mark.asyncio
async def test_capability_rejection_latches_rich_off_and_still_never_retries():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Method Not Found"))

    result = await rq.send_rich_reply(host, "42", "answer", _params())

    assert not result.success and not result.retryable
    assert result.raw_response["rejected"] is True
    assert host._rich_send_disabled is True
    host._bot.do_api_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_transient_failure_is_ambiguous_and_non_retryable():
    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=ConnectionResetError("connection reset by peer"))

    result = await rq.send_rich_reply(host, "42", "answer", _params())

    assert not result.success and not result.retryable
    assert result.raw_response["ambiguous"] is True
    assert result.error_kind == "transient"
    host._bot.do_api_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_flood_refusal_arms_chat_window_and_reports_retry_after():
    class Flood(Exception):
        retry_after = 12

    host = Host()
    host._bot.do_api_request = AsyncMock(side_effect=Flood("Too Many Requests: retry after 12"))

    result = await rq.send_rich_reply(host, "42", "answer", _params())

    assert not result.success and not result.retryable
    assert result.retry_after == 12
    assert result.raw_response["flood_control"] is True
    assert host.flood_notes == [(42, 12.0)]
    host._bot.do_api_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_open_flood_window_fails_closed_without_an_api_call():
    host = Host(cooldown=7.0)

    result = await rq.send_rich_reply(host, "42", "answer", _params())

    assert not result.success
    assert result.error.startswith("flood_control:")
    assert result.retry_after == 7.0
    host._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_reply_parameters_fail_before_any_request():
    host = Host()

    result = await rq.send_rich_reply(
        host, "42", "answer",
        {"message_id": 1, "quote_rich_texts": [{"type": "code", "text": "not quotable"}]})

    assert not result.success and not result.retryable
    host._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_or_extra_reply_parameter_fields_fail_before_any_request():
    host = Host()
    for bad in (
        {"message_id": 1},
        {"message_id": 1, "quote_rich_texts": ["x"], "quote": "x"},
        {"quote_rich_texts": ["x"]},
        ["not", "an", "object"],
    ):
        result = await rq.send_rich_reply(host, "42", "answer", bad)
        assert not result.success and not result.retryable
    host._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_content_fails_before_any_request():
    host = Host()
    result = await rq.send_rich_reply(host, "42", "   ", _params())
    assert not result.success and not result.retryable
    host._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_response_without_message_id_is_ambiguous_and_not_retried():
    host = Host()
    host._bot.do_api_request = AsyncMock(return_value={"ok": True})

    result = await rq.send_rich_reply(host, "42", "answer", _params())

    assert not result.success and not result.retryable
    assert result.raw_response["ambiguous"] is True
    host._record_rich_sent.assert_not_awaited()
    host._bot.do_api_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnected_adapter_fails_without_a_request():
    host = Host()
    host._bot = None
    result = await rq.send_rich_reply(host, "42", "answer", _params())
    assert not result.success and result.error == "Not connected"


@pytest.mark.asyncio
async def test_rich_payload_build_failure_is_reported_without_a_request():
    host = Host()
    host._rich_message_payload = lambda *a, **k: (_ for _ in ()).throw(ValueError("bad rich body"))
    result = await rq.send_rich_reply(host, "42", "answer", _params())
    assert not result.success and not result.retryable
    host._bot.do_api_request.assert_not_awaited()


# --- real PTB serialization --------------------------------------------------------------

def test_real_ptb_serializes_reply_parameters_with_nested_quote_and_rich_body():
    out = _probe_real_ptb()
    assert out["ok"] is True and out["message_id"] == "555"
    json_parameters = out["json"]

    # RequestData maps each api_kwargs value to its wire form.
    assert str(json_parameters["chat_id"]) == "-100123"
    assert str(json_parameters["message_thread_id"]) == "8"
    assert str(json_parameters["disable_notification"]).lower() == "true"
    reply = json.loads(json_parameters["reply_parameters"])
    assert reply == {
        "message_id": 99,
        "quote_rich_texts": [[{"type": "bold", "text": "quoted"}], ["second"]],
        "quote_position": 3,
    }
    assert json.loads(json_parameters["rich_message"]) == {"markdown": "answer **here**"}
    assert "reply_to_message_id" not in json_parameters
