"""Bot API 10.4 widget sends and data edits on the Telegram adapter (opt-in ``extra.widgets``).

The helpers validate locally before any request (Telegram refuses oversized or NUL-bearing widgets
anyway, after a round-trip), reuse the rich-message send routing for forum and DM topics, and never
re-send after an ambiguous failure: a request that timed out may already have reached Telegram.
"""

import random
import string
import zlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

from gateway.config import PlatformConfig
from plugins.platforms.telegram import telegram_widgets as widgets
from plugins.platforms.telegram.adapter import TelegramAdapter

_MESSAGE = {"message_id": 55, "date": 1800000000, "chat": {"id": 42, "type": "private"},
            "widget": {"html": "<b>hi</b>"}}


def _adapter(**extra) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:widget-test", extra={"widgets": True, **extra}))
    adapter._bot = MagicMock()
    adapter._bot.do_api_request = AsyncMock(return_value=dict(_MESSAGE))
    return adapter


def _api_calls(adapter) -> list:
    return [(c.args[0], c.kwargs["api_kwargs"]) for c in adapter._bot.do_api_request.await_args_list]


_HELPERS = {
    "send_widget": lambda adapter: adapter.send_widget("42", "<b>hi</b>"),
    "send_shared_widget": lambda adapter: adapter.send_shared_widget("42", "w-1"),
    "edit_widget_data": lambda adapter: adapter.edit_widget_data("42", "7", chat_data="x"),
    "edit_shared_widget_data": lambda adapter: adapter.edit_shared_widget_data("w-1", "x"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", sorted(_HELPERS))
@pytest.mark.parametrize("extra", [{}, {"widgets": False}])
async def test_helpers_refuse_while_widgets_are_off(helper, extra):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:widget-test", extra=extra))
    adapter._bot = MagicMock()
    adapter._bot.do_api_request = AsyncMock()

    result = await _HELPERS[helper](adapter)

    assert result.success is False and "disabled" in result.error
    adapter._bot.do_api_request.assert_not_awaited()


def _incompressible(length: int) -> str:
    rng = random.Random(0)
    return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(length))


def _compressed_size(html: str) -> int:
    return len(zlib.compress(html.encode("utf-8"), 9))


def test_incompressible_sample_trips_only_the_compressed_limit_on_any_zlib_build():
    sample = _incompressible(8000)

    assert len(sample.encode("utf-8")) <= 32768
    assert _compressed_size(sample) > 4096 + 4096 // 16  # past even zlib-ng's tolerance band


def _html_compressing_into(low: int, high: int) -> str:
    """A prefix of incompressible text whose zlib-9 size (this interpreter's zlib) lands in (low, high]."""
    text = _incompressible(9000)
    for length in range(4000, len(text), 8):
        if low < _compressed_size(text[:length]) <= high:
            return text[:length]
    raise AssertionError(f"no sample compresses into ({low}, {high}]")


@pytest.mark.asyncio
@pytest.mark.parametrize(("slack", "sent"), [(0, False), (256, True)], ids=["classic-zlib", "zlib-ng"])
async def test_html_just_past_the_compressed_limit_is_left_to_telegram_only_where_zlib_counts_differently(
        monkeypatch, slack, sent):
    # Telegram counts with classic zlib, which CPython's usual zlib matches byte for byte; zlib-ng (the
    # PM-pinned Python's) compresses a few percent larger, so a local count just over 4096 is not proof.
    monkeypatch.setattr(widgets, "_COMPRESSED_SLACK", slack)
    adapter = _adapter()

    result = await adapter.send_widget("42", _html_compressing_into(4096, 4096 + 128))

    assert result.success is sent
    assert adapter._bot.do_api_request.await_count == int(sent)


_INVALID = {
    "html over 32768 bytes": lambda a: a.send_widget("42", "x" * 32769),
    "html over 4096 compressed": lambda a: a.send_widget("42", _incompressible(8000)),
    "empty html": lambda a: a.send_widget("42", ""),
    "NUL in html": lambda a: a.send_widget("42", "<b>\x00</b>"),
    "chat_data over 1024 bytes": lambda a: a.send_widget("42", "<b>hi</b>", chat_data="é" * 513),
    "NUL in public_data": lambda a: a.send_widget("42", "<b>hi</b>", public_data="a\x00b"),
    "negative width": lambda a: a.send_widget("42", "<b>hi</b>", width=-1),
    "non-integer height": lambda a: a.send_widget("42", "<b>hi</b>", height="tall"),
    "shared chat_data over 1024 bytes": lambda a: a.send_shared_widget("42", "w-1", chat_data="x" * 1025),
    "empty widget id": lambda a: a.send_shared_widget("42", ""),
    "edit without changes": lambda a: a.edit_widget_data("42", "7"),
    "edit with a bad message id": lambda a: a.edit_widget_data("42", "seven", chat_data="x"),
    "NUL in edited public_data": lambda a: a.edit_widget_data("42", "7", public_data="\x00"),
    "shared public_data over 1024 bytes": lambda a: a.edit_shared_widget_data("w-1", "x" * 1025),
    "shared edit without widget id": lambda a: a.edit_shared_widget_data("", "x"),
    "cache_time under 5": lambda a: a.edit_shared_widget_data("w-1", "x", cache_time=4),
    "cache_time over 3600": lambda a: a.edit_shared_widget_data("w-1", "x", cache_time=3601),
    "boolean cache_time": lambda a: a.edit_shared_widget_data("w-1", "x", cache_time=True),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(_INVALID))
async def test_invalid_input_fails_locally_without_a_request(case):
    adapter = _adapter()

    result = await _INVALID[case](adapter)

    assert result.success is False and result.error and result.retryable is False
    adapter._bot.do_api_request.assert_not_awaited()


_AT_THE_LIMITS = {
    "html of exactly 32768 bytes": lambda a: a.send_widget("42", "x" * 32768),
    "data of exactly 1024 bytes": lambda a: a.send_widget("42", "<b>hi</b>", chat_data="é" * 512, public_data="x" * 1024),
    "empty data clears": lambda a: a.edit_widget_data("42", "7", chat_data="", public_data=""),
    "cache_time 5": lambda a: a.edit_shared_widget_data("w-1", "x", cache_time=5),
    "cache_time 3600": lambda a: a.edit_shared_widget_data("w-1", "", cache_time=3600),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(_AT_THE_LIMITS))
async def test_input_at_the_limits_is_sent(case):
    adapter = _adapter()
    adapter._bot.do_api_request.return_value = True if "edit" in case or "cache" in case or "clears" in case else dict(_MESSAGE)

    result = await _AT_THE_LIMITS[case](adapter)

    assert result.success is True
    assert adapter._bot.do_api_request.await_count == 1


@pytest.mark.asyncio
async def test_send_widget_payload_and_result():
    adapter = _adapter()
    adapter._bot.do_api_request.return_value = {**_MESSAGE, "widget": {"id": "w-1", "html": "<b>hi</b>"}}

    result = await adapter.send_widget(
        "-1001234", "<b>hi</b>", metadata={"thread_id": "17"}, caption="Price", chat_data="{}",
        public_data='{"p":1}', allow_queries=True, is_shared=True, width=320, height=200)

    assert _api_calls(adapter) == [("sendWidget", {
        "chat_id": -1001234, "html": "<b>hi</b>", "caption": "Price", "chat_data": "{}", "public_data": '{"p":1}',
        "allow_queries": True, "is_shared": True, "width": 320, "height": 200, "message_thread_id": 17,
        "disable_notification": True})]
    assert result.success and result.message_id == "55" and result.raw_response == {"widget_id": "w-1"}


@pytest.mark.asyncio
async def test_send_widget_omits_everything_not_asked_for():
    adapter = _adapter()

    result = await adapter.send_widget("42", "<b>hi</b>")

    assert _api_calls(adapter) == [("sendWidget", {"chat_id": 42, "html": "<b>hi</b>", "disable_notification": True})]
    assert result.success and result.message_id == "55" and result.raw_response is None


@pytest.mark.asyncio
async def test_send_widget_replies_through_reply_parameters():
    adapter = _adapter()

    await adapter.send_widget("42", "<b>hi</b>", reply_to="5", metadata={"notify": True})

    assert _api_calls(adapter) == [("sendWidget", {"chat_id": 42, "html": "<b>hi</b>", "reply_parameters": {"message_id": 5}})]


@pytest.mark.asyncio
async def test_send_widget_routes_into_a_private_chat_topic():
    adapter = _adapter()
    metadata = {"thread_id": "9", "telegram_dm_topic_reply_fallback": True, "telegram_reply_to_message_id": "77"}

    result = await adapter.send_widget("42", "<b>hi</b>", metadata=metadata)

    assert _api_calls(adapter) == [("sendWidget", {
        "chat_id": 42, "html": "<b>hi</b>", "message_thread_id": 9, "reply_parameters": {"message_id": 77},
        "disable_notification": True})]
    assert result.success


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [
    {"thread_id": "9", "telegram_dm_topic_reply_fallback": True},  # Hermes DM topic without a reply anchor
    {"direct_messages_topic_id": "555"},  # a channel direct-messages topic: sendWidget has no such parameter
], ids=["dm-topic-without-anchor", "direct-messages-topic"])
async def test_widget_routes_it_cannot_honour_fail_without_a_request(metadata):
    adapter = _adapter()

    result = await adapter.send_widget("42", "<b>hi</b>", metadata=metadata)

    assert result.success is False and result.error
    adapter._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_shared_widget_payload_and_result():
    adapter = _adapter()

    result = await adapter.send_shared_widget("42", "w-1", caption="Again", chat_data="c", metadata={"thread_id": "17"})

    assert _api_calls(adapter) == [("sendSharedWidget", {
        "chat_id": 42, "widget_id": "w-1", "caption": "Again", "chat_data": "c", "message_thread_id": 17,
        "disable_notification": True})]
    assert result.success and result.message_id == "55" and result.raw_response == {"widget_id": "w-1"}


@pytest.mark.asyncio
async def test_edit_widget_data_tells_omit_from_clear():
    adapter = _adapter()
    adapter._bot.do_api_request.return_value = True

    results = [
        await adapter.edit_widget_data("42", "7", chat_data=""),
        await adapter.edit_widget_data("42", "7", public_data='{"v":2}'),
        await adapter.edit_widget_data("-1001234", 8, chat_data="c", public_data=""),
    ]

    assert _api_calls(adapter) == [
        ("editMessageWidgetData", {"chat_id": 42, "message_id": 7, "chat_data": ""}),
        ("editMessageWidgetData", {"chat_id": 42, "message_id": 7, "public_data": '{"v":2}'}),
        ("editMessageWidgetData", {"chat_id": -1001234, "message_id": 8, "chat_data": "c", "public_data": ""})]
    assert [(r.success, r.message_id) for r in results] == [(True, "7"), (True, "7"), (True, "8")]


@pytest.mark.asyncio
async def test_edit_shared_widget_data_payload():
    adapter = _adapter()
    adapter._bot.do_api_request.return_value = True

    first = await adapter.edit_shared_widget_data("w-1", '{"p":5.47}', cache_time=60)
    second = await adapter.edit_shared_widget_data("w-1", "")

    assert _api_calls(adapter) == [
        ("editSharedWidgetData", {"widget_id": "w-1", "public_data": '{"p":5.47}', "cache_time": 60}),
        ("editSharedWidgetData", {"widget_id": "w-1", "public_data": ""})]
    assert first.success and second.success


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", ["edit_widget_data", "edit_shared_widget_data"])
async def test_an_edit_that_changes_nothing_counts_as_done(helper):
    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=BadRequest(
        "Message is not modified: specified new message content and reply markup are exactly the same"))

    result = await _HELPERS[helper](adapter)

    assert result.success is True
    assert adapter._bot.do_api_request.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", sorted(_HELPERS))
async def test_a_timed_out_request_is_never_sent_again(helper):
    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=TimedOut("Timed out"))

    result = await _HELPERS[helper](adapter)

    assert adapter._bot.do_api_request.await_count == 1
    assert result.success is False and result.retryable is False
    assert result.raw_response == {"delivery_uncertain": True}


@pytest.mark.asyncio
async def test_flood_control_reports_its_wait():
    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=RetryAfter(7))

    result = await adapter.send_widget("42", "<b>hi</b>")

    assert adapter._bot.do_api_request.await_count == 1
    assert result.success is False and result.retryable is True and result.retry_after == 7


@pytest.mark.asyncio
async def test_a_rejected_widget_is_a_definite_failure():
    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=BadRequest("INPUT_MEDIA_WIDGET_HTML_INVALID"))

    result = await adapter.send_widget("42", "<b>hi</b>")

    assert adapter._bot.do_api_request.await_count == 1
    assert result.success is False and result.retryable is False
    assert "INPUT_MEDIA_WIDGET_HTML_INVALID" in result.error
    assert result.raw_response is None


@pytest.mark.asyncio
async def test_error_text_never_carries_the_bot_token():
    token = "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw0"
    adapter = _adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=NetworkError(
        f"POST https://api.telegram.org/bot{token}/sendWidget: connection reset"))

    result = await adapter.send_widget("42", "<b>hi</b>")

    assert result.success is False and token not in result.error


@pytest.mark.asyncio
async def test_disconnected_adapter_fails_without_raising():
    adapter = _adapter()
    adapter._bot = None

    result = await adapter.send_widget("42", "<b>hi</b>")

    assert result.success is False and result.retryable is True
