"""Tests for Bot API 10.1 Rich Messages (sendRichMessage) on Telegram.

Final / new-message replies opportunistically use ``sendRichMessage`` with the
RAW agent markdown so tables, task lists, etc. render natively. The legacy
MarkdownV2 ``send_message`` path stays as the fallback for unsupported /
oversized content and for transports that lack the endpoint.

The ``telegram`` package is mocked by ``tests/gateway/conftest.py``
(:func:`_ensure_telegram_mock`), so these tests construct a real
``TelegramAdapter`` and wire a mock bot.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.telegram import TelegramAdapter
from telegram.error import BadRequest, NetworkError, TimedOut


# Content exercising rich-only constructs: a heading, a real Markdown table,
# and a task list. Pipes / brackets must survive untouched into the payload.
RICH_CONTENT = "## Results\n\n| Case | Status |\n|---|---|\n| rich | ✅ |\n\n- [x] table renders"


def _make_adapter(extra=None):
    """Build a TelegramAdapter with a mock bot wired for the rich path.

    Rich messages are opt-in (default off) while the Bot API 10.1 endpoint
    is validated live, so tests that exercise the rich path enable it
    explicitly here; opt-out tests pass their own ``extra``.
    """
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={"rich_messages": True} if extra is None else extra,
    )
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    # do_api_request as an AsyncMock makes inspect.iscoroutinefunction(...) True,
    # so _bot_supports_rich() is satisfied (real Bot.do_api_request is async too).
    bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=123))
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_chat_action = AsyncMock()  # keeps the post-send typing re-trigger quiet
    bot.send_message_draft = AsyncMock(return_value=True)  # legacy draft fallback
    adapter._bot = bot
    return adapter


def _rich_api_kwargs(adapter):
    """Return the api_kwargs dict from the single sendRichMessage call."""
    call = adapter._bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessage"
    return call.kwargs["api_kwargs"]


@pytest.mark.asyncio
async def test_rich_happy_path_sends_raw_markdown():
    adapter = _make_adapter()

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    assert result.message_id == "123"
    adapter._bot.do_api_request.assert_awaited_once()
    api_kwargs = _rich_api_kwargs(adapter)
    # Raw markdown — NOT MarkdownV2-escaped. Table pipes still present.
    assert api_kwargs["rich_message"]["markdown"] == RICH_CONTENT
    assert "| Case | Status |" in api_kwargs["rich_message"]["markdown"]
    assert "- [x] table renders" in api_kwargs["rich_message"]["markdown"]
    # Legacy path must not run on rich success.
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_rich_opt_out_uses_legacy():
    adapter = _make_adapter(extra={"rich_messages": False})

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_rich_opt_out_accepts_string_false():
    adapter = _make_adapter(extra={"rich_messages": "false"})

    await adapter.send("12345", RICH_CONTENT)

    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_oversized_content_skips_rich_and_chunks():
    adapter = _make_adapter()
    # > 32,768 UTF-8 bytes -> rich pre-check fails, legacy chunking takes over.
    oversized = "a" * 40000
    assert len(oversized.encode("utf-8")) > TelegramAdapter.RICH_MESSAGE_MAX_BYTES

    result = await adapter.send("12345", oversized)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    # Oversized content is split into multiple legacy chunks.
    assert adapter._bot.send_message.await_count > 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        BadRequest("can't parse rich message"),
        BadRequest("Method not found"),
    ],
)
async def test_permanent_rich_error_falls_back_to_legacy(exc):
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=exc)

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_awaited_once()
    adapter._bot.send_message.assert_awaited()  # legacy fallback ran


@pytest.mark.asyncio
async def test_unknown_endpoint_error_falls_back_to_legacy():
    """A non-BadRequest 'Method not found' (old PTB/endpoint) degrades gracefully."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Method not found"))

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    adapter._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_capability_error_latches_rich_send_off():
    """Endpoint-missing errors latch rich off so later sends skip the
    doomed extra roundtrip entirely."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=RuntimeError("Method not found"))

    result = await adapter.send("12345", RICH_CONTENT)
    assert result.success is True
    assert adapter._rich_send_disabled is True

    # Second send skips rich entirely (no second do_api_request call).
    adapter._bot.do_api_request.reset_mock()
    adapter._bot.send_message.reset_mock()
    result2 = await adapter.send("12345", RICH_CONTENT)
    assert result2.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_per_message_bad_request_does_not_latch_off():
    """A parser/limit BadRequest is per-message — rich must stay enabled
    for subsequent messages."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=BadRequest("can't parse rich message"))

    result = await adapter.send("12345", RICH_CONTENT)
    assert result.success is True
    assert adapter._rich_send_disabled is False

    # Next message re-attempts rich.
    adapter._bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=124))
    result2 = await adapter.send("12345", RICH_CONTENT)
    assert result2.success is True
    adapter._bot.do_api_request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [TimedOut("timed out"), NetworkError("connection reset")])
async def test_transient_rich_error_does_not_legacy_resend(exc):
    """Transient transport errors must NOT trigger a legacy resend (duplicate risk)."""
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=exc)

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is False
    adapter._bot.do_api_request.assert_awaited_once()
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_transient_timeout_is_not_retryable():
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=TimedOut("timed out"))

    result = await adapter.send("12345", RICH_CONTENT)

    # A plain timeout may have reached Telegram -> non-retryable (no auto-resend).
    assert result.success is False
    assert result.retryable is False


@pytest.mark.asyncio
async def test_routing_thread_id_maps_to_message_thread_id():
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT, metadata={"thread_id": "5"})

    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["message_thread_id"] == 5
    assert "direct_messages_topic_id" not in api_kwargs


@pytest.mark.asyncio
async def test_routing_direct_messages_topic_id_drops_message_thread_id():
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT, metadata={"direct_messages_topic_id": "20189"})

    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["direct_messages_topic_id"] == 20189
    # _thread_kwargs_for_send pairs the topic id with message_thread_id=None;
    # the rich payload must drop the None key, not send a stray field.
    assert "message_thread_id" not in api_kwargs


@pytest.mark.asyncio
async def test_reply_to_propagates_as_reply_parameters():
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT, reply_to="999")

    api_kwargs = _rich_api_kwargs(adapter)
    # Spec: sendRichMessage documents reply_parameters (ReplyParameters), not
    # the legacy reply_to_message_id scalar — unknown params are silently
    # ignored, which would quietly drop the reply anchor.
    assert api_kwargs["reply_parameters"] == {"message_id": 999}
    assert "reply_to_message_id" not in api_kwargs


@pytest.mark.asyncio
async def test_notification_silent_by_default():
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT)

    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["disable_notification"] is True


@pytest.mark.asyncio
async def test_notification_opt_in_drops_disable_flag():
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT, metadata={"notify": True})

    api_kwargs = _rich_api_kwargs(adapter)
    assert "disable_notification" not in api_kwargs


@pytest.mark.asyncio
async def test_rich_gate_tolerates_missing_enabled_attr():
    """Adapters missing _rich_messages_enabled (object.__new__ in some tests)
    must not raise — the gate reads it via getattr(default=True), and a bot
    without an async do_api_request falls through to the legacy path."""
    adapter = _make_adapter()
    del adapter._rich_messages_enabled  # simulate object.__new__ construction
    # SimpleNamespace bot has no do_api_request -> _bot_supports_rich() False.
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)),
        send_chat_action=AsyncMock(),
    )

    result = await adapter.send("12345", "hello world")

    assert result.success is True
    assert result.message_id == "42"


# ── Streaming drafts: sendRichMessageDraft ─────────────────────────────


@pytest.mark.asyncio
async def test_rich_draft_happy_path_sends_raw_markdown():
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(return_value=True)

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_awaited_once()
    call = adapter._bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessageDraft"
    api_kwargs = call.kwargs["api_kwargs"]
    assert api_kwargs["draft_id"] == 7
    assert api_kwargs["rich_message"]["markdown"] == RICH_CONTENT
    # Legacy plain-text draft must not run when rich draft succeeds.
    adapter._bot.send_message_draft.assert_not_called()


@pytest.mark.asyncio
async def test_rich_draft_capability_failure_falls_back_and_latches_off():
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=BadRequest("Method not found"))

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True  # legacy plain-text draft delivered the frame
    adapter._bot.send_message_draft.assert_awaited_once()
    assert adapter._rich_draft_disabled is True

    # A subsequent frame skips the rich attempt entirely (latched off).
    adapter._bot.do_api_request.reset_mock()
    adapter._bot.send_message_draft.reset_mock()
    result2 = await adapter.send_draft("12345", draft_id=8, content=RICH_CONTENT)
    assert result2.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message_draft.assert_awaited_once()


@pytest.mark.asyncio
async def test_rich_draft_transient_failure_does_not_latch_off():
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=TimedOut("timed out"))

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True  # legacy draft carried this frame
    adapter._bot.send_message_draft.assert_awaited_once()
    # Transient errors must NOT permanently disable rich drafts.
    assert adapter._rich_draft_disabled is False


@pytest.mark.asyncio
async def test_rich_draft_opt_out_uses_legacy():
    adapter = _make_adapter(extra={"rich_messages": False})

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message_draft.assert_awaited_once()


@pytest.mark.asyncio
async def test_rich_draft_oversized_uses_legacy():
    adapter = _make_adapter()
    oversized = "a" * 40000

    result = await adapter.send_draft("12345", draft_id=7, content=oversized)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message_draft.assert_awaited_once()
