"""Tests for guest-mode two-phase reply delivery (Bot API 10.0).

Three delivery paths exercised:
1. Streaming  — stub succeeds: send() returns inline_message_id so the stream
   consumer takes ownership and drives progressive editMessageText calls via
   edit_message().
2. Buffer fallback — stub fails (_imi is None): send() buffers text; when
   on_processing_complete fires it delivers the full buffer via answerGuestQuery.
3. Clean completion — stream consumer delivered everything; buffer is empty:
   on_processing_complete skips the flush entirely.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Telegram stub factory (mirrors the pattern used across gateway test suite)
# ---------------------------------------------------------------------------

def _build_telegram_stubs() -> dict:
    telegram_mod = types.ModuleType("telegram")
    telegram_mod.Update = object
    telegram_mod.Bot = object
    telegram_mod.Message = object
    telegram_mod.InlineKeyboardButton = object
    telegram_mod.InlineKeyboardMarkup = object
    telegram_mod.LinkPreviewOptions = object

    ext_mod = types.ModuleType("telegram.ext")
    ext_mod.Application = object
    ext_mod.CommandHandler = object
    ext_mod.CallbackQueryHandler = object
    ext_mod.MessageHandler = object
    ext_mod.ContextTypes = SimpleNamespace(DEFAULT_TYPE=type(None))
    ext_mod.filters = SimpleNamespace()

    const_mod = types.ModuleType("telegram.constants")
    const_mod.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    const_mod.ChatType = SimpleNamespace(
        GROUP="group", SUPERGROUP="supergroup", CHANNEL="channel", PRIVATE="private",
    )

    req_mod = types.ModuleType("telegram.request")
    req_mod.HTTPXRequest = object

    telegram_mod.ext = ext_mod
    telegram_mod.constants = const_mod
    telegram_mod.request = req_mod

    return {
        "telegram": telegram_mod,
        "telegram.ext": ext_mod,
        "telegram.constants": const_mod,
        "telegram.request": req_mod,
    }


@pytest.fixture
def TelegramAdapter(monkeypatch):
    module_name = "plugins.platforms.telegram.adapter"
    existing = sys.modules.get(module_name)
    if existing is not None:
        yield existing.TelegramAdapter
        return

    pkg = sys.modules.get("telegram")
    installed = isinstance(getattr(pkg, "__file__", None), str)
    if pkg is None:
        try:
            installed = importlib.util.find_spec("telegram") is not None
        except ValueError:
            installed = False
    if not installed:
        for name, mod in _build_telegram_stubs().items():
            monkeypatch.setitem(sys.modules, name, mod)

    module = importlib.import_module(module_name)
    try:
        yield module.TelegramAdapter
    finally:
        if not installed:
            sys.modules.pop(module_name, None)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

CHAT_ID = "-100999888777"
GUEST_QUERY_ID = "gq_AAAAABBBBB"
INLINE_MESSAGE_ID = "AAMCAgADsample"


def _make_adapter(TelegramAdapter):
    a = object.__new__(TelegramAdapter)
    a.platform = Platform.TELEGRAM
    a._bot = AsyncMock()
    a._send_path_degraded = False
    a._pending_guest_queries = {CHAT_ID: GUEST_QUERY_ID}
    a._guest_only_chats = {CHAT_ID}
    a._guest_reply_buffer = {}
    a._guest_inline_message_ids = {}
    return a


def _make_event() -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=CHAT_ID,
            chat_type="supergroup",
            user_id="user-1",
        ),
    )


# ---------------------------------------------------------------------------
# 1. Streaming path: stub succeeded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_returns_imi_to_stream_consumer(TelegramAdapter):
    """send() hands inline_message_id back to the stream consumer as message_id."""
    a = _make_adapter(TelegramAdapter)
    a._guest_inline_message_ids[CHAT_ID] = INLINE_MESSAGE_ID

    result = await a.send(CHAT_ID, "Thinking...", metadata={"expect_edits": True})

    assert result.success is True
    assert result.message_id == INLINE_MESSAGE_ID


@pytest.mark.asyncio
async def test_edit_message_routes_to_editMessageText_via_imi(TelegramAdapter):
    """edit_message() calls editMessageText(inline_message_id=…) for the guest bubble."""
    a = _make_adapter(TelegramAdapter)
    a._guest_inline_message_ids[CHAT_ID] = INLINE_MESSAGE_ID
    a._bot.do_api_request = AsyncMock(return_value=None)

    result = await a.edit_message(CHAT_ID, INLINE_MESSAGE_ID, "Here is the answer.")

    a._bot.do_api_request.assert_awaited_once()
    method, = a._bot.do_api_request.await_args.args
    kwargs = a._bot.do_api_request.await_args.kwargs["api_kwargs"]
    assert method == "editMessageText"
    assert kwargs["inline_message_id"] == INLINE_MESSAGE_ID
    assert "Here is the answer." in kwargs["text"]
    assert result.success is True


@pytest.mark.asyncio
async def test_on_processing_complete_noop_when_stream_consumer_delivered(TelegramAdapter):
    """on_processing_complete does not call any API when imi is set and buffer is empty."""
    a = _make_adapter(TelegramAdapter)
    a._guest_inline_message_ids[CHAT_ID] = INLINE_MESSAGE_ID
    # _guest_reply_buffer is empty — stream consumer handled every chunk
    a._bot.do_api_request = AsyncMock(return_value=None)

    await a.on_processing_complete(_make_event(), ProcessingOutcome.SUCCESS)

    a._bot.do_api_request.assert_not_awaited()


# ---------------------------------------------------------------------------
# 2. Fallback path: stub failed (_imi is None)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_buffers_text_when_stub_failed(TelegramAdapter):
    """send() accumulates text in _guest_reply_buffer when no imi is available."""
    a = _make_adapter(TelegramAdapter)
    a._guest_inline_message_ids[CHAT_ID] = None

    result = await a.send(CHAT_ID, "Answer text", metadata={"expect_edits": True})

    assert result.success is True
    assert result.message_id is None
    assert "Answer text" in a._guest_reply_buffer[CHAT_ID]


@pytest.mark.asyncio
async def test_on_processing_complete_flushes_via_answerGuestQuery(TelegramAdapter):
    """on_processing_complete delivers buffered text via answerGuestQuery when stub failed."""
    a = _make_adapter(TelegramAdapter)
    a._guest_inline_message_ids[CHAT_ID] = None
    a._guest_reply_buffer[CHAT_ID] = "The actual answer from the LLM."
    a._bot.do_api_request = AsyncMock(return_value=None)

    await a.on_processing_complete(_make_event(), ProcessingOutcome.SUCCESS)

    a._bot.do_api_request.assert_awaited_once()
    method, = a._bot.do_api_request.await_args.args
    body = a._bot.do_api_request.await_args.kwargs["api_kwargs"]
    assert method == "answerGuestQuery"
    assert body["guest_query_id"] == GUEST_QUERY_ID
    assert "The actual answer from the LLM." in (
        body["result"]["input_message_content"]["message_text"]
    )


@pytest.mark.asyncio
async def test_on_processing_complete_clears_guest_state(TelegramAdapter):
    """on_processing_complete removes all guest state so the next query starts fresh."""
    a = _make_adapter(TelegramAdapter)
    a._guest_inline_message_ids[CHAT_ID] = None
    a._guest_reply_buffer[CHAT_ID] = "some reply"
    a._bot.do_api_request = AsyncMock(return_value=None)

    await a.on_processing_complete(_make_event(), ProcessingOutcome.SUCCESS)

    assert CHAT_ID not in a._pending_guest_queries
    assert CHAT_ID not in a._guest_inline_message_ids
    assert CHAT_ID not in a._guest_reply_buffer
    assert CHAT_ID not in a._guest_only_chats


# ---------------------------------------------------------------------------
# 4. Streaming-finalize robustness (text-mode hardening)
# ---------------------------------------------------------------------------

def _adapter_module(TelegramAdapter):
    return sys.modules[TelegramAdapter.__module__]


def test_strip_mdv2_strips_incomplete_bold(TelegramAdapter):
    """An unclosed ** from a stream cut mid-token must not survive as visible text."""
    _strip_mdv2 = _adapter_module(TelegramAdapter)._strip_mdv2
    # The exact reported truncation: "- **2" (bold opened, never closed).
    assert _strip_mdv2("- **2") == "- 2"
    # Balanced bold still collapses, and a trailing unclosed marker is removed.
    assert _strip_mdv2("**bold** then **trunc") == "bold then trunc"
    # Plain text is untouched.
    assert _strip_mdv2("just plain text") == "just plain text"


@pytest.mark.asyncio
async def test_send_normalizes_int_chat_id(TelegramAdapter):
    """send() routes an int chat_id to the (str-keyed) guest buffer — no silent miss."""
    a = _make_adapter(TelegramAdapter)
    a._guest_inline_message_ids[CHAT_ID] = None  # stub failed → buffer mode

    result = await a.send(int(CHAT_ID), "Answer text", metadata={"expect_edits": True})

    assert result.success is True
    assert result.message_id is None
    assert "Answer text" in a._guest_reply_buffer[CHAT_ID]


@pytest.mark.asyncio
async def test_send_strips_media_residual_from_buffer(TelegramAdapter):
    """A MEDIA: tag that slips past the stream consumer must not buffer as text."""
    a = _make_adapter(TelegramAdapter)
    a._guest_inline_message_ids[CHAT_ID] = None

    await a.send(CHAT_ID, "Here you go MEDIA:/workspace/out.png", metadata={"expect_edits": True})

    assert "MEDIA:" not in a._guest_reply_buffer[CHAT_ID]
    assert "Here you go" in a._guest_reply_buffer[CHAT_ID]


@pytest.mark.asyncio
async def test_edit_message_populates_buffer_for_finalize_fallback(TelegramAdapter):
    """Each streaming edit keeps the buffer current so a final re-render can recover."""
    a = _make_adapter(TelegramAdapter)
    a._guest_inline_message_ids[CHAT_ID] = INLINE_MESSAGE_ID
    a._bot.do_api_request = AsyncMock(return_value=None)

    await a.edit_message(CHAT_ID, INLINE_MESSAGE_ID, "partial answer")

    assert a._guest_reply_buffer[CHAT_ID] == "partial answer"


@pytest.mark.asyncio
async def test_on_processing_complete_re_renders_truncated_buffer(TelegramAdapter):
    """If the stream left a truncation artifact, the final edit re-renders it clean."""
    a = _make_adapter(TelegramAdapter)
    a._guest_inline_message_ids[CHAT_ID] = INLINE_MESSAGE_ID
    a._guest_reply_buffer[CHAT_ID] = "Result:\n- **2"  # unclosed bold from a stream cut
    a._bot.do_api_request = AsyncMock(return_value=None)

    await a.on_processing_complete(_make_event(), ProcessingOutcome.SUCCESS)

    a._bot.do_api_request.assert_awaited_once()
    method, = a._bot.do_api_request.await_args.args
    text = a._bot.do_api_request.await_args.kwargs["api_kwargs"]["text"]
    assert method == "editMessageText"
    assert "**" not in text          # artifact stripped
    assert "2" in text               # content preserved
