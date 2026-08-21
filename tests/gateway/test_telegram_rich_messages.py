"""Tests for Bot API 10.1 Rich Messages (sendRichMessage) on Telegram.

Final / new-message replies opportunistically use ``sendRichMessage`` with the
RAW agent markdown so tables, task lists, etc. render natively. The legacy
MarkdownV2 ``send_message`` path stays as the fallback for unsupported /
oversized content and for transports that lack the endpoint.

The ``telegram`` package is mocked by ``tests/gateway/conftest.py``
(:func:`_ensure_telegram_mock`), so these tests construct a real
``TelegramAdapter`` and wire a mock bot.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from telegram.error import BadRequest, NetworkError, TimedOut


# Content exercising rich-only constructs: a heading, a real Markdown table,
# and a task list. Pipes / brackets must survive untouched into the payload.
RICH_CONTENT = "## Results\n\n| Case | Status |\n|---|---|\n| rich | ✅ |\n\n- [x] table renders"
CJK_RICH_CONTENT = "## 持仓\n\n| 项目 | 状态 |\n|---|---|\n| 早盘 | 正常 |"
ASTRAL_CJK_RICH_CONTENT = "## Rare Han\n\n| glyph | status |\n|---|---|\n| \U00030000 | ok |"
TABLE_ONLY_CONTENT = (
    "| Team | W | L | GB |\n"
    "|---|---|---|---|\n"
    "| Red Sox | 36 | 34 | 6.0 |\n"
    "| Dodgers | 40 | 30 | 2.0 |"
)
DANGEROUS_DETAILS_MATH = (
    "<details><summary>Complex proof</summary>\n\n"
    "$$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$\n\n"
    "And inline \\(\\alpha + \\beta\\)\n"
    "</details>"
)

# PTB 22.6's real unknown-endpoint errors: do_api_request can raise
# EndPointNotFound for Bot API 404s, and the request layer can wrap that same
# missing endpoint as InvalidToken. Use class names here so the tests don't
# depend on optional PTB internals.
EndPointNotFound = type("EndPointNotFound", (Exception,), {})
InvalidToken = type("InvalidToken", (Exception,), {})
PTB_ENDPOINT_NOT_FOUND = EndPointNotFound(
    "Endpoint 'sendRichMessage' not found in Bot API"
)
PTB_INVALID_TOKEN_404 = InvalidToken(
    "Either the bot token was rejected by Telegram or the endpoint "
    "'sendRichMessage' does not exist."
)


def _make_adapter(extra=None):
    """Build a TelegramAdapter with a mock bot wired for the rich path."""
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={"rich_messages": True, **(extra or {})},
    )
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    # do_api_request as an AsyncMock makes inspect.iscoroutinefunction(...) True,
    # so _bot_supports_rich() is satisfied (real Bot.do_api_request is async too).
    bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=123))
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_chat_action = AsyncMock()  # keeps the post-send typing re-trigger quiet
    bot.send_message_draft = AsyncMock(return_value=True)  # legacy draft fallback
    bot.edit_message_text = AsyncMock(return_value=MagicMock(message_id=1))  # legacy edit path
    bot.delete_message = AsyncMock(return_value=True)
    adapter._bot = bot
    return adapter


def _rich_api_kwargs(adapter):
    """Return the api_kwargs dict from the single sendRichMessage call."""
    call = adapter._bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessage"
    return call.kwargs["api_kwargs"]


@pytest.mark.asyncio
async def test_details_without_math_still_uses_rich_send():
    adapter = _make_adapter()

    result = await adapter.send(
        "12345",
        "<details><summary>Notes</summary>\nNo equations here.\n</details>",
    )

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_awaited_once()
    bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_math_outside_details_still_uses_rich_send():
    adapter = _make_adapter()

    result = await adapter.send("12345", "Outside details: $$x^2 + y^2$$")

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_awaited_once()
    bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_astral_cjk_rich_content_skips_rich_send_to_avoid_tdesktop_garble():
    adapter = _make_adapter()

    result = await adapter.send("12345", ASTRAL_CJK_RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_plain_markdown_stays_on_legacy_path():
    """Ordinary replies (no table/task-list/details/math) stay on the legacy
    MarkdownV2 path for consistent client rendering, even with rich enabled."""
    adapter = _make_adapter()

    result = await adapter.send("12345", "Hello **there**\n\nA normal reply.")

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_not_called()
    bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_expect_edits_metadata_keeps_preview_on_legacy_path():
    adapter = _make_adapter()

    result = await adapter.send(
        "12345",
        RICH_CONTENT,
        metadata={"expect_edits": True},
    )

    assert result.success is True
    # Streaming preview sends will be edited later, so they must not be born as
    # rich messages until Hermes wires rich_message edits directly.
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_not_called()
    bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_oversized_content_skips_rich_and_chunks():
    adapter = _make_adapter()
    # > 32,768 characters -> rich pre-check fails, legacy chunking takes over.
    oversized = "a" * 40000
    assert len(oversized) > TelegramAdapter.RICH_MESSAGE_MAX_CHARS

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
@pytest.mark.parametrize("exc", [PTB_ENDPOINT_NOT_FOUND, PTB_INVALID_TOKEN_404])
async def test_real_ptb_endpoint_missing_falls_back_and_latches_off(exc):
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(side_effect=exc)

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_awaited_once()
    bot.send_message.assert_awaited()
    assert adapter._rich_send_disabled is True


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
async def test_rich_transport_error_redacts_bot_token_even_when_redaction_disabled(monkeypatch):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(
        side_effect=NetworkError(
            f"Timed out requesting https://api.telegram.org/bot{token}/sendRichMessage"
        )
    )

    result = await adapter.send("12345", RICH_CONTENT)

    assert result.success is False
    assert result.error is not None
    assert token not in result.error
    assert "bot123456789:***/sendRichMessage" in result.error
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_send_error_redacts_bot_token_without_traceback(monkeypatch, caplog):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
    adapter = _make_adapter({"rich_messages": False})
    adapter._bot.send_message = AsyncMock(
        side_effect=BadRequest(
            f"Bad Request: https://api.telegram.org/bot{token}/sendMessage"
        )
    )

    with caplog.at_level(logging.ERROR):
        result = await adapter.send("12345", "Plain legacy content.")

    assert result.success is False
    assert result.error is not None
    assert token not in result.error
    assert "bot123456789:***/sendMessage" in result.error
    assert token not in caplog.text
    assert "bot123456789:***/sendMessage" in caplog.text
    adapter._bot.do_api_request.assert_not_called()


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
async def test_notification_silent_by_default():
    adapter = _make_adapter()

    await adapter.send("-100123", RICH_CONTENT)

    api_kwargs = _rich_api_kwargs(adapter)
    assert api_kwargs["disable_notification"] is True


@pytest.mark.asyncio
async def test_table_only_uses_legacy_with_default_config():
    """Default config (rich_messages unset → False) keeps tables on legacy path."""
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = TelegramAdapter(config)
    bot = MagicMock()
    bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=123))
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot

    result = await adapter.send("12345", TABLE_ONLY_CONTENT)

    assert result.success is True
    bot.do_api_request.assert_not_called()
    bot.send_message.assert_awaited()


# ── Streaming drafts: sendRichMessageDraft ─────────────────────────────


@pytest.mark.asyncio
async def test_cjk_rich_content_skips_rich_draft_to_avoid_tdesktop_garble():
    adapter = _make_adapter(extra={"rich_drafts": True})
    adapter._bot.do_api_request = AsyncMock(return_value=True)

    result = await adapter.send_draft("12345", draft_id=7, content=CJK_RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message_draft.assert_awaited_once()


# ----------------------------------------------------------------------
# prefers_fresh_final_streaming: root DMs stay on the no-duplicate edit/draft
# path (#47048). DM topics that degrade off drafts still need a fresh
# sendRichMessage so tables are not flattened by format_message.
# ----------------------------------------------------------------------
def test_prefers_fresh_final_streaming_stays_disabled_when_rich_enabled():
    adapter = _make_adapter()
    assert adapter.prefers_fresh_final_streaming(RICH_CONTENT) is False
    assert adapter.prefers_fresh_final_streaming(RICH_CONTENT, None) is False


def test_prefers_fresh_final_streaming_for_dm_topic_tables():
    adapter = _make_adapter()
    topic_meta = {
        "thread_id": "20189",
        "telegram_dm_topic_reply_fallback": True,
        "direct_messages_topic_id": "20189",
        "telegram_reply_to_message_id": "42",
    }
    assert adapter.prefers_fresh_final_streaming(RICH_CONTENT, topic_meta) is True
    assert adapter.prefers_fresh_final_streaming("Just a sentence.", topic_meta) is False
    assert adapter.prefers_fresh_final_streaming(
        RICH_CONTENT, {"direct_messages_topic_id": "20189"}
    ) is True
    # The documented telegram_-prefixed alias is honored through the same
    # canonical accessor the send path uses (gateway/delivery.py treats the
    # two keys as equivalent) — an alias-only lane must not flatten tables.
    assert adapter.prefers_fresh_final_streaming(
        RICH_CONTENT, {"telegram_direct_messages_topic_id": "20189"}
    ) is True


@pytest.mark.asyncio
async def test_legacy_draft_stream_finalizes_with_persistent_rich_message():
    """A plain draft must not force the persistent final to MarkdownV2."""
    adapter = _make_adapter()  # rich messages on, rich drafts off
    assert adapter.supports_draft_streaming(chat_type="dm") is True

    consumer = GatewayStreamConsumer(
        adapter,
        "12345",
        StreamConsumerConfig(transport="auto", chat_type="dm", cursor=""),
    )
    consumer._use_draft_streaming = True

    delivered = await consumer._send_or_edit(RICH_CONTENT, finalize=True)

    assert delivered is True
    bot = adapter._bot
    assert bot is not None
    bot.do_api_request.assert_awaited_once()
    assert bot.do_api_request.call_args.args[0] == "sendRichMessage"
    bot.send_message.assert_not_called()


# ----------------------------------------------------------------------
# supports_draft_streaming: rich_drafts controls draft rendering, not whether
# Telegram's ephemeral DM draft transport is available.  Keeping that transport
# lets the persistent final use sendRichMessage instead of relying on an
# edit-in-place conversion from a plain message.
# ----------------------------------------------------------------------


def test_supports_plain_draft_streaming_when_rich_without_rich_drafts():
    adapter = _make_adapter()  # rich_messages True, rich_drafts default False
    assert adapter.supports_draft_streaming(chat_type="dm") is True
    assert adapter.supports_draft_streaming(chat_type="private") is True


@pytest.mark.asyncio
async def test_rich_table_uses_raw_plain_draft_before_persistent_rich_final():
    adapter = _make_adapter()  # rich messages on, rich drafts off

    result = await adapter.send_draft("12345", draft_id=7, content=RICH_CONTENT)

    assert result.success is True
    adapter._bot.do_api_request.assert_not_called()
    adapter._bot.send_message_draft.assert_awaited_once_with(
        chat_id=12345,
        draft_id=7,
        text=RICH_CONTENT,
    )


@pytest.mark.asyncio
async def test_dm_table_stream_persists_through_send_rich_message():
    """Exercise the reporter's transport: ephemeral DM draft, then rich final."""
    adapter = _make_adapter()  # rich messages on, rich drafts off
    consumer = GatewayStreamConsumer(
        adapter,
        "12345",
        StreamConsumerConfig(
            transport="auto",
            chat_type="dm",
            edit_interval=0.01,
            buffer_threshold=1,
            cursor="",
        ),
    )

    task = asyncio.create_task(consumer.run())
    consumer.on_delta(RICH_CONTENT)
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    adapter._bot.send_message_draft.assert_awaited()
    draft_kwargs = adapter._bot.send_message_draft.call_args.kwargs
    assert draft_kwargs["text"] == RICH_CONTENT
    assert "parse_mode" not in draft_kwargs
    rich_endpoints = [call.args[0] for call in adapter._bot.do_api_request.await_args_list]
    assert rich_endpoints == ["sendRichMessage"]
    adapter._bot.edit_message_text.assert_not_called()
    adapter._bot.send_message.assert_not_called()


TOPIC_METADATA = {
    "thread_id": "20189",
    "telegram_dm_topic_reply_fallback": True,
    "direct_messages_topic_id": "20189",
    "telegram_reply_to_message_id": "42",
}

# Shape from the Telegram iOS DM-topic report: blank line, then a GFM table.
TOPIC_TABLE = (
    "Here's a table:\n"
    "\n"
    "| Sport | Followed? | Notes |\n"
    "|---|---|---|\n"
    "| F1 | ✅ | |\n"
    "| MLB | ✅ | |\n"
    "| LoL | ✅ | |\n"
)


@pytest.mark.asyncio
async def test_send_draft_routes_dm_topic_thread_id_as_int():
    """Drafts must use the same integer thread routing as send(), not the
    raw string thread_id. Telegram rejects the string on private topics."""
    adapter = _make_adapter()

    result = await adapter.send_draft(
        "12345", draft_id=7, content=TOPIC_TABLE, metadata=TOPIC_METADATA,
    )

    assert result.success is True
    kwargs = adapter._bot.send_message_draft.call_args.kwargs
    assert kwargs["message_thread_id"] == 20189
    assert kwargs["text"] == TOPIC_TABLE
    assert "parse_mode" not in kwargs


@pytest.mark.asyncio
async def test_dm_topic_table_stream_uses_send_rich_message():
    """Happy-path topic stream: drafts land, persistent final is rich."""
    adapter = _make_adapter()
    consumer = GatewayStreamConsumer(
        adapter,
        "12345",
        StreamConsumerConfig(
            transport="auto",
            chat_type="dm",
            edit_interval=0.01,
            buffer_threshold=1,
            cursor="",
        ),
        metadata=dict(TOPIC_METADATA),
        initial_reply_to_id="42",
    )

    task = asyncio.create_task(consumer.run())
    consumer.on_delta(TOPIC_TABLE)
    await asyncio.sleep(0.05)
    consumer.finish()
    await task

    adapter._bot.send_message_draft.assert_awaited()
    draft_kwargs = adapter._bot.send_message_draft.call_args.kwargs
    assert draft_kwargs["text"] == TOPIC_TABLE
    assert draft_kwargs["message_thread_id"] == 20189
    rich_endpoints = [call.args[0] for call in adapter._bot.do_api_request.await_args_list]
    # Invariant, not a frozen call list: the persistent final goes through
    # sendRichMessage, and no rich DRAFT frames fire (rich_drafts is off).
    assert "sendRichMessage" in rich_endpoints
    assert "sendRichMessageDraft" not in rich_endpoints
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_dm_topic_table_survives_when_drafts_degrade_to_edit():
    """Reporter path: sendMessageDraft fails in a private topic, Telegram
    then rejects a rich edit of the plain MarkdownV2 preview. The final
    must still persist through sendRichMessage — not convert_table_to_bullets.
    """
    adapter = _make_adapter()
    adapter._bot.send_message_draft = AsyncMock(
        side_effect=BadRequest("Bad Request: message thread not found")
    )

    async def _api(endpoint, api_kwargs=None, **kwargs):
        if endpoint == "editMessageText" and api_kwargs and "rich_message" in api_kwargs:
            raise BadRequest("can't parse rich message")
        if endpoint == "sendRichMessage":
            return SimpleNamespace(message_id=123)
        return SimpleNamespace(message_id=1)

    adapter._bot.do_api_request = AsyncMock(side_effect=_api)

    consumer = GatewayStreamConsumer(
        adapter,
        "12345",
        StreamConsumerConfig(
            transport="auto",
            chat_type="dm",
            edit_interval=0.01,
            buffer_threshold=1,
            cursor="",
        ),
        metadata=dict(TOPIC_METADATA),
        initial_reply_to_id="42",
    )

    task = asyncio.create_task(consumer.run())
    consumer.on_delta(TOPIC_TABLE)
    await asyncio.sleep(0.08)
    consumer.finish()
    await task

    rich_endpoints = [call.args[0] for call in adapter._bot.do_api_request.await_args_list]
    assert "sendRichMessage" in rich_endpoints
    rich_kwargs = None
    for call in adapter._bot.do_api_request.await_args_list:
        if call.args[0] == "sendRichMessage":
            rich_kwargs = call.kwargs["api_kwargs"]
            break
    assert rich_kwargs is not None
    assert "| F1 |" in rich_kwargs["rich_message"]["markdown"]
    # Degraded preview is deleted so the user is not left with the bullet rewrite.
    adapter._bot.delete_message.assert_awaited()


def test_supports_draft_streaming_enabled_when_rich_drafts_opt_in():
    adapter = _make_adapter(extra={"rich_drafts": True})
    assert adapter.supports_draft_streaming(chat_type="dm") is True
    assert adapter.supports_draft_streaming(chat_type="group") is False


def test_supports_draft_streaming_legacy_when_rich_messages_off():
    adapter = _make_adapter(extra={"rich_messages": False})
    assert adapter.supports_draft_streaming(chat_type="dm") is True


# ----------------------------------------------------------------------
# streaming_overflow_limit: with rich on, the stream consumer may accumulate up
# to the 32,768-char rich cap before splitting, so a reply that fits one
# sendRichMessage / sendRichMessageDraft isn't fragmented at the 4,096 limit.
# ----------------------------------------------------------------------


def test_streaming_overflow_limit_none_when_rich_latched_off():
    adapter = _make_adapter()
    adapter._rich_send_disabled = True
    assert adapter.streaming_overflow_limit() is None


# ----------------------------------------------------------------------------
# Rich finalize via editMessageText (Bot API 10.1 rich_message edit param).
# Streamed previews finalize by editing the existing message IN PLACE as rich,
# so tables/task lists survive without a fresh send + delete (no duplicate).
# ----------------------------------------------------------------------------


def _rich_edit_kwargs(adapter):
    """Return the api_kwargs dict from the single editMessageText rich call."""
    call = adapter._bot.do_api_request.call_args
    assert call.args[0] == "editMessageText"
    return call.kwargs["api_kwargs"]


@pytest.mark.asyncio
async def test_finalize_edit_uses_rich_for_table_content():
    """Finalizing a streamed preview whose content is a table edits the
    existing message IN PLACE via editMessageText's rich_message param —
    no fresh send, no delete, no duplicate."""
    adapter = _make_adapter()

    result = await adapter.edit_message(
        "12345", "555", RICH_CONTENT, finalize=True,
    )

    assert result.success is True
    assert result.message_id == "555"  # same message, edited in place
    api_kwargs = _rich_edit_kwargs(adapter)
    assert api_kwargs["message_id"] == 555
    # RAW markdown is passed through so table pipes survive.
    assert api_kwargs["rich_message"]["markdown"] == RICH_CONTENT
    # No fresh send / delete — the whole point of the in-place rich edit.
    adapter._bot.edit_message_text.assert_not_called()
    adapter._bot.delete_message.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_edit_error_logs_redacted_bot_token_without_traceback(monkeypatch, caplog):
    import agent.redact as redact

    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
    adapter = _make_adapter()
    adapter._bot.edit_message_text = AsyncMock(
        side_effect=BadRequest(
            f"Bad Request: https://api.telegram.org/bot{token}/editMessageText"
        )
    )

    with caplog.at_level(logging.WARNING):
        result = await adapter.edit_message(
            "12345", "555", "Just a normal answer.", finalize=True,
        )

    assert result.success is False
    assert result.error is not None
    assert token not in result.error
    assert "bot123456789:***/editMessageText" in result.error
    assert token not in caplog.text
    assert "bot123456789:***/editMessageText" in caplog.text


# --------------------------------------------------------------------------
# Rich-reply recovery (#47375): Telegram does not echo a sendRichMessage's
# content in reply_to_message (.text/.caption empty, .api_kwargs None), so we
# record message_id -> text at send time and recover it on inbound reply.
# --------------------------------------------------------------------------


def _reply_message(reply_to_id, *, reply_text=None, reply_caption=None, quote_text=None):
    """Build a mock inbound reply Message for _build_message_event."""
    replied = SimpleNamespace(
        message_id=int(reply_to_id),
        text=reply_text,
        caption=reply_caption,
    )
    quote = SimpleNamespace(text=quote_text) if quote_text is not None else None
    return SimpleNamespace(
        message_id=999,
        chat=SimpleNamespace(id=12345, type="private", title=None, full_name="U"),
        from_user=SimpleNamespace(
            id=42, username="u", first_name="U", last_name=None,
            full_name="U", is_bot=False,
        ),
        text="what did this mean?",
        caption=None,
        reply_to_message=replied,
        quote=quote,
        message_thread_id=None,
        is_topic_message=False,
        entities=[],
        date=None,
    )


def _reply_message_with_rich_blocks(
    reply_to_id,
    *,
    blocks,
    quote_text=None,
    api_kwargs_factory=dict,
):
    """Build a reply whose echoed content lives only in api_kwargs.rich_message."""
    replied = SimpleNamespace(
        message_id=int(reply_to_id),
        text=None,
        caption=None,
        api_kwargs=api_kwargs_factory({"rich_message": {"blocks": blocks}}),
    )
    quote = SimpleNamespace(text=quote_text) if quote_text is not None else None
    return SimpleNamespace(
        message_id=999,
        chat=SimpleNamespace(id=12345, type="private", title=None, full_name="U"),
        from_user=SimpleNamespace(
            id=42, username="u", first_name="U", last_name=None,
            full_name="U", is_bot=False,
        ),
        text="what did this mean?",
        caption=None,
        reply_to_message=replied,
        quote=quote,
        message_thread_id=None,
        is_topic_message=False,
        entities=[],
        date=None,
    )


@pytest.mark.asyncio
async def test_rich_reply_records_and_recovers_text(monkeypatch, tmp_path):
    """A reply to a rich-sent message resolves the original text via the index."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.platforms.base import MessageType
    from gateway import rich_sent_store

    adapter = _make_adapter()

    # _try_send_rich records (chat_id, message_id) -> content on a successful
    # rich send. Drive that path directly so the test doesn't depend on send()
    # gating heuristics (length, content shape) choosing the rich path.
    adapter._bot.do_api_request = AsyncMock(
        return_value=SimpleNamespace(message_id=678)
    )
    send_result = await adapter._try_send_rich(
        "12345", "Your morning briefing: CI is green.", None, None,
    )
    assert send_result is not None and send_result.success is True
    assert send_result.message_id == "678"
    assert rich_sent_store.lookup("12345", "678") == "Your morning briefing: CI is green."

    # Inbound reply carries NO text/caption (the rich-message blind spot).
    event = adapter._build_message_event(
        _reply_message("678"), MessageType.TEXT,
    )
    assert event.reply_to_message_id == "678"
    assert event.reply_to_text == "Your morning briefing: CI is green."
