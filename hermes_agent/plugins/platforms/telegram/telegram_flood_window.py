"""Streaming drafts and message deletes honour the chat's Telegram flood window (#107612).

A 429 ``retry_after`` arms the chat's send flood window (``_record_send_flood_cooldown``); sends,
edits, typing and native media already stay out of it. Draft frames and ``deleteMessage`` fired
straight into it, and Telegram lengthens a penalty that keeps getting hit. Now a draft frame inside
the window is skipped (the next frame or the final send supersedes it, so nothing is lost), and a
delete waits the window out: inline when it is short, else once from a tracked background task so
the caller (the end-of-turn preview cleanup) is not parked for minutes. A 429 on either call arms
the window for every other path too.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from hermes_agent.gateway.platforms.base import SendResult
from hermes_agent.plugins.platforms.telegram.telegram_media_flood import _MAX_WINDOW_WAITS, flood_wait_seconds

logger = logging.getLogger("hermes_agent.plugins.platforms.telegram.adapter")


def note_flood(adapter: Any, chat_id: Any, error: Exception) -> bool:
    """Arm the chat's flood window when ``error`` is a ``retry_after`` refusal; True if it was."""
    wait = flood_wait_seconds(error)
    if wait is None:
        return False
    adapter._record_send_flood_cooldown(chat_id, wait)
    return True


def skipped_draft(adapter: Any, chat_id: Any) -> Optional[SendResult]:
    """A skipped-frame result while the chat is inside its flood window, else ``None`` (may fire)."""
    remaining = adapter._send_flood_cooldown_remaining(chat_id)
    if remaining is None:
        return None
    logger.debug("[%s] skipping draft frame for chat %s: flood window %.1fs left", adapter.name, chat_id, remaining)
    return SendResult(success=True, message_id=None, raw_response={"skipped": True})


async def _after_window(adapter: Any, chat_id: Any, delete: Callable[[], Awaitable[bool]], *, inline: bool) -> bool:
    """``inline`` (the caller awaits it): every wake re-checks the cap, so a window another request
    extended meanwhile is handed to the background task instead of being slept on the caller's turn."""
    from hermes_agent.plugins.platforms.telegram.adapter import _FLOOD_INLINE_WAIT_CAP_SECS
    for waits in range(_MAX_WINDOW_WAITS + 2):  # re-check after the last allowed wait
        remaining = adapter._send_flood_cooldown_remaining(chat_id)
        if remaining is None:
            return await delete()
        if waits > _MAX_WINDOW_WAITS:
            break
        if inline and remaining > _FLOOD_INLINE_WAIT_CAP_SECS:
            return _defer(adapter, chat_id, delete, remaining)
        await asyncio.sleep(remaining)
    logger.warning("[%s] message delete in chat %s dropped: flood window kept re-arming", adapter.name, chat_id)
    return False


async def delete_outside_flood_window(adapter: Any, chat_id: Any, delete: Callable[[], Awaitable[bool]]) -> bool:
    """Run ``delete`` once the chat's flood window is closed. A window longer than the inline cap is
    waited out by a tracked background task (shutdown cancels it) and the call returns True: the
    delete is owned by that task, so a caller's retry-on-False must not fire a second one into it."""
    return await _after_window(adapter, chat_id, delete, inline=True)


def _defer(adapter: Any, chat_id: Any, delete: Callable[[], Awaitable[bool]], remaining: float) -> bool:
    logger.info("[%s] deferring message delete in chat %s past its flood window (%.0fs)", adapter.name, chat_id, remaining)
    task = asyncio.get_running_loop().create_task(_after_window(adapter, chat_id, delete, inline=False))
    tracked = getattr(adapter, "_background_tasks", None)
    if isinstance(tracked, set):
        tracked.add(task)
        task.add_done_callback(tracked.discard)
    return True


async def delete_message(adapter: Any, chat_id: str, message_id: str) -> bool:
    """``TelegramAdapter.delete_message`` body: ``deleteMessage`` once the chat's flood window is closed."""
    from hermes_agent.plugins.platforms.telegram.adapter import _redact_telegram_error_text, normalize_telegram_chat_id
    if not adapter._bot:
        live = adapter._replacement_telegram_adapter()
        return await live.delete_message(chat_id, message_id) if live is not None else False

    async def _delete() -> bool:
        try:
            await adapter._bot.delete_message(chat_id=normalize_telegram_chat_id(chat_id), message_id=int(message_id))
            return True
        except Exception as e:  # health: allow BLE001 -- best-effort delete (moved from adapter.py): every failure is a logged False
            note_flood(adapter, chat_id, e)
            logger.debug("[%s] Failed to delete Telegram message %s: %s", adapter.name, message_id, _redact_telegram_error_text(e))
            return False
    return await delete_outside_flood_window(adapter, chat_id, _delete)


async def send_draft(adapter: Any, chat_id: str, draft_id: int, content: str, metadata: Optional[dict[str, Any]]) -> SendResult:
    """``TelegramAdapter.send_draft`` body: rich draft, else MarkdownV2 then plain ``sendMessageDraft``."""
    from hermes_agent.plugins.platforms.telegram.adapter import (
        _TEXT_SEND_DEADLINE, ParseMode, _await_with_thread_deadline, _redact_telegram_error_text, normalize_telegram_chat_id,
        utf16_len)
    if not adapter._bot:
        return SendResult(success=False, error="not_connected")
    # A frame inside the chat's flood window is skipped (#107612): the next frame or the final send supersedes it.
    if (skipped := skipped_draft(adapter, chat_id)) is not None:
        return skipped
    # Rich draft fast-path; any failure degrades to the plain draft below. Drafts have no message_id.
    if adapter._should_attempt_rich_draft(content) and await adapter._try_send_rich_draft(chat_id, draft_id, content, metadata):
        return SendResult(success=True, message_id=None)
    if (skipped := skipped_draft(adapter, chat_id)) is not None:  # the rich frame was refused with retry_after
        return skipped
    if not hasattr(adapter._bot, "send_message_draft"):
        return SendResult(success=False, error="api_unavailable")
    # Drafts share the regular-send UTF-16 length contract.
    text = content if len(
        content) <= adapter.MAX_MESSAGE_LENGTH else adapter.truncate_message(content, adapter.MAX_MESSAGE_LENGTH, len_fn=utf16_len)[0]
    # Same MarkdownV2 conversion as ``send`` (MarkdownV2 then plain) so the draft doesn't snap at the end. Exception: a Rich
    # final with rich drafts disabled previews raw — the legacy formatter would turn pipe tables into bullets.
    plain_rich_preview = bool(
        getattr(adapter, "_rich_messages_enabled", False) and not getattr(adapter, "_rich_drafts_enabled", False)
        and adapter._needs_rich_rendering(text))
    draft_thread_kwargs = adapter._thread_kwargs_for_draft(chat_id, metadata)
    for use_markdown in ((False,) if plain_rich_preview else (True, False)):
        kwargs: dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id), "draft_id": int(draft_id),
            "text": adapter.format_message(text) if use_markdown else text}
        if use_markdown:
            kwargs["parse_mode"] = ParseMode.MARKDOWN_V2
        kwargs.update(draft_thread_kwargs)
        try:
            if await _await_with_thread_deadline(
                adapter._bot.send_message_draft(**kwargs), timeout=_TEXT_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False):
                return SendResult(success=True, message_id=None)
            return SendResult(success=False, error="draft_rejected")
        except Exception as e:  # health: allow BLE001 -- draft boundary (moved from adapter.py): any failure falls back to edit streaming
            # MarkdownV2 parse failure → retry once as plain text; anything else returns to the caller,
            # which falls back to edit-based streaming for this response.
            if use_markdown and adapter._is_bad_request_error(e):
                logger.debug(
                    "[%s] sendMessageDraft MarkdownV2 rejected, retrying as plain text (chat=%s draft_id=%s): %s",
                    adapter.name, chat_id, draft_id, _redact_telegram_error_text(e))
                continue
            logger.debug("[%s] sendMessageDraft failed (chat=%s draft_id=%s): %s", adapter.name, chat_id, draft_id, e)
            if note_flood(adapter, chat_id, e):  # keep the draft transport; frames skip the window
                return skipped_draft(adapter, chat_id)
            return SendResult(success=False, error=_redact_telegram_error_text(e))
    return SendResult(success=False, error="draft_rejected")
