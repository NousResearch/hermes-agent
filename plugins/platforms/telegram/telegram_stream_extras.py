"""Opt-in Telegram stop buttons, completion effects, whispers and draft delivery."""
from __future__ import annotations

import logging
from typing import Any, Optional
from gateway.platforms.base import SendResult, utf16_len
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

logger = logging.getLogger("plugins.platforms.telegram.adapter")

# Bot API message effects (private chats only): emoji name → effect id. Ids are verified
# live against the Bot API — an unknown id is rejected server-side, so keep this list to
# the verified set. Paired with sendMessage/sendRichMessage's ``message_effect_id``.
_MESSAGE_EFFECT_IDS: dict[str, str] = {
    "👍": "5107584321108051014",
    "👎": "5104858069142078462",
    # "❤" fires only on paid-message sends (EFFECT_ID_INVALID on regular messages) — excluded.
    "🔥": "5104841245755180586",
    "🎉": "5046509860389126442",
    "🎊": "5046589136895476101",
}


class TelegramStreamExtrasMixin:
    async def _maybe_handle_generation_stopped(self, update) -> None:
        """Stop button (Bot API 10.3): the user pressed it on a live draft. Re-dispatch as a
        synthetic ``/stop`` through the normal inbound path so the running turn is cancelled
        with the standard semantics (session lock cleanup, interrupt reason, feedback).

        PTB 22.8 predates this update type; the payload survives in ``Update.api_kwargs``.
        The update carries only chat/thread/draft ids — in DMs the chat id doubles as the
        user id; group presses arrive without a user and are dropped by the auth chain
        (fail closed) until group buttons are enabled."""
        if not getattr(self, "_stop_button_enabled", False):
            return
        payload = (getattr(update, "api_kwargs", None) or {}).get("stopped_message_generation")
        if not isinstance(payload, dict):
            return
        chat = payload.get("chat") if isinstance(payload.get("chat"), dict) else {}
        chat_id = chat.get("id")
        if chat_id is None:
            return
        telegram_chat_type = str(chat.get("type") or "private")
        chat_type = "group" if telegram_chat_type in {"group", "supergroup"} else (
            "channel" if telegram_chat_type == "channel" else "dm")
        thread_id = payload.get("message_thread_id")
        try:
            source = self.build_source(
                chat_id=str(chat_id), chat_type=chat_type,
                user_id=(str(chat_id) if chat_type == "dm" else None),
                thread_id=(str(thread_id) if thread_id else None),
            )
            event = MessageEvent(text="/stop", message_type=MessageType.COMMAND, source=source)
            await self.handle_message(event)
            logger.info("[%s] Stop button pressed (chat=%s draft_id=%s) — dispatched /stop",
                        self.name, chat_id, payload.get("draft_id"))
        except Exception:
            logger.debug("[%s] Stop-button dispatch failed", self.name, exc_info=True)

    def _message_effect_id_for(self, chat_id: str, metadata: Optional[dict[str, Any]]) -> Optional[str]:
        """``message_effect_id`` for a fresh send (Bot API: private chats only), from
        ``metadata['message_effect']`` (an emoji name). Unknown names are skipped."""
        emoji = (metadata or {}).get("message_effect")
        if not emoji:
            return None
        effect_id = emoji if str(emoji).isdigit() else _MESSAGE_EFFECT_IDS.get(str(emoji))
        if not effect_id:
            logger.debug("[%s] Unknown message effect %r — skipping", self.name, emoji)
            return None
        try:
            cid = normalize_telegram_chat_id(chat_id)
        except (TypeError, ValueError):
            return None
        if not (isinstance(cid, int) and cid > 0):  # groups/channels: effect unsupported
            return None
        return effect_id

    async def _maybe_send_ephemeral(self, chat_id: str, content: str,
                                    metadata: Optional[dict[str, Any]]) -> Optional[SendResult]:
        """Group-only ephemeral whisper (opt-in; ``telegram.extra.ephemeral_messages``).
        None = not applicable → normal send. Raw call — PTB 22.8 predates
        ``ephemeral_message_parameters``. Failures do NOT fall back to a normal send:
        a whisper that would leak to the whole group is worse than a missing notice."""
        if not getattr(self, "_ephemeral_messages_enabled", False):
            return None
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text, _strip_mdv2
        receiver = (metadata or {}).get("ephemeral_for")
        if not receiver:
            return None
        try:
            cid = normalize_telegram_chat_id(chat_id)
            if isinstance(cid, int) and cid > 0:
                return None  # DMs have nothing to hide — normal send
            payload = {
                "chat_id": cid,
                "text": _strip_mdv2(self.format_message(content)) if content else content,
                "ephemeral_message_parameters": {"receiver_user_id": int(receiver)},
            }
        except Exception:
            logger.debug("[%s] ephemeral payload build failed", self.name, exc_info=True)
            return SendResult(success=False, error="ephemeral_invalid", retryable=False)
        try:
            ok = bool(await self._bot.do_api_request("sendMessage", api_kwargs=payload))
        except Exception as e:
            logger.debug("[%s] ephemeral send failed: %s", self.name, _redact_telegram_error_text(e), exc_info=True)
            return SendResult(success=False, error="ephemeral_failed", retryable=False)
        if ok:
            return SendResult(success=True, message_id=None)
        return SendResult(success=False, error="ephemeral_rejected", retryable=False)

    async def send_draft(self, chat_id: str, draft_id: int, content: str, metadata: Optional[dict[str, Any]] = None) -> SendResult:
        """Stream a partial message via ``sendRichMessageDraft`` (when rich is enabled and supported) else
        ``sendMessageDraft``; reusing ``draft_id`` animates the preview. The caller sends the final text."""
        from plugins.platforms.telegram.adapter import (
            ParseMode, _TEXT_SEND_DEADLINE, _await_with_thread_deadline, _redact_telegram_error_text,
        )
        if not self._bot:
            return SendResult(success=False, error="not_connected")
        # Rich draft fast-path; any failure degrades to the plain draft below. Drafts have no message_id.
        if self._should_attempt_rich_draft(content) and await self._try_send_rich_draft(chat_id, draft_id, content, metadata):
            return SendResult(success=True, message_id=None)
        if not hasattr(self._bot, "send_message_draft"):
            return SendResult(success=False, error="api_unavailable")
        # Drafts share the regular-send UTF-16 length contract.
        text = content if len(
            content) <= self.MAX_MESSAGE_LENGTH else self.truncate_message(content, self.MAX_MESSAGE_LENGTH, len_fn=utf16_len)[0]
        # Same MarkdownV2 conversion as ``send`` (MarkdownV2 then plain) so the draft doesn't snap at the end. Exception: a Rich
        # final with rich drafts disabled previews raw — the legacy formatter would turn pipe tables into bullets.
        plain_rich_preview = bool(
            getattr(self, "_rich_messages_enabled", False) and not getattr(self, "_rich_drafts_enabled", False)
            and self._needs_rich_rendering(text))
        draft_thread_kwargs = self._thread_kwargs_for_draft(chat_id, metadata)
        for use_markdown in ((False,) if plain_rich_preview else (True, False)):
            kwargs: dict[str, Any] = {
                "chat_id": normalize_telegram_chat_id(chat_id), "draft_id": int(draft_id),
                "text": self.format_message(text) if use_markdown else text}
            if use_markdown:
                kwargs["parse_mode"] = ParseMode.MARKDOWN_V2
            kwargs.update(draft_thread_kwargs)
            try:
                if self._stop_button_enabled:
                    # PTB predates can_stop; keep raw requests under the send deadline too.
                    raw_kwargs = {k: ("MarkdownV2" if k == "parse_mode" else v) for k, v in kwargs.items()}
                    raw_kwargs["can_stop"] = True
                    accepted = await _await_with_thread_deadline(
                        self._bot.do_api_request("sendMessageDraft", api_kwargs=raw_kwargs),
                        timeout=_TEXT_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False)
                    return SendResult(success=True, message_id=None) if accepted else SendResult(
                        success=False, error="draft_rejected")
                if await _await_with_thread_deadline(
                    self._bot.send_message_draft(**kwargs), timeout=_TEXT_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False):
                    return SendResult(success=True, message_id=None)
                return SendResult(success=False, error="draft_rejected")
            except Exception as e:
                # MarkdownV2 parse failure → retry once as plain text; anything else returns to the caller,
                # which falls back to edit-based streaming for this response.
                if use_markdown and self._is_bad_request_error(e):
                    logger.debug(
                        "[%s] sendMessageDraft MarkdownV2 rejected, retrying as plain text (chat=%s draft_id=%s): %s",
                        self.name, chat_id, draft_id, _redact_telegram_error_text(e))
                    continue
                logger.debug("[%s] sendMessageDraft failed (chat=%s draft_id=%s): %s", self.name, chat_id, draft_id, _redact_telegram_error_text(e), exc_info=True)
                return SendResult(success=False, error=_redact_telegram_error_text(e))
        return SendResult(success=False, error="draft_rejected")
