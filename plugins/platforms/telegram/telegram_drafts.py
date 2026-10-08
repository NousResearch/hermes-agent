"""Prefix-stable native drafts and their shared Telegram typing-action budget."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from typing import Any, Optional

from gateway.platforms.base import SendResult, _prefix_within_utf16_limit
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

logger = logging.getLogger(__name__)


def _ai_action_now() -> float:
    return asyncio.get_running_loop().time()


class TelegramDraftMixin:
    DRAFT_STREAM_PREFIX_STABLE = True

    def _claim_ai_action_slot(self, chat_id: Any) -> float:
        """Reserve one actual draft/typing request, or return its non-blocking retry delay.

        Telegram shares 20 calls/5s and 40 calls/30s per peer across sendMessageDraft and
        sendChatAction: https://core.telegram.org/api/bots/ai. Reserving before the await
        prevents concurrent topics or typing refreshes from spending the same slot.
        """
        key = str(normalize_telegram_chat_id(chat_id))
        now = _ai_action_now()
        cooldowns = self.__dict__.setdefault("_telegram_ai_action_cooldown_until", {})
        remaining = cooldowns.get(key, now) - now
        if remaining > 0:
            return remaining
        cooldowns.pop(key, None)
        calls = self.__dict__.setdefault("_telegram_ai_action_calls", {}).setdefault(key, deque())
        while calls and now - calls[0] >= 30.0:
            calls.popleft()
        short_delay = calls[-20] + 5.0 - now if len(calls) >= 20 else 0.0
        long_delay = calls[0] + 30.0 - now if len(calls) >= 40 else 0.0
        delay = max(short_delay, long_delay, 0.0)
        if delay == 0:
            calls.append(now)
        return delay

    def _record_ai_action_cooldown(self, chat_id: Any, exc: Exception) -> Optional[float]:
        """Respect RetryAfter for drafts and typing without delaying completed messages."""
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is None:
            return None
        if hasattr(retry_after, "total_seconds"):
            retry_after = retry_after.total_seconds()
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(delay):
            return None
        delay = max(1.0, delay)
        key = str(normalize_telegram_chat_id(chat_id))
        cooldowns = self.__dict__.setdefault("_telegram_ai_action_cooldown_until", {})
        cooldowns[key] = max(cooldowns.get(key, 0.0), _ai_action_now() + delay)
        return delay

    @staticmethod
    def _draft_skipped(delay: float) -> SendResult:
        return SendResult(success=True, raw_response={"skipped": True}, retry_after=delay)

    def _native_draft_recent(self, chat_id: Any) -> bool:
        """A native draft is itself a typing action; skip redundant indicator refreshes."""
        key = str(normalize_telegram_chat_id(chat_id))
        sent_at = self.__dict__.setdefault("_telegram_native_draft_sent_at", {})
        last_sent = sent_at.get(key)
        if last_sent is not None and _ai_action_now() - last_sent < 4.0:
            return True
        sent_at.pop(key, None)
        return False

    def _record_native_draft(self, chat_id: Any) -> SendResult:
        sent_at = self.__dict__.setdefault("_telegram_native_draft_sent_at", {})
        sent_at[str(normalize_telegram_chat_id(chat_id))] = _ai_action_now()
        return SendResult(success=True)

    async def _try_send_rich_draft(
        self, chat_id: str, draft_id: int, content: str, metadata: Optional[dict[str, Any]],
    ) -> Optional[SendResult]:
        """Preserve explicit rich drafts, falling back only outside a flood cooldown."""
        from plugins.platforms.telegram.adapter import (
            _TEXT_SEND_DEADLINE, _await_with_thread_deadline, _redact_telegram_error_text,
        )

        delay = self._claim_ai_action_slot(chat_id)
        if delay:
            return self._draft_skipped(delay)
        payload = {
            "chat_id": normalize_telegram_chat_id(chat_id), "draft_id": int(draft_id),
            "rich_message": self._rich_message_payload(content),
        }
        payload.update(self._thread_kwargs_for_draft(chat_id, metadata))
        try:
            accepted = await _await_with_thread_deadline(
                self._bot.do_api_request("sendRichMessageDraft", api_kwargs=payload),
                timeout=_TEXT_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False,
            )
        except Exception as exc:  # health: allow BLE001 -- API boundary; raw tracebacks may expose bot-token URLs, so log redacted text.
            delay = self._record_ai_action_cooldown(chat_id, exc)
            if delay is not None:
                return self._draft_skipped(delay)
            if self._is_rich_capability_error(exc):
                self._rich_draft_disabled = True
            logger.debug(
                "[%s] sendRichMessageDraft rejected; using legacy draft: %s",
                self.name, _redact_telegram_error_text(exc),
            )
            return None
        return self._record_native_draft(chat_id) if accepted else None

    async def send_draft(
        self, chat_id: str, draft_id: int, content: str, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send an append-only plain preview; persistent finals retain their normal formatting."""
        from plugins.platforms.telegram.adapter import (
            _TEXT_SEND_DEADLINE, _await_with_thread_deadline, _redact_telegram_error_text,
        )

        if not self._bot:
            return SendResult(success=False, error="not_connected")
        if self._should_attempt_rich_draft(content):
            rich_result = await self._try_send_rich_draft(chat_id, draft_id, content, metadata)
            if rich_result is not None:
                return rich_result
        if not hasattr(self._bot, "send_message_draft"):
            return SendResult(success=False, error="api_unavailable")
        delay = self._claim_ai_action_slot(chat_id)
        if delay:
            return self._draft_skipped(delay)
        # The Markdown renderer and chunker can rewrite earlier characters as fences/tables grow.
        # A raw UTF-16-bounded prefix lets Telegram animate only newly appended characters.
        kwargs = {
            "chat_id": normalize_telegram_chat_id(chat_id), "draft_id": int(draft_id),
            "text": _prefix_within_utf16_limit(content, self.MAX_MESSAGE_LENGTH),
            **self._thread_kwargs_for_draft(chat_id, metadata),
        }
        try:
            accepted = await _await_with_thread_deadline(
                self._bot.send_message_draft(**kwargs), timeout=_TEXT_SEND_DEADLINE,
                label="telegram-send", dump_on_blocked_loop=False,
            )
        except Exception as exc:  # health: allow BLE001 -- API boundary; raw tracebacks may expose bot-token URLs, so log redacted text.
            delay = self._record_ai_action_cooldown(chat_id, exc)
            if delay is not None:
                return self._draft_skipped(delay)
            safe_error = _redact_telegram_error_text(exc)
            logger.debug("[%s] sendMessageDraft failed: %s", self.name, safe_error)
            return SendResult(success=False, error=safe_error)
        if accepted:
            return self._record_native_draft(chat_id)
        return SendResult(success=False, error="draft_rejected")
