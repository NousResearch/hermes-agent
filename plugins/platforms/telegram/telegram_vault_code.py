"""The browser vault's one-time-code prompt on Telegram: a reply the bot deletes on receipt.

Telegram has no private form a bot can open, so the code arrives as a normal message. The gateway asks
through ``send_vault_code_prompt`` (``gateway/session_vault_code_card.py``); the card is a ForceReply
prompt aimed at the turn's author. Their next text in that chat (a reply to the card, or any message in a
private chat) is taken off the inbound path before batching or the agent see it: the bot deletes it, hands
the code to ``submit``, and never turns it into a MessageEvent. Bots may delete incoming messages in private
chats; in a group the bot needs delete rights, and the author is told to remove it when the delete fails.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)

_HINT_LIMIT = 300
_PLACEHOLDER_LIMIT = 64  # Bot API cap on ForceReply.input_field_placeholder


@dataclass
class _CodeReply:
    submit: Callable[[str], bool]
    user_id: str
    chat_id: str
    prompt_message_id: str
    prompt_text: str
    expires_at: float


class _DropPendingCodeUpdates(logging.Filter):
    """python-telegram-bot logs every getUpdates result (``telegram.Bot`` / ``telegram.ext.ExtBot``) and every
    webhook body (``telegram.ext.Updater``) at DEBUG, before any handler runs. While a code reply is pending,
    drop those records so ``logging.level: DEBUG`` never writes the code into agent.log."""

    pending = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if not _DropPendingCodeUpdates.pending:
            return True
        text = record.getMessage()
        return "getUpdates" not in text and "Webhook received data" not in text


for _name in ("telegram.Bot", "telegram.ext.ExtBot", "telegram.ext.Updater"):
    _logger = logging.getLogger(_name)
    if not any(isinstance(f, _DropPendingCodeUpdates) for f in _logger.filters):
        _logger.addFilter(_DropPendingCodeUpdates())


def _t(key: str, **kwargs: Any) -> str:
    from plugins.platforms.telegram.adapter import t
    return t(key, **kwargs)


class TelegramVaultCodeMixin:
    """``send_vault_code_prompt`` (the capability the gateway detects) and the inbound interception."""

    _bot: Any

    def _vault_code_replies(self) -> dict:
        replies = getattr(self, "_vault_code_reply_state", None)
        if replies is None:
            replies = self._vault_code_reply_state = {}
        return replies

    async def send_vault_code_prompt(self, chat_id: str, site: str, hint: str, *, user_id: str,
                                     submit: Callable[[str], bool], timeout_s: float,
                                     metadata: Optional[dict] = None) -> SendResult:
        from telegram import ForceReply  # late: test doubles of the telegram package need not define it
        from plugins.platforms.telegram.adapter import ParseMode
        import html as _html

        text = f"<b>{_html.escape(_t('platform.shared.vault_code.title', site=site))}</b>\n" + _html.escape(
            _t("platform.telegram.vault_code.body", site=site))
        if hint:
            text += "\n" + _html.escape(hint[:_HINT_LIMIT])
        reply_markup = ForceReply(selective=True, input_field_placeholder=_t(
            "platform.shared.vault_code.input_label")[:_PLACEHOLDER_LIMIT])
        pending: dict = {}

        def build():
            def remember(msg):
                pending["message_id"] = str(msg.message_id)
                pending["chat_id"] = str(getattr(msg, "chat_id", chat_id))
            return text, reply_markup, remember
        result = await self._send_prompt("send_vault_code_prompt", chat_id, metadata, build, parse_mode=ParseMode.HTML,
                                         thread_id=self._metadata_thread_id(metadata))
        if result.success and pending:
            key = (pending["chat_id"], str(user_id))
            self._vault_code_replies()[key] = _CodeReply(
                submit=submit, user_id=str(user_id), chat_id=pending["chat_id"],
                prompt_message_id=pending["message_id"], prompt_text=text, expires_at=time.monotonic() + timeout_s)
            _DropPendingCodeUpdates.pending += 1
            asyncio.get_running_loop().call_later(
                timeout_s, lambda: asyncio.ensure_future(self._retire_vault_code_reply(key, "ended")))
        return result

    def _pop_vault_code_reply(self, key: tuple) -> Optional[_CodeReply]:
        reply = self._vault_code_replies().pop(key, None)
        if reply is not None:
            _DropPendingCodeUpdates.pending = max(0, _DropPendingCodeUpdates.pending - 1)
        return reply

    async def _retire_vault_code_reply(self, key: tuple, outcome: str) -> None:
        reply = self._pop_vault_code_reply(key)
        if reply is not None:
            await self._mark_vault_code_card(reply, outcome)

    async def _mark_vault_code_card(self, reply: _CodeReply, outcome: str) -> None:
        notice = _t("platform.shared.vault_code.submitted_footer" if outcome == "submitted"
                    else "platform.shared.vault_code.ended")
        try:
            from plugins.platforms.telegram.adapter import ParseMode
            await self._bot.edit_message_text(chat_id=int(reply.chat_id), message_id=int(reply.prompt_message_id),
                                              text=f"{reply.prompt_text}\n\n{notice}", parse_mode=ParseMode.HTML)
        except Exception:  # health: allow BLE001 -- the card's outcome line is cosmetic; the request is already settled
            logger.debug("[%s] vault code card edit failed", getattr(self, "name", "Telegram"), exc_info=True)

    def _pending_vault_code_reply(self, msg) -> Optional[tuple]:
        chat_id = str(getattr(msg, "chat_id", "") or "")
        user_id = str(getattr(getattr(msg, "from_user", None), "id", "") or "")
        key = (chat_id, user_id)
        reply = self._vault_code_replies().get(key)
        if reply is None or time.monotonic() >= reply.expires_at:
            return None
        replied_to = getattr(getattr(msg, "reply_to_message", None), "message_id", None)
        private = str(getattr(getattr(msg, "chat", None), "type", "")) == "private"
        # A group message counts only as a reply to the card; a private chat has no one else to answer.
        if not private and str(replied_to) != reply.prompt_message_id:
            return None
        return key

    async def consume_vault_code_reply(self, msg) -> bool:
        """True when ``msg`` answered a pending code prompt: deleted, handed over, never a turn."""
        key = self._pending_vault_code_reply(msg)
        reply = self._pop_vault_code_reply(key) if key is not None else None
        if reply is None:
            return False
        self._accept_update()
        code = (getattr(msg, "text", "") or "").strip()
        deleted = await self.delete_message(reply.chat_id, str(msg.message_id))
        accepted = bool(code) and reply.submit(code)
        code = ""
        await self._mark_vault_code_card(reply, "submitted" if accepted else "ended")
        if not deleted:
            await self._vault_code_notice(reply, "platform.telegram.vault_code.delete_failed")
        return True

    async def _vault_code_notice(self, reply: _CodeReply, key: str) -> None:
        try:
            await self._bot.send_message(chat_id=int(reply.chat_id), text=_t(key),
                                         reply_to_message_id=int(reply.prompt_message_id))
        except Exception:  # health: allow BLE001 -- a notice the chat refuses must not fail the interception
            logger.debug("[%s] vault code notice failed", getattr(self, "name", "Telegram"), exc_info=True)
