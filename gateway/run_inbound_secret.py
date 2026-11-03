"""Secret answers to a clarify prompt (vault password, master password, one-time code): deleted
from the chat on receipt, only from the person asked, never routed to the agent as a turn.

Moved out of ``gateway/run_inbound.py``; the prompts themselves are ``gateway/vault_prompts.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from typing import Optional

from agent.i18n import t
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayInboundSecretMixin:
    """The inbound side of a secret prompt: take the answer, delete it, report what happened."""

    async def _hm_secret_clarify_reply(self, event: MessageEvent, source: SessionSource, pending) -> Optional[str]:
        """Answer a secret prompt (vault password, master password, one-time code) from its owner.

        The message is deleted on receipt, BEFORE the value is handed to the waiting tool, and the
        deletion is checked: when it fails the user is told in the chat to delete it. A message that
        is not an answer is never let through to the agent once deleted. Three cases:
        a real slash command falls through (``/stop`` must still work) and the prompt stays armed;
        a voice note is deleted, its cached audio removed and NOT transcribed, and the prompt stays
        armed; any other text is the secret, raw (``skip`` included: it is a decline only for an
        ordinary clarify)."""
        from tools import clarify_gateway as _clarify_mod
        if self._is_known_gateway_command(event):
            return None
        adapter = self._delivery_adapter_for(source)
        audio = self._pending_event_audio_paths(event)
        text = (event.text or "").strip()
        if not audio and not text:
            return None  # a sticker or a photo answers nothing: it is not taken, not deleted
        deleted = await self._delete_secret_reply(adapter, source, event)
        if audio:
            for path in audio:
                with suppress(OSError):
                    os.remove(path)
            logger.info("Secret prompt answered by a voice note: deleted=%s, not transcribed (id=%s)",
                        deleted, pending.clarify_id)
            await self._secret_reply_notice(adapter, source, t("gateway.vault_prompt.voice_refused"), deleted)
            return ""
        if not _clarify_mod.resolve_gateway_clarify(pending.clarify_id, text):
            # Timed out or cancelled between lookup and now: the value is dropped, never routed.
            logger.info("Secret prompt ended before its answer arrived: deleted=%s, value dropped (id=%s)",
                        deleted, pending.clarify_id)
            await self._secret_reply_notice(adapter, source, t("gateway.vault_prompt.expired"), deleted)
            return ""
        logger.info("Gateway took a secret clarify answer: deleted=%s (id=%s)", deleted, pending.clarify_id)
        if adapter:
            adapter.resume_typing_for_chat(source.chat_id)
            if callable(getattr(type(adapter), "retire_clarify_card", None)):
                try:
                    # Never the answer itself, not even in the retired card's summary line.
                    await adapter.retire_clarify_card(pending.clarify_id, t("gateway.clarify.answered_secret"))
                except Exception:
                    logger.debug("Failed to retire clarify card after secret answer", exc_info=True)
        if not deleted:
            await self._secret_reply_notice(adapter, source, None, deleted)
        return ""

    async def _hm_late_secret_reply(self, event: MessageEvent, source: SessionSource, _quick_key: str) -> Optional[str]:
        """The sender's secret prompt ended unanswered moments ago (timeout, /stop, an interrupt from
        another member of a shared topic): their next text message may be the secret, typed while the
        prompt was going away. Inside the short window (``clarify_gateway.SECRET_GRACE_S``) it is
        deleted and dropped instead of becoming a turn. Commands, voice notes and media pass."""
        from tools import clarify_gateway as _clarify_mod
        if self._is_known_gateway_command(event) or self._pending_event_audio_paths(event) \
                or not (event.text or "").strip() or not _clarify_mod.take_secret_grace(_quick_key, source.user_id):
            return None
        adapter = self._delivery_adapter_for(source)
        deleted = await self._delete_secret_reply(adapter, source, event)
        logger.info("Message right after an unanswered secret prompt taken as a late secret: deleted=%s, "
                    "value dropped (session=%s)", deleted, _quick_key)
        await self._secret_reply_notice(adapter, source, t("gateway.vault_prompt.expired"), deleted)
        return ""

    @staticmethod
    def _is_known_gateway_command(event: MessageEvent) -> bool:
        """A real slash command (``/stop``), as opposed to text that merely starts with "/" — which a
        password can."""
        from hermes_cli.commands import resolve_command
        cmd = event.get_command()
        return bool(cmd) and resolve_command(cmd) is not None

    async def _delete_secret_reply(self, adapter, source, event) -> bool:
        """Delete every message the answer was built from (a text batch merges several into one event);
        one retry. True only when each deletion was confirmed. Logs ids, never content."""
        ids = event.chat_message_ids()
        if adapter is None or not ids:
            logger.warning("Secret answer could not be deleted: %s", "no adapter" if adapter is None else "no message id")
            return False
        all_deleted = True
        for message_id in ids:
            ok = False
            for attempt in range(2):
                try:
                    ok = bool(await adapter.delete_message(chat_id=source.chat_id, message_id=message_id))
                except Exception:
                    logger.debug("Secret answer deletion raised", exc_info=True)
                if ok:
                    break
                if attempt == 0:
                    await asyncio.sleep(1.0)
            if not ok:
                logger.warning("Secret answer NOT deleted (chat=%s, message=%s)", source.chat_id, message_id)
            all_deleted = all_deleted and ok
        return all_deleted

    async def _secret_reply_notice(self, adapter, source, text: Optional[str], deleted: bool) -> None:
        """Tell the user what happened to their secret message, in its topic, never as a reply to it
        (a reply would quote the secret, or point at a message that no longer exists)."""
        lines = [line for line in (text, None if deleted else t("gateway.vault_prompt.not_deleted")) if line]
        if adapter is None or not lines:
            return
        metadata = {k: v for k, v in (self._thread_metadata_for_source(source) or {}).items() if "reply_to" not in k}
        try:
            await adapter.send(source.chat_id, "\n\n".join(lines), metadata=metadata or None)
        except Exception:
            logger.warning("Could not post the secret-answer notice (chat=%s)", source.chat_id, exc_info=True)
