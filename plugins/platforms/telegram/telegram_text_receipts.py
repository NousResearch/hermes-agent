"""Skip an edited Telegram text whose content matches what the core last received.

A reaction makes Telegram send an ``edited_message`` with unchanged text
(tdlib/telegram-bot-api#388, #773). Comparing content keeps genuine edits as turns.

Imports no Telegram SDK: the text and command handlers call ``unchanged_text`` on
every message, including adapters driven without the ``messaging`` extra.
"""

import asyncio
import hashlib
import json
import time


def message_text_digest(message) -> str:
    """Hash exact text and entity semantics, not transport IDs or user display names."""
    entities = [
        [entity.type, entity.offset, entity.length, entity.url,
         entity.user.id if entity.user is not None else None,
         entity.language, entity.custom_emoji_id]
        for entity in message.entities
    ]
    payload = json.dumps([message.text, entities], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unchanged_text(adapter, update) -> bool:
    """Compare latest accepted content only: A -> B -> A remains two genuine edits."""
    admission = getattr(adapter, "_update_admission", None)
    claim = admission.get() if admission is not None else None
    if claim is None:
        return False
    # A claim exists only under TelegramApplication, so the SDK is loaded by now.
    from plugins.platforms.telegram.update_admission import RECEIPT_TTL_SECONDS
    message = update.effective_message
    key = f"{claim.key.split(':', 1)[0]}:{message.chat_id}:{message.message_id}"
    digest = message_text_digest(message)
    latest = adapter._latest_text_receipts.get(key)
    if latest is not None and latest[1] <= time.time() - RECEIPT_TTL_SECONDS:
        latest = None
    is_edit = update.edited_message is not None or update.edited_channel_post is not None
    if is_edit and latest is not None and latest[0] == digest:
        return True
    claim.text_receipt = (key, digest)
    claim.text_receipt_task = asyncio.current_task()
    return False
