"""Telegram ForceReply input with exact prompt correlation and no chat-wide capture."""

import asyncio
import re
import unicodedata

from agent.i18n import SUPPORTED_LANGUAGES, t
from gateway.native_reply_input import NativeReplySubmission, ReplyInput, prompt_token, text
from gateway.platforms.base import MessageType, SendResult


def _is_friendly_prompt(body):
    """Recognize an orphan's complete localized shape for rejection, never authorization."""
    if not isinstance(body, str):
        return False
    for language in SUPPORTED_LANGUAGES:
        title = t("gateway.group_compose.title", lang=language)
        prompt = t("gateway.group_compose.prompt", lang=language)
        # The bounded, single-line slot is a label, not a parsed room reference.
        pattern = re.escape(title).replace(re.escape("{group}"), r"(?P<label>[^\r\n]{1,80})")
        match = re.fullmatch(pattern + re.escape("\n\n" + prompt), body)
        if match is None:
            continue
        label = match["label"]
        if (
            label == " ".join(label.split())
            and not any(unicodedata.category(char) in {"Cc", "Cf"} for char in label)
        ):
            return True
    return False


async def send_reply_input(adapter, event, title, on_reply):
    from telegram import ForceReply, ReplyParameters
    from .adapter import normalize_telegram_chat_id

    request = ReplyInput(adapter, on_reply)
    source = event.source
    bot_id = getattr(adapter._bot, "id", None)
    if not bot_id or not event.message_id or source.chat_type == "channel":
        return None, SendResult(success=False, error=text("unavailable"))
    request.chat_id, request.bot_id = str(source.chat_id), str(bot_id)
    await asyncio.to_thread(request.register)
    metadata = adapter.gateway_runner._thread_metadata_for_source(source, event.message_id)
    try:
        # Explicit reply is functional input targeting, independent of cosmetic reply_to_mode.
        msg = await adapter._bot.send_message(
            chat_id=normalize_telegram_chat_id(source.chat_id),
            text=f"{title}\n\n{text('prompt')}",
            parse_mode=None,
            reply_parameters=ReplyParameters(message_id=int(event.message_id), allow_sending_without_reply=False),
            reply_markup=ForceReply(selective=True, input_field_placeholder=text("placeholder")[:64]),
            **adapter._thread_kwargs_for_send(
                source.chat_id, source.thread_id, metadata,
                reply_to_message_id=int(event.message_id),
            ),
        )
        await asyncio.to_thread(request.bind_prompt, msg.message_id)
        return request, SendResult(success=True, message_id=request.prompt_id)
    except Exception:
        # A timeout may have delivered the friendly prompt. Its shape can only close a
        # later own-bot reply; without the returned message ID it cannot authorize input.
        await asyncio.to_thread(request.cancel)
        return None, SendResult(success=False, error=text("unavailable"))


async def dispatch_reply_input(adapter, message, update_id):
    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return False
    author = getattr(reply, "from_user", None)
    own_bot_reply = (
        getattr(author, "id", None) == getattr(adapter._bot, "id", None)
        and getattr(author, "is_bot", None) is True
    )
    if not own_bot_reply:
        return False
    try:
        token = await asyncio.to_thread(
            prompt_token, adapter, adapter._bot.id, message.chat.id, reply.message_id,
        )
    except Exception:
        # Receipt failure cannot authorize input or claim every ordinary bot reply.
        # Retained exact IDs and the friendly orphan shape can still reject known input.
        known = any(
            request.bot_id == str(adapter._bot.id)
            and request.chat_id == str(message.chat.id)
            and request.prompt_id == str(reply.message_id)
            for request in getattr(adapter, "_native_reply_inputs", {}).values()
        )
        token = "unavailable" if known else None
    if token is None and not _is_friendly_prompt(getattr(reply, "text", None)):
        return False
    request = getattr(adapter, "_native_reply_inputs", {}).get(token)
    kind = MessageType.TEXT if getattr(message, "text", None) else adapter._media_message_type(message)
    event = adapter._build_message_event(message, kind, update_id=update_id)
    event.text = message.text or ""
    # Real commands retain their ordinary control semantics, including emergency stops.
    command = event.get_command()
    if command:
        from hermes_cli.commands import resolve_command
        if resolve_command(command) is not None:
            return False
    if not adapter._should_process_message(message):
        return True
    if getattr(message, "caption", None) or event.message_type != MessageType.TEXT:
        event.source.message_had_attachments = True
    valid = token is not None and (request is None or request.bot_id == str(adapter._bot.id))
    event._native_reply_submission = NativeReplySubmission(adapter, token, valid)
    # Bypass BOTH normal session guards via the existing profile-scoped handler. Do not
    # acquire/release an agent turn or send this through client-split text batching.
    await adapter._dispatch_inline_reply(event)
    return True
