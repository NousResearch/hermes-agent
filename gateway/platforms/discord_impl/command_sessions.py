"""Discord command-session helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import re
from typing import Any, Optional, Tuple

from gateway.platforms.base import MessageEvent, MessageType

try:
    import discord
except ImportError:  # pragma: no cover - import guard
    discord = None


INLINE_SHORTCUT_COMMANDS = ("help", "commands", "status", "whoami", "id")
_INLINE_SHORTCUT_RE = re.compile(
    r"(?<!\S)(/(?:" + "|".join(INLINE_SHORTCUT_COMMANDS) + r"))(?=\s|$)",
    re.IGNORECASE,
)


def extract_inline_shortcut(text: str) -> Tuple[Optional[str], str]:
    """Return the first supported inline shortcut and the stripped remaining text."""
    match = _INLINE_SHORTCUT_RE.search(text)
    if not match:
        return None, text

    command = match.group(1).lstrip("/").lower()
    remaining = (text[:match.start()] + " " + text[match.end():]).strip()
    remaining = re.sub(r"\s{2,}", " ", remaining)
    return command, remaining


def _resolve_target_chat(adapter: Any, interaction: Any) -> tuple[str, str, Optional[str]]:
    dm_channel_cls = getattr(discord, "DMChannel", None) if discord else None
    thread_cls = getattr(discord, "Thread", None) if discord else None

    channel = interaction.channel
    channel_id = str(getattr(channel, "id", getattr(interaction, "channel_id", "")) or "")
    is_dm = isinstance(channel, dm_channel_cls) if dm_channel_cls else False
    is_thread = isinstance(channel, thread_cls) if thread_cls else False
    thread_id = channel_id if is_thread else None

    if is_dm:
        chat_type = "dm"
        chat_name = getattr(interaction.user, "display_name", None) or str(interaction.user.id)
        return chat_type, chat_name, thread_id

    if is_thread:
        chat_type = "thread"
        formatter = getattr(adapter, "_format_thread_chat_name", None)
        if callable(formatter):
            return chat_type, formatter(channel), thread_id
        return chat_type, getattr(channel, "name", channel_id or "unknown"), thread_id

    chat_type = "group"
    chat_name = getattr(channel, "name", channel_id or "unknown")
    guild = getattr(channel, "guild", None)
    if guild:
        chat_name = f"{guild.name} / #{chat_name}"
    return chat_type, chat_name, thread_id


def build_slash_event(adapter: Any, interaction: Any, text: str) -> MessageEvent:
    """Build a slash MessageEvent with isolated command-session metadata."""
    chat_type, chat_name, thread_id = _resolve_target_chat(adapter, interaction)
    chat_topic = getattr(interaction.channel, "topic", None)
    user_id = str(interaction.user.id)

    target_source = adapter.build_source(
        chat_id=str(interaction.channel_id),
        chat_name=chat_name,
        chat_type=chat_type,
        user_id=user_id,
        user_name=interaction.user.display_name,
        thread_id=thread_id,
        chat_topic=chat_topic,
    )
    session_namespace = f"slash:{user_id}"
    if is_dataclass(target_source):
        session_source = replace(target_source, session_namespace=session_namespace)
    else:
        source_kwargs = dict(asdict(target_source)) if hasattr(target_source, "__dataclass_fields__") else dict(
            getattr(target_source, "__dict__", {})
        )
        source_kwargs["session_namespace"] = session_namespace
        session_source = adapter.build_source(**source_kwargs)

    msg_type = MessageType.COMMAND if text.startswith("/") else MessageType.TEXT
    return MessageEvent(
        text=text,
        message_type=msg_type,
        source=target_source,
        raw_message=interaction,
        metadata={
            "session_source": session_source,
            "command_target_source": target_source,
            "is_native_slash": True,
        },
    )
