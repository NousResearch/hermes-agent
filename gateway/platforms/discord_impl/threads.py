"""Discord thread creation and routing helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

from gateway.platforms.base import MessageEvent, MessageType


logger = logging.getLogger(__name__)

VALID_THREAD_AUTO_ARCHIVE_MINUTES = {60, 1440, 4320, 10080}

try:
    import discord
except ImportError:  # pragma: no cover - import guard
    discord = None


async def auto_create_thread(message: Any, max_name_len: int = 80) -> Optional[Any]:
    """Create a thread from a Discord message for auto-threading."""
    content = (message.content or "").strip()
    thread_name = content[:max_name_len] if content else "Hermes"
    if len(content) > max_name_len:
        thread_name = thread_name[: max_name_len - 3] + "..."

    try:
        return await message.create_thread(name=thread_name, auto_archive_duration=1440)
    except Exception as exc:
        logger.warning("Auto-thread creation failed: %s", exc)
        return None


async def create_thread(
    client: Any,
    interaction: Any,
    name: str,
    message: str,
    auto_archive_duration: int,
    resolve_channel_fn,
) -> dict[str, Any]:
    """Create a Discord thread in the interaction's current channel."""
    name = (name or "").strip()
    if not name:
        return {"error": "Thread name is required."}

    if auto_archive_duration not in VALID_THREAD_AUTO_ARCHIVE_MINUTES:
        allowed = ", ".join(str(v) for v in sorted(VALID_THREAD_AUTO_ARCHIVE_MINUTES))
        return {"error": f"auto_archive_duration must be one of: {allowed}."}

    channel = await resolve_channel_fn(client, interaction)
    if channel is None:
        return {"error": "Could not resolve the current Discord channel."}

    dm_channel_cls = getattr(discord, "DMChannel", None) if discord else None
    if dm_channel_cls and isinstance(channel, dm_channel_cls):
        return {"error": "Discord threads can only be created inside server text channels, not DMs."}

    parent_channel = thread_parent_channel(channel)
    if parent_channel is None:
        return {"error": "Could not determine a parent text channel for the new thread."}

    display_name = getattr(getattr(interaction, "user", None), "display_name", None) or "unknown user"
    reason = f"Requested by {display_name} via /thread"
    starter_message = (message or "").strip()

    try:
        thread = await parent_channel.create_thread(
            name=name,
            auto_archive_duration=auto_archive_duration,
            reason=reason,
        )
        if starter_message:
            await thread.send(starter_message)
        return {
            "success": True,
            "thread_id": str(thread.id),
            "thread_name": getattr(thread, "name", None) or name,
        }
    except Exception as direct_error:
        try:
            seed_content = starter_message or f"\U0001f9f5 Thread created by Hermes: **{name}**"
            seed_msg = await parent_channel.send(seed_content)
            thread = await seed_msg.create_thread(
                name=name,
                auto_archive_duration=auto_archive_duration,
                reason=reason,
            )
            return {
                "success": True,
                "thread_id": str(thread.id),
                "thread_name": getattr(thread, "name", None) or name,
            }
        except Exception as fallback_error:
            return {
                "error": (
                    "Discord rejected direct thread creation and the fallback also failed. "
                    f"Direct error: {direct_error}. Fallback error: {fallback_error}"
                )
            }


async def handle_thread_create_slash(
    adapter: Any,
    interaction: Any,
    name: str,
    message: str = "",
    auto_archive_duration: int = 1440,
) -> None:
    """Create a Discord thread from a slash command and start a session in it."""
    result = await adapter._create_thread(
        interaction,
        name=name,
        message=message,
        auto_archive_duration=auto_archive_duration,
    )

    if not result.get("success"):
        error = result.get("error", "unknown error")
        await interaction.followup.send(f"Failed to create thread: {error}", ephemeral=True)
        return

    thread_id = result.get("thread_id")
    thread_name = result.get("thread_name") or name

    link = f"<#{thread_id}>" if thread_id else f"**{thread_name}**"
    await interaction.followup.send(f"Created thread {link}", ephemeral=True)

    if thread_id:
        adapter._track_thread(thread_id)

    starter = (message or "").strip()
    if starter and thread_id:
        await adapter._dispatch_thread_session(interaction, thread_id, thread_name, starter)


async def dispatch_thread_session(
    adapter: Any,
    interaction: Any,
    thread_id: str,
    thread_name: str,
    text: str,
) -> None:
    """Build a thread MessageEvent and send it through the adapter handler."""
    guild_name = ""
    if hasattr(interaction, "guild") and interaction.guild:
        guild_name = interaction.guild.name

    chat_name = f"{guild_name} / {thread_name}" if guild_name else thread_name
    source = adapter.build_source(
        chat_id=thread_id,
        chat_name=chat_name,
        chat_type="thread",
        user_id=str(interaction.user.id),
        user_name=interaction.user.display_name,
        thread_id=thread_id,
    )

    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=interaction,
    )
    await adapter.handle_message(event)


def thread_parent_channel(channel: Any) -> Any:
    """Return the parent text channel when invoked from a thread."""
    return getattr(channel, "parent", None) or channel


async def resolve_interaction_channel(client: Any, interaction: Any) -> Optional[Any]:
    """Return the interaction channel, fetching it if the payload is partial."""
    channel = getattr(interaction, "channel", None)
    if channel is not None:
        return channel
    if not client:
        return None

    channel_id = getattr(interaction, "channel_id", None)
    if channel_id is None:
        return None

    channel = client.get_channel(int(channel_id))
    if channel is not None:
        return channel

    try:
        return await client.fetch_channel(int(channel_id))
    except Exception:
        return None
