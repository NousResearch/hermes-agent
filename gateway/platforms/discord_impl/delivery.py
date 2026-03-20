"""Discord message delivery helpers."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from gateway.platforms.base import SendResult


logger = logging.getLogger(__name__)

try:
    import discord
except ImportError:  # pragma: no cover - import guard
    discord = None


async def resolve_channel(client: Any, chat_id: str) -> Optional[Any]:
    """Get or fetch a Discord channel by ID."""
    if not client:
        return None

    try:
        channel_id = int(chat_id)
    except (TypeError, ValueError):
        return None

    channel = client.get_channel(channel_id)
    if channel is not None:
        return channel

    try:
        return await client.fetch_channel(channel_id)
    except Exception:
        return None


async def send_text_message(channel: Any, content: str, reference: Any = None) -> Any:
    """Send a single Discord text message chunk with reply fallback."""
    try:
        return await channel.send(content=content, reference=reference)
    except Exception as exc:
        err_text = str(exc)
        if (
            reference is not None
            and "error code: 50035" in err_text
            and "Cannot reply to a system message" in err_text
        ):
            logger.warning(
                "Reply target is a Discord system message; retrying send without reply reference"
            )
            return await channel.send(content=content, reference=None)
        raise


async def send_file_attachment(
    client: Any,
    chat_id: str,
    file_path: str,
    caption: str | None = None,
    file_name: str | None = None,
) -> SendResult:
    """Send a local file as a Discord attachment."""
    if not client:
        return SendResult(success=False, error="Not connected")

    channel = await resolve_channel(client, chat_id)
    if not channel:
        return SendResult(success=False, error=f"Channel {chat_id} not found")

    if discord is None:  # pragma: no cover - import guard
        return SendResult(success=False, error="discord.py not installed")

    filename = file_name or os.path.basename(file_path)
    with open(file_path, "rb") as fh:
        file = discord.File(fh, filename=filename)
        msg = await channel.send(content=caption if caption else None, file=file)
    return SendResult(success=True, message_id=str(msg.id))
