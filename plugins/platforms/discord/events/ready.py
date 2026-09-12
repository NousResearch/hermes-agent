"""Discord ``on_ready`` event."""

import asyncio
import logging

logger = logging.getLogger("plugins.platforms.discord.adapter")


async def handle(adapter) -> None:
    """Complete the post-connect transition after Discord announces readiness."""
    logger.info("[%s] Connected as %s", adapter.name, adapter._client.user)
    await adapter._resolve_allowed_usernames()
    adapter._ready_event.set()
    if adapter._post_connect_task and not adapter._post_connect_task.done():
        adapter._post_connect_task.cancel()
    adapter._post_connect_task = asyncio.create_task(adapter._run_post_connect_initialization())
    if adapter._missed_message_backfill_enabled():
        adapter._ensure_missed_message_backfill_task()


def register(client, adapter) -> None:
    @client.event
    async def on_ready():
        await handle(adapter)
