"""Teardown for discord.py tasks that survive Client.close()."""

import asyncio
from typing import Any


async def close_discord_client(client: Any) -> None:
    """Drain delayed READY before close replaces the client's loop with MISSING."""
    state = getattr(client, "_connection", None)
    ready_task = getattr(state, "_ready_task", None)
    try:
        if isinstance(ready_task, asyncio.Task) and ready_task is not asyncio.current_task():
            if not ready_task.done():
                ready_task.cancel()
            # gather distinguishes the child's cancellation from cancellation of
            # this teardown, which must still propagate to its timeout owner.
            result, = await asyncio.gather(ready_task, return_exceptions=True)
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                raise result
    finally:
        await client.close()
