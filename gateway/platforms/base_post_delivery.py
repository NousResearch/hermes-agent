"""Running a turn's one-shot post-delivery callback (``BasePlatformAdapter.register_post_delivery_callback``).

Shared by the adapter task that delivered a messaging reply (``_process_message_background``) and the
session authority's drain, which settles every turn no adapter task brackets (``session_ingress``).
"""
import asyncio
import contextlib
import inspect
from typing import Any


async def run_post_delivery_callback(callback: Any, timeout: float) -> None:
    """Run *callback* (a no-op when None): bounded, errors swallowed, so a stuck or failing callback
    never holds up the session whose reply it follows."""
    if callable(callback):
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            result = callback()
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=timeout)
