"""Running a turn's one-shot post-delivery callback (``BasePlatformAdapter.register_post_delivery_callback``).

Shared by the adapter task that delivered a messaging reply (``_process_message_background``) and the
session authority's drain, which settles every turn no adapter task brackets (``session_ingress``).
"""
import asyncio
import contextlib
import inspect
from typing import Any


def turn_run_generation(event: Any) -> int | None:
    """The run generation stamped on one turn's own event (``_hmwa_prepare_turn``; on authority routes
    ``session_ingress._hand_over_turn_marker`` hands it to the event of the lifecycle that sends the
    reply), or None. Per turn, unlike the adapter's session guard Event, which every turn running
    while one adapter task holds the guard shares."""
    generation = getattr(event, "_hermes_run_generation", None)
    return generation if isinstance(generation, int) and not isinstance(generation, bool) else None


async def run_post_delivery_callback(callback: Any, timeout: float) -> None:
    """Run *callback* (a no-op when None): bounded, errors swallowed, so a stuck or failing callback
    never holds up the session whose reply it follows."""
    if callable(callback):
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            result = callback()
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=timeout)
