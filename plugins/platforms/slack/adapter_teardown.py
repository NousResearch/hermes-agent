"""Best-effort teardown chain for ``SlackAdapter.disconnect()``, kept out of the adapter facade.

The gateway disconnects an adapter through a bounded wait that *cancels* ``disconnect()`` at its
deadline (``gateway/run_adapters._safe_adapter_disconnect`` -> ``_wait_or_detach``), so a Slack API
call that raises or hangs there used to skip every later step of the chain — including the scoped
platform-lock release, which then leaves the app token owned by a process that is going away
(#147845). Two layers close that: every step runs isolated from its siblings, and the state clear
plus the lock release sit in a strictly synchronous ``finally`` that has no suspension point to
skip. The same shape as the Feishu adapter's ``_local_teardown`` finalizer (#96801 / #96809).
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable, Iterable

logger = logging.getLogger(__name__)

# Passed to ``_cancel_socket_watchdog``: says which teardown phase saw the watchdog fail.
WATCHDOG_DISCONNECT_FAILURE = "[Slack] Watchdog task raised during disconnect"

Step = tuple[str, Callable[[], Awaitable[Any]]]


def teardown_steps(adapter: Any) -> list[Step]:
    """The ordered chain: streams and task cards first (their workspace clients are still live),
    then the socket-mode handler and the clients themselves. ``_active_streams`` is snapshotted
    and dropped up front so a stream stranded by a cancel cannot leak into the next ``connect()``.
    """
    streams = [(f"seal stream {key[1]}", functools.partial(adapter._seal_stream, key, stream))
               for key, stream in list(adapter._active_streams.items())]
    adapter._active_streams.clear()
    cards = [(f"stop task card {key[1]}",
              functools.partial(adapter._stop_native_task_card_stream, key, stream))
             for key, stream in list(adapter._native_task_card_streams.items())]
    return [
        *streams,
        ("cancel socket watchdog",
         functools.partial(adapter._cancel_socket_watchdog, WATCHDOG_DISCONNECT_FAILURE)),
        *cards,
        ("stop socket mode", adapter._stop_socket_mode_handler),
        ("close workspace clients", adapter._close_workspace_clients),
    ]


async def run_steps(steps: Iterable[Step]) -> bool:
    """Run every step isolated from the others; True when one of them was cancelled.

    ``Exception`` is logged with its traceback and the chain continues. ``CancelledError`` is
    caught for the same reason — the lock release that follows the chain is exactly what a bounded
    teardown must not lose — and the caller re-raises it, so the task still reports as cancelled.
    """
    cancelled = False
    for label, step in steps:
        try:
            await step()
        except asyncio.CancelledError:
            cancelled = True
            logger.warning("[Slack] Teardown step '%s' was cancelled; continuing with the rest",
                           label)
        except Exception:
            logger.warning("[Slack] Teardown step '%s' failed; continuing with the rest",
                           label, exc_info=True)
    return cancelled


async def disconnect_teardown(adapter: Any) -> None:
    """Run the teardown chain, always clearing the connection state and releasing the platform
    lock, then re-raise a cancellation the chain absorbed (one step per suspension point)."""
    cancelled = False
    try:
        cancelled = await run_steps(teardown_steps(adapter))
    finally:
        adapter._app = adapter._app_token = adapter._proxy_url = adapter._bot_user_id = None
        adapter._team_clients, adapter._team_bot_user_ids = {}, {}
        adapter._channel_team, adapter._dm_conversation_cache = {}, {}
        _release_platform_lock(adapter)
        logger.info("[Slack] Disconnected")
    if cancelled:
        raise asyncio.CancelledError


def _release_platform_lock(adapter: Any) -> None:
    """Release the scoped platform lock, logging rather than raising on failure: the state is
    already cleared and a pending cancellation must still reach the caller."""
    try:
        adapter._release_platform_lock()
    except Exception:
        logger.warning("[Slack] Releasing the scoped platform lock failed", exc_info=True)
