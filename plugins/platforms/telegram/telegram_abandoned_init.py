"""Telegram connect cleanup: an ``initialize()`` whose connect was cancelled keeps one owner until its
transports are closed."""

from __future__ import annotations

import asyncio

# How long an abandoned initialize() may unwind before its transports are closed under it: long enough for
# a cancellation-shielded httpcore connect to return, bounded so a wedged one cannot hold the pools open.
_ABANDONED_INIT_UNWIND_TIMEOUT = 15.0


def retain_abandoned_initialize(adapter, init_task: asyncio.Future, app) -> None:
    """Own an ``initialize()`` abandoned by its caller's cancellation until the app's transports close.

    ``run_bounded_async`` cancels and abandons the child when the CALLER is cancelled (runner connect
    timeout, startup abort) but runs no ``on_abandon`` there, and ``app.shutdown()`` no-ops for an app
    that never finished initializing, so nothing else closes its httpx pools. The owner lets the child
    unwind first: ``HTTPXRequest.initialize`` rebuilds a closed client, so closing under a child still
    inside ``Bot.initialize`` leaves a pool no-one owns. The owner is kept out of ``_background_tasks``
    (base teardown cancels those) and the app is detached, so ``disconnect()`` waits on the owner
    instead of racing it."""
    owners = getattr(adapter, "_abandoned_initialize_owners", None)
    if owners is None:
        owners = adapter._abandoned_initialize_owners = set()
    owner = asyncio.get_running_loop().create_task(_close_after_initialize_exits(init_task, app))
    owners.add(owner)
    owner.add_done_callback(owners.discard)
    if adapter._app is app:
        adapter._app = adapter._bot = None


async def _close_after_initialize_exits(init_task: asyncio.Future, app) -> None:
    await asyncio.wait({init_task}, timeout=_ABANDONED_INIT_UNWIND_TIMEOUT)
    from plugins.platforms.telegram.adapter import _shutdown_abandoned_app

    await _shutdown_abandoned_app(app)


async def await_abandoned_initialize_owners(adapter, timeout: float) -> None:
    """Give retained initialize owners ``timeout`` to finish; never cancel them (they are the only
    thing that closes those transports)."""
    owners = [task for task in getattr(adapter, "_abandoned_initialize_owners", ()) if not task.done()]
    if owners:
        await asyncio.wait(owners, timeout=timeout)
