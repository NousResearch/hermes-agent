"""Manage Telegram requests after ``initialize()`` is abandoned."""

from __future__ import annotations

import asyncio

# How long an abandoned initialize() may unwind before its transports are closed under it: long enough for
# a cancellation-shielded httpcore connect to return, bounded so a wedged one cannot hold the pools open.
_ABANDONED_INIT_UNWIND_TIMEOUT = 15.0


def abandonment_aware_request_class(request_class):
    """Let a discarded app close its requests without a later initialize reopening them."""
    class AbandonmentAwareRequest(request_class):
        __slots__ = ("_abandoned",)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._abandoned = False

        def abandon(self) -> None:
            self._abandoned = True

        async def initialize(self) -> None:
            if self._abandoned:
                return
            await super().initialize()
            if self._abandoned:
                await self.shutdown()

    return AbandonmentAwareRequest


def abandon_app_requests(app) -> None:
    bot = getattr(app, "bot", None)
    for request in (getattr(bot, "_request", None) if bot is not None else None) or ():
        abandon = getattr(request, "abandon", None)
        if abandon is not None:
            abandon()


def retain_abandoned_initialize(adapter, init_task: asyncio.Future, app) -> None:
    """Own an ``initialize()`` abandoned by its caller's cancellation until the app's transports close.

    ``run_bounded_async`` cancels and abandons the child when the CALLER is cancelled (runner connect
    timeout, startup abort) but runs no ``on_abandon`` there, and ``app.shutdown()`` no-ops for an app
    that never finished initializing, so nothing else closes its httpx pools. An initialize
    already in progress can still rebuild a client, so the owner also closes after the child
    exits. The owner is kept out of ``_background_tasks``
    (base teardown cancels those). ``disconnect()`` closes the detached app immediately, and the
    owner closes it again if the child later reopens its request transports."""
    owners = getattr(adapter, "_abandoned_initialize_owners", None)
    if owners is None:
        owners = adapter._abandoned_initialize_owners = {}
    abandon_app_requests(app)
    owner = asyncio.get_running_loop().create_task(_close_after_initialize_exits(init_task, app))
    owners[owner] = app
    owner.add_done_callback(lambda task: owners.pop(task, None))
    if adapter._app is app:
        adapter._app = adapter._bot = None


async def _close_after_initialize_exits(init_task: asyncio.Future, app) -> None:
    from plugins.platforms.telegram.adapter import _shutdown_abandoned_app

    async def close() -> None:
        while True:
            try:
                await _shutdown_abandoned_app(app)
                return
            except asyncio.CancelledError:
                # A second cancellation can land during request.shutdown() itself.
                continue

    deadline = asyncio.get_running_loop().time() + _ABANDONED_INIT_UNWIND_TIMEOUT
    try:
        while not init_task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait({init_task}, timeout=remaining)
            except asyncio.CancelledError:
                # asyncio.run cancels owners at exit, but the init child can still reopen a pool.
                continue
    finally:
        await close()

    if not init_task.done():
        # The first close bounds pool lifetime even for a wedged initialize(). If it eventually
        # resumes and rebuilds a request client, close that new client before releasing ownership.
        while not init_task.done():
            try:
                await asyncio.wait({init_task})
            except asyncio.CancelledError:
                continue
        await close()


async def close_abandoned_initialize_apps(adapter, timeout: float) -> None:
    """Close detached apps within disconnect's step budget, regardless of init unwind time."""
    apps = list(getattr(adapter, "_abandoned_initialize_owners", {}).values())
    if apps:
        from plugins.platforms.telegram.adapter import _close_app_requests

        for app in apps:
            abandon_app_requests(app)
        await adapter._await_disconnect_step(
            asyncio.gather(*(_close_app_requests(app) for app in apps), return_exceptions=True),
            timeout, "abandoned initialize request shutdown")
