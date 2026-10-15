"""The attached classic view as the plugin manager's CLI slot.

Main's classic REPL registered itself as ``PluginManager._cli_ref``, so ``ctx.inject_message(text)``
with no ``session_key``/``origin`` (the documented "Active CLI conversation" form) queued into the
REPL. The classic chat is now a gateway client: the same call submits into the attached session at
the gateway (same FIFO as a typed line) from whatever thread the plugin calls on."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class _InjectSlot:
    _agent_running = False  # one queue: the gateway FIFO orders it behind a running turn

    def __init__(self, view, loop):
        self._view, self._loop = view, loop
        self._pending_input = self._interrupt_queue = self

    def put(self, message):
        asyncio.run_coroutine_threadsafe(self._submit(message), self._loop)

    async def _submit(self, message):
        print(f"\n{'─' * 40}\n● {message}", flush=True)
        try:
            await self._view.submit(message)
        except Exception:
            logger.warning("plugin inject_message: submit to %s failed", self._view.session_id, exc_info=True)


@contextmanager
def plugin_inject_slot(view):
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    slot = manager._cli_ref = _InjectSlot(view, asyncio.get_running_loop())
    try:
        yield
    finally:
        if manager._cli_ref is slot:
            manager._cli_ref = None
