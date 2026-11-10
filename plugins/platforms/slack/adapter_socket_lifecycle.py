"""Socket Mode lifecycle pieces of ``SlackAdapter``: watchdog cancel and the drain-time inbound stop."""

import asyncio
import logging

logger = logging.getLogger("plugins.platforms.slack.adapter")


class SlackSocketLifecycleMixin:
    async def _cancel_socket_watchdog(self, failure_msg: str) -> None:
        """Cancel and await the watchdog task (if any); exceptions are debug-logged."""
        watchdog_task = self._socket_watchdog_task
        self._socket_watchdog_task = None
        if watchdog_task is None or watchdog_task.done():
            return
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(failure_msg, exc_info=True)

    async def stop_inbound(self) -> None:
        """Gateway drain: close Socket Mode while the message handler is still installed (#143122).

        Bolt acks an envelope before the listener runs, so one received after the handler
        withdrawal is dropped and never redelivered. ``_running=False`` first keeps the watchdog and
        the task-done callback from reconnecting; the reconnect lock waits out a restart already past
        its ``_running`` check, so its replacement socket is the one closed here. Web API clients
        (egress) stay open.
        """
        self._running = False
        await self._cancel_socket_watchdog("[Slack] Watchdog task raised during stop_inbound")
        async with self._socket_reconnect_lock:
            await self._stop_socket_mode_handler()
