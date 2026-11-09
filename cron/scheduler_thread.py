"""Supervised daemon thread for the in-process cron ticker.

The gateway (and the Desktop backend) run ``InProcessCronScheduler.start`` on a bare daemon thread.
Every guard inside ``start`` keeps the loop alive on a per-tick failure, but nothing outside it could
notice a thread that had already ended — the gateway kept serving while ``ticker_heartbeat`` froze and
no job fired again until a restart (#111010). The supervisor is the missing outer layer: the
housekeeping loop asks it once a cycle to respawn a ticker that died while shutdown was not requested.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


class SupervisedTickerThread:
    """``threading.Thread``-shaped handle whose ``restart_if_dead`` respawns a dead ticker."""

    def __init__(self, target: Callable[..., Any], *, args: tuple = (), kwargs: Mapping[str, Any] | None = None,
                 stop_event: threading.Event, name: str = "cron-scheduler",
                 restart_on_return: bool | None = None) -> None:
        self._target, self._args, self._kwargs = target, args, dict(kwargs or {})
        self._stop_event, self._name = stop_event, name
        # Chronos start() arms one-shots and RETURNS by design. Re-running it every housekeeping
        # tick would reconcile against NAS again. The in-process ticker is the only local trigger,
        # so a normal return while shutdown was not requested is still a dead ticker.
        self._crashed = False
        if restart_on_return is None:
            from cron.scheduler_provider import InProcessCronScheduler
            restart_on_return = isinstance(getattr(target, "__self__", None), InProcessCronScheduler)
        self._restart_on_return = restart_on_return
        self._thread = self._spawn()
        self.restarts = 0

    def _run(self) -> None:
        try:
            self._target(*self._args, **self._kwargs)
        except BaseException:
            self._crashed = True
            raise

    def _spawn(self) -> threading.Thread:
        self._crashed = False
        return threading.Thread(target=self._run, daemon=True, name=self._name)

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def restart_if_dead(self) -> bool:
        """Respawn a ticker that ended without ``stop_event``.

        An exception always counts. A normal return counts only for the in-process ticker.
        True when a restart happened.
        """
        if self._stop_event.is_set() or self._thread.is_alive():
            return False
        if not self._crashed and not self._restart_on_return:
            return False
        self.restarts += 1
        if self._crashed:
            detail = "check errors.log for the escaping exception."
        else:
            detail = "it returned while shutdown was not requested."
        logger.error(
            "Cron ticker thread %r died without a stop request; restarting (restart #%d). Scheduled jobs "
            "did not fire while it was down — %s",
            self._name, self.restarts, detail,
        )
        self._thread = self._spawn()
        self._thread.start()
        return True
