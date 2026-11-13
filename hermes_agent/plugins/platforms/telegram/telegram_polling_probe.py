"""Telegram pending-queue heartbeat probe: a getUpdates consumer that is wedged or stopped while the
general request path stays healthy is only visible through ``getWebhookInfo.pending_update_count``."""

import asyncio
import logging

logger = logging.getLogger("hermes_agent.plugins.platforms.telegram.adapter")


class TelegramPendingProbeMixin:
    """Stuck-consumer probe for ``TelegramAdapter``'s polling heartbeat, plus the conflict quiet window."""

    _polling_conflict_quiet_banked = 0.0  # healthy polling seconds proven in earlier generations since the last 409

    async def _probe_pending_updates(self, bot, probe_timeout: float) -> bool:
        """Detect a wedged or stopped getUpdates consumer via pending_update_count.

        Returns True when this probe observed a first stuck signal (queued updates the poller isn't
        consuming, or a stopped updater) that the heartbeat loop should re-check promptly — after
        ``STUCK_CONFIRM_INTERVAL``, not a full heartbeat interval, so the confirming probe can land in
        the same awake window on power-napping hosts (#139281).

        PTB can report ``updater.running`` while the long-poll is stuck; get_me() stays healthy yet DMs queue in the
        Bot API. A stuck queue over two probes ⇒ dead consumer. Also covers the updater having stopped
        entirely (``running=False``, no reconnect in flight).

        PTB can report ``updater.running == True`` while its long-poll task is silently stuck (e.g. a socket
        that epoll keeps reporting readable on WSL2). ``get_me()`` stays healthy because it uses the general
        request path, so the CLOSE-WAIT heartbeat never fires — yet DMs queue in the Bot API and never reach
        handlers (#42909).
        We detect the stopped updater directly and feed the same ladder (#55769).
        """
        # Polling mode only: in webhook mode Telegram pushes and holds no server-side queue.
        if self._teardown_started or self._webhook_mode:
            return False
        # An in-flight reconnect owns recovery — don't double-trigger, and don't misread its brief
        # stop()->start_polling() window (updater.running transiently False) as dead.
        if self._recovery_in_flight():
            self._polling_not_running_count = 0
            return False
        updater = getattr(self._app, "updater", None) if self._app else None
        if updater is None:
            self._polling_pending_stuck_count = 0
            return False
        if not getattr(updater, "running", False):
            # Long-poll task gone, general-path calls still succeed, so no error_callback/probe ever
            # fires. Debounced over two probes so a just-starting updater never trips it.
            self._polling_pending_stuck_count = 0
            # We are in polling mode with no reconnect in flight, yet PTB's updater has stopped entirely.
            # This is distinct from the wedged-but-running consumer handled below: the long-poll task is
            # gone, get_me()/get_webhook_info() on the general request path still succeed, so no
            # error_callback or connectivity probe ever fires and the gateway silently stops receiving
            # messages while the process stays alive (#55769).
            self._polling_not_running_count += 1
            logger.warning(
                "[%s] Telegram polling heartbeat: updater stopped while in polling mode (stuck probe %d/2)", self.name,
                self._polling_not_running_count)
            if self._polling_not_running_count >= 2:
                self._polling_not_running_count = 0
                self._escalate_stuck_consumer(
                    "[%s] Telegram updater is not running (long-poll task gone); triggering polling restart",
                    "Telegram updater stopped while in polling mode")
                return False
            return True
        self._polling_not_running_count = 0
        get_webhook_info = getattr(bot, "get_webhook_info", None)
        if not callable(get_webhook_info):
            return False
        try:
            info = await asyncio.wait_for(get_webhook_info(), probe_timeout)  # type: ignore[arg-type]
        except (TimeoutError, OSError):
            return False  # connectivity symptom for the get_me() path, not a stuck-queue signal
        pending = int(getattr(info, "pending_update_count", 0) or 0)
        if pending <= 0:
            self._polling_pending_stuck_count = 0
            return False
        self._polling_pending_stuck_count += 1
        logger.warning(
            "[%s] Telegram polling heartbeat: %d update(s) queued but not consumed (stuck probe %d/2)", self.name,
            pending, self._polling_pending_stuck_count)
        if self._polling_pending_stuck_count >= 2:
            self._polling_pending_stuck_count = 0
            self._escalate_stuck_consumer(
                "[%s] getUpdates consumer appears wedged (queue not draining); triggering polling restart",
                "getUpdates consumer wedged: pending updates not draining")
            return False
        return True

    def _bank_conflict_quiet_window(self) -> None:
        """A new polling generation pauses the conflict quiet window: the healthy polling proven so far stays banked and
        the recovery downtime before the next progress is not counted. Only a 409 empties the bank, so network recoveries
        closer together than the window can no longer keep an episode open for hours (#151829)."""
        since, last = self._polling_conflict_quiet_since, getattr(self, "_polling_last_progress_monotonic", None)
        if since is not None and last is not None:
            self._polling_conflict_quiet_banked += last - since
        self._polling_conflict_quiet_since = None

    def _note_conflict_quiet_progress(self, now: float) -> None:
        """End a conflict episode after a quiet period of getUpdates progress (#82303): getUpdates also succeeds
        between the 409s of a live storm, so any-progress resets pinned the ladder at (1/5). The window reopens at
        each generation's first progress, so stop/backoff/drain downtime never counts as healthy polling."""
        from hermes_agent.plugins.platforms.telegram.adapter import _POLLING_CONFLICT_QUIET_PERIOD
        if not self._polling_conflict_count:
            return
        since = self._polling_conflict_quiet_since
        if since is None:
            self._polling_conflict_quiet_since = now
        elif self._polling_conflict_quiet_banked + now - since >= _POLLING_CONFLICT_QUIET_PERIOD:
            self._polling_conflict_count = 0
            self._polling_conflict_quiet_since, self._polling_conflict_quiet_banked = None, 0.0

    def _escalate_stuck_consumer(self, log_message: str, reason: str) -> None:
        """Second consecutive stuck probe: restart polling via the network-error ladder (unless tearing down)."""
        if self._teardown_started:
            return
        logger.warning(log_message, self.name)
        self._polling_error_task = asyncio.get_running_loop().create_task(self._handle_polling_network_error(RuntimeError(reason)))
