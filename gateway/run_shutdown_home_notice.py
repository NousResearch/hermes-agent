"""Home-channel half of the shutdown notice for GatewayRunner.

Split out of ``gateway/run_shutdown.py``; bound onto ``GatewayRunner`` via ``GatewayShutdownMixin``.
The broadcast is the one lifecycle message nobody asked for, so it is the one that must stay quiet when
a stop is not an outage (a ``--replace`` takeover) and when stops repeat in a loop, and the one that owes
a "Gateway online" on the next boot once it was delivered.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Optional

logger = logging.getLogger("gateway.run")

# Outcome of this stop's home-channel broadcast, read by ``_stop_persist_exit_state``.
HOME_NOTICE_SENT = "sent"
HOME_NOTICE_MUTED = "muted"


def settle_home_online_notice_debt(runner) -> None:
    """Make the next boot's "Gateway online" match this stop's "shutting down".

    A home chat told the gateway is going down is owed the matching "back online", whatever stopped
    it: a supervisor restart (``systemctl restart``, launchd, Docker) sets no restart flag and the CLI
    stamps nothing, so the next boot stayed silent after a delivered goodbye. A loop-muted broadcast
    owes nothing, and drops a marker the CLI already stamped, so the loop stays quiet on both ends.
    """
    from gateway.restart_notice import planned_restart_marker_path, write_planned_restart_marker
    outcome = getattr(runner, "_home_shutdown_notice", None)
    if outcome == HOME_NOTICE_MUTED:
        # Best-effort like the write: an undeletable marker costs one extra notice, never the stop.
        try:
            planned_restart_marker_path().unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not drop the planned-restart marker after a loop-muted stop", exc_info=True)
    elif outcome == HOME_NOTICE_SENT and not runner._restart_requested:
        write_planned_restart_marker()
    if outcome == HOME_NOTICE_SENT and getattr(runner, "_home_notice_loop_checked", False):
        # The loop cap counts goodbyes users saw, not stops that tried (a cycle with no adapter up).
        from gateway import restart_loop_guard as rlg
        rlg.record_notice_delivered()
    runner._home_notice_loop_checked = False


class GatewayHomeShutdownNoticeMixin:
    """Broadcast "shutting down" to every served profile's home channel, loop- and takeover-aware."""

    # Set by the signal handler when SIGTERM is a planned ``--replace`` takeover (gateway/run_bootstrap.py).
    _planned_takeover_shutdown: bool = False
    _home_shutdown_notice: Optional[str] = None
    _home_notice_loop_checked: bool = False  # this stop was linked into the notice chain

    def _home_shutdown_notice_muted_by_loop(self) -> bool:
        """Record this broadcast in the persisted notice chain; True once the chain is a loop.

        A supervisor that keeps killing the gateway (a stuck post-update helper, two services fighting
        over one home) otherwise posts one notice per cycle into the home chat, thousands overnight. The
        chain and its limits are the restart-loop breaker's (``gateway.restart_loop_guard``)."""
        from gateway import restart_loop_guard as rlg
        max_restarts, window, max_gap = self._restart_loop_guard_config()
        self._home_notice_loop_checked = True
        return rlg.notice_loop_tripped(max_restarts, window, max_gap_seconds=max_gap)

    def _home_shutdown_broadcast_skip_reason(self) -> Optional[str]:
        """Why this stop sends no home-channel notice, or None to broadcast."""
        if self._restart_requested and self._restart_command_source is not None:
            return "in-chat restart"
        if self._planned_takeover_shutdown:
            # Another gateway is taking this home over: the service never went away.
            return "planned --replace takeover"
        # A quiet drain (routine fleet auto-update) suppresses ONLY the home-channel broadcast; per-session
        # pings stay. Current-epoch marker only; a failing check fails toward the louder behaviour.
        from gateway.run_shutdown import _log_suppressed
        with _log_suppressed(logging.DEBUG, "drain_notification_suppressed check failed: %s"):
            from gateway.drain_control import drain_notification_suppressed
            if drain_notification_suppressed():
                return "drain marker (suppress_notification=true)"
        with _log_suppressed(logging.DEBUG, "Lifecycle notice loop check failed: %s"):
            if self._home_shutdown_notice_muted_by_loop():
                self._home_shutdown_notice = HOME_NOTICE_MUTED
                return "restart loop (gateway.restart_loop_guard)"
        return None

    def _home_got_active_session_notice(self, delivered: set) -> bool:
        """True when the active-session pass already told a home chat (a task was running there)."""
        from gateway.run_shutdown import _delivery_target_key
        return any(
            _delivery_target_key(platform.value, cfg.home_channel.chat_id, cfg.home_channel.thread_id,
                                 profile=profile) in delivered
            for profile, platform, cfg in self._served_home_channel_configs()
            if cfg.home_channel and cfg.home_channel.chat_id
        )

    async def _notify_home_channels_of_shutdown(
        self, msg: str, notified: set, delivered: set, *, skip_reason: Optional[str]
    ) -> None:
        """Send ``msg`` to EVERY served profile's home channel through that profile's OWN bot.

        ``notified`` holds the targets the active-session pass already handled; ``delivered`` the subset
        that really got the notice (a home among them is owed "online" like one this pass reaches). ``self.adapters`` and
        ``self.config`` are the launch profile's alone, so iterating them left the secondaries' channels
        silent (#118233). ``list(...)`` snapshots the adapter maps: adapter.send() can hit a fatal path
        (_handle_fatal) that pops the adapter -> "dictionary changed size during iteration".
        """
        from gateway.run import _async_profile_runtime_scope
        from gateway.run_shutdown import _delivery_target_key
        from gateway.warning_notifications import present_notification

        if self._home_got_active_session_notice(delivered):
            self._home_shutdown_notice = HOME_NOTICE_SENT
        if skip_reason is not None:
            logger.info("Home-channel shutdown broadcast suppressed: %s", skip_reason)
            return
        profile_adapters = getattr(self, "_profile_adapters", None) or {}
        for profile, platform, platform_cfg in list(self._served_home_channel_configs()):
            home = platform_cfg.home_channel
            if not home or not home.chat_id:
                continue
            adapter = (self.adapters if profile is None else profile_adapters.get(profile) or {}).get(platform)
            if adapter is None or not self._notice_allowed(platform, "home channel", platform_cfg):
                continue
            dedup_key = _delivery_target_key(platform.value, home.chat_id, home.thread_id, profile=profile)
            if dedup_key in notified:
                continue
            try:
                metadata = self._thread_metadata_for_target(platform, home.chat_id, home.thread_id, adapter=adapter)
            except Exception as e:  # health: allow BLE001 -- moved verbatim from run_shutdown.py: one bad home channel never blocks the rest of shutdown
                logger.debug(
                    "Failed to send shutdown notification to home channel %s:%s: %s", platform.value, home.chat_id, e,
                )
                continue

            async def _send_home(adapter=adapter, home=home, platform=platform, metadata=metadata,
                                 dedup_key=dedup_key):
                if await self._send_shutdown_notice(
                    adapter, str(home.chat_id), msg, "home channel", platform.value, metadata=metadata,
                ):
                    notified.add(dedup_key)
                    self._home_shutdown_notice = HOME_NOTICE_SENT

            # present_notification reads the ACTIVE profile's display settings: bind the served one's.
            profile_home = (getattr(self, "_served_profile_homes", None) or {}).get(profile) if profile else None
            async with _async_profile_runtime_scope(profile_home) if profile_home else nullcontext():
                await present_notification(_send_home, platform=platform)
