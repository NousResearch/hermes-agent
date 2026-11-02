"""The multiplex secondary-profile reconnect lane for GatewayRunner (mixin bound via the MRO).

A secondary profile's platform lands here when its connect failed, it went fatal mid-run, or its
adapter was never built because no plugin registered it in that profile (its load failed or overran
its deadline, #137268). One task per (profile, platform) rebuilds it under that profile's own runtime
scope until it connects or can no longer heal.

Split out of ``gateway/run_adapters.py`` (its size ratchet is at its target). ``gateway.run``
internals are imported lazily inside method bodies (import cycle), so ``patch("gateway.run.X")``
keeps intercepting them at call time.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Optional

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.helpers import carry_inbound_dedup, hand_over_held_inbound
from gateway.restart import is_global_startup_conflict
from gateway.run_shutdown import _log_suppressed

logger = logging.getLogger("gateway.run")
_MAX_PLUGIN_LOAD_REARMS = 3  # per queued platform: transient load failures heal, a broken plugin stops re-importing


def _adapter_unavailable_message(platform: Platform, *, retrying: bool = True) -> str:
    """Actionable ``adapter_unavailable`` status text, shared by startup and the reconnect watchers so
    ``hermes status`` keeps the plugin/deps/credentials hint after the first retry."""
    message = (
        f"No adapter available for enabled {platform.value}; check the plugin, dependencies, and credentials."
    )
    return f"{message} Retrying in the background." if retrying else message


class GatewaySecondaryReconnectMixin:
    """Queue, rebuild and reconnect a secondary profile's platform, plus the capped plugin-load re-arm
    that both reconnect lanes (launch profile and secondaries) share."""

    async def _rearm_failed_plugin_load(self, platform: Platform, info: dict) -> None:
        """Re-arm a failed plugin load for a queued platform, capped per queue entry, so the next attempt's
        import can succeed. The CALLER's runtime scope picks the plugin manager (one per profile home).
        Off the loop: the re-arm takes the manager's discovery lock, which a sweep or deferred load holds."""
        from hermes_cli.plugins import get_plugin_manager
        rearms = info.get("load_rearms", 0)
        if rearms < _MAX_PLUGIN_LOAD_REARMS and await self._run_in_executor_with_context(
                get_plugin_manager().rearm_failed_platform, platform.value):
            info["load_rearms"] = rearms + 1

    def _note_unbuilt_secondary_platform(self, profile_name: str, platform: Platform, heals: bool) -> None:
        """A secondary platform whose ``_create_adapter`` returned None at startup: queue it when it can
        heal (no plugin registered it in this profile's scope, so a later import can still succeed).
        Anything else needs a config change: retrying it would only re-warn at the backoff cap (#5196)."""
        logger.warning(
            "[MULTIPLEX] Profile '%s': skipping platform '%s' - adapter creation returned None",
            profile_name, platform.value,
        )
        if not heals:
            return
        logger.warning(
            "[MULTIPLEX] Profile '%s': platform '%s' has no registered adapter — no plugin registered it "
            "(its load failed or overran its deadline); queued for reconnect",
            profile_name, platform.value,
        )
        self._schedule_secondary_profile_startup_reconnect(profile_name, platform, None)

    async def _secondary_reconnect_attempt(self, profile_name: str, platform: Platform, predecessor=None):
        """One scoped attempt to rebuild+connect a secondary adapter → ``(adapter, success)``;
        ``(None, None)`` = give up for good (disabled, credential removed, or adapter unavailable for a
        reason only a config change fixes); ``(None, False)`` = no adapter yet
        but the platform can still heal (no plugin registered it, so a later import can succeed). Caller
        tears down a RETURNED adapter; one whose configure/connect raised is torn down here."""
        from gateway.run import _platform_has_bot_credential, _profile_runtime_scope
        # Lazy + per-attempt: keeps test monkeypatches on these modules live.
        from hermes_cli.profiles import get_profile_dir
        from hermes_cli.env_loader import hydrate_profile_secret_sources
        from gateway.config import load_gateway_config
        profile_home = get_profile_dir(profile_name)
        # Hydrate external secret sources off-loop so they cannot starve heartbeats.
        await asyncio.to_thread(hydrate_profile_secret_sources, profile_home)
        with _profile_runtime_scope(profile_home, hydrate_secrets=False):
            profile_config = load_gateway_config().platforms.get(platform)
            if profile_config is None or not profile_config.enabled:
                return None, None
            # Startup credential gate mirror: a removed credential must not rebuild.
            # Mirrors the startup credential gate (#84079): a credential removed from this profile's scope
            # must not rebuild an adapter that would fan out turns.
            if not _platform_has_bot_credential(platform, profile_config):
                logger.info(
                    "Secondary %s reconnect skipped: no bot credential (profile: %s)",
                    platform.value, profile_name,
                )
                return None, None
            adapter = self._create_adapter(platform, profile_config)
            if adapter is None:
                # An unregistered plugin (a load that failed transiently) can still heal, so report the
                # attempt as RETRYABLE and let the caller keep its slot and re-arm the load — mirroring
                # the primary queue's ``adapter_unavailable`` branch. Anything else (builtin probe
                # failure, registered plugin still returning None) needs a config change.
                healable = self._adapter_may_heal(platform, profile_config)
                logger.warning(
                    "Secondary %s reconnect skipped: adapter unavailable (profile: %s)%s",
                    platform.value, profile_name, " — will retry" if healable else "",
                )
                return (None, False) if healable else (None, None)
            carry_inbound_dedup(predecessor, adapter)
            try:
                self._configure_profile_adapter(adapter, profile_name, platform)
                success = await self._connect_adapter_with_timeout(adapter, platform, is_reconnect=True)
            except BaseException:
                # Caller never sees this adapter; release its partial resources and return its held inbound.
                await self._safe_adapter_disconnect(adapter, platform)
                hand_over_held_inbound(adapter, predecessor)
                raise
            return adapter, success

    async def _run_secondary_profile_reconnect(
        self, profile_name: str, platform: Platform, predecessor=None
    ) -> None:
        """Reconnect a retryable secondary adapter under its own profile scope."""
        from gateway.run import _profile_runtime_scope, _reconnect_backoff
        attempts = 0
        # Same escalation shape as the primary queue entry; ``queued_at`` is this task's start.
        queue_info = {"queued_at": time.monotonic(), "attempts": 0}
        current_task = asyncio.current_task()
        try:
            while self._running:
                adapter = None
                rearm = False
                try:
                    adapter, success = await self._secondary_reconnect_attempt(
                        profile_name, platform, predecessor
                    )
                    if adapter is None:
                        if success is None:
                            return  # gave up for good: needs a config change (disabled, credential, deps)
                        rearm = True  # no adapter yet but it can heal: re-arm the failed load, then retry
                    elif success and self._running:
                        profile_map = self._profile_adapters.setdefault(profile_name, {})
                        if platform not in profile_map:
                            profile_map[platform] = adapter
                            hand_over_held_inbound(predecessor, adapter)
                            self._sync_voice_mode_state_to_adapter(adapter)
                            logger.info("✓ %s reconnected (profile: %s)", platform.value, profile_name)
                            await self._redeliver_failed_obligations_for_platform(
                                platform, profile=profile_name
                            )
                            # What a primary reconnect replays too: the owed notice spans served profiles' home
                            # channels, and sessions boot skipped for this offline adapter wait for this call.
                            self._schedule_planned_restart_replay()
                            try:
                                self._schedule_resume_pending_sessions(platform=platform)
                            except Exception:
                                logger.debug("resume-pending reschedule after %s reconnect failed (profile: %s)",
                                             platform.value, profile_name, exc_info=True)
                            return
                    if adapter is not None:
                        # Not installed (newer reconnect won the slot, shutdown began, or connect failed):
                        # release partial resources and return its held inbound; stop only for a non-retryable fatal.
                        await self._safe_adapter_disconnect(adapter, platform)
                        hand_over_held_inbound(adapter, predecessor)
                        if success or (
                            getattr(adapter, "has_fatal_error", False)
                            and not getattr(adapter, "fatal_error_retryable", True)
                        ):
                            return
                except BaseException as exc:
                    if adapter is not None:
                        await self._safe_adapter_disconnect(adapter, platform)
                        if self._profile_adapters.get(profile_name, {}).get(platform) is not adapter:
                            hand_over_held_inbound(adapter, predecessor)  # an installed adapter owns its queue
                    if not isinstance(exc, Exception):
                        raise  # CancelledError (and other BaseExceptions) propagate after release
                    logger.debug(
                        "Secondary %s reconnect attempt failed (profile: %s)", platform.value,
                        profile_name, exc_info=True,
                    )
                if not self._running:
                    return
                attempts += 1
                queue_info["attempts"] = attempts
                profile_home = self._routed_profile_home(profile_name)
                # The attempt above already hydrated this profile's secret sources off-loop. The re-arm runs in
                # this scope too: the plugin manager whose load failed is this profile's, not the launch's.
                with self._scope_or_null(
                        functools.partial(_profile_runtime_scope, hydrate_secrets=False), profile_home):
                    if rearm:
                        with _log_suppressed(logging.DEBUG, "plugin load re-arm for %s failed (profile: %s): %s",
                                             platform.value, profile_name):
                            await self._rearm_failed_plugin_load(platform, queue_info)
                    self._flag_reconnect_needs_attention(
                        platform, queue_info, time.monotonic(), status_key=f"{profile_name}:{platform.value}")
                backoff = _reconnect_backoff(attempts)
                logger.info(
                    "Secondary %s reconnect retry in %ds (profile: %s)", platform.value, backoff, profile_name
                )
                await asyncio.sleep(backoff)
        finally:
            # Release our slot unless a newer task already owns it.
            pending = self._profile_failed_platforms
            profile_pending = pending.get(profile_name) if isinstance(pending, dict) else None
            if isinstance(profile_pending, dict):
                task = profile_pending.get(platform)
                if not isinstance(task, asyncio.Task) or task is current_task:
                    profile_pending.pop(platform, None)
                    if not profile_pending:
                        pending.pop(profile_name, None)

    def _schedule_secondary_profile_startup_reconnect(
        self, profile_name: str, platform: Platform, adapter: Optional[BasePlatformAdapter] = None
    ) -> None:
        """Queue a cold-start reconnect: startup failures happen BEFORE ``_running`` flips True (the
        regular scheduler would drop them), so park a task and hand off once live. ``adapter`` is None
        when nothing was built at all (a platform whose creation may still heal)."""
        if not getattr(adapter, "fatal_error_retryable", True):
            return
        if is_global_startup_conflict(getattr(adapter, "fatal_error_code", None)):
            # A live foreign token holder is an ownership conflict, not a blip: park it fatal.
            logger.error(
                # Park it fatal (like ``duplicate_credential``) instead of retry-storming the token every
                # backoff (#83183).
                "[MULTIPLEX] Profile '%s': %s credential is held by another "
                "gateway (%s) — parked, not retried. %s", profile_name, platform.value,
                adapter.fatal_error_code, adapter.fatal_error_message or "",
            )
            self._mark_platform_fatal(f"{profile_name}:{platform.value}", adapter)
            return

        def _handoff() -> None:
            try:
                self._schedule_secondary_profile_reconnect(profile_name, platform, adapter)
            except Exception:
                # A raise here would die as an unretrieved-task exception logged only at GC; surface it.
                logger.exception(
                    "secondary-startup-reconnect handoff failed (profile=%s platform=%s)",
                    profile_name, platform.value,
                )

        async def _await_running_then_schedule() -> None:
            if self._running:  # fast path (also the only path for bare runners without _shutdown_event)
                _handoff()
                return
            # Poll (startup completion has no event); bounded so a wedged startup cannot spin.
            while not self._running and not self._shutdown_event.is_set():
                await asyncio.sleep(0.1)
            if self._running and not self._shutdown_event.is_set():
                _handoff()

        self._retain_background_task(asyncio.create_task(
            _await_running_then_schedule(),
            name=f"secondary-startup-reconnect:{profile_name}:{platform.value}",
        ))

    def _schedule_secondary_profile_reconnect(
        self, profile_name: str, platform: Platform, adapter: Optional[BasePlatformAdapter] = None
    ) -> None:
        """Schedule one runner-owned reconnect without sharing primary secrets. ``adapter`` is the
        failed predecessor, or None when the platform has no adapter yet (creation returned None for a
        platform that can still heal): the queued entry is claim enough — it also stops the runtime
        re-scan from building the same platform twice."""
        if not self._running or not getattr(adapter, "fatal_error_retryable", True):
            return
        pending = self._profile_failed_platforms
        if not isinstance(pending, dict):
            pending = self._profile_failed_platforms = {}
        profile_pending = pending.setdefault(profile_name, {})
        if platform in profile_pending:
            return
        profile_pending[platform] = self._retain_background_task(asyncio.create_task(
            self._run_secondary_profile_reconnect(profile_name, platform, adapter),
            name=f"secondary-reconnect:{profile_name}:{platform.value}",
        ))
