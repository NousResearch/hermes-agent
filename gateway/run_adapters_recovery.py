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

    def _mark_adapter_unavailable(
        self, status_key: str, platform: Platform, *, retrying: bool, needs_attention: Optional[bool] = None,
    ) -> None:
        """Stamp ``adapter_unavailable`` runtime status (``retrying`` or ``fatal``) with the actionable hint, for
        an enabled platform whose ``_create_adapter`` returned None. Shared by both lanes' startup and retries."""
        self._update_platform_runtime_status(
            status_key, platform_state="retrying" if retrying else "fatal", error_code="adapter_unavailable",
            error_message=_adapter_unavailable_message(platform, retrying=retrying), needs_attention=needs_attention,
        )

    def _stand_down_unbuilt_secondary(self, profile_name: str, platform: Platform, reason: str) -> None:
        """A never-built platform whose retry gave up for a config reason (disabled, credential removed): replace
        the ``retrying`` stamp startup gave it, or status keeps promising a retry nothing runs."""
        self._update_platform_runtime_status(
            f"{profile_name}:{platform.value}", platform_state="disabled", error_code=None, error_message=reason,
            needs_attention=False,
        )

    async def _rearm_failed_plugin_load(self, platform: Platform, info: dict) -> None:
        """Re-arm a failed plugin load for a queued platform, capped per queue entry, so the next attempt's
        import can succeed. The CALLER's runtime scope picks the plugin manager (one per profile home).
        Off the loop: the re-arm takes the manager's discovery lock, which a sweep or deferred load holds."""
        from hermes_cli.plugins import get_plugin_manager
        rearms = info.get("load_rearms", 0)
        if rearms < _MAX_PLUGIN_LOAD_REARMS and await self._run_in_executor_with_context(
                get_plugin_manager().rearm_failed_platform, platform.value):
            info["load_rearms"] = rearms + 1

    def _claim_secondary_platform(
        self, profile_name: str, profile_home, platform: Platform, adapter: BasePlatformAdapter,
        claimed: dict[tuple, str],
    ) -> Optional[tuple]:
        """Refuse a secondary adapter whose credential/listener another profile owns (a second poller on
        one bot, a second bind on one port); otherwise record the profile as the bot's owner and return
        its ``(credential_claim, listener_claim)``. None = refused (parked fatal)."""
        credential_claim, listener_claim = self._adapter_claims(platform, adapter)
        owner_name = claimed.get(credential_claim) if credential_claim is not None else None
        owner_origin, incoming_origin = (None, None)
        if owner_name:
            owner_origin, incoming_origin = self._duplicate_credential_origins(
                owner_name, profile_name, profile_home, platform, adapter,
            )
        if self._refuse_duplicate_claim(
            credential_claim, claimed, profile_name, platform, "credential",
            owner_origin=owner_origin, incoming_origin=incoming_origin,
        ) or self._refuse_duplicate_claim(listener_claim, claimed, profile_name, platform, "listener"):
            return None
        # The profile owns this bot from here on, connected or not: cron and the delivery drain read
        # ownership here from another thread, never from the live map a fatal error rewrites.
        bot_platforms = getattr(self, "_profile_bot_platforms", None)
        if bot_platforms is None:
            bot_platforms = self._profile_bot_platforms = {}
        bot_platforms.setdefault(profile_name, set()).add(platform)
        return credential_claim, listener_claim

    def _disown_secondary_platform(self, profile_name: str, platform: Platform) -> None:
        """Undo ``_claim_secondary_platform``'s ownership record for a never-built platform that gave up for good:
        cron and the delivery drain read it to decide that the profile owns the bot."""
        (getattr(self, "_profile_bot_platforms", None) or {}).get(profile_name, set()).discard(platform)

    def _adapter_claims(self, platform: Platform, adapter: BasePlatformAdapter) -> tuple[Optional[tuple], Optional[tuple]]:
        """``(credential_claim, listener_claim)`` of one adapter; either may be None. Never raises: unserve and a
        fatal teardown compute it after the adapter left the map."""
        return self._adapter_credential_claim(platform, adapter), self._adapter_listener_claim(platform, adapter)

    def _hold_secondary_claims(self, profile_name: str, platform: Platform, owner, claims: tuple) -> None:
        """Hold a secondary platform's credential/listener claims while it is not in ``_profile_adapters``
        (initial connect, failed boot connect awaiting the handoff, fatal teardown, unserve teardown, queued
        to reconnect, or mid-heal), so no other profile takes the same bot meanwhile. A task owner's hold is
        released when that task finishes, however it finishes (cancelled before it ever ran included); an
        adapter owner (being connected or torn down) releases its own, so it never drops a newer hold on the
        same platform. Unserve drops all of the profile's holds once its adapters are torn down."""
        held = getattr(self, "_secondary_held_claims", None)
        if held is None:
            held = self._secondary_held_claims = {}
        key = (profile_name, platform)
        previous = held.get(key)
        held[key] = (owner, claims)
        if isinstance(owner, asyncio.Future) and (previous is None or previous[0] is not owner):
            owner.add_done_callback(lambda done: self._release_secondary_claims(profile_name, platform, done))

    def _release_secondary_claims(self, profile_name: str, platform: Platform, owner) -> None:
        """Drop the hold on (profile, platform) if ``owner`` still owns it (a newer owner keeps its hold)."""
        held = getattr(self, "_secondary_held_claims", None) or {}
        entry = held.get((profile_name, platform))
        if entry is not None and entry[0] is owner:
            del held[(profile_name, platform)]

    def _held_claim_owners(self, exclude: Optional[tuple] = None) -> dict[tuple, str]:
        """Claim -> profile for every held secondary claim but the ``exclude`` (profile, platform) hold."""
        held = getattr(self, "_secondary_held_claims", None) or {}
        return {
            claim: key[0] for key, (_owner, claims) in held.items() if key != exclude
            for claim in claims if claim is not None
        }

    def _current_claim_owners(self, exclude: Optional[tuple] = None) -> dict[tuple, str]:
        """Who owns each bot/listener right now: held claims (but ``exclude``'s) plus everything live. Read at
        the moment of a claim check, never from a snapshot taken before an await (an owner may connect or let
        go meanwhile)."""
        active = getattr(self, "_primary_profile_name", None) or "default"
        return {**self._held_claim_owners(exclude), **self._live_resource_claims(active)}

    def _note_unbuilt_secondary_platform(self, profile_name: str, platform: Platform, heals: bool) -> None:
        """A secondary platform whose ``_create_adapter`` returned None at startup: say why, stamp its
        ``<profile>:<platform>`` status, and queue it when it can heal (no plugin registered it in this
        profile's scope, so a later import can still succeed). Anything else needs a config change:
        retrying it would only re-warn at the backoff cap (#5196)."""
        pv = platform.value
        if heals:
            logger.warning(
                "[MULTIPLEX] Profile '%s': no plugin registered platform '%s' - is the plugin installed? (a load "
                "that failed or overran its deadline also leaves it unregistered); retrying in the background",
                profile_name, pv,
            )
        else:
            logger.warning(
                "[MULTIPLEX] Profile '%s': skipping platform '%s' - adapter creation returned None",
                profile_name, pv,
            )
        self._mark_adapter_unavailable(f"{profile_name}:{pv}", platform, retrying=heals, needs_attention=True)
        if heals:
            self._schedule_secondary_profile_startup_reconnect(profile_name, platform, None)

    async def _secondary_reconnect_attempt(self, profile_name: str, platform: Platform, predecessor=None):
        """One scoped attempt to rebuild+connect a secondary adapter → ``(adapter, success)``;
        ``(None, None)`` = give up for good (disabled, credential removed, refused as a duplicate, or
        adapter unavailable for a reason only a config change fixes); ``(None, False)`` = no adapter yet
        but the platform can still heal (no plugin registered it, so a later import can succeed). Caller
        tears down a RETURNED adapter; one whose configure/connect raised is torn down here."""
        from gateway.platform_registry import platform_registry
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
                if predecessor is None:
                    self._stand_down_unbuilt_secondary(profile_name, platform, "disabled in config")
                return None, None
            # Startup credential gate mirror: a removed credential must not rebuild.
            # Mirrors the startup credential gate (#84079): a credential removed from this profile's scope
            # must not rebuild an adapter that would fan out turns.
            if not _platform_has_bot_credential(platform, profile_config):
                logger.info(
                    "Secondary %s reconnect skipped: no bot credential (profile: %s)",
                    platform.value, profile_name,
                )
                if predecessor is None:
                    self._stand_down_unbuilt_secondary(profile_name, platform, "no bot credential configured")
                return None, None
            # A re-armed plugin import + register() joins its load deadline: run it off the event loop,
            # in this profile's scope (the registry keeps deferred loaders per profile home).
            await self._run_in_executor_with_context(platform_registry.get, platform.value)
            adapter = await self._create_adapter_off_loop_deps(platform, profile_config)
            if adapter is None:
                if self._adapter_may_heal(platform, profile_config):
                    logger.info("Secondary %s: no adapter yet (profile: %s)", platform.value, profile_name)
                    return None, False
                logger.warning(
                    "Secondary %s reconnect skipped: adapter unavailable (profile: %s)",
                    platform.value, profile_name,
                )
                self._mark_adapter_unavailable(
                    f"{profile_name}:{platform.value}", platform, retrying=False, needs_attention=True)
                return None, None
            if predecessor is None:
                # Queued without an adapter, so it never passed the startup ownership check: run it before
                # connect() against live and held owners (but this platform's own hold from an earlier tick),
                # and hold the claim (no await between the check and the hold) so a second healing profile
                # with the same bot is refused meanwhile.
                claims = self._claim_secondary_platform(
                    profile_name, profile_home, platform, adapter,
                    self._current_claim_owners(exclude=(profile_name, platform)))
                if claims is None:
                    return None, None
                self._hold_secondary_claims(profile_name, platform, asyncio.current_task(), claims)
            else:  # re-hold what the rebuilt adapter claims now (its token may have changed; as on main, a
                # reconnect with a predecessor is not re-checked against other owners)
                self._hold_secondary_claims(
                    profile_name, platform, asyncio.current_task(), self._adapter_claims(platform, adapter))
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

    def _end_secondary_reconnect(self, profile_name: str, platform: Platform, predecessor, adapter, success) -> bool:
        """Decide whether a reconnect ends after an attempt that installed nothing, and settle ownership: it
        gave up for good (``(None, None)``: needs a config change), its connected adapter was not installed (a
        newer task won the slot, or shutdown began), or it hit a non-retryable fatal. A never-built platform
        that ends for good (not the connected case) drops its bot ownership, which cron and the delivery drain
        read."""
        if adapter is None:
            final = success is None
        elif success:
            return True  # connected but not installed: a newer task owns the platform, or shutdown began
        else:
            final = getattr(adapter, "has_fatal_error", False) and not getattr(adapter, "fatal_error_retryable", True)
        if final and predecessor is None:
            self._disown_secondary_platform(profile_name, platform)
        return final

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
                        if self._end_secondary_reconnect(profile_name, platform, predecessor, None, success):
                            return
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
                        if self._end_secondary_reconnect(profile_name, platform, predecessor, adapter, success):
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
                    flagged = self._flag_reconnect_needs_attention(
                        platform, queue_info, time.monotonic(), status_key=f"{profile_name}:{platform.value}")
                    if flagged and rearm:  # the flag rewrote the entry without an error code: put the hint back
                        self._mark_adapter_unavailable(f"{profile_name}:{platform.value}", platform, retrying=True)
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
        if adapter is not None and is_global_startup_conflict(getattr(adapter, "fatal_error_code", None)):
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

        # The bridge belongs to this serve of the profile: once it is unserved (its map popped) or served again
        # (a new map), a bridge still waiting for startup to finish hands nothing over.
        profile_map = (getattr(self, "_profile_adapters", None) or {}).get(profile_name)

        def _handoff() -> None:
            if (getattr(self, "_profile_adapters", None) or {}).get(profile_name) is not profile_map:
                return
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

        bridge = self._retain_background_task(asyncio.create_task(
            _await_running_then_schedule(),
            name=f"secondary-startup-reconnect:{profile_name}:{platform.value}",
        ))
        if adapter is not None:  # it owns its bot from the boot sweep on; the handoff passes the hold on
            self._hold_secondary_claims(profile_name, platform, bridge, self._adapter_claims(platform, adapter))

    def _schedule_secondary_profile_reconnect(
        self, profile_name: str, platform: Platform, adapter: Optional[BasePlatformAdapter] = None
    ) -> None:
        """Schedule one runner-owned reconnect without sharing primary secrets. ``adapter`` is the
        failed predecessor, or None when the platform has no adapter yet (creation returned None for a
        platform that can still heal): the queued entry stops the runtime re-scan from building the same
        platform twice, and the attempt claims the bot before it connects. A profile being unserved
        (``_retiring_profiles``) gets none: its unserve has already popped the reconnects it cancels."""
        if (
            not self._running
            or not getattr(adapter, "fatal_error_retryable", True)
            or profile_name in (getattr(self, "_retiring_profiles", None) or ())
        ):
            return
        pending = self._profile_failed_platforms
        if not isinstance(pending, dict):
            pending = self._profile_failed_platforms = {}
        profile_pending = pending.setdefault(profile_name, {})
        if platform in profile_pending:
            return
        task = profile_pending[platform] = self._retain_background_task(asyncio.create_task(
            self._run_secondary_profile_reconnect(profile_name, platform, adapter),
            name=f"secondary-reconnect:{profile_name}:{platform.value}",
        ))
        if adapter is not None:  # the failed predecessor still owns its bot while it waits to reconnect
            self._hold_secondary_claims(profile_name, platform, task, self._adapter_claims(platform, adapter))
