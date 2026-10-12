"""Recovery for a secondary profile's platform whose adapter was never built.

Split out of ``gateway/run_adapters.py`` (whose size ratchet is at its target): these members
belong to one lane — a secondary platform whose ``_create_adapter`` returned ``None`` at boot,
classified, parked for reconnect, and (when the registry holds no registration for it) re-armed so
a later import can still succeed (#137268).
"""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import TYPE_CHECKING, Optional

from gateway.config import Platform
from gateway.restart import is_global_startup_conflict
from gateway.run_shutdown import _log_suppressed

if TYPE_CHECKING:  # annotations only; importing the adapters at runtime would cycle
    from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger("gateway.run")


class SecondaryReconnectSchedulingMixin:
    """Queue a secondary profile's platform for reconnect — the half of that lane that only parks
    work. The attempt, the watcher and the re-arm live with the helpers below and in
    ``run_adapters``; these two methods schedule the task and own the cold-start handoff."""

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


def queue_unbuilt_secondary_platform(
    runner, profile_name: str, platform: Platform, platform_config, profile_home
) -> None:
    """Park a secondary platform whose adapter was never built for reconnect, when it can heal.

    Only an unregistered plugin can: the registry forgot it (its load failed or overran its deadline),
    so a later import may still succeed. The queued entry is also the claim the runtime re-scan defers
    to, so the platform is never built twice. Anything else — a builtin probe failure, a credential-less
    config, a registered plugin still returning None — needs a config change and stays a plain skip:
    retrying it would only re-warn at the backoff cap (#5196).
    """
    from gateway.run import _profile_runtime_scope
    # Classification runs in the profile's own scope: plugin registration is scope-keyed, so
    # "unregistered here" is exactly the transient-load case that can heal.
    with runner._scope_or_null(
            functools.partial(_profile_runtime_scope, hydrate_secrets=False), profile_home):
        if not runner._adapter_may_heal(platform, platform_config):
            return
    logger.warning(
        "[MULTIPLEX] Profile '%s': platform '%s' has no registered adapter — no plugin registered it "
        "(its load failed or overran its deadline); queued for reconnect",
        profile_name, platform.value,
    )
    runner._schedule_secondary_profile_startup_reconnect(profile_name, platform, None)


async def rearm_secondary_plugin_load(
    runner, profile_name: str, platform: Platform, queue_info: dict
) -> None:
    """Re-arm a failed plugin load for a queued secondary platform (capped), so the next reconnect
    tick's import can succeed. Mirrors the primary queue's ``adapter_unavailable`` branch: a plugin
    whose load died transiently comes back, a permanently broken one stops re-importing."""
    from gateway.run import _profile_runtime_scope
    from gateway.run_adapters import _MAX_PLUGIN_LOAD_REARMS
    from hermes_cli.plugins import get_plugin_manager
    if queue_info.get("load_rearms", 0) >= _MAX_PLUGIN_LOAD_REARMS:
        return
    # Off the loop and under the profile's own scope: the re-arm takes the manager's discovery lock
    # (which a sweep or a deferred load holds) and plugin state is per profile.
    profile_home = runner._routed_profile_home(profile_name)
    with runner._scope_or_null(
            functools.partial(_profile_runtime_scope, hydrate_secrets=False), profile_home):
        with _log_suppressed(
            logging.DEBUG, "plugin load re-arm for %s failed (profile: %s)", platform.value, profile_name
        ):
            if await asyncio.to_thread(get_plugin_manager().rearm_failed_platform, platform.value):
                queue_info["load_rearms"] = queue_info.get("load_rearms", 0) + 1
