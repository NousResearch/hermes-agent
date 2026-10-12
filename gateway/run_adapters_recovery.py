"""Recovery for a secondary profile's platform whose adapter was never built.

Split out of ``gateway/run_adapters.py`` (whose size ratchet is at its target): both helpers belong
to one lane — a secondary platform whose ``_create_adapter`` returned ``None`` at boot, classified,
parked for reconnect, and (when the registry holds no registration for it) re-armed so a later
import can still succeed (#137268).
"""
from __future__ import annotations

import asyncio
import functools
import logging

from gateway.config import Platform
from gateway.run_shutdown import _log_suppressed

logger = logging.getLogger("gateway.run")


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
