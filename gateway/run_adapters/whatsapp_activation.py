"""Hot activation of a newly paired WhatsApp adapter on a running gateway (mixin bound via the MRO).

``hermes whatsapp`` and the dashboard call ``activate-whatsapp`` on the control socket after pairing;
only WhatsApp is (re)created, so other adapters and their in-flight turns keep running.
``gateway.run`` internals are imported lazily inside method bodies (import cycle).
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from gateway.config import Platform

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayWhatsAppActivationMixin:
    """Activate WhatsApp for the primary or a multiplexed profile without a gateway restart."""

    async def activate_whatsapp(self, profile_name: str) -> dict[str, Any]:
        """Activate WhatsApp for one served profile after its pairing session is saved."""
        if self._draining:
            return {"active": False, "reason": "gateway_stopping"}
        if not self._running:
            self.__dict__.setdefault("_pending_whatsapp_activation_profiles", set()).add(profile_name)
            return {"pending": True}
        primary = getattr(self, "_primary_profile_name", None) or "default"
        if profile_name == primary:
            return await self.activate_primary_whatsapp()
        if not self._multiplex_on():
            return {"active": False, "reason": "profile_not_served"}
        result = await self.reconcile_served_profiles(
            reason="control-socket", refresh_whatsapp=profile_name)
        if profile_name not in self.served_profile_names():
            return {"active": False, "reason": "profile_not_served"}
        active = Platform.WHATSAPP in (self._profile_adapters or {}).get(profile_name, {})
        retrying = Platform.WHATSAPP in (
            (getattr(self, "_profile_failed_platforms", None) or {}).get(profile_name) or {}) or (
                profile_name, Platform.WHATSAPP) in (getattr(self, "_secondary_held_claims", None) or {})
        return {"active": active, "pending": bool(result.get("pending")), "retrying": retrying}

    def _launch_pending_whatsapp_activations(self) -> list[asyncio.Task]:
        """Honor pairing requests received while the startup adapter set was still being built."""
        tasks = []
        for profile in self.__dict__.pop("_pending_whatsapp_activation_profiles", set()):
            task = self._retain_background_task(asyncio.create_task(self.activate_whatsapp(profile)))
            task.add_done_callback(self._late_failure_callback(
                f"WhatsApp activation after startup failed for profile {profile}"))
            tasks.append(task)
        return tasks

    async def activate_primary_whatsapp(self) -> dict[str, Any]:
        """Add a newly paired primary WhatsApp adapter without draining active turns.

        The onboarding process saves the session and config before calling this method. Keep the
        connect inside the owning home's secret scope, and serialize duplicate activation requests
        with profile reconciliation (which can start secondary WhatsApp adapters concurrently).
        """
        platform = Platform.WHATSAPP
        if not self._running or self._draining:
            return {"pending": True}
        async with self._reconcile_lock():
            from gateway.config import load_gateway_config
            from gateway.run import _async_profile_runtime_scope, _platform_has_bot_credential
            from hermes_constants import get_default_hermes_root, get_process_hermes_home

            home = (get_default_hermes_root() if self._multiplex_on()
                    else get_process_hermes_home())
            async with _async_profile_runtime_scope(Path(home)):
                config = load_gateway_config()
                platform_config = config.platforms.get(platform)
                if platform_config is None or not platform_config.enabled:
                    return {"active": False, "reason": "whatsapp_disabled"}
                if not _platform_has_bot_credential(platform, platform_config):
                    return {"active": False, "reason": "whatsapp_unpaired"}
                platform_config.extra.setdefault("group_sessions_per_user", self.config.group_sessions_per_user)
                platform_config.extra.setdefault(
                    "thread_sessions_per_user", getattr(self.config, "thread_sessions_per_user", False))
                current = self.adapters.get(platform)
                if current is not None:
                    from plugins.platforms.whatsapp.activation import whatsapp_activation_signature
                    session_path = getattr(current, "_session_path", Path(home) / "whatsapp" / "session")
                    same_pairing = getattr(current, "_activation_signature", None) == (
                        whatsapp_activation_signature(Path(home), session_path))
                    if self.config.platforms.get(platform) == platform_config and same_pairing:
                        return {"active": True, "already_active": True}
                    # Only WhatsApp is recycled. Other adapters and their chat turns keep running.
                    self.adapters.pop(platform)
                    await self._bounded_adapter_teardown(current, platform)
                    self.delivery_router.adapters = self.adapters
                # The reconnect watcher must use the newly saved configuration, not the boot snapshot.
                self.config.platforms[platform] = platform_config
                pending = self._failed_platforms.get(platform)
                if pending is not None:
                    pending["config"] = platform_config
                    pending["next_retry"] = time.monotonic()
                    pending["retry_now"] = True
                    if pending.get("paused"):
                        self._resume_paused_platform(platform)
                    return {"active": False, "retrying": True}
                adapter = self._create_adapter(platform, platform_config)
                if adapter is None:
                    self._mark_adapter_unavailable(platform.value, platform, retrying=False,
                                                   needs_attention=True)
                    return {"active": False, "reason": "adapter_unavailable"}
                self._wire_adapter_handlers(adapter)
                self._update_platform_runtime_status(
                    platform.value, platform_state="connecting", error_code=None, error_message=None)
                try:
                    connected = await self._connect_adapter_with_timeout(adapter, platform)
                except Exception as exc:
                    logger.warning("WhatsApp activation failed; queuing reconnect: %s", exc, exc_info=True)
                    connected = False
                    retryable = True
                else:
                    retryable = not adapter.has_fatal_error or adapter.fatal_error_retryable

                if connected:
                    self._publish_primary_adapter(platform, adapter)
                    self.delivery_router.adapters = self.adapters
                    degraded = adapter.send_path_degraded
                    self._update_platform_runtime_status(
                        platform.value, platform_state="retrying" if degraded else "connected",
                        error_code=None,
                        error_message=adapter.DEGRADED_STATUS_MESSAGE if degraded else None,
                        needs_attention=False, retrying_since=None)
                    if self._startup_parked_platforms:
                        from gateway.status_runtime_snapshot import primary_fatal_platforms_snapshot
                        if primary_fatal_platforms_snapshot() == set():
                            self._startup_parked_platforms = False
                            if not (self._external_drain_active or self._draining or not self._running):
                                self._update_runtime_status(self._serving_state())
                    with suppress(Exception):
                        from gateway.channel_directory import build_channel_directory
                        await build_channel_directory(self.adapters)
                    self._schedule_resume_pending_sessions(platform=platform)
                    logger.info("WhatsApp activated without restarting the gateway")
                    return {"active": True}

                await self._safe_adapter_disconnect(adapter, platform)
                if retryable:
                    self._failed_platforms[platform] = self._startup_retry_entry(
                        platform, adapter, platform_config)
                    self._ensure_reconnect_watcher_running()
                    self._update_platform_runtime_status(
                        platform.value, platform_state="retrying",
                        error_code=adapter.fatal_error_code,
                        error_message=adapter.fatal_error_message or "failed to connect")
                    return {"active": False, "retrying": True}
                self._mark_platform_fatal(platform.value, adapter)
                self._startup_parked_platforms = True
                if not (self._external_drain_active or self._draining or not self._running):
                    self._update_runtime_status(self._serving_state())
                return {"active": False, "reason": adapter.fatal_error_code or "connect_failed"}
