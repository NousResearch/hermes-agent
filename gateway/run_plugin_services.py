"""Profile-local plugin event dispatch and service lifecycle for GatewayRunner."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")

# Upper bound on waiting for cancelled profile services during shutdown.
PROFILE_SERVICE_STOP_TIMEOUT = 5.0


class GatewayPluginServicesMixin:
    """Keep plugin-owned execution explicitly local to a served profile."""

    @staticmethod
    def _validate_internal_plugin_event(event: MessageEvent) -> SessionSource:
        """Validate the narrow event shape allowed for profile-local plugins."""
        if not isinstance(event, MessageEvent):
            raise TypeError("internal plugin dispatch requires a MessageEvent")
        if event.internal is not True:
            raise PermissionError("internal plugin events must set internal=True")
        if event.allow_gateway_control is not False:
            raise PermissionError(
                "internal plugin events must set allow_gateway_control=False"
            )
        if event.message_type is not MessageType.TEXT:
            raise ValueError("internal plugin dispatch accepts text events only")

        source = event.source
        if source is None:
            raise ValueError("internal plugin events require a SessionSource")
        if source.platform is not Platform.LOCAL:
            raise PermissionError("internal plugin events must use the local platform")
        if not str(getattr(source, "profile", "") or "").strip():
            raise ValueError("internal plugin events require an explicit profile")
        if not str(getattr(source, "chat_id", "") or "").strip():
            raise ValueError("internal plugin events require a session chat_id")
        return source

    async def prepare_internal_plugin_session(self, event: MessageEvent) -> str:
        """Persist a plugin-owned session without beginning an agent turn."""
        from gateway.run import _profile_runtime_scope

        source = self._validate_internal_plugin_event(event)
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            profile_home = self._resolve_profile_home_for_source(source)
            with _profile_runtime_scope(profile_home):
                entry = await self.async_session_store.get_or_create_session(
                    source, touch_activity=False
                )
        else:
            entry = await self.async_session_store.get_or_create_session(
                source, touch_activity=False
            )
        return str(entry.session_id)

    async def dispatch_internal_plugin_event(
        self, event: MessageEvent
    ) -> Optional[str]:
        """Dispatch a validated plugin event through the normal scoped handler."""
        self._validate_internal_plugin_event(event)
        return await self._primary_message_handler()(event)

    def _start_plugin_profile_services(self) -> None:
        """Start each registered profile service in its owning profile scope."""
        from gateway.run import _multiplex_profile_homes, _profile_runtime_scope
        from hermes_cli.plugins import get_plugin_manager
        from hermes_constants import get_hermes_home

        # Startup must not lose references to still-running services on a repeated call.
        if any(not task.done() for task in getattr(self, "_profile_service_tasks", ())):
            return
        stop_event = getattr(self, "_profile_service_stop", None)
        if stop_event is None or stop_event.is_set():
            stop_event = asyncio.Event()
            self._profile_service_stop = stop_event
        self._profile_service_tasks = []

        active_profile = (
            os.getenv("HERMES_PROFILE") or os.getenv("HERMES_AGENT_PROFILE") or "default"
        ).strip() or "default"
        profiles = [(active_profile, Path(get_hermes_home()))]
        if getattr(self.config, "multiplex_profiles", False):
            profiles.extend(
                (name, Path(home))
                for name, home in _multiplex_profile_homes(self.config)
                if name != active_profile
            )

        seen_homes: set[Path] = set()
        for profile_name, profile_home in profiles:
            # A logical launch name and `default` may refer to the same home.
            # Home ownership, not the spelling of a routing alias, owns services.
            resolved_home = profile_home.resolve()
            if resolved_home in seen_homes:
                continue
            seen_homes.add(resolved_home)
            with _profile_runtime_scope(profile_home):
                runtime = SimpleNamespace(
                    profile_home=str(get_hermes_home()),
                    profile_name=profile_name,
                    stop_event=stop_event,
                    gateway=self,
                )
                try:
                    services = list(get_plugin_manager()._profile_services)
                except Exception:
                    logger.debug(
                        "Could not load plugin profile services for %s",
                        profile_name,
                        exc_info=True,
                    )
                    continue
                for name, factory in services:
                    try:
                        task = asyncio.create_task(
                            factory(runtime), name=f"profile-service:{profile_name}:{name}"
                        )
                    except Exception:
                        logger.exception(
                            "Failed to start plugin profile service %s for profile %s",
                            name,
                            profile_name,
                        )
                        continue
                    self._profile_service_tasks.append(task)
                    logger.info(
                        "Started plugin profile service %s for profile %s",
                        name,
                        profile_name,
                    )

    async def _stop_plugin_profile_services(self) -> None:
        stop_event = getattr(self, "_profile_service_stop", None)
        if stop_event is not None and not stop_event.is_set():
            stop_event.set()
        tasks = list(getattr(self, "_profile_service_tasks", []) or [])
        self._profile_service_tasks = []
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        # Bounded: a profile service that swallows CancelledError (or blocks in a finally)
        # must not hold the whole shutdown path open forever.
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=PROFILE_SERVICE_STOP_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "plugin profile services did not stop within %.0fs: %s",
                PROFILE_SERVICE_STOP_TIMEOUT,
                ", ".join(sorted(t.get_name() for t in tasks if not t.done())) or "unknown",
            )
