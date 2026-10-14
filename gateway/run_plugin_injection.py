"""Plugin-triggered gateway turns (``PluginContext.inject_message``): the process-wide injector this
runner publishes, the thread-safe scheduler, and the dispatch through the adapter message path.

Two targets: an existing session (``session_key``; route and session id pinned from the store) or a
messaging chat (``origin``) whose session is created — or continued — through the normal session path
in the calling plugin's OWN profile. The origin path never chooses a profile: the gateway's routing
for that chat through that profile's adapter must already land in the plugin's profile, so a human
follow-up in the same chat keys into the same session."""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import weakref
from pathlib import Path
from typing import Any, Mapping, Optional

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


class GatewayPluginInjectionMixin:
    def _install_plugin_message_injector(self) -> None:
        """Publish this live gateway's plugin message scheduler process-wide."""
        from hermes_cli.plugins import publish_gateway_message_host

        publish_gateway_message_host(self, self._schedule_plugin_message_injection)

    def _clear_plugin_message_injector(self) -> None:
        """Remove this runner's scheduler without clobbering a newer owner."""
        from hermes_cli.plugins import clear_published_gateway_message_host

        clear_published_gateway_message_host(self)

    def _schedule_plugin_message_injection(
        self, *, content: str, plugin_id: str, session_key: Optional[str] = None,
        origin: Optional[Mapping[str, Any]] = None, plugin_home: Optional[Path] = None,
    ) -> bool:
        """Schedule a plugin-triggered turn on the live gateway loop (thread-safe).

        With ``origin`` the profile, origin and adapter checks run here, synchronously, so the plugin
        gets ``False`` for a chat it may not start; authorization is rechecked at dispatch."""
        from gateway.run import safe_schedule_threadsafe
        loop = getattr(self, "_gateway_loop", None)
        if not getattr(self, "_running", False) or loop is None or loop.is_closed():
            return False

        if origin is not None:
            source = self._plugin_origin_source(origin, plugin_home=plugin_home, plugin_id=plugin_id)
            if source is None:
                return False
            target = f"origin={source.platform.value}/{source.chat_id}"
            coro = self._dispatch_plugin_origin_injection(source=source, content=content, plugin_id=plugin_id)
        else:
            target = f"session={session_key}"
            coro = self._dispatch_plugin_message_injection(
                session_key=session_key, content=content, plugin_id=plugin_id,
            )
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is loop:
            try:
                future = loop.create_task(coro)
            except Exception:
                coro.close()
                logger.warning("Plugin message injection scheduling failed", exc_info=True)
                return False
            self._background_tasks.add(future)
            future.add_done_callback(self._background_tasks.discard)
        else:
            future = safe_schedule_threadsafe(
                coro, loop, logger=logger, log_message="Plugin message injection scheduling failed",
                log_level=logging.WARNING,
            )
            if future is None:
                return False

        def _log_result(completed) -> None:
            try:
                if completed.result():
                    return
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                return
            except Exception:  # dispatch boundary: the plugin already got True; log, never raise
                logger.warning("Plugin message injection failed: plugin=%s %s", plugin_id, target, exc_info=True)
                return
            logger.warning("Plugin message injection was not routed: plugin=%s %s", plugin_id, target)

        future.add_done_callback(_log_result)
        return True

    def _plugin_injection_accepting(self) -> bool:
        return getattr(self, "_running", False) and not getattr(self, "_draining", False)

    def _plugin_injection_scope(self, source: SessionSource):
        """Multiplex: enter the source's profile runtime scope around ``handle_message`` so the
        background task it spawns — the turn AND the reply send after the handler returns — inherits
        that profile's home and secrets (this coroutine runs from the default loop context, not from
        the profile adapter's own intake task)."""
        if getattr(getattr(self, "config", None), "multiplex_profiles", False) is not True:
            return contextlib.nullcontext()
        from gateway.run import _async_profile_runtime_scope
        return _async_profile_runtime_scope(self._resolve_profile_home_for_source(source))

    def _plugin_injection_authorized(self, source: SessionSource, *, plugin_id: str, target: str) -> bool:
        """Current gateway authorization for the route the injected turn will run as. Host-side
        plugin code is trusted to ask, never to widen access: a stored or supplied route must pass
        today's allowlists / pairing / allow-all, and adapter-time grants (Discord roles) do not count."""
        try:
            authorized = self._is_user_authorized_for_source(source, allow_adapter_delegation=False)
        except Exception:
            logger.warning("Plugin message injection authorization check failed: plugin=%s %s",
                           plugin_id, target, exc_info=True)
            return False
        if not authorized:
            logger.warning("Plugin message injection denied by current gateway authorization: plugin=%s %s",
                           plugin_id, target)
        return authorized

    async def _dispatch_plugin_message_injection(
        self, *, session_key: str, content: str, plugin_id: str
    ) -> bool:
        """Route a plugin-triggered turn through the session's live adapter."""
        if not self._plugin_injection_accepting():
            return False
        entry = await self.async_session_store.lookup_by_session_key(session_key)
        if entry is None or entry.origin is None or not self._plugin_injection_accepting():
            return False

        from gateway.session_identity import replace_source
        source = replace_source(self._restored_source(entry))
        if not self._plugin_injection_authorized(source, plugin_id=plugin_id, target=f"session={session_key}"):
            return False

        adapter = self._delivery_adapter_for(source)
        if adapter is None:
            return False

        async with self._plugin_injection_scope(source):
            await adapter.handle_message(MessageEvent(
                text=content, message_type=MessageType.TEXT, source=source, internal=True,
                allow_gateway_control=False,
                metadata={
                    "hermes_plugin_id": plugin_id, "hermes_plugin_injection": True,
                    "gateway_session_key": session_key, "gateway_session_id": entry.session_id,
                    "gateway_session_strict": True,
                },
            ))
        logger.info(
            "Plugin message injection dispatched: plugin=%s session=%s session_id=%s",
            plugin_id, session_key, entry.session_id,
        )
        return True

    def _served_profile_for_home(self, home: Optional[Path]) -> Optional[str]:
        """Name of the profile this gateway serves at *home* (a plugin manager's immutable home), or
        ``None`` when this gateway does not serve it."""
        if home is None:
            return None
        from hermes_constants import get_routing_process_hermes_home, hermes_home_key
        key = hermes_home_key(home)
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            from gateway.run import _multiplex_profile_homes
            try:
                served = _multiplex_profile_homes(self.config)
            except Exception:
                logger.warning("Plugin message injection: served-profile set unavailable", exc_info=True)
                return None
            return next((name for name, served_home in served if hermes_home_key(served_home) == key), None)
        if hermes_home_key(get_routing_process_hermes_home()) != key:
            return None
        return getattr(self, "_primary_profile_name", None) or self._active_profile_name()

    def _plugin_origin_source(
        self, origin: Mapping[str, Any], *, plugin_home: Optional[Path], plugin_id: str,
    ) -> Optional[SessionSource]:
        """The canonicalized source for a plugin-supplied *origin*, or ``None`` (logged) when the
        plugin may not start a turn there: its profile is not served, the origin is malformed or
        names another profile, that profile has no live adapter for the platform, or the gateway's
        own routing for the chat lands in a different profile."""
        def _refuse(reason: str, *args: Any) -> None:
            logger.warning("Plugin message injection refused: plugin=%s " + reason, plugin_id, *args)

        profile = self._served_profile_for_home(plugin_home)
        if profile is None:
            return _refuse("its profile home %s is not served by this gateway", plugin_home)
        fields = dict(origin)
        requested = fields.pop("profile", None)
        if requested and requested != profile:
            return _refuse("origin names profile %r; plugins may only start sessions in their own profile (%s)",
                           requested, profile)
        try:
            source = SessionSource.from_dict(fields)
        except (KeyError, TypeError, ValueError) as exc:
            return _refuse("invalid origin (%s)", exc)
        where = f"{source.platform.value}/{source.chat_id}"
        adapter = self._adapters_for_profile(profile).get(source.platform)
        if adapter is None:
            return _refuse("no connected %s adapter for profile %s", source.platform.value, profile)
        # Same provenance build_source() stamps on a live inbound event from this adapter, so the
        # identity (receiving bot, runtime profile, authorization home) resolves exactly as it would
        # for a human message in this chat.
        source._transport_adapter_ref = weakref.ref(adapter)
        identity = self._canonicalize(source)
        if identity is None:
            return _refuse("origin %s does not resolve to a served profile", where)
        if identity.runtime_profile != profile:
            return _refuse("origin %s routes to profile %s, not the plugin's profile %s",
                           where, identity.runtime_profile, profile)
        return source

    async def _dispatch_plugin_origin_injection(
        self, *, source: SessionSource, content: str, plugin_id: str
    ) -> bool:
        """Start (or continue) the session at a plugin-supplied origin through the adapter's normal
        message path: ``get_or_create_session`` creates it, the reply goes back to that chat."""
        target = f"origin={source.platform.value}/{source.chat_id}"
        if not self._plugin_injection_accepting():
            return False
        adapter = self._delivery_adapter_for(source)
        if adapter is None:
            logger.warning("Plugin message injection: adapter for %s went away before dispatch", target)
            return False
        if not self._plugin_injection_authorized(source, plugin_id=plugin_id, target=target):
            return False
        session_key = self._session_key_for_source(source)
        async with self._plugin_injection_scope(source):
            await adapter.handle_message(MessageEvent(
                text=content, message_type=MessageType.TEXT, source=source, internal=True,
                allow_gateway_control=False,
                # Non-strict: the key is pinned against topic recovery, the session may not exist yet.
                metadata={
                    "hermes_plugin_id": plugin_id, "hermes_plugin_injection": True,
                    "gateway_session_key": session_key,
                },
            ))
        logger.info("Plugin message injection dispatched: plugin=%s %s session=%s", plugin_id, target, session_key)
        return True
