"""Late plugin loads re-wire the running adapters (#87770).

A plugin that finishes loading after an adapter connected — a tool-triggered force re-discovery, or
``hermes plugins install/enable`` / the Desktop / ``plugins.manage`` nudging the gateway over the control
socket — used to leave its platform handlers (slash commands, button callbacks, inbound transforms)
silently unwired until a restart. The runner subscribes to ``PluginManager.on_plugin_loaded`` per served
profile and calls every live adapter's idempotent ``rewire_plugin_handlers()`` on the event loop.

Scope is handlers only: a mid-run-loaded plugin's tools and system-prompt sections stay deferred to the
next session (prompt-cache invariant; same as ``/skills install``).
"""

from __future__ import annotations

import asyncio
import contextlib
from contextvars import ContextVar
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Only suppress this runner's redundant loaded-event pass during a control reload.
_control_reload_scope: ContextVar[Optional[tuple[int, str]]] = ContextVar(
    "_control_reload_scope", default=None)


class GatewayPluginRewireMixin:
    """Subscribe once per plugin manager; re-wire that profile's adapters on every loaded event."""

    _plugin_rewire_unsubscribe: Optional[dict[str, Callable[[], None]]] = None

    def _subscribe_plugin_rewire(self, manager: Any, profile_name: Optional[str] = None,
                                 profile_home: Optional[Path] = None) -> None:
        """Idempotent per manager scope: a served-profile rescan re-enters ``_load_secondary_profile_config``
        and must not stack a second listener (two listeners = two re-wire passes, still deduped, but noise)."""
        subs = self._plugin_rewire_unsubscribe
        if subs is None:
            subs = self._plugin_rewire_unsubscribe = {}
        scope = str(getattr(manager, "scope_key", "") or id(manager))
        if scope in subs:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def _on_loaded(_names) -> None:
            # Fires on the discovering thread (tool worker, control-socket executor, or the loop itself):
            # adapters own PTB/Bolt handler tables, so always hop onto the gateway loop.
            if _control_reload_scope.get() == (id(self), scope):
                return
            if loop is None or loop.is_closed():
                return
            loop.call_soon_threadsafe(self._rewire_plugin_handlers, profile_name, profile_home)

        subs[scope] = manager.on_plugin_loaded(_on_loaded)

    def _rewire_plugin_handlers(self, profile_name: Optional[str] = None,
                                profile_home: Optional[Path] = None) -> int:
        """Call ``rewire_plugin_handlers()`` on every live adapter of one profile (``None`` = the primary
        adapter map, ``self.adapters``); a secondary's adapters read their own manager, so bind its scope.
        Returns the number of adapters re-wired."""
        if profile_name is None:
            adapters = dict(getattr(self, "adapters", None) or {})
        else:
            adapters = dict((getattr(self, "_profile_adapters", None) or {}).get(profile_name) or {})
        if not adapters:
            return 0
        from gateway.run import _profile_runtime_scope
        scope = (_profile_runtime_scope(profile_home, hydrate_secrets=False)
                 if profile_home is not None else contextlib.nullcontext())
        count = 0
        with scope:
            for platform, adapter in adapters.items():
                try:
                    adapter.rewire_plugin_handlers()
                    count += 1
                except Exception:
                    logger.warning("[%s] plugin handler re-wire failed", getattr(platform, "value", platform),
                                   exc_info=True)
        logger.info("Re-wired plugin handlers on %d adapter(s)%s", count,
                    f" for profile '{profile_name}'" if profile_name else "")
        return count


def reload_plugins_verb(runner: Any, loop: asyncio.AbstractEventLoop) -> Callable[..., dict]:
    """Control-socket ``reload-plugins``: force re-discovery under the requested home's scope so a plugin
    installed/enabled by another process (CLI, Desktop, ``plugins.manage``) loads now; an explicit
    loop pass re-wires that profile even when an existing plugin emits no newcomer event.
    Other subscribers retain the loaded event. Only the gateway and served homes are accepted.
    Answer: ``{"reloaded", "home", "plugins", "activations", "adapters_rewired"}``. Runs on the socket executor thread
    (discovery is blocking); the count is completed adapter method calls, not native handler readiness.
    Tools and prompt sections stay deferred to the next session."""

    def _handler(params: Optional[dict] = None) -> dict:
        from hermes_constants import get_hermes_home, hermes_home_key
        from hermes_cli.plugins import discover_plugins, get_plugin_manager
        from hermes_cli.plugins_activation import activation_summaries
        from gateway.run import _profile_runtime_scope
        params = params or {}
        gateway_home = Path(get_hermes_home())
        requested = Path(str(params.get("home") or gateway_home)).expanduser()
        req_key = hermes_home_key(requested)
        served = getattr(runner, "_served_profile_homes", None) or {}
        profile_name = next((str(n) for n, h in served.items() if hermes_home_key(h) == req_key), None)
        if profile_name is None and (served or req_key != hermes_home_key(gateway_home)):
            return {"reloaded": False, "error": "home is not served by this gateway", "home": str(requested)}
        # The launch home may belong to a secondary; only the primary uses self.adapters.
        if profile_name == (getattr(runner, "_primary_profile_name", None) or "default"):
            profile_name = None
        with _profile_runtime_scope(requested, hydrate_secrets=False):
            manager = get_plugin_manager()
            manager_scope = str(getattr(manager, "scope_key", "") or id(manager))
            token = _control_reload_scope.set((id(runner), manager_scope))
            try:
                discover_plugins(force=True)
            finally:
                _control_reload_scope.reset(token)
            names, activations = sorted(manager._plugins), activation_summaries(manager)
        try:
            rewired = asyncio.run_coroutine_threadsafe(
                _rewire_adapters(runner, profile_name, requested), loop).result(timeout=5.0)
        except Exception:
            rewired = None
        return {"reloaded": True, "home": str(requested), "plugins": names, "activations": activations,
                "adapters_rewired": rewired}

    return _handler


async def _rewire_adapters(runner: Any, profile_name: Optional[str], profile_home: Path) -> int:
    """Run the requested-profile pass even when discovery added no new plugin key."""
    return runner._rewire_plugin_handlers(profile_name, profile_home)
