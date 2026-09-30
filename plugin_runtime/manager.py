"""Plugin manager composition and manager-local runtime lifecycle."""

from __future__ import annotations

import importlib
import logging
import re
import threading
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from hermes_constants import get_hermes_home, hermes_home_key
from plugin_runtime.activation import activation_summaries
from plugin_runtime import compat as plugin_compat
from plugin_runtime.config_bridge import load_plugin_config
from plugin_runtime.context import PluginContext
from plugin_runtime.discovery import (
    _get_disabled_plugins,
    _get_enabled_plugins,
    collect_directory_manifests,
    discover_entrypoint_manifests,
    gate_manifest,
    resolve_manifest_winners,
    scan_directory,
)
from plugin_runtime.dispatch import (
    PluginDispatchMixin,
    PluginSystemPromptSection,
    resolve_plugin_command_result,
)
from plugin_runtime.loading import LoadedPlugin, PluginLoaderMixin, in_plugin_load_worker
from plugin_runtime.manifest import PluginManifest, manifest_key, resolve_plugin_load_order
from plugin_runtime.ownership import PluginOwnershipMixin
from plugin_runtime.registration import PluginRegistration
from plugin_runtime.contracts import RegisteredApprovalTransport
from plugin_runtime.relay_policy import RELAY_PLUGINS_CONFIG_ENV, legacy_relay_plugin_keys
from plugin_runtime.scope import plugin_home_scope as _plugin_home_scope
from plugin_runtime.host_bindings import get_plugin_host_callback
from utils import env_var_enabled

logger = logging.getLogger("hermes_cli.plugins")


def _installed_plugin_removal(name: str, plugin_dir: Any):
    """Call the host-owned catalog recall policy when that host is available."""
    callback = get_plugin_host_callback("installed_plugin_removal")
    return callback(name, plugin_dir) if callback is not None else None


class PluginManager(PluginLoaderMixin, PluginDispatchMixin, PluginOwnershipMixin):
    """Central manager that discovers, loads, and invokes plugins."""

    @staticmethod
    def _plugin_dispatch_safe_worker_enabled() -> bool:
        """Supply the host-owned safe-worker admission policy to runtime dispatch."""
        from agent.safe_worker_policy import safe_worker_enabled

        return safe_worker_enabled()

    @staticmethod
    def _plugin_dispatch_resolve_result(result: Any) -> Any:
        """Supply the host-owned plugin await/result resolver to runtime dispatch."""
        return resolve_plugin_command_result(result)

    def __init__(self, scope_key: Optional[str] = None) -> None:
        # Home is captured immutably: unload may run from another profile context, but every
        # inverse must target the registration's original scope.
        self.scope_key = hermes_home_key(scope_key)
        self.home_path = Path(self.scope_key)
        self._discovery_lock = threading.RLock()
        self._discovered: bool = False
        self._cli_ref = None  # Set by CLI after plugin discovery
        self._gateway_message_injector: tuple[object, Callable] | None = None
        # Ink TUI / desktop. Must not alias ``_gateway_message_injector``: a live
        # messaging gateway and a TUI in one process would otherwise clobber each other.
        self._tui_message_injector: tuple[object, Callable] | None = None
        self._context_engine = None  # Set by a plugin via register_context_engine()
        # Manager-local registries keyed by name (see the matching ``PluginContext.register_*``):
        # plugins, hooks, middleware, CLI + slash commands, prompt sections, skills (qualified name ->
        # metadata), portable MCP servers, auxiliary tasks, approval transports, Slack action handlers
        # (matcher, callback, plugin_name), platform handler factories (lowercase platform -> list).
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        # Fallback hooks registered by a memory provider before general discovery.
        self._memory_hook_registrations: Dict[Tuple[str, str], List[PluginRegistration]] = {}
        self._middleware: Dict[str, List[Callable]] = {}
        self._plugin_tool_names: Set[str] = set()
        self._plugin_platform_names: Set[str] = set()
        self._cli_commands: Dict[str, dict] = {}
        self._plugin_commands: Dict[str, dict] = {}
        self._system_prompt_sections: Dict[str, PluginSystemPromptSection] = {}
        self._plugin_skills: Dict[str, Dict[str, Any]] = {}
        self._portable_mcp_servers: Dict[str, Dict[str, Any]] = {}
        self._portable_mcp_server_plugins: Dict[str, str] = {}
        self._aux_tasks: Dict[str, Dict[str, Any]] = {}
        self._approval_transports: Dict[str, Any] = {}
        self._slack_action_handlers: List[tuple] = []
        self._platform_handler_factories: Dict[str, List[tuple]] = {}
        # Process-owned discovery listeners (``on_plugin_loaded``); never cleared by unload().
        self._plugin_loaded_listeners: List[Callable] = []
        self._init_dispatch_runtime_state()
        # Ledger per plugin (ownership) plus global order (reverse teardown across plugins). Process-
        # global registries are shared across profiles while several managers coexist, so the ledger
        # is keyed per (hermes_home, plugin_id) and every inverse is identity-conditional — one
        # profile's unload can never clear another's. Persistent registrations that survived an
        # unload-all park in ``_persistent_carryover`` until force re-discovery evicts the stale ones.
        # Registration handles are kept both per plugin (ownership lookup) and globally (reverse-order
        # teardown for overrides spanning plugins). Registry overlays keyed by scope_key (see
        # tools/registry.py and gateway/platform_registry.py) carry the profile dimension; anything still
        # process-global is guarded by the identity checks. TODO(#64178): extend explicit profile keying to
        # any remaining process-global slots when the symmetric force-reload lands.
        self._ownership_ledger: Dict[str, List[PluginRegistration]] = {}
        self._registration_order: List[PluginRegistration] = []
        # Force re-discovery drains this via _evict_stale_persistent_registrations(): entries whose plugin
        # re-registered the same (kind, key) are kept (the upsert rotated them in place), the rest are
        # disposed so a disabled/removed auth plugin's provider does not outlive its plugin (#91701
        # follow-up).
        self._persistent_carryover: List[PluginRegistration] = []
        # Deferred platforms whose client tools registered at discovery (see
        # _register_deferred_platform_tools): imported package (don't re-execute on materialize)
        # and contributed tool names (so `hermes plugins list` still attributes them).
        self._predeclared_modules: Dict[str, types.ModuleType] = {}
        self._predeclared_tools: Dict[str, List[str]] = {}

    def context_for(self, manifest: PluginManifest) -> PluginContext:
        """Construct the runtime-owned plugin context requested by plugin loading."""
        return PluginContext(manifest, self)

    @staticmethod
    def _plugin_load_disable_reason(manifest: PluginManifest) -> Optional[str]:
        """Apply runtime-owned plugin compatibility policy."""
        return plugin_compat.disable_reason(manifest)

    def on_plugin_loaded(self, callback: Callable[[List[Dict[str, Any]]], Any]) -> Callable[[], None]:
        """Subscribe to discovery sweeps that load plugins this process did not already have."""
        if not callable(callback):
            raise ValueError("on_plugin_loaded requires a callable")
        listeners = self._plugin_loaded_listeners
        listeners.append(callback)

        def _unsubscribe() -> None:
            try:
                listeners.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    def _notify_plugin_loaded(self, loaded_before: frozenset) -> None:
        """Notify process-owned listeners about plugins newly loaded by this discovery sweep."""
        if not self._plugin_loaded_listeners:
            return
        summaries = [s for s in activation_summaries(self) if s["key"] not in loaded_before]
        if not summaries:
            return
        for callback in list(self._plugin_loaded_listeners):
            try:
                callback(summaries)
            except Exception:
                logger.warning("plugin-loaded listener %r raised", callback, exc_info=True)

    @property
    def has_gateway_message_injector(self) -> bool:
        """Return whether a live gateway can accept plugin-triggered turns."""
        return self._gateway_message_injector is not None

    def set_gateway_message_injector(self, owner: object, injector: Callable[..., bool]) -> None:
        """Publish a live gateway injector and its lifecycle owner."""
        self._gateway_message_injector = (owner, injector)

    def clear_gateway_message_injector(self, owner: object) -> None:
        """Clear the injector only when it still belongs to ``owner``."""
        if self._gateway_message_injector is not None and self._gateway_message_injector[0] is owner:
            self._gateway_message_injector = None

    def inject_gateway_message(self, **kwargs: Any) -> bool:
        """Submit a plugin-triggered turn to the live gateway."""
        registered = self._gateway_message_injector
        return registered is not None and bool(registered[1](**kwargs))

    @property
    def has_tui_message_injector(self) -> bool:
        """Return whether a live Ink TUI / desktop host can accept plugin-triggered turns."""
        return self._tui_message_injector is not None

    def set_tui_message_injector(self, owner: object, injector: Callable[..., bool]) -> None:
        """Publish a live TUI/desktop injector. Does not touch the messaging-gateway slot."""
        self._tui_message_injector = (owner, injector)

    def clear_tui_message_injector(self, owner: object) -> None:
        """Clear the TUI injector only when it still belongs to ``owner``."""
        if self._tui_message_injector is not None and self._tui_message_injector[0] is owner:
            self._tui_message_injector = None

    def inject_tui_message(self, **kwargs: Any) -> bool:
        """Submit a plugin-triggered turn to the live TUI/desktop host."""
        registered = self._tui_message_injector
        return registered is not None and bool(registered[1](**kwargs))

    def discover_and_load(self, force: bool = False) -> None:
        """Scan all plugin sources and load each plugin found; ``force`` unloads first so config
        changes / new bundled backends become visible in long-lived sessions."""
        from agent.safe_worker_policy import safe_worker_enabled

        if safe_worker_enabled():
            return
        if self._discovered and not force and in_plugin_load_worker():
            # A plugin whose register() re-enters discovery (importing model_tools does) runs on a
            # deadline worker that cannot re-acquire the sweep's RLock; the flag is already set for the
            # whole sweep, so return where the locked re-entry used to. Every other caller still waits.
            return
        with self._discovery_lock, _plugin_home_scope(self.home_path):
            if self._discovered and not force:
                return
            # ``on_plugin_loaded`` reports the plugins this sweep loads that the process did not have before
            # (boot: everything; a mid-run install/enable: just the newcomer), keyed on the pre-sweep set.
            loaded_before = frozenset(k for k, p in self._plugins.items() if not p.error and not p.deferred)
            if force:
                self.unload()  # the ledger owns teardown of process-global registries
            if env_var_enabled("HERMES_SAFE_MODE"):
                logger.info("HERMES_SAFE_MODE=1 — plugin discovery skipped")
                self._discovered = True
                return
            # Flag set up front as a re-entrancy guard (register() can trigger discovery again) but
            # reset on failure so a failed scan is NOT cached as "discovered with an empty registry"
            # — callers swallow the exception and would be stranded on the early return above.
            self._discovered = True
            try:
                self._discover_and_load_inner()
                # Persistent registrations survived the unload-all; now that plugins re-registered,
                # dispose the ones whose plugin did not come back.
                # Now that plugins have had their chance to re-register, dispose the ones whose plugin did
                # not come back (disabled, removed, or omitted from this discovery pass) so e.g. a disabled
                # auth plugin's provider does not stay live process-wide until restart. See #91701.
                self._evict_stale_persistent_registrations()
                # load_hermes_dotenv() ran at import, before plugin secret sources existed: re-pull.
                # Plugin secret sources register during discover; the initial load_hermes_dotenv() already
                # ran at import time. Re-pull so the first process sees plugin backends (tracking #64177).
                self._refresh_secret_sources_after_discovery()
                if force:
                    # config.yaml shell hooks / outbound webhooks live in ``_hooks`` but are
                    # config-owned; unload() wiped them and cannot restore them.
                    # Re-register so force-reload is symmetric (#60036; tracking #64178 — salvaged from PR
                    # #64188; outbound webhooks added per #92682 review).
                    self._re_register_config_hooks_after_force()
            except BaseException:
                self._discovered = False
                raise
        # Outside the lock: a listener (the gateway's re-wire) may read the registry from another thread.
        self._notify_plugin_loaded(loaded_before)

    def _re_register_config_hooks_after_force(self) -> None:
        """Restore config-owned shell hooks/outbound webhooks after a force clear; each guarded
        independently so one failing does not skip the other."""
        for label, module_name in (("shell-hook", "agent.shell_hooks"),
                                   ("outbound-webhook", "agent.outbound_webhooks")):
            try:
                importlib.import_module(module_name).re_register_config_hooks()
            except Exception as exc:
                logger.debug("force-reload %s re-register skipped: %s", label, exc)

    def _refresh_secret_sources_after_discovery(self) -> None:
        """If any plugin secret source is enabled (per its own ``is_enabled(cfg)``, honoring custom
        activation), reset the cache and re-apply. Fail-open: never raises into discover_and_load."""
        try:
            from agent.secret_sources.registry import list_plugin_sources
            plugin_sources = list_plugin_sources()
        except Exception:
            return
        if not plugin_sources:
            return
        try:
            secrets = (load_plugin_config() or {}).get("secrets") or {}
        except Exception:
            secrets = {}

        def _enabled(source) -> bool:
            section = secrets.get(getattr(source, "name", ""))
            try:
                return bool(source.is_enabled(section if isinstance(section, dict) else {}))
            except Exception:
                return False  # mirrors the orchestrator: a raising is_enabled() is skipped

        enabled_names = [getattr(s, "name", "") for s in plugin_sources if _enabled(s)]
        if not enabled_names:
            return
        try:
            # Reset and reload the SAME home the process (or routed turn) resolves to: under multiplex this
            # runs at gateway boot after sibling profiles may already have hydrated, and a global clear
            # wiped their snapshots; a routed discovery must rebuild the profile it just dropped.
            home = get_hermes_home()
            refresh = get_plugin_host_callback("refresh_secret_sources")
            if refresh is None:
                return
            refresh(home)
            logger.debug("Re-applied secret sources after plugin discovery for: %s",
                         ", ".join(sorted(enabled_names)))
        except Exception as exc:
            logger.debug("secret source re-apply after discovery failed: %s", exc)

    def _discover_and_load_inner(self) -> None:
        """The actual discovery sweep — see :meth:`discover_and_load`."""
        manifests: List[PluginManifest] = self._collect_directory_manifests()
        # Entry points are separate from the directory scan: the startup MCP probe must not import
        # or register them.
        # An installed directory plugin keeps its identity when its own pip dependency also ships an
        # entry point under the same name (the pyproject wrapper shape): the directory is what the
        # user installed, carries catalog provenance and is what update/remove act on.
        directory_keys = {manifest_key(m) for m in manifests}
        ep_manifests = [m for m in self._scan_entry_points() if manifest_key(m) not in directory_keys]
        logger.debug("  entrypoints: %d manifest(s)", len(ep_manifests))
        manifests.extend(ep_manifests)
        disabled = _get_disabled_plugins()
        enabled = _get_enabled_plugins()  # None = opt-in default (nothing enabled)
        stale_relay_keys = legacy_relay_plugin_keys(enabled)
        if stale_relay_keys:
            logger.warning("Removed Hermes plugin %s is still listed in plugins.enabled; "
                           "remove it and configure native Relay plugins with %s",
                           ", ".join(stale_relay_keys), RELAY_PLUGINS_CONFIG_ENV)
        # Later sources win on key collision (project > user > bundled) except a flat impostor claiming a
        # bundled key from another directory (resolve_manifest_winners); gate the winners, then
        # load survivors in requires_plugins order (see resolve_plugin_load_order).
        winners = resolve_manifest_winners(manifests)
        to_load = {k: m for k, m in winners.items() if self._gate_manifest(m, disabled, enabled)}
        for lookup_key in resolve_plugin_load_order(to_load):
            manifest = to_load[lookup_key]
            self._warn_python_dependencies(manifest)
            self._validate_plugin_config_schema(manifest)
            self._load_plugin(manifest)
        if manifests:
            logger.info("Plugin discovery complete: %d found, %d enabled", len(self._plugins),
                        sum(1 for p in self._plugins.values() if p.enabled))
        self._refresh_plugin_compat_report(list(to_load.values()))

    def _refresh_plugin_compat_report(self, manifests: List[PluginManifest]) -> None:
        """Refresh HERMES_HOME/.plugin-compat-report.json from this discovery pass.

        The Desktop boot modal has no Python runtime of its own and reads that file after the ``serve``
        backend is up, so the scan must run wherever plugins are discovered — not only under the CLI
        banner / doctor / update, which never run inside the Desktop's backend. Fail-open: never raises.
        """
        try:
            plugin_compat.compat_report(manifests, force=True)
        except Exception as exc:
            logger.debug("plugin compat report refresh skipped: %s", exc)

    def _gate_manifest(
        self, manifest: PluginManifest, disabled: Set[str], enabled: Optional[Set[str]],
    ) -> bool:
        """Route one winning manifest per :func:`gate_manifest`: load now, defer, or record as
        skipped (introspection-only placeholder). Returns True only for plugins that go through the
        dependency-ordered load pass."""
        verdict = gate_manifest(
            manifest, disabled, enabled, installed_plugin_removal=_installed_plugin_removal,
        )
        if verdict.action == "load":
            return True
        if verdict.action == "load_now":
            self._load_plugin(manifest)
        elif verdict.action == "defer":
            self._register_deferred_platform(manifest)
        else:
            self._plugins[manifest_key(manifest)] = LoadedPlugin(
                manifest=manifest, enabled=verdict.enabled, error=verdict.error)
        if verdict.log:
            logger.log(*verdict.log)
        return False

    def register_approval_transport(self, name: str, present_fn: Callable, *, plugin_id: str) -> None:
        """Manager-level registration (public API kept for out-of-tree plugins); the PluginContext
        method is the tracked path plugins normally use. Same validation, no unload tracking."""
        clean = str(name).strip().lower()
        if clean == "builtin":
            raise ValueError("approval transport name 'builtin' is reserved")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", clean):
            raise ValueError("approval transport name must match [a-z0-9][a-z0-9_-]{0,63}")
        if not callable(present_fn):
            raise TypeError("approval transport present_fn must be callable")
        if clean in self._approval_transports:
            owner = self._approval_transports[clean].plugin_id
            raise ValueError(f"approval transport {clean!r} is already registered by {owner!r}")
        self._approval_transports[clean] = RegisteredApprovalTransport(
            name=clean, present=present_fn, plugin_id=plugin_id, profile_home=str(get_hermes_home().resolve()),
        )
        logger.info("Plugin %s registered approval transport: %s", plugin_id, clean)

    def get_approval_transport(self, name: str):
        """Return a transport only inside the profile that registered it."""
        registered = self._approval_transports.get(str(name).strip().lower())
        if registered is None or registered.profile_home != str(get_hermes_home().resolve()):
            return None
        return registered

    def _collect_directory_manifests(self) -> List[PluginManifest]:
        """Directory manifests in full-discovery order (see :func:`collect_directory_manifests`)."""
        return collect_directory_manifests()

    def has_enabled_portable_mcp(self, raw_config: Mapping[str, Any]) -> bool:
        """Probe enabled portable MCP packages without loading plugins (shares the full-discovery
        manifest collection so precedence/gating cannot diverge)."""
        if env_var_enabled("HERMES_SAFE_MODE"):
            return False
        plugins_config = raw_config.get("plugins")
        if not isinstance(plugins_config, dict):
            return False

        def _names(value: Any) -> Set[str]:
            return {v for v in value if isinstance(v, str)} if isinstance(value, list) else set()

        enabled = _names(plugins_config.get("enabled"))
        disabled = _names(plugins_config.get("disabled", []))
        if not enabled:
            return False
        for lookup_key, manifest in {manifest_key(m): m for m in self._collect_directory_manifests()}.items():
            names = {lookup_key, manifest.name}
            if not manifest.portable or names & disabled or not names & enabled:
                continue
            try:  # lazy: this is a startup probe, keep agent_plugins unimported unless needed
                from plugin_runtime.portable import _discover_mcp
                if _discover_mcp(Path(manifest.path), get_hermes_home() / "plugin-data"
                                 / (manifest.skill_namespace or lookup_key), [], create_data=False):
                    return True
            except (OSError, RuntimeError, ValueError):
                continue  # fail closed on an unreadable package; full discovery reports it
        return False

    def _scan_directory(
        self, path: Path, source: str, skip_names: Optional[Set[str]] = None,
    ) -> List[PluginManifest]:
        """Read manifests under *path* (see :func:`scan_directory`)."""
        return scan_directory(path, source, skip_names=skip_names)

    def _scan_entry_points(self) -> List[PluginManifest]:
        """Read installed plugin entry points (see :func:`discover_entrypoint_manifests`)."""
        return discover_entrypoint_manifests()

    def get_slack_action_handlers(self) -> List[tuple]:
        """``(action_id, callback, plugin_name)`` tuples for the Slack adapter to wire at connect."""
        return list(self._slack_action_handlers)

    def get_platform_handler_factories(self, platform: str) -> List[tuple]:
        """``(factory, plugin_name)`` tuples for one platform; adapters call ``factory(native,
        adapter)`` at connect (see :meth:`PluginContext.register_platform_handler`)."""
        return list(self._platform_handler_factories.get((platform or "").strip().lower(), []))

    def get_telegram_handler_factories(self) -> List[tuple]:
        """Back-compat alias for ``get_platform_handler_factories("telegram")``."""
        return self.get_platform_handler_factories("telegram")

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return a list of info dicts for all discovered plugins."""
        return [
            {
                "name": p.manifest.name, "key": manifest_key(p.manifest), "kind": p.manifest.kind,
                "version": p.manifest.version, "description": p.manifest.description,
                "source": p.manifest.source, "enabled": p.enabled, "tools": len(p.tools_registered),
                "hooks": len(p.hooks_registered), "middleware": len(p.middleware_registered),
                "commands": len(p.commands_registered), "error": p.error,
            } for _key, p in sorted(self._plugins.items())
        ]

    def find_plugin_skill(self, qualified_name: str) -> Optional[Path]:
        """Return the ``Path`` to a plugin skill's SKILL.md, or ``None``."""
        entry = self._plugin_skills.get(qualified_name)
        return entry["path"] if entry else None

    def list_plugin_skills(self, plugin_name: str) -> List[str]:
        """Return sorted bare names of all skills registered by *plugin_name*."""
        prefix = f"{plugin_name}:"
        return sorted(e["bare_name"] for qn, e in self._plugin_skills.items() if qn.startswith(prefix))

    def list_plugin_skill_metadata(self) -> List[Dict[str, Any]]:
        """Return progressive-disclosure metadata for registered plugin skills."""
        return [
            {
                "name": qualified, "description": str(entry.get("description", "")),
                "category": "plugin", "frontmatter": dict(entry.get("frontmatter", {})),
            } for qualified, entry in sorted(self._plugin_skills.items())
        ]

    def has_portable_mcp_servers(self) -> bool:
        return bool(self._portable_mcp_servers)

    def get_portable_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Return a defensive copy of enabled portable MCP server configs."""
        return {name: dict(config) for name, config in self._portable_mcp_servers.items()}

    def get_portable_mcp_server_plugins(self) -> Dict[str, str]:
        return dict(self._portable_mcp_server_plugins)

    def remove_plugin_skill(self, qualified_name: str) -> None:
        """Remove a stale registry entry (silently ignores missing keys)."""
        self._plugin_skills.pop(qualified_name, None)