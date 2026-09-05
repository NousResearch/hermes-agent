"""Plugin loading: directory/entry-point module import, deferred bundled platforms, portable packages,
dependency/config-schema warnings. Mixed into :class:`hermes_cli.plugins.PluginManager`.

Origin-internal names (``PluginContext``, ``LoadedPlugin``, ``_PLUGINS_DEBUG`` …) are imported lazily
through ``hermes_cli.plugins`` so tests that patch them on the origin keep working.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import logging
import re
import sys
import threading
import types
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from registration_lifecycle import replacement_coordinator
from hermes_cli.plugins_discovery import ENTRY_POINTS_GROUP, _select_entry_point_group
from hermes_cli.plugins_manifest import PluginManifest, manifest_key, validate_config_schema
from hermes_cli.plugins_state import _plugin_settings_entry

if TYPE_CHECKING:  # pragma: no cover
    from hermes_cli.plugins import LoadedPlugin

logger = logging.getLogger("hermes_cli.plugins")

_NS_PARENT = "hermes_plugins"
_MODULE_NAMESPACE_LOCK = threading.RLock()
_BARE_MODULE_SCOPE: Dict[str, str] = {}  # bare module name -> owning scope_key


def _evict_modules(module_name: str) -> None:
    """Drop ``module_name`` and every ``module_name.*`` submodule from ``sys.modules``."""
    prefix = f"{module_name}."
    for name in [n for n in sys.modules if n == module_name or n.startswith(prefix)]:
        del sys.modules[name]


def _serialized_replacement(method):
    """Make snapshot → write → lease attachment one atomic transaction."""
    @wraps(method)
    def wrapped(*args, **kwargs):
        with replacement_coordinator.transaction():
            return method(*args, **kwargs)

    return wrapped


@contextmanager
def _plugin_home_scope(home: Path):
    """Bind discovery and loading to the manager's immutable Hermes home."""
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _dist_installed(req: str) -> Optional[bool]:
    """Best-effort presence probe on a requirement's distribution name; ``None`` when unprobeable."""
    dist = re.split(r"[<>=!~\[;\s]", req, maxsplit=1)[0].strip()
    if not dist:
        return None
    try:
        importlib.metadata.version(dist)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
    except Exception:
        return None


class PluginLoaderMixin:
    @staticmethod
    def _platform_name_from_manifest(manifest: PluginManifest) -> str:
        """Derive the platform name without importing the adapter: strip a trailing ``-platform`` from the
        manifest name, else the directory basename (the bundled convention)."""
        name = manifest.name or ""
        if name.endswith("-platform"):
            return name[: -len("-platform")]
        return Path(manifest.path).name if manifest.path else name

    @_serialized_replacement
    def _register_deferred_platform(self, manifest: PluginManifest) -> None:
        """Register a lazy loader for a bundled platform: the adapter imports only when the
        ``platform_registry`` is first asked for it; a placeholder ``LoadedPlugin`` keeps it visible in
        ``hermes plugins list`` until then."""
        from hermes_cli.plugins import LoadedPlugin
        lookup_key = manifest_key(manifest)
        platform_name = self._platform_name_from_manifest(manifest)
        loaded = LoadedPlugin(manifest=manifest, enabled=True, deferred=True)
        self._plugins[lookup_key] = loaded
        try:
            from gateway.platform_registry import platform_registry
            scope = self.scope_key

            def _loader(_manifest: PluginManifest = manifest) -> None:
                # Lock before checking cancellation: if an unload won the race it restored the predecessor
                # and this loader must publish nothing; if loading won, unload waits and disposes the set.
                with self._discovery_lock, _plugin_home_scope(self.home_path):
                    if platform_registry.is_deferred_load_cancelled(platform_name, scope=scope):
                        return
                    self._load_plugin_scoped(_manifest)

            previous = platform_registry.snapshot_registration(platform_name, scope=scope)
            platform_registry.register_deferred(platform_name, _loader, scope=scope)
            current = platform_registry.snapshot_registration(platform_name, scope=scope)
            if current[0] is None and current[1] is _loader:
                self._plugin_platform_names.add(platform_name)
                self._track_scoped_registration(
                    manifest, "platform", platform_name, platform_registry, current, previous,
                    finalize=lambda: self._remove_platform_name_if_unowned(platform_name),
                )
            logger.debug("Registered deferred platform loader: %s (plugin=%s)", platform_name, lookup_key)
        except Exception:
            # Fall back to eager loading so the platform is never silently lost.
            logger.debug(
                "Deferred platform registration failed for '%s'; eager-loading", lookup_key, exc_info=True)
            self._load_plugin(manifest)
            return
        self._register_deferred_platform_tools(manifest, loaded)

    def _register_deferred_platform_tools(self, manifest: PluginManifest, loaded: LoadedPlugin) -> None:
        """Register a deferred platform's *client* tools without its adapter. Deferring the plugin would
        otherwise defer its outbound tools too, so CLI/TUI processes (which never materialize platforms)
        would miss them in ``hermes tools`` / ``platform_toolsets``. Opt-in is explicit via ``provides_tools``;
        tools live in a ``tools`` submodule so ``__init__`` stays import-light.

        A platform plugin can ship two independent things: an inbound adapter (heavy — it imports the
        platform SDK) and outbound client tools the agent calls like any other tool. Deferring the plugin
        defers both, so in a CLI/TUI process the client tools never register at all: ``resolve_toolset()``
        returns ``[]``, the toolset is missing from the ``hermes tools`` checklist, and even an explicit
        ``platform_toolsets`` entry is dropped because the key is unknown. The same tools work in
        gateway/web processes only because those materialize every platform at startup (issue #78050).
        Opting in is explicit: the manifest must declare ``provides_tools`` (the field the plugin list and
        web server already read to name a plugin's tools, per #78538). Keying off the mere presence of a
        ``tools.py`` would opt a plugin in by accident — a platform is free to put internal helpers there —
        and would leave the contract invisible to anyone reading the manifest. ``tools.py`` remains where
        the code is imported from; ``provides_tools`` is what asks for it. A platform that does not declare
        the field is untouched and stays fully deferred.
        """
        from hermes_cli.plugins import PluginContext, _PLUGINS_DEBUG
        if not manifest.provides_tools:
            return
        lookup_key = manifest_key(manifest)
        # Never let a client-tool import break discovery — the platform stays deferred and behaves exactly
        # as it did before. But a broken tools.py produces the #78050 symptom itself (declared tools missing
        # from the session), so this has to be visible without turning on debug logging to find it. Where it
        # failed is the first thing an operator needs: nothing registered points at the import or the module
        # body, a partial run points at one tool's definition, and a full run that still raised points past
        # the registrations entirely.
        declared = list(manifest.provides_tools)
        plugin_dir = Path(manifest.path) if manifest.path else None
        if plugin_dir is None or not (plugin_dir / "tools.py").is_file():
            # Declared but undeliverable — staying quiet reproduces the very symptom this fixes.
            logger.warning(
                # Staying quiet here reproduces the exact symptom this path exists to fix — tools the
                # manifest promises, silently absent from the session (#78050) — so say so.
                "Plugin '%s' declares provides_tools %s but has no tools.py; "
                "those tools will not be available in CLI/TUI sessions.", lookup_key, declared,
            )
            return
        before = set(self._plugin_tool_names)  # lets the failure path credit partial registrations

        def _credit() -> List[str]:
            """Attribute every tool registered since ``before`` to this plugin."""
            registered = [t for t in self._plugin_tool_names if t not in before]
            if registered:
                loaded.tools_registered = registered
                self._predeclared_tools[lookup_key] = registered
            return registered

        try:
            module = self._load_directory_module(manifest)
            # Record the module even if nothing registers: the package body has run, so materializing the
            # adapter later must reuse it rather than execute it twice.
            loaded.module = module
            self._predeclared_modules[lookup_key] = module
            tools_module = importlib.import_module(f"{module.__name__}.tools")
            register_tools = getattr(tools_module, "register_tools", None)
            if register_tools is None:
                logger.warning(
                    "Plugin '%s' declares provides_tools %s but its tools.py "
                    "has no register_tools(ctx); those tools will not be "
                    "available in CLI/TUI sessions.", lookup_key, declared,
                )
                return
            register_tools(PluginContext(manifest, self))
            registered = _credit()
            logger.debug(
                "Deferred platform '%s': pre-registered %d client tool(s) %s", lookup_key, len(registered),
                registered,
            )
        except Exception as exc:
            # Tools registered before the raise are live: credit them or `hermes plugins list` under-reports
            # (and _load_plugin's later diff would miss them too). Never break discovery (the platform stays
            # deferred), but a broken tools.py IS the symptom, so warn — and say where it failed first.
            partial, total = _credit(), len(declared)
            complete = len(partial) >= total
            scope = (
                f"before registering any of its {total} declared tool(s)" if not partial
                else f"after registering all {total} declared tool(s)" if complete
                else f"after registering {len(partial)} of {total} declared tool(s)"
            )
            logger.warning(
                "Plugin '%s': client-tool pre-registration failed %s (%s).%s", lookup_key, scope, exc,
                "" if complete else " The remainder will be missing from CLI/TUI sessions.",
                exc_info=_PLUGINS_DEBUG,
            )

    def _warn_python_dependencies(self, manifest: PluginManifest) -> None:
        """Warn about missing declared pip dependencies with an install hint — NEVER auto-install.

        See #64165.
        python_dependencies is a declaration seam ONLY: Hermes validates and prints the requirements with an
        install hint but NEVER auto-installs them. The isolation design (constraints installs vs. vendored
        dirs vs. conflict-detection-and-refusal) is an explicitly deferred follow-up — see the round-2
        review on #64165 and #15220.
        """
        deps = manifest.python_dependencies
        if not deps:
            return
        key = manifest_key(manifest)
        missing = [req for req in deps if _dist_installed(req) is False]
        if missing:
            logger.warning(
                "Plugin %s declares Python dependencies that are not "
                "installed: %s. Hermes does not install plugin dependencies "
                "automatically; install them yourself, e.g.: pip install %s",
                key, ", ".join(missing), " ".join(f"'{m}'" for m in missing),
            )
        else:
            logger.debug("Plugin %s python_dependencies satisfied: %s", key, ", ".join(deps))

    def _validate_plugin_config_schema(self, manifest: PluginManifest) -> None:
        """Warn (never block) on plugins.entries.<id> settings that violate config_schema.

        See #64165.
        """
        if not manifest.config_schema:
            return
        plugin_id = manifest_key(manifest)
        settings: Mapping[str, Any] = {}
        try:
            from hermes_cli.config import load_config
            entry = _plugin_settings_entry(load_config() or {}, plugin_id) or {}
            raw = entry.get("settings")
            if not isinstance(raw, Mapping):
                raw = entry.get("config")  # migration fallback mirroring ctx.get_config
            settings = raw if isinstance(raw, Mapping) else {}
        except Exception:
            settings = {}
        for warning in validate_config_schema(plugin_id, manifest.config_schema, settings):
            logger.warning("Plugin %s config: %s", plugin_id, warning)

    def _load_plugin(self, manifest: PluginManifest) -> None:
        """Import a plugin module and call its ``register(ctx)`` function."""
        with self._discovery_lock, _plugin_home_scope(self.home_path):
            self._load_plugin_scoped(manifest)

    def _load_plugin_scoped(self, manifest: PluginManifest) -> None:
        """Load one plugin with the manager's home bound as current."""
        from hermes_cli.plugins import LoadedPlugin, PluginContext, _PLUGINS_DEBUG
        loaded = LoadedPlugin(manifest=manifest)
        plugin_key = manifest_key(manifest)
        logger.debug(
            "Loading plugin '%s' (source=%s, kind=%s, path=%s)",
            plugin_key, manifest.source, manifest.kind, manifest.path,
        )
        if manifest.portable:
            self._load_portable_plugin(manifest, loaded)
            return
        # After the compat-removal date an external plugin that still imports pre-decomposition paths is
        # skipped with a clear reason instead of dying on ImportError mid-register (hermes_cli.plugin_compat).
        from hermes_cli.plugin_compat import disable_reason
        reason = disable_reason(manifest)
        if reason:
            loaded.error = reason
            logger.warning("Plugin '%s' not loaded: %s", manifest.name, reason)
            self._plugins[plugin_key] = loaded
            return
        registration_start = len(self._registration_order)
        module_name = self._policy_module_name(manifest)
        self._track_tool_override_policy(manifest, module_name)
        try:
            # Reuse a deferred platform's already-imported package so its body doesn't run twice.
            # See #78050.
            module = self._predeclared_modules.pop(plugin_key, None)
            if module is None and manifest.source in {"user", "project", "bundled"}:
                module = self._load_directory_module(manifest, module_name=module_name)
            elif module is None:
                module = self._load_entrypoint_module(manifest)
            loaded.module = module
            register_fn = getattr(module, "register", None)
            if register_fn is None:
                loaded.error = "no register() function"
                logger.warning("Plugin '%s' has no register() function", manifest.name)
            else:
                register_fn(PluginContext(manifest, self))
                self._attribute_registrations(loaded, plugin_key, registration_start)
                loaded.enabled = True
        except Exception as exc:
            owned = [r for r in self._registration_order if r.plugin_key == plugin_key]
            self._dispose_registrations(owned)
            self._forget_registrations(owned)
            loaded.error = str(exc)
            # register() may have subscribed before raising; a failed plugin must leave no callable reachable
            # from later event dispatch.
            self._remove_plugin_subscriptions(plugin_key)
            logger.warning("Failed to load plugin '%s': %s", manifest.name, exc, exc_info=_PLUGINS_DEBUG)
        # The failure path swept this plugin's whole ledger (not just the registration_start slice), so
        # discovery-time pre-registrations are gone too.
        # There is no live tool left to credit — attribution and the registry agree at zero. Only the
        # success path pops _predeclared_tools, so drop the entry here rather than let the bookkeeping
        # outlive the load attempt (#78050).
        if not loaded.enabled:
            self._predeclared_tools.pop(plugin_key, None)
        self._plugins[plugin_key] = loaded

    def _track_tool_override_policy(self, manifest: PluginManifest, module_name: str) -> None:
        """Install the plugin's tool-override policy in tools.registry as a ledger-owned lease."""
        from hermes_cli.plugins import PluginContext
        from tools.registry import registry as _registry
        scope = self.scope_key
        with replacement_coordinator.transaction():
            previous_policy = _registry.snapshot_plugin_override_policy(module_name, scope=scope)
            current_policy = _registry.register_plugin_override_policy(
                module_name, PluginContext(manifest, self)._tool_override_allowed(""), scope=scope,
            )
            policy_lease = replacement_coordinator.acquire(
                ("tool_override_policy", scope, module_name), current=current_policy,
                previous=previous_policy,
                restore=lambda replacement: _registry.restore_plugin_override_policy(
                    module_name, current_policy, replacement, scope=scope,
                ),
            )
            self._track_registration(manifest, "tool_override_policy", module_name, policy_lease.dispose)

    def _attribute_registrations(
        self, loaded: LoadedPlugin, plugin_key: str, registration_start: int
    ) -> None:
        """Fill ``loaded.*_registered`` from the ledger slice this plugin's register() produced."""
        registrations = [
            r for r in self._registration_order[registration_start:]
            if r.plugin_key == plugin_key and r.active
        ]

        def _keys(kind: str) -> List[str]:
            return [r.key for r in registrations if r.kind == kind]

        # Discovery-time tools predate registration_start; credit them back or `hermes plugins list`
        # under-reports once the deferred adapter materializes.
        predeclared = [t for t in self._predeclared_tools.pop(plugin_key, []) if t in self._plugin_tool_names]
        loaded.tools_registered = predeclared + [k for k in _keys("tool") if k not in predeclared]
        loaded.hooks_registered = _keys("hook")
        loaded.middleware_registered = _keys("middleware")
        loaded.commands_registered = _keys("command")
        logger.debug(
            "  registered: %d tool(s), %d hook(s), %d middleware, %d slash command(s), %d CLI command(s)",
            len(loaded.tools_registered), len(loaded.hooks_registered),
            len(loaded.middleware_registered), len(loaded.commands_registered),
            sum(1 for c in self._cli_commands if c in _keys("cli_command")),
        )

    def _load_portable_plugin(self, manifest: PluginManifest, loaded: LoadedPlugin) -> None:
        """Load validated portable components without importing Python code."""
        from hermes_cli.plugins import PluginContext
        lookup_key = manifest_key(manifest)
        try:
            from hermes_cli.agent_plugins import load_agent_plugin
            package = load_agent_plugin(
                Path(manifest.path), get_hermes_home() / "plugin-data" / manifest.skill_namespace)
            ctx = PluginContext(manifest, self)
            for diagnostic in package.diagnostics:
                logger.warning("Agent Plugin '%s' [%s]: %s", lookup_key, diagnostic.scope, diagnostic.message)
            for skill in package.skills:
                try:
                    ctx.register_skill(skill.name, skill.skill_md, skill.description, skill.frontmatter)
                except Exception as exc:
                    logger.warning("Agent Plugin '%s' skill '%s' skipped: %s", lookup_key, skill.name, exc)
            for server_name, config in package.mcp_servers.items():
                internal_name = f"{manifest.skill_namespace}__{server_name}"
                if internal_name in self._portable_mcp_servers:
                    logger.warning("Agent Plugin '%s' MCP server collision: %s", lookup_key, internal_name)
                    continue
                self._portable_mcp_servers[internal_name] = dict(config)
            loaded.enabled = True
        except Exception as exc:
            loaded.error = str(exc)
            logger.warning("Failed to load Agent Plugin '%s': %s", lookup_key, exc)
        self._plugins[lookup_key] = loaded

    def _directory_module_name(self, manifest: PluginManifest) -> str:
        """Profile-safe import namespace for a directory plugin: the bare ``hermes_plugins.<slug>`` for the
        first scope that claims it, a ``__home_<digest>`` suffix for any other scope."""
        slug = manifest_key(manifest).replace("/", "__").replace("-", "_")
        bare_name = f"{_NS_PARENT}.{slug}"
        with _MODULE_NAMESPACE_LOCK:
            if _BARE_MODULE_SCOPE.setdefault(bare_name, self.scope_key) == self.scope_key:
                return bare_name
            digest = hashlib.sha256(self.scope_key.encode("utf-8")).hexdigest()[:12]
            return f"{bare_name}__home_{digest}"

    def _policy_module_name(self, manifest: PluginManifest) -> str:
        """Return the module prefix whose callbacks inherit plugin policy."""
        if manifest.source == "entrypoint" and manifest.path:
            module_name = str(manifest.path).partition(":")[0].strip()
            if module_name:
                return module_name
        return self._directory_module_name(manifest)

    def _load_directory_module(
        self, manifest: PluginManifest, *, module_name: Optional[str] = None,
    ) -> types.ModuleType:
        """Import a directory plugin as ``hermes_plugins.<slug>`` (slug from ``manifest.key`` so
        ``image_gen/openai`` cannot collide with ``tts/openai``)."""
        plugin_dir = Path(manifest.path)  # type: ignore[arg-type]
        init_file = plugin_dir / "__init__.py"
        if not init_file.exists():
            raise FileNotFoundError(f"No __init__.py in {plugin_dir}")
        if _NS_PARENT not in sys.modules:
            ns_pkg = types.ModuleType(_NS_PARENT)
            ns_pkg.__path__ = []  # type: ignore[attr-defined]
            ns_pkg.__package__ = _NS_PARENT
            sys.modules[_NS_PARENT] = ns_pkg
        module_name = module_name or self._directory_module_name(manifest)
        # Evict stale entries for this slug (same slug cached from another Hermes home, or an earlier force
        # reload). Replacing only sys.modules[module_name] is not enough: the plugin's relative imports are
        # cached as "module_name.sub" and resolve from sys.modules first, so a stale submodule would keep
        # serving the previous load's code/state.
        _evict_modules(module_name)
        spec = importlib.util.spec_from_file_location(
            module_name, init_file, submodule_search_locations=[str(plugin_dir)])
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {init_file}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = module_name
        module.__path__ = [str(plugin_dir)]  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # Don't leave a half-initialized module (or its partially imported relative submodules) cached — a
            # retry or a same-slug plugin in another profile would inherit broken state.
            _evict_modules(module_name)
            raise
        return module

    def _load_entrypoint_module(self, manifest: PluginManifest) -> types.ModuleType:
        """Load a pip-installed plugin via its entry-point reference."""
        for ep in _select_entry_point_group(importlib.metadata.entry_points(), ENTRY_POINTS_GROUP):
            if ep.name == manifest.name:
                return ep.load()
        raise ImportError(f"Entry point '{manifest.name}' not found in group '{ENTRY_POINTS_GROUP}'")
