"""Plugin loading: directory/entry-point module import, deferred bundled platforms, portable packages,
dependency/config-schema warnings. Mixed into :class:`hermes_cli.plugins.PluginManager`.

Origin-internal names (``PluginContext``, ``LoadedPlugin``, ``_PLUGINS_DEBUG`` …) are imported lazily
through ``hermes_cli.plugins`` so tests that patch them on the origin keep working.
"""

from __future__ import annotations

import contextvars
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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Union

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from registration_lifecycle import replacement_coordinator
from hermes_cli.plugins_discovery import ENTRY_POINTS_GROUP, _select_entry_point_group
from hermes_cli.plugins_manifest import PluginManifest, manifest_key, validate_config_schema
from hermes_cli.plugins_state import _plugin_settings_entry

if TYPE_CHECKING:  # pragma: no cover
    from hermes_cli.plugins import LoadedPlugin, PluginContext

logger = logging.getLogger("hermes_cli.plugins")

_NS_PARENT = "hermes_plugins"
_MODULE_NAMESPACE_LOCK = threading.RLock()
_BARE_MODULE_SCOPE: Dict[str, str] = {}  # bare module name -> owning scope_key

# Per-plugin deadline on import + register(): ``plugins.load_timeout_seconds`` (default 10s, 0 disables,
# clamped to the max). A plugin that never returns is skipped with a named reason and loading moves on
# (#108139). Python cannot kill a thread, so the worker is abandoned as a daemon; the cap bounds how many
# abandoned loaders one process may accumulate (#98382) — past it, further loads are refused, not run inline.
_LOAD_TIMEOUT_SECS = 10.0
_MAX_LOAD_TIMEOUT_SECS = 600.0
_MAX_ABANDONED_LOADERS = 8
_ABANDONED_LOADERS: List[threading.Thread] = []
_ABANDONED_LOADERS_LOCK = threading.Lock()
_IN_PLUGIN_LOAD = threading.local()  # ``.active`` on a loader worker thread


class PluginLoadTimeout(Exception):
    """Raised on the loading thread when a plugin's import + ``register()`` overran its deadline."""


def in_plugin_load_worker() -> bool:
    """True on a deadline worker thread; re-entrant discovery from there must not block on its own parent."""
    return bool(getattr(_IN_PLUGIN_LOAD, "active", False))


def _resolve_plugin_load_timeout() -> float:
    """Effective per-plugin load deadline from ``plugins.load_timeout_seconds`` (default 10s; ``0`` runs
    loads inline with no deadline; clamped to ``_MAX_LOAD_TIMEOUT_SECS``)."""
    default = _LOAD_TIMEOUT_SECS
    try:
        from hermes_cli.config import load_config_readonly
        plugins_cfg = (load_config_readonly() or {}).get("plugins")
        if not isinstance(plugins_cfg, dict) or plugins_cfg.get("load_timeout_seconds") is None:
            return default
        timeout = float(plugins_cfg["load_timeout_seconds"])
    except (TypeError, ValueError):
        logger.warning("plugins.load_timeout_seconds is not a number; using default %gs", default)
        return default
    except Exception:
        return default
    if timeout < 0:
        logger.warning("plugins.load_timeout_seconds=%g is negative; using default %gs", timeout, default)
        return default
    if timeout > _MAX_LOAD_TIMEOUT_SECS:
        logger.warning("plugins.load_timeout_seconds=%g exceeds max %gs; clamping", timeout,
                       _MAX_LOAD_TIMEOUT_SECS)
        return _MAX_LOAD_TIMEOUT_SECS
    return timeout


def _reserve_abandoned_loader_slot() -> None:
    """Drop finished abandoned loaders; refuse the load once the live cap is reached. Refusing beats
    loading inline: at the cap the process already holds several hung loaders, so an inline load is the
    exact startup hang this deadline exists to prevent."""
    with _ABANDONED_LOADERS_LOCK:
        _ABANDONED_LOADERS[:] = [t for t in _ABANDONED_LOADERS if t.is_alive()]
        if len(_ABANDONED_LOADERS) < _MAX_ABANDONED_LOADERS:
            return
    raise PluginLoadTimeout(
        f"not loaded: {_MAX_ABANDONED_LOADERS} abandoned plugin loader thread(s) are still running "
        f"(plugins.load_timeout_seconds); restart Hermes to retry"
    )


def run_with_load_deadline(plugin_key: str, ctx: "PluginContext", fn: Callable[[], Any]) -> Any:
    """Run ``fn`` (a plugin's import + ``register()``) under the per-plugin deadline.

    The worker inherits the caller's context (the Hermes-home override is a ContextVar). On timeout the
    worker is abandoned as a daemon, ``ctx`` is marked so any registration it still attempts is ignored,
    and :class:`PluginLoadTimeout` is raised on the calling thread so the usual failure path records the
    reason and disposes whatever was registered before the hang.
    """
    timeout = _resolve_plugin_load_timeout()
    if timeout <= 0:
        return fn()
    _reserve_abandoned_loader_slot()
    outcome: List[Any] = []
    failure: List[BaseException] = []

    def _worker() -> None:
        _IN_PLUGIN_LOAD.active = True
        try:
            outcome.append(fn())
        except BaseException as exc:  # re-raised on the loading thread, KeyboardInterrupt included
            failure.append(exc)

    worker = threading.Thread(
        target=contextvars.copy_context().run, args=(_worker,), name=f"plugin-load:{plugin_key}", daemon=True,
    )
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        ctx._abandon_load()
        with _ABANDONED_LOADERS_LOCK:
            _ABANDONED_LOADERS.append(worker)
        raise PluginLoadTimeout(f"load timed out after {timeout:g}s (import + register() never returned)")
    if failure:
        raise failure[0]
    return outcome[0]


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


def _load_error_text(exc: BaseException) -> str:
    """Human-readable load failure; ``sys.exit(0)`` has an empty ``str()`` so name the class and code."""
    if isinstance(exc, SystemExit):
        return f"SystemExit({exc.code!r}) raised during import/register()"
    return str(exc)


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

    def _register_deferred_platform(self, manifest: PluginManifest) -> None:
        """Register a lazy loader for a bundled platform: the adapter imports only when the
        ``platform_registry`` is first asked for it; a placeholder ``LoadedPlugin`` keeps it visible in
        ``hermes plugins list`` until then."""
        from hermes_cli.plugins import LoadedPlugin
        lookup_key = manifest_key(manifest)
        loaded = LoadedPlugin(manifest=manifest, enabled=True, deferred=True)
        self._plugins[lookup_key] = loaded
        if not self._lease_deferred_platform(manifest, lookup_key):
            # Fall back to eager loading so the platform is never silently lost. Runs outside the
            # replacement transaction: the eager load's register() executes on a deadline worker, whose
            # registrations need the coordinator lock this thread would otherwise still hold.
            self._load_plugin(manifest)
            return
        self._register_deferred_platform_tools(manifest, loaded)

    @_serialized_replacement
    def _lease_deferred_platform(self, manifest: PluginManifest, lookup_key: str) -> bool:
        """Publish the deferred loader as a ledger-owned lease; False when the registry refused it."""
        platform_name = self._platform_name_from_manifest(manifest)
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
            logger.debug(
                "Deferred platform registration failed for '%s'; eager-loading", lookup_key, exc_info=True)
            return False
        return True

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
        except (Exception, SystemExit) as exc:
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
        """Warn about declared pip dependencies missing at load time. Installing happens at
        ``hermes plugins install``/``enable`` and after ``hermes update`` (``hermes_cli.plugin_python_deps``)
        under core constraints; the loader itself never installs — import time is not a consent point.
        """
        deps = manifest.python_dependencies
        if not deps:
            return
        key = manifest_key(manifest)
        missing = [req for req in deps if _dist_installed(req) is False]
        if missing:
            logger.warning(
                "Plugin %s declares Python dependencies that are not "
                "installed: %s. Run `hermes plugins enable %s` to install them, "
                "or install them yourself: pip install %s",
                key, ", ".join(missing), key, " ".join(f"'{m}'" for m in missing),
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
        # requires_hermes gate: skip cleanly (no import, no traceback) on a version mismatch.
        from hermes_cli.plugins_manifest import requires_hermes_error
        reason = requires_hermes_error(manifest)
        if reason:
            loaded.error = reason
            logger.warning("Plugin '%s' skipped: %s", plugin_key, reason)
            self._plugins[plugin_key] = loaded
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
        ctx = PluginContext(manifest, self)

        def _import_and_register() -> bool:
            """Import + register() — the part a plugin controls, so the part the deadline covers."""
            # Reuse a deferred platform's already-imported package so its body doesn't run twice.
            # See #78050.
            module = self._predeclared_modules.pop(plugin_key, None)
            if module is None and manifest.source in {"user", "project", "bundled"}:
                module = self._load_directory_module(manifest, module_name=module_name)
            elif module is None:
                module = self._load_entrypoint_module(manifest)
            register_fn = None
            if module is not None and not isinstance(module, types.ModuleType) and callable(module):
                # An entry point declared as ``module:function`` resolves to the function object itself via
                # ``ep.load()``, not its module (#72052).
                register_fn = module
                module = sys.modules.get(getattr(register_fn, "__module__", ""))
            loaded.module = module
            if register_fn is None:
                register_fn = getattr(module, "register", None)
            if register_fn is None:
                loaded.error = "no register() function"
                logger.warning("Plugin '%s' has no register() function", manifest.name)
                return False
            register_fn(ctx)
            return True

        try:
            if run_with_load_deadline(plugin_key, ctx, _import_and_register):
                self._attribute_registrations(loaded, plugin_key, registration_start)
                loaded.enabled = True
                from hermes_cli.plugins_ledger import _hook_source_of

                self._drop_fallback_hooks(_hook_source_of(manifest.name, loaded.module))
        except (Exception, SystemExit) as exc:
            # SystemExit too: a plugin module with an unguarded ``main()``/``sys.exit()`` must not take the
            # whole process (and every other plugin's registry) down with it; KeyboardInterrupt still propagates.
            # PluginLoadTimeout lands here as well: the abandoned worker's later registrations are refused
            # by ``ctx``, and whatever it registered before hanging is disposed below.
            owned = [r for r in self._registration_order if r.plugin_key == plugin_key]
            self._dispose_registrations(owned)
            self._forget_registrations(owned)
            loaded.error = _load_error_text(exc)
            # register() may have subscribed before raising; a failed plugin must leave no callable reachable
            # from later event dispatch.
            self._remove_plugin_subscriptions(plugin_key)
            logger.warning("Failed to load plugin '%s': %s", manifest.name, _load_error_text(exc), exc_info=_PLUGINS_DEBUG)
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
                self._portable_mcp_server_plugins[internal_name] = lookup_key
            loaded.enabled = True
        except (Exception, SystemExit) as exc:
            loaded.error = _load_error_text(exc)
            logger.warning("Failed to load Agent Plugin '%s': %s", lookup_key, loaded.error)
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

    def _load_entrypoint_module(self, manifest: PluginManifest) -> Union[types.ModuleType, Callable[..., Any]]:
        """Load a pip-installed plugin via its entry-point reference: the module for a bare ``module`` target,
        the referenced attribute (normally ``register``) for the ``module:function`` form."""
        for ep in _select_entry_point_group(importlib.metadata.entry_points(), ENTRY_POINTS_GROUP):
            if ep.name == manifest.name:
                return ep.load()
        raise ImportError(f"Entry point '{manifest.name}' not found in group '{ENTRY_POINTS_GROUP}'")
