"""
Hermes Plugin System
====================

Discovers, loads, and manages plugins from three sources:

1. **User plugins**   – ``~/.hermes/plugins/<name>/``
2. **Project plugins** – ``./.hermes/plugins/<name>/`` (opt-in via
   ``HERMES_ENABLE_PROJECT_PLUGINS``)
3. **Pip plugins**     – packages that expose the ``hermes_agent.plugins``
   entry-point group.

Each directory plugin must contain a ``plugin.yaml`` manifest **and** an
``__init__.py`` with a ``register(ctx)`` function.

Lifecycle hooks
---------------
Plugins may register callbacks for any of the hooks in ``VALID_HOOKS``.
The agent core calls ``invoke_hook(name, **kwargs)`` at the appropriate
points.

Tool registration
-----------------
``PluginContext.register_tool()`` delegates to ``tools.registry.register()``
so plugin-defined tools appear alongside the built-in tools.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import yaml
except ImportError:  # pragma: no cover – yaml is optional at import time
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_HOOKS: Set[str] = {
    "pre_tool_call",
    "post_tool_call",
    "pre_llm_call",
    "post_llm_call",
    "on_session_start",
    "on_session_end",
}

ENTRY_POINTS_GROUP = "hermes_agent.plugins"

_NS_PARENT = "hermes_plugins"


def _env_enabled(name: str) -> bool:
    """Return True when an env var is set to a truthy opt-in value."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PluginManifest:
    """Parsed representation of a plugin.yaml manifest."""

    name: str
    version: str = ""
    description: str = ""
    author: str = ""
    requires_env: List[str] = field(default_factory=list)
    provides_tools: List[str] = field(default_factory=list)
    provides_hooks: List[str] = field(default_factory=list)
    source: str = ""        # "user", "project", or "entrypoint"
    path: Optional[str] = None


@dataclass
class LoadedPlugin:
    """Runtime state for a single loaded plugin."""

    manifest: PluginManifest
    module: Optional[types.ModuleType] = None
    tools_registered: List[str] = field(default_factory=list)
    hooks_registered: List[str] = field(default_factory=list)
    enabled: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# PluginContext  – handed to each plugin's ``register()`` function
# ---------------------------------------------------------------------------

class PluginContext:
    """Facade given to plugins so they can register tools and hooks."""

    def __init__(self, manifest: PluginManifest, manager: "PluginManager"):
        self.manifest = manifest
        self._manager = manager

    # -- tool registration --------------------------------------------------

    def register_tool(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable | None = None,
        requires_env: list | None = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
    ) -> None:
        """Register a tool in the global registry **and** track it as plugin-provided."""
        from tools.registry import registry

        registry.register(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            requires_env=requires_env,
            is_async=is_async,
            description=description,
            emoji=emoji,
        )
        self._manager._plugin_tool_names.add(name)
        logger.debug("Plugin %s registered tool: %s", self.manifest.name, name)

    # -- hook registration --------------------------------------------------

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a lifecycle hook callback.

        Unknown hook names produce a warning but are still stored so
        forward-compatible plugins don't break.
        """
        if hook_name not in VALID_HOOKS:
            logger.warning(
                "Plugin '%s' registered unknown hook '%s' "
                "(valid: %s)",
                self.manifest.name,
                hook_name,
                ", ".join(sorted(VALID_HOOKS)),
            )
        self._manager._hooks.setdefault(hook_name, []).append(callback)
        logger.debug("Plugin %s registered hook: %s", self.manifest.name, hook_name)


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------

class PluginManager:
    """Central manager that discovers, loads, and invokes plugins."""

    def __init__(self) -> None:
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._plugin_tool_names: Set[str] = set()
        self._discovered: bool = False

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def discover_and_load(self) -> None:
        """Scan all plugin sources and load each plugin found."""
        if self._discovered:
            return
        self._discovered = True

        manifests: List[PluginManifest] = []

        # 1. User plugins (~/.hermes/plugins/)
        hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
        user_dir = Path(hermes_home) / "plugins"
        manifests.extend(self._scan_directory(user_dir, source="user"))

        # 2. Project plugins (./.hermes/plugins/)
        if _env_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
            project_dir = Path.cwd() / ".hermes" / "plugins"
            manifests.extend(self._scan_directory(project_dir, source="project"))

        # 3. Pip / entry-point plugins
        manifests.extend(self._scan_entry_points())

        # Load each manifest
        for manifest in manifests:
            self._load_plugin(manifest)

        if manifests:
            logger.info(
                "Plugin discovery complete: %d found, %d enabled",
                len(self._plugins),
                sum(1 for p in self._plugins.values() if p.enabled),
            )

    # -----------------------------------------------------------------------
    # Directory scanning
    # -----------------------------------------------------------------------

    def _scan_directory(self, path: Path, source: str) -> List[PluginManifest]:
        """Read ``plugin.yaml`` manifests from subdirectories of *path*."""
        manifests: List[PluginManifest] = []
        if not path.is_dir():
            return manifests

        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            manifest_file = child / "plugin.yaml"
            if not manifest_file.exists():
                manifest_file = child / "plugin.yml"
            if not manifest_file.exists():
                logger.debug("Skipping %s (no plugin.yaml)", child)
                continue

            try:
                if yaml is None:
                    logger.warning("PyYAML not installed – cannot load %s", manifest_file)
                    continue
                data = yaml.safe_load(manifest_file.read_text()) or {}
                manifest = PluginManifest(
                    name=data.get("name", child.name),
                    version=str(data.get("version", "")),
                    description=data.get("description", ""),
                    author=data.get("author", ""),
                    requires_env=data.get("requires_env", []),
                    provides_tools=data.get("provides_tools", []),
                    provides_hooks=data.get("provides_hooks", []),
                    source=source,
                    path=str(child),
                )
                manifests.append(manifest)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", manifest_file, exc)

        return manifests

    # -----------------------------------------------------------------------
    # Entry-point scanning
    # -----------------------------------------------------------------------

    def _scan_entry_points(self) -> List[PluginManifest]:
        """Check ``importlib.metadata`` for pip-installed plugins."""
        manifests: List[PluginManifest] = []
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups; earlier returns dict
            if hasattr(eps, "select"):
                group_eps = eps.select(group=ENTRY_POINTS_GROUP)
            elif isinstance(eps, dict):
                group_eps = eps.get(ENTRY_POINTS_GROUP, [])
            else:
                group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]

            for ep in group_eps:
                manifest = PluginManifest(
                    name=ep.name,
                    source="entrypoint",
                    path=ep.value,
                )
                manifests.append(manifest)
        except Exception as exc:
            logger.debug("Entry-point scan failed: %s", exc)

        return manifests

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def _load_plugin(self, manifest: PluginManifest) -> None:
        """Import a plugin module and call its ``register(ctx)`` function."""
        loaded = LoadedPlugin(manifest=manifest)

        try:
            if manifest.source in ("user", "project"):
                module = self._load_directory_module(manifest)
            else:
                module = self._load_entrypoint_module(manifest)

            loaded.module = module

            # Call register()
            register_fn = getattr(module, "register", None)
            if register_fn is None:
                loaded.error = "no register() function"
                logger.warning("Plugin '%s' has no register() function", manifest.name)
            else:
                ctx = PluginContext(manifest, self)
                register_fn(ctx)
                loaded.tools_registered = [
                    t for t in self._plugin_tool_names
                    if t not in {
                        n
                        for name, p in self._plugins.items()
                        for n in p.tools_registered
                    }
                ]
                loaded.hooks_registered = list(
                    {
                        h
                        for h, cbs in self._hooks.items()
                        if cbs  # non-empty
                    }
                    - {
                        h
                        for name, p in self._plugins.items()
                        for h in p.hooks_registered
                    }
                )
                loaded.enabled = True

        except Exception as exc:
            loaded.error = str(exc)
            logger.warning("Failed to load plugin '%s': %s", manifest.name, exc)

        self._plugins[manifest.name] = loaded

    def _load_directory_module(self, manifest: PluginManifest) -> types.ModuleType:
        """Import a directory-based plugin as ``hermes_plugins.<name>``."""
        plugin_dir = Path(manifest.path)  # type: ignore[arg-type]
        init_file = plugin_dir / "__init__.py"
        if not init_file.exists():
            raise FileNotFoundError(f"No __init__.py in {plugin_dir}")

        # Ensure the namespace parent package exists
        if _NS_PARENT not in sys.modules:
            ns_pkg = types.ModuleType(_NS_PARENT)
            ns_pkg.__path__ = []  # type: ignore[attr-defined]
            ns_pkg.__package__ = _NS_PARENT
            sys.modules[_NS_PARENT] = ns_pkg

        module_name = f"{_NS_PARENT}.{manifest.name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_file,
            submodule_search_locations=[str(plugin_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {init_file}")

        module = importlib.util.module_from_spec(spec)
        module.__package__ = module_name
        module.__path__ = [str(plugin_dir)]  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _load_entrypoint_module(self, manifest: PluginManifest) -> types.ModuleType:
        """Load a pip-installed plugin via its entry-point reference."""
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            group_eps = eps.select(group=ENTRY_POINTS_GROUP)
        elif isinstance(eps, dict):
            group_eps = eps.get(ENTRY_POINTS_GROUP, [])
        else:
            group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]

        for ep in group_eps:
            if ep.name == manifest.name:
                return ep.load()

        raise ImportError(
            f"Entry point '{manifest.name}' not found in group '{ENTRY_POINTS_GROUP}'"
        )

    # -----------------------------------------------------------------------
    # Hook invocation
    # -----------------------------------------------------------------------

    def invoke_hook(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """Call all registered callbacks for *hook_name*.

        Each callback is wrapped in its own try/except so a misbehaving
        plugin cannot break the core agent loop.

        Returns a list of non-``None`` return values from callbacks.
        This allows hooks like ``pre_llm_call`` to contribute context
        that the agent core can collect and inject.
        """
        callbacks = self._hooks.get(hook_name, [])
        results: List[Any] = []
        for cb in callbacks:
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception as exc:
                logger.warning(
                    "Hook '%s' callback %s raised: %s",
                    hook_name,
                    getattr(cb, "__name__", repr(cb)),
                    exc,
                )
        return results

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return a list of info dicts for all discovered plugins."""
        result: List[Dict[str, Any]] = []
        for name, loaded in sorted(self._plugins.items()):
            result.append(
                {
                    "name": name,
                    "version": loaded.manifest.version,
                    "description": loaded.manifest.description,
                    "source": loaded.manifest.source,
                    "enabled": loaded.enabled,
                    "tools": len(loaded.tools_registered),
                    "hooks": len(loaded.hooks_registered),
                    "error": loaded.error,
                }
            )
        return result


# ---------------------------------------------------------------------------
# Module-level singleton & convenience functions
# ---------------------------------------------------------------------------

_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Return (and lazily create) the global PluginManager singleton."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def discover_plugins() -> None:
    """Discover and load all plugins (idempotent)."""
    get_plugin_manager().discover_and_load()


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Invoke a lifecycle hook on all loaded plugins.

    Returns a list of non-``None`` return values from plugin callbacks.
    """
    return get_plugin_manager().invoke_hook(hook_name, **kwargs)


def get_plugin_tool_names() -> Set[str]:
    """Return the set of tool names registered by plugins."""
    return get_plugin_manager()._plugin_tool_names


def get_plugin_toolsets() -> List[tuple]:
    """Return plugin toolsets as ``(key, label, description)`` tuples.

    Used by the ``hermes tools`` TUI so plugin-provided toolsets appear
    alongside the built-in ones and can be toggled on/off per platform.
    """
    manager = get_plugin_manager()
    if not manager._plugin_tool_names:
        return []

    try:
        from tools.registry import registry
    except Exception:
        return []

    # Group plugin tool names by their toolset
    toolset_tools: Dict[str, List[str]] = {}
    toolset_plugin: Dict[str, LoadedPlugin] = {}
    for tool_name in manager._plugin_tool_names:
        entry = registry._tools.get(tool_name)
        if not entry:
            continue
        ts = entry.toolset
        toolset_tools.setdefault(ts, []).append(entry.name)

    # Map toolsets back to the plugin that registered them
    for _name, loaded in manager._plugins.items():
        for tool_name in loaded.tools_registered:
            entry = registry._tools.get(tool_name)
            if entry and entry.toolset in toolset_tools:
                toolset_plugin.setdefault(entry.toolset, loaded)

    result = []
    for ts_key in sorted(toolset_tools):
        plugin = toolset_plugin.get(ts_key)
        label = f"🔌 {ts_key.replace('_', ' ').title()}"
        if plugin and plugin.manifest.description:
            desc = plugin.manifest.description
        else:
            desc = ", ".join(sorted(toolset_tools[ts_key]))
        result.append((ts_key, label, desc))

    return result
