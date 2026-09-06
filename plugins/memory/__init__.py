"""Memory provider plugin discovery: bundled ``plugins/memory/<name>/``, user
``$HERMES_HOME/plugins/<name>/``, project ``./.hermes/plugins/<name>/`` (opt-in via
HERMES_ENABLE_PROJECT_PLUGINS), then ``hermes_agent.memory_providers`` entry points.
Precedence is deliberately the REVERSE of PluginManager's later-source-wins:
bundled wins, then user, project, entry point — a provider is activated by name
(``memory.provider``, one at a time), so a directory dropped into the working tree
must never shadow a shipped provider. Changing this order is a breaking change.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

from hermes_cli.config import cfg_get
from plugins import plugin_loader as _loader

if TYPE_CHECKING:
    from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_MEMORY_PLUGINS_DIR = Path(__file__).parent
ENTRY_POINTS_GROUP = "hermes_agent.memory_providers"
_REGISTERED_MEMORY_PROVIDER_SKILLS: dict[str, Path] = {}

# Synthetic parent package so user-installed providers don't collide with bundled ones.
_USER_NAMESPACE = "_hermes_user_memory"

_register_synthetic_package = _loader.register_synthetic_package
_get_user_plugins_dir = _loader.user_plugins_dir


def _get_project_plugins_dir() -> Optional[Path]:
    """``./.hermes/plugins/`` or None. Gated on HERMES_ENABLE_PROJECT_PLUGINS like the
    PluginManager scan: a repo you merely ``cd`` into must not offer a memory backend."""
    try:
        from hermes_cli.plugins import _env_enabled

        if not _env_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
            return None
        d = Path.cwd() / ".hermes" / "plugins"
        return d if d.is_dir() else None
    except Exception:
        return None


def _is_memory_provider_dir(path: Path) -> bool:
    """Cheap text heuristic (no import): ``__init__.py`` mentions the memory provider contract."""
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(errors="replace", encoding="utf-8")[:8192]
        return "register_memory_provider" in source or "MemoryProvider" in source
    except Exception:
        return False


def _is_bundled(provider_dir: Path) -> bool:
    return _MEMORY_PLUGINS_DIR in provider_dir.parents or provider_dir.parent == _MEMORY_PLUGINS_DIR


def _module_name(provider_dir: Path, name: str) -> str:
    """``plugins.memory.<name>`` for bundled providers, else under the synthetic user namespace."""
    return f"plugins.memory.{name}" if _is_bundled(provider_dir) else f"{_USER_NAMESPACE}.{name}"


def _external_source_dirs() -> List[Path]:
    """User then project plugin roots that exist (precedence order)."""
    return [d for d in (_get_user_plugins_dir(), _get_project_plugins_dir()) if d]


def _iter_provider_dirs() -> List[Tuple[str, Path]]:
    """``(name, path)`` for bundled, then user-installed, then project-local; first-seen wins."""
    dirs = [(child.name, child) for child in _loader.iter_plugin_dirs(_MEMORY_PLUGINS_DIR)]
    seen = {name for name, _ in dirs}
    for source_dir in _external_source_dirs():
        for child in sorted(source_dir.iterdir()):
            if (
                child.is_dir()
                and not child.name.startswith(("_", "."))
                and child.name not in seen
                and _is_memory_provider_dir(child)
            ):
                seen.add(child.name)
                dirs.append((child.name, child))
    return dirs


def _iter_entry_points():
    """Yield pip-installed memory provider entry points."""
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            return list(eps.select(group=ENTRY_POINTS_GROUP))
        return list(eps.get(ENTRY_POINTS_GROUP, [])) if isinstance(eps, dict) else [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]
    except Exception as exc:
        logger.debug("Memory provider entry-point scan failed: %s", exc)
        return []


def find_provider_dir(name: str) -> Optional[Path]:
    """Provider name -> directory: bundled, user, project, then a pip entry point's
    package dir. The entry-point case matters because ``config_schema.py`` and
    ``cli.py`` are read from disk, not imported; without a directory a pip-installed
    provider silently loses its dashboard panel and ``hermes <provider>`` commands."""
    bundled = _MEMORY_PLUGINS_DIR / name
    if bundled.is_dir() and (bundled / "__init__.py").exists():
        return bundled
    for source_dir in _external_source_dirs():
        candidate = source_dir / name
        if candidate.is_dir() and _is_memory_provider_dir(candidate):
            return candidate
    return _entry_point_package_dir(find_provider_entry_point(name))


def _entry_point_package_dir(entry_point) -> Optional[Path]:
    """Directory of an entry point's module, resolved WITHOUT importing it (discovery
    runs from the dashboard/argparse before a provider is selected; importing every
    candidate would run arbitrary code). Only package entry points yield a
    directory — a bare ``module.py`` has nowhere for a sibling ``config_schema.py``."""
    if entry_point is None:
        return None
    try:
        from hermes_cli.plugins import resolve_module_origin

        module_name = (entry_point.value or "").split(":")[0].strip()
        origin = resolve_module_origin(module_name)
        if not origin:
            return None
        path = Path(origin)
        return path.parent if path.name == "__init__.py" else None
    except Exception as exc:
        logger.debug("Could not resolve directory for entry point '%s': %s",
                     getattr(entry_point, "name", "?"), exc)
        return None


def find_provider_entry_point(name: str):
    """Resolve a provider name to a pip entry point, if installed."""
    return next((ep for ep in _iter_entry_points() if ep.name == name), None)


def list_memory_provider_names() -> List[str]:
    """Cheap name-only listing (directory scan + entry-point enumeration, no import or
    availability check) — safe at module-import time (dashboard dropdown)."""
    names = {name for name, _ in _iter_provider_dirs()}
    names.update(ep.name for ep in _iter_entry_points())
    return sorted(names)


def discover_memory_providers() -> List[Tuple[str, str, bool]]:
    """``[(name, description, is_available), ...]``; bundled wins on name collisions,
    then user directories, then pip."""
    results = [
        (name, _loader.read_plugin_description(child),
         _loader.probe_availability(lambda c=child: _load_provider_from_dir(c, register_skills=False)))
        for name, child in _iter_provider_dirs()
    ]
    seen = {name for name, _, _ in results}
    for entry_point in _iter_entry_points():
        if entry_point.name not in seen:
            seen.add(entry_point.name)
            results.append((entry_point.name, "", _loader.probe_availability(
                lambda ep=entry_point: _load_provider_from_entry_point(ep, register_skills=False)
            )))
    return results


def load_memory_provider(name: str, *, register_skills: Optional[bool] = None) -> Optional["MemoryProvider"]:
    """Load a MemoryProvider by name (bundled, user, project, then pip entry point);
    None if not found or failing to load. Skills register only for the configured
    active provider unless ``register_skills`` is explicit, so inspecting inactive
    providers leaves no registry side effects."""
    if register_skills is None:
        register_skills = name == _get_active_memory_provider()

    provider_dir = find_provider_dir(name)
    entry_point = None if provider_dir else find_provider_entry_point(name)
    if not provider_dir and entry_point is None:
        logger.debug("Memory provider '%s' not found in bundled, user plugins, or entry points", name)
        return None

    def _load(_dir):
        if provider_dir:
            return _load_provider_from_dir(provider_dir, register_skills=register_skills)
        return _load_provider_from_entry_point(entry_point, register_skills=register_skills)

    return _loader.load_named(name, provider_dir, _load, kind="Memory provider", noun="provider", logger=logger)


def _instantiate_subclass(namespace) -> Optional["MemoryProvider"]:
    """First instantiable ``MemoryProvider`` subclass found among *namespace*'s attributes."""
    from agent.memory_provider import MemoryProvider

    for attr_name in dir(namespace):
        attr = getattr(namespace, attr_name, None)
        if isinstance(attr, type) and issubclass(attr, MemoryProvider) and attr is not MemoryProvider:
            try:
                return attr()
            except Exception:
                pass
    return None


def _load_provider_from_entry_point(entry_point, *, register_skills: bool = True) -> Optional["MemoryProvider"]:
    """Import a provider entry point and extract the MemoryProvider instance: an
    instance, a subclass, a module with ``register(ctx)``, a factory / ``register``
    callable, or a namespace holding a subclass — in that order."""
    from agent.memory_provider import MemoryProvider

    loaded = entry_point.load()
    if isinstance(loaded, MemoryProvider):
        return loaded
    if isinstance(loaded, type) and issubclass(loaded, MemoryProvider):
        try:
            return loaded()
        except Exception:
            pass
    if hasattr(loaded, "register"):
        collector = _ProviderCollector(entry_point.name, register_skills=register_skills)
        loaded.register(collector)
        if collector.provider:
            return collector.provider
    if callable(loaded):
        try:
            provider = loaded()
            if isinstance(provider, MemoryProvider):
                return provider
        except TypeError:
            pass
        collector = _ProviderCollector(entry_point.name, register_skills=register_skills)
        loaded(collector)
        return collector.provider

    provider = _instantiate_subclass(loaded)
    if provider is None:
        logger.debug("Memory provider entry point '%s' loaded no provider", entry_point.name)
    return provider


def _load_provider_from_dir(provider_dir: Path, *, register_skills: bool = True) -> Optional["MemoryProvider"]:
    """Import a provider module; ``register(ctx)`` first, else a top-level subclass."""
    name = provider_dir.name
    mod = _loader.load_plugin_module(
        _module_name(provider_dir, name), provider_dir,
        parents=("plugins", "plugins.memory"),
        logger=logger,
        synthetic_namespace=None if _is_bundled(provider_dir) else _USER_NAMESPACE,
    )
    if mod is None:
        return None

    if hasattr(mod, "register"):
        collector = _ProviderCollector(name, register_skills=register_skills)
        try:
            mod.register(collector)
        except Exception as e:
            # A raise AFTER register_memory_provider() must not cost us the provider:
            # falling through to the subclass scan would hand back a bare second
            # instance — a silent downgrade that looks like success.
            if collector.provider is None:
                logger.debug("register() failed for %s: %s", name, e)
            else:
                logger.warning(
                    "Memory provider '%s' raised after registering (%s) — "
                    "using the registered provider; later registrations were skipped",
                    name, e,
                )
        if collector.provider:
            return collector.provider

    return _instantiate_subclass(mod)


class _ProviderCollector:
    """Plugin context for memory providers: captures ``register_memory_provider``
    (the one call the activation path owns) and delegates other ``register_*``
    calls to a real ``PluginContext`` so providers have the full plugin surface."""

    def __init__(self, name: str, *, register_skills: bool = True):
        self.name = name
        self.provider = None
        self._register_skills = register_skills
        self._context = None

    def register_memory_provider(self, provider):
        self.provider = provider

    def register_skill(self, *args, **kwargs):
        """Forward skills to the plugin registry, tracking qualified name + path so
        switching the active provider can retract the previous one's skills. Gated
        on ``register_skills`` so inspecting an inactive provider has no side effects."""
        if not self._register_skills:
            return
        try:
            self._plugin_context().register_skill(*args, **kwargs)
            qualified_name = f"{self.name}:{args[0] if args else kwargs.get('name')}"

            from hermes_cli.plugins import get_plugin_manager

            registered_path = get_plugin_manager().find_plugin_skill(qualified_name)
            if registered_path is not None:
                _REGISTERED_MEMORY_PROVIDER_SKILLS[qualified_name] = registered_path
        except Exception as exc:
            logger.debug("Memory provider '%s' failed to register skill: %s", self.name, exc)

    def register_cli_command(self, *args, **kwargs):
        pass  # CLI registration happens via discover_plugin_cli_commands()

    def __getattr__(self, attr: str):
        """Delegate any other ``register_*`` call to a real ``PluginContext`` (a
        hand-maintained stub drifted: unknown calls raised and cost the provider).
        Non-``register_*`` attributes raise normally so a typo still fails loudly."""
        if not attr.startswith("register_"):
            raise AttributeError(attr)

        def _forward(*args, **kwargs):
            try:
                return self._plugin_context().__getattribute__(attr)(*args, **kwargs)
            except Exception as exc:
                # A secondary registration must not cost the provider itself.
                logger.warning("Memory provider '%s' failed to %s: %s", self.name, attr, exc)
                return None

        return _forward

    def _plugin_context(self):
        """A real ``PluginContext``, built once on demand: the common provider that only
        calls ``register_memory_provider`` must not pay for importing the plugin manager."""
        if self._context is None:
            from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager

            manifest = PluginManifest(name=self.name, key=self.name)
            self._context = PluginContext(manifest, get_plugin_manager())
        return self._context


def _get_active_memory_provider() -> Optional[str]:
    """Active provider name from config.yaml (``memory.provider``), or None. Reads config only."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return cfg_get(config, "memory", "provider") or None
    except Exception:
        return None


def _prune_inactive_memory_provider_skills(active_provider: Optional[str] = None) -> None:
    """Remove tracked skills that no longer belong to the active provider."""
    if active_provider is None:
        active_provider = _get_active_memory_provider()

    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    for qualified_name, registered_path in list(_REGISTERED_MEMORY_PROVIDER_SKILLS.items()):
        if qualified_name.partition(":")[0] == active_provider:
            continue
        if manager.find_plugin_skill(qualified_name) == registered_path:
            manager.remove_plugin_skill(qualified_name)
        _REGISTERED_MEMORY_PROVIDER_SKILLS.pop(qualified_name, None)


def discover_plugin_cli_commands() -> List[dict]:
    """CLI commands for the **active** memory plugin only. Imports just its ``cli.py``
    (``register_cli(subparser)``), never the provider module, so it is safe during
    argparse setup. At most one dict: name/help/description/setup_fn/handler_fn/plugin."""
    active_provider = _get_active_memory_provider() if _MEMORY_PLUGINS_DIR.is_dir() else None
    plugin_dir = find_provider_dir(active_provider) if active_provider else None
    if not plugin_dir or not (plugin_dir / "cli.py").exists():
        return []

    module_name = _module_name(plugin_dir, active_provider) + ".cli"
    try:
        cli_mod = sys.modules.get(module_name)
        if cli_mod is None:
            if not _is_bundled(plugin_dir):
                # cli.py imports as _hermes_user_memory.<name>.cli, usually before the
                # provider is loaded: register parent packages so its relative imports
                # resolve without executing the plugin's __init__.py (the shell has no
                # __file__, so _load_provider_from_dir() still loads the real module).
                _register_synthetic_package(_USER_NAMESPACE, [])
                _register_synthetic_package(f"{_USER_NAMESPACE}.{active_provider}", [str(plugin_dir)])
            spec = importlib.util.spec_from_file_location(module_name, str(plugin_dir / "cli.py"))
            if not spec or not spec.loader:
                return []
            cli_mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = cli_mod
            spec.loader.exec_module(cli_mod)

        register_cli = getattr(cli_mod, "register_cli", None)
        if not callable(register_cli):
            return []
        desc = _loader.read_plugin_description(plugin_dir)
        return [{
            "name": active_provider,
            "help": desc or f"Manage {active_provider} memory plugin",
            "description": desc or "",
            "setup_fn": register_cli,
            "handler_fn": getattr(cli_mod, f"{active_provider}_command", None) or getattr(cli_mod, "honcho_command", None),
            "plugin": active_provider,
        }]
    except Exception as e:
        logger.debug("Failed to scan CLI for memory plugin '%s': %s", active_provider, e)
        return []
