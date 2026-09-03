"""Memory provider plugin discovery.

Scans four sources for memory provider plugins:

1. Bundled providers: ``plugins/memory/<name>/`` (shipped with hermes-agent)
2. User-installed providers: ``$HERMES_HOME/plugins/<name>/``
3. Project-local providers: ``./.hermes/plugins/<name>/``, opt-in via
   ``HERMES_ENABLE_PROJECT_PLUGINS``
4. Pip-installed providers: ``hermes_agent.memory_providers`` entry points

Directory providers must contain ``__init__.py`` with a class implementing
the MemoryProvider ABC. Pip packages expose a provider or ``register(ctx)``
callback through the entry-point group.

These are the same four sources the general ``PluginManager`` scans, but the
precedence is deliberately the reverse of its later-source-wins order: here
**bundled wins**, then user, then project, then entry point. A memory provider
is activated by name, so letting a directory dropped into the working tree
shadow a shipped provider would silently redirect the agent's memory. Changing
this order is a breaking change, not a cleanup.

Only ONE provider can be active at a time, selected via
``memory.provider`` in config.yaml.

Usage:
    from plugins.memory import discover_memory_providers, load_memory_provider

    available = discover_memory_providers()   # [(name, desc, available), ...]
    provider = load_memory_provider("mnemosyne")  # MemoryProvider instance
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING
from hermes_cli.config import cfg_get

if TYPE_CHECKING:
    from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_MEMORY_PLUGINS_DIR = Path(__file__).parent
ENTRY_POINTS_GROUP = "hermes_agent.memory_providers"
_REGISTERED_MEMORY_PROVIDER_SKILLS: dict[str, Path] = {}

# Synthetic parent package for user-installed providers, so they don't
# collide with bundled providers in sys.modules.
_USER_NAMESPACE = "_hermes_user_memory"


def _register_synthetic_package(name: str, search_locations: List[str]) -> None:
    """Register an empty package shell in sys.modules.

    User-installed providers import as ``_hermes_user_memory.<name>``, a
    dotted name whose parents exist nowhere on disk.  Unless those parents
    are present in ``sys.modules``, any relative import inside the plugin
    (``from . import config``) fails with
    ``ModuleNotFoundError: No module named '_hermes_user_memory'`` — the
    same reason the loader already registers ``plugins`` and
    ``plugins.memory`` for bundled providers.
    """
    if name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = search_locations
    sys.modules[name] = importlib.util.module_from_spec(spec)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _get_user_plugins_dir() -> Optional[Path]:
    """Return ``$HERMES_HOME/plugins/`` or None if unavailable."""
    try:
        from hermes_constants import get_hermes_home
        d = get_hermes_home() / "plugins"
        return d if d.is_dir() else None
    except Exception:
        return None


def _get_project_plugins_dir() -> Optional[Path]:
    """Return ``./.hermes/plugins/`` or None if unavailable or not opted in.

    Gated on ``HERMES_ENABLE_PROJECT_PLUGINS`` exactly as the general
    ``PluginManager`` gates its own project scan — a repository you merely
    ``cd`` into must not be able to offer the agent a memory backend.
    """
    try:
        from hermes_cli.plugins import _env_enabled

        if not _env_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
            return None
        d = Path.cwd() / ".hermes" / "plugins"
        return d if d.is_dir() else None
    except Exception:
        return None


def _is_memory_provider_dir(path: Path) -> bool:
    """Heuristic: does *path* look like a memory provider plugin?

    Checks for ``register_memory_provider`` or ``MemoryProvider`` in the
    ``__init__.py`` source.  Cheap text scan — no import needed.
    """
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(errors="replace", encoding="utf-8")[:8192]
        return "register_memory_provider" in source or "MemoryProvider" in source
    except Exception:
        return False


def _iter_provider_dirs() -> List[Tuple[str, Path]]:
    """Yield ``(name, path)`` for all discovered provider directories.

    Scans bundled, then user-installed, then project-local.  Bundled takes
    precedence on name collisions (first-seen wins via ``seen`` set).
    """
    seen: set = set()
    dirs: List[Tuple[str, Path]] = []

    # 1. Bundled providers (plugins/memory/<name>/)
    if _MEMORY_PLUGINS_DIR.is_dir():
        for child in sorted(_MEMORY_PLUGINS_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if not (child / "__init__.py").exists():
                continue
            seen.add(child.name)
            dirs.append((child.name, child))

    # 2. User-installed providers ($HERMES_HOME/plugins/<name>/)
    # 3. Project-local providers (./.hermes/plugins/<name>/), opt-in
    for source_dir in (_get_user_plugins_dir(), _get_project_plugins_dir()):
        if not source_dir:
            continue
        for child in sorted(source_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if child.name in seen:
                continue  # earlier source wins
            if not _is_memory_provider_dir(child):
                continue  # skip non-memory plugins
            seen.add(child.name)
            dirs.append((child.name, child))

    return dirs


def _iter_entry_points():
    """Yield pip-installed memory provider entry points."""
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            return list(eps.select(group=ENTRY_POINTS_GROUP))
        if isinstance(eps, dict):
            return list(eps.get(ENTRY_POINTS_GROUP, []))
        return [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]
    except Exception as exc:
        logger.debug("Memory provider entry-point scan failed: %s", exc)
        return []


def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider name to the directory holding its files.

    Checks bundled, then user-installed, then project-local, then the package
    directory of a pip entry-point provider.

    The entry-point case matters because two of a provider's files are read
    from disk rather than imported: ``config_schema.py`` (loaded by path so the
    web server never pulls in the agent runtime — see
    ``plugins/memory/config_schema.py``) and ``cli.py`` (loaded by
    ``discover_plugin_cli_commands`` at argparse time). Without a directory, a
    pip-installed provider silently loses its dashboard config panel and its
    ``hermes <provider>`` subcommands — working, but a second-class citizen next
    to a directory install.
    """
    # Bundled
    bundled = _MEMORY_PLUGINS_DIR / name
    if bundled.is_dir() and (bundled / "__init__.py").exists():
        return bundled
    # User-installed, then project-local
    for source_dir in (_get_user_plugins_dir(), _get_project_plugins_dir()):
        if not source_dir:
            continue
        candidate = source_dir / name
        if candidate.is_dir() and _is_memory_provider_dir(candidate):
            return candidate
    # Pip entry point
    return _entry_point_package_dir(find_provider_entry_point(name))


def _entry_point_package_dir(entry_point) -> Optional[Path]:
    """The directory of an entry point's module, resolved WITHOUT importing it.

    Discovery must stay free of third-party imports: ``find_provider_dir`` is
    called from the dashboard and from argparse setup, long before the operator
    has selected a provider, so importing every installed candidate would run
    arbitrary code on the strength of a package merely being present.
    ``resolve_module_origin`` walks the module's file layout instead.

    Only package entry points (``pkg/__init__.py``) yield a directory — a
    provider pointed at a bare ``module.py`` has nowhere to put a sibling
    ``config_schema.py``, so it correctly resolves to None.
    """
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
    for entry_point in _iter_entry_points():
        if entry_point.name == name:
            return entry_point
    return None


def _warm_import_provider_from_dir(provider_dir: Path) -> bool:
    """Import a provider package without constructing a provider instance."""
    name = provider_dir.name
    _is_bundled = _MEMORY_PLUGINS_DIR in provider_dir.parents or provider_dir.parent == _MEMORY_PLUGINS_DIR
    module_name = (
        f"plugins.memory.{name}"
        if _is_bundled
        else f"{_USER_NAMESPACE}.{name}"
    )
    init_file = provider_dir / "__init__.py"

    if not init_file.exists():
        return False

    cached = sys.modules.get(module_name)
    if cached is not None and getattr(cached, "__file__", None):
        return True

    for parent in ("plugins", "plugins.memory"):
        if parent not in sys.modules:
            parent_path = Path(__file__).parent
            if parent == "plugins":
                parent_path = parent_path.parent
            parent_init = parent_path / "__init__.py"
            if parent_init.exists():
                spec = importlib.util.spec_from_file_location(
                    parent,
                    str(parent_init),
                    submodule_search_locations=[str(parent_path)],
                )
                if spec:
                    parent_mod = importlib.util.module_from_spec(spec)
                    sys.modules[parent] = parent_mod
                    try:
                        spec.loader.exec_module(parent_mod)
                    except Exception:
                        pass

    if not _is_bundled:
        _register_synthetic_package(_USER_NAMESPACE, [])

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(init_file),
        submodule_search_locations=[str(provider_dir)],
    )
    if not spec or not spec.loader:
        return False

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod

    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        logger.debug("Failed to warm-import memory provider '%s': %s", name, e)
        sys.modules.pop(module_name, None)
        return False

    return True


def warmup_import_memory_provider_module(name: str) -> bool:
    """Import a provider module only, avoiding construction side effects.

    Returns True if the module import succeeded or was already loaded.
    """
    provider_dir = find_provider_dir(name)
    entry_point = None if provider_dir else find_provider_entry_point(name)
    if not provider_dir and entry_point is None:
        logger.debug(
            "Memory provider '%s' not found for warm-up import", name
        )
        return False

    try:
        if provider_dir:
            return _warm_import_provider_from_dir(provider_dir)
        if entry_point is not None:
            entry_point.load()
            return True
    except Exception as exc:
        logger.debug(
            "Warm-up import of memory provider '%s' failed: %s", name, exc
        )
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_memory_provider_names() -> List[str]:
    """Cheap name-only listing of discoverable memory providers.

    Unlike :func:`discover_memory_providers`, this does NOT import provider
    modules or run availability checks — a directory scan plus entry-point
    *enumeration*, which reads distribution metadata without executing any of
    it. Safe to call at module-import time (e.g. when building the dashboard
    config schema, where it fills the ``memory.provider`` dropdown).
    """
    names = {name for name, _ in _iter_provider_dirs()}
    names.update(ep.name for ep in _iter_entry_points())
    return sorted(names)


def discover_memory_providers() -> List[Tuple[str, str, bool]]:
    """Scan directory and pip entry-point memory providers.

    Returns list of (name, description, is_available) tuples.
    Bundled providers take precedence on name collisions, followed by
    user-installed directory providers, then pip entry-point providers.
    """
    results = []
    seen: set[str] = set()

    for name, child in _iter_provider_dirs():
        # Read description from plugin.yaml if available
        desc = ""
        yaml_file = child / "plugin.yaml"
        if yaml_file.exists():
            try:
                import yaml
                with open(yaml_file, encoding="utf-8-sig") as f:
                    meta = yaml.safe_load(f) or {}
                desc = meta.get("description", "")
            except Exception:
                pass

        # Quick availability check — try loading and calling is_available()
        available = True
        try:
            provider = _load_provider_from_dir(child, register_skills=False)
            if provider:
                available = provider.is_available()
            else:
                available = False
        except Exception:
            available = False

        results.append((name, desc, available))
        seen.add(name)

    for entry_point in _iter_entry_points():
        name = entry_point.name
        if name in seen:
            continue
        desc = ""
        available = True
        try:
            provider = _load_provider_from_entry_point(
                entry_point,
                register_skills=False,
            )
            if provider:
                available = provider.is_available()
            else:
                available = False
        except Exception:
            available = False

        results.append((name, desc, available))
        seen.add(name)

    return results


def load_memory_provider(
    name: str,
    *,
    register_skills: Optional[bool] = None,
) -> Optional["MemoryProvider"]:
    """Load and return a MemoryProvider instance by name.

    Checks bundled (``plugins/memory/<name>/``), user-installed
    (``$HERMES_HOME/plugins/<name>/``), and pip entry-point providers.
    Bundled providers take precedence on name collisions.

    Skills register only when *name* is the configured active provider unless
    ``register_skills`` is passed explicitly. This keeps status and setup
    inspection of inactive providers free of registry side effects.

    Returns None if the provider is not found or fails to load.
    """
    if register_skills is None:
        register_skills = name == _get_active_memory_provider()

    provider_dir = find_provider_dir(name)
    entry_point = None if provider_dir else find_provider_entry_point(name)
    if not provider_dir and entry_point is None:
        logger.debug(
            "Memory provider '%s' not found in bundled, user plugins, or entry points",
            name,
        )
        return None

    try:
        provider = (
            _load_provider_from_dir(provider_dir, register_skills=register_skills)
            if provider_dir
            else _load_provider_from_entry_point(
                entry_point,
                register_skills=register_skills,
            )
        )
        if provider:
            return provider
        logger.warning("Memory provider '%s' loaded but no provider instance found", name)
        return None
    except Exception as e:
        logger.warning("Failed to load memory provider '%s': %s", name, e)
        return None


def _load_provider_from_entry_point(
    entry_point,
    *,
    register_skills: bool = True,
) -> Optional["MemoryProvider"]:
    """Import a provider entry point and extract the MemoryProvider instance."""
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

    for attr_name in dir(loaded):
        attr = getattr(loaded, attr_name, None)
        if (isinstance(attr, type) and issubclass(attr, MemoryProvider)
                and attr is not MemoryProvider):
            try:
                return attr()
            except Exception:
                pass

    logger.debug("Memory provider entry point '%s' loaded no provider", entry_point.name)
    return None


def _load_provider_from_dir(
    provider_dir: Path,
    *,
    register_skills: bool = True,
) -> Optional["MemoryProvider"]:
    """Import a provider module and extract the MemoryProvider instance.

    The module must have either:
    - A register(ctx) function (plugin-style) — we simulate a ctx
    - A top-level class that extends MemoryProvider — we instantiate it
    """
    name = provider_dir.name
    # Use a separate namespace for user-installed plugins so they don't
    # collide with bundled providers in sys.modules.
    _is_bundled = _MEMORY_PLUGINS_DIR in provider_dir.parents or provider_dir.parent == _MEMORY_PLUGINS_DIR
    module_name = f"plugins.memory.{name}" if _is_bundled else f"{_USER_NAMESPACE}.{name}"
    init_file = provider_dir / "__init__.py"

    if not init_file.exists():
        return None

    # Check if already loaded.  A synthetic package shell registered by
    # discover_plugin_cli_commands() for relative-import support has no
    # __file__; only reuse modules that were actually loaded from disk.
    cached = sys.modules.get(module_name)
    if cached is not None and getattr(cached, "__file__", None):
        mod = cached
    else:
        # Handle relative imports within the plugin
        # First ensure the parent packages are registered
        for parent in ("plugins", "plugins.memory"):
            if parent not in sys.modules:
                parent_path = Path(__file__).parent
                if parent == "plugins":
                    parent_path = parent_path.parent
                parent_init = parent_path / "__init__.py"
                if parent_init.exists():
                    spec = importlib.util.spec_from_file_location(
                        parent, str(parent_init),
                        submodule_search_locations=[str(parent_path)]
                    )
                    if spec:
                        parent_mod = importlib.util.module_from_spec(spec)
                        sys.modules[parent] = parent_mod
                        try:
                            spec.loader.exec_module(parent_mod)
                        except Exception:
                            pass

        # User-installed plugins need their synthetic parent registered the
        # same way, or relative imports inside the plugin cannot resolve.
        if not _is_bundled:
            _register_synthetic_package(_USER_NAMESPACE, [])

        # Now load the provider module
        spec = importlib.util.spec_from_file_location(
            module_name, str(init_file),
            submodule_search_locations=[str(provider_dir)]
        )
        if not spec:
            return None

        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod

        # Register submodules so relative imports work
        # e.g., "from .store import MemoryStore" in holographic plugin
        for sub_file in provider_dir.glob("*.py"):
            if sub_file.name == "__init__.py":
                continue
            sub_name = sub_file.stem
            full_sub_name = f"{module_name}.{sub_name}"
            if full_sub_name not in sys.modules:
                sub_spec = importlib.util.spec_from_file_location(
                    full_sub_name, str(sub_file)
                )
                if sub_spec:
                    sub_mod = importlib.util.module_from_spec(sub_spec)
                    sys.modules[full_sub_name] = sub_mod
                    try:
                        sub_spec.loader.exec_module(sub_mod)
                    except Exception as e:
                        logger.debug("Failed to load submodule %s: %s", full_sub_name, e)

        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            logger.debug("Failed to exec_module %s: %s", module_name, e)
            sys.modules.pop(module_name, None)
            return None

    # Try register(ctx) pattern first (how our plugins are written)
    if hasattr(mod, "register"):
        collector = _ProviderCollector(name, register_skills=register_skills)
        try:
            mod.register(collector)
        except Exception as e:
            # A raise AFTER register_memory_provider() must not cost us the
            # provider. Falling through to the subclass scan below would
            # discard the instance the plugin configured and hand back a bare
            # second one — a silent downgrade that looks like success.
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

    # Fallback: find a MemoryProvider subclass and instantiate it
    from agent.memory_provider import MemoryProvider
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if (isinstance(attr, type) and issubclass(attr, MemoryProvider)
                and attr is not MemoryProvider):
            try:
                return attr()
            except Exception:
                pass

    return None


class _ProviderCollector:
    """Plugin context for memory providers.

    Captures ``register_memory_provider`` directly — that is the one call the
    exclusive activation path owns — and delegates everything else to a real
    ``PluginContext`` (see ``__getattr__``), so a memory provider has the same
    registration surface as any other plugin.
    """

    def __init__(self, name: str, *, register_skills: bool = True):
        self.name = name
        self.provider = None
        self._register_skills = register_skills
        self._context = None

    def register_memory_provider(self, provider):
        self.provider = provider

    def register_skill(self, *args, **kwargs):
        """Forward plugin-provided skills to the general plugin registry.

        Handled explicitly rather than through ``__getattr__`` because skills
        are tracked for pruning: switching the active provider has to retract
        the skills the previous one registered, which needs the qualified name
        and resolved path recorded here.

        Gated on ``register_skills`` so merely *inspecting* an inactive
        provider — ``hermes memory status``, the setup picker — leaves no
        registry side effects behind.
        """
        if not self._register_skills:
            return
        try:
            manager_context = self._plugin_context()
            manager_context.register_skill(*args, **kwargs)
            skill_name = args[0] if args else kwargs.get("name")
            qualified_name = f"{self.name}:{skill_name}"

            from hermes_cli.plugins import get_plugin_manager

            registered_path = get_plugin_manager().find_plugin_skill(qualified_name)
            if registered_path is not None:
                _REGISTERED_MEMORY_PROVIDER_SKILLS[qualified_name] = registered_path
        except Exception as exc:
            logger.debug("Memory provider '%s' failed to register skill: %s", self.name, exc)

    def register_cli_command(self, *args, **kwargs):
        pass  # CLI registration happens via discover_plugin_cli_commands()

    def __getattr__(self, attr: str):
        """Delegate any other ``register_*`` call to a real ``PluginContext``.

        Memory providers used to get a hand-maintained stub of three no-ops
        here, which had two failure modes. Calls it *did* know about
        (``register_tool``, ``register_hook``) were silently dropped, so a
        provider's tools simply never appeared. Calls it did *not* know about
        raised ``AttributeError`` — and ``register_auxiliary_task`` is one of
        them, despite ``PluginContext.register_auxiliary_task`` documenting a
        memory provider (hindsight's pre-retain dedup) as its worked example.
        That exception surfaces as "register() failed" and costs the provider.

        Delegating instead of enumerating means this can never drift behind
        ``PluginContext`` again: a capability added there works for memory
        providers on the same commit, which is what the "widen the generic
        plugin surface" rule in AGENTS.md asks for.

        Only ``register_*`` is forwarded. Everything else raises normally, so a
        typo still fails loudly rather than being absorbed.
        """
        if not attr.startswith("register_"):
            raise AttributeError(attr)

        def _forward(*args, **kwargs):
            try:
                return self._plugin_context().__getattribute__(attr)(*args, **kwargs)
            except Exception as exc:
                # A secondary registration must not cost the provider itself —
                # by the time these run, register_memory_provider has usually
                # already handed us the instance the agent needs.
                logger.warning(
                    "Memory provider '%s' failed to %s: %s", self.name, attr, exc
                )
                return None

        return _forward

    def _plugin_context(self):
        """A real ``PluginContext`` for this provider, built once on demand.

        Lazy because the common case — a provider that only calls
        ``register_memory_provider`` — must not pay for importing the general
        plugin manager, which discovery touches on every hermes startup.
        """
        if self._context is None:
            from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager

            manifest = PluginManifest(name=self.name, key=self.name)
            self._context = PluginContext(manifest, get_plugin_manager())
        return self._context


def _get_active_memory_provider() -> Optional[str]:
    """Read the active memory provider name from config.yaml.

    Returns the provider name (e.g. ``"honcho"``) or None if no
    external provider is configured.  Lightweight — only reads config,
    no plugin loading.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return cfg_get(config, "memory", "provider") or None
    except Exception:
        return None


def _prune_inactive_memory_provider_skills(
    active_provider: Optional[str] = None,
) -> None:
    """Remove tracked skills that no longer belong to the active provider."""
    if active_provider is None:
        active_provider = _get_active_memory_provider()

    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    for qualified_name, registered_path in list(
        _REGISTERED_MEMORY_PROVIDER_SKILLS.items()
    ):
        namespace, _, _ = qualified_name.partition(":")
        if namespace == active_provider:
            continue
        if manager.find_plugin_skill(qualified_name) == registered_path:
            manager.remove_plugin_skill(qualified_name)
        _REGISTERED_MEMORY_PROVIDER_SKILLS.pop(qualified_name, None)


def discover_plugin_cli_commands() -> List[dict]:
    """Return CLI commands for the **active** memory plugin only.

    Only one memory provider can be active at a time (set via
    ``memory.provider`` in config.yaml).  This function reads that
    value and only loads CLI registration for the matching plugin.
    If no provider is active, no commands are registered.

    Looks for a ``register_cli(subparser)`` function in the active
    plugin's ``cli.py``.  Returns a list of at most one dict with
    keys: ``name``, ``help``, ``description``, ``setup_fn``,
    ``handler_fn``.

    This is a lightweight scan — it only imports ``cli.py``, not the
    full plugin module.  Safe to call during argparse setup before
    any provider is loaded.
    """
    results: List[dict] = []
    if not _MEMORY_PLUGINS_DIR.is_dir():
        return results

    active_provider = _get_active_memory_provider()
    if not active_provider:
        return results

    # Only look at the active provider's directory
    plugin_dir = find_provider_dir(active_provider)
    if not plugin_dir:
        return results

    cli_file = plugin_dir / "cli.py"
    if not cli_file.exists():
        return results

    _is_bundled = _MEMORY_PLUGINS_DIR in plugin_dir.parents or plugin_dir.parent == _MEMORY_PLUGINS_DIR
    module_name = f"plugins.memory.{active_provider}.cli" if _is_bundled else f"{_USER_NAMESPACE}.{active_provider}.cli"
    try:
        # Import the CLI module (lightweight — no SDK needed)
        if module_name in sys.modules:
            cli_mod = sys.modules[module_name]
        else:
            if not _is_bundled:
                # cli.py imports as _hermes_user_memory.<name>.cli, usually
                # before the provider itself is loaded.  Register its parent
                # packages so relative imports inside cli.py
                # ("from . import config") resolve without executing the
                # plugin's __init__.py.  The package shell has no __file__,
                # so _load_provider_from_dir() will still load the real
                # module later instead of reusing the shell.
                _register_synthetic_package(_USER_NAMESPACE, [])
                _register_synthetic_package(
                    f"{_USER_NAMESPACE}.{active_provider}", [str(plugin_dir)]
                )
            spec = importlib.util.spec_from_file_location(
                module_name, str(cli_file)
            )
            if not spec or not spec.loader:
                return results
            cli_mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = cli_mod
            spec.loader.exec_module(cli_mod)

        register_cli = getattr(cli_mod, "register_cli", None)
        if not callable(register_cli):
            return results

        # Read metadata from plugin.yaml if available
        help_text = f"Manage {active_provider} memory plugin"
        description = ""
        yaml_file = plugin_dir / "plugin.yaml"
        if yaml_file.exists():
            try:
                import yaml
                with open(yaml_file, encoding="utf-8-sig") as f:
                    meta = yaml.safe_load(f) or {}
                desc = meta.get("description", "")
                if desc:
                    help_text = desc
                    description = desc
            except Exception:
                pass

        handler_fn = getattr(cli_mod, f"{active_provider}_command", None) or \
                     getattr(cli_mod, "honcho_command", None)

        results.append({
            "name": active_provider,
            "help": help_text,
            "description": description,
            "setup_fn": register_cli,
            "handler_fn": handler_fn,
            "plugin": active_provider,
        })
    except Exception as e:
        logger.debug("Failed to scan CLI for memory plugin '%s': %s", active_provider, e)

    return results
