"""Memory provider plugin discovery.

Scans two directories for memory provider plugins:

1. Bundled providers: ``plugins/memory/<name>/`` (shipped with hermes-agent)
2. User-installed providers: ``$HERMES_HOME/plugins/<name>/``

Each subdirectory must contain ``__init__.py`` with a class implementing
the MemoryProvider ABC.  On name collisions, bundled providers take
precedence.

Only ONE provider can be active at a time, selected via
``memory.provider`` in config.yaml.

Usage:
    from plugins.memory import discover_memory_providers, load_memory_provider

    available = discover_memory_providers()   # [(name, desc, available), ...]
    provider = load_memory_provider("mnemosyne")  # MemoryProvider instance
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)

_MEMORY_PLUGINS_DIR = Path(__file__).parent


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


def _is_memory_provider_dir(path: Path) -> bool:
    """Heuristic: does *path* look like a memory provider plugin?

    Checks for ``register_memory_provider`` or ``MemoryProvider`` in the
    ``__init__.py`` source.  Cheap text scan — no import needed.
    """
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(errors="replace")[:8192]
        return "register_memory_provider" in source or "MemoryProvider" in source
    except Exception:
        return False


def _iter_provider_dirs() -> List[Tuple[str, Path]]:
    """Yield ``(name, path)`` for all discovered provider directories.

    Scans bundled first, then user-installed.  Bundled takes precedence
    on name collisions (first-seen wins via ``seen`` set).
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
    user_dir = _get_user_plugins_dir()
    if user_dir:
        for child in sorted(user_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if child.name in seen:
                continue  # bundled takes precedence
            if not _is_memory_provider_dir(child):
                continue  # skip non-memory plugins
            dirs.append((child.name, child))

    return dirs


def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider name to its directory.

    Checks bundled first, then user-installed.
    """
    # Bundled
    bundled = _MEMORY_PLUGINS_DIR / name
    if bundled.is_dir() and (bundled / "__init__.py").exists():
        return bundled
    # User-installed
    user_dir = _get_user_plugins_dir()
    if user_dir:
        user = user_dir / name
        if user.is_dir() and _is_memory_provider_dir(user):
            return user
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_memory_providers() -> List[Tuple[str, str, bool]]:
    """Scan bundled and user-installed directories for available providers.

    Returns list of (name, description, is_available) tuples.
    Bundled providers take precedence on name collisions.
    """
    results = []

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
            provider = _load_provider_from_dir(child)
            if provider:
                available = provider.is_available()
            else:
                available = False
        except Exception:
            available = False

        results.append((name, desc, available))

    return results


def load_memory_provider(name: str) -> Optional["MemoryProvider"]:
    """Load and return a MemoryProvider instance by name.

    Checks both bundled (``plugins/memory/<name>/``) and user-installed
    (``$HERMES_HOME/plugins/<name>/``) directories.  Bundled takes
    precedence on name collisions.

    Returns None if the provider is not found or fails to load.
    """
    provider_dir = find_provider_dir(name)
    if not provider_dir:
        logger.debug("Memory provider '%s' not found in bundled or user plugins", name)
        return None

    try:
        provider = _load_provider_from_dir(provider_dir)
        if provider:
            return provider
        logger.warning("Memory provider '%s' loaded but no provider instance found", name)
        return None
    except Exception as e:
        logger.warning("Failed to load memory provider '%s': %s", name, e)
        return None


def _load_provider_from_dir(provider_dir: Path) -> Optional["MemoryProvider"]:
    """Import a provider module and extract the MemoryProvider instance.

    The module must have either:
    - A register(ctx) function (plugin-style) — we simulate a ctx
    - A top-level class that extends MemoryProvider — we instantiate it
    """
    name = provider_dir.name
    # Use a separate namespace for user-installed plugins so they don't
    # collide with bundled providers in sys.modules.
    _is_bundled = _MEMORY_PLUGINS_DIR in provider_dir.parents or provider_dir.parent == _MEMORY_PLUGINS_DIR
    module_name = f"plugins.memory.{name}" if _is_bundled else f"_hermes_user_memory.{name}"
    init_file = provider_dir / "__init__.py"

    if not init_file.exists():
        return None

    # Check if already loaded
    if module_name in sys.modules:
        mod = sys.modules[module_name]
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
        collector = _ProviderCollector()
        try:
            mod.register(collector)
            if collector.provider:
                return collector.provider
        except Exception as e:
            logger.debug("register() failed for %s: %s", name, e)

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
    """Fake plugin context that captures register_memory_provider calls."""

    def __init__(self):
        self.provider = None

    def register_memory_provider(self, provider):
        self.provider = provider

    # No-op for other registration methods
    def register_tool(self, *args, **kwargs):
        pass

    def register_hook(self, *args, **kwargs):
        pass

    def register_cli_command(self, *args, **kwargs):
        pass  # CLI registration happens via discover_plugin_cli_commands()


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
    module_name = f"plugins.memory.{active_provider}.cli" if _is_bundled else f"_hermes_user_memory.{active_provider}.cli"
    try:
        # Import the CLI module (lightweight — no SDK needed)
        if module_name in sys.modules:
            cli_mod = sys.modules[module_name]
        else:
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
