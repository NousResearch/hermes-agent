"""Memory provider plugin discovery.

Scans bundled ``plugins/memory/<name>/`` directories and user-installed
``$HERMES_HOME/plugins/<name>/`` directories for memory provider plugins.
Each subdirectory must contain ``__init__.py`` with a class implementing
the MemoryProvider ABC.

Bundled providers take precedence when a user plugin reuses the same name.
Only ONE memory provider can be active at a time, selected via
``memory.provider`` in config.yaml.

Usage:
    from plugins.memory import discover_memory_providers, load_memory_provider

    available = discover_memory_providers()   # [(name, desc, available), ...]
    provider = load_memory_provider("openviking")  # MemoryProvider instance
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_MEMORY_PLUGINS_DIR = Path(__file__).parent


def _get_user_memory_plugins_dir() -> Path:
    """Return the third-party memory provider directory under HERMES_HOME."""
    return get_hermes_home() / "plugins"


def _iter_provider_dirs() -> Iterator[tuple[Path, bool]]:
    """Yield candidate provider directories with bundled providers first.

    Returns ``(provider_dir, is_bundled)`` tuples. Name collisions are
    de-duplicated so bundled providers always win.
    """
    seen_names: set[str] = set()
    for root, is_bundled in ((_MEMORY_PLUGINS_DIR, True), (_get_user_memory_plugins_dir(), False)):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if child.name in seen_names:
                continue
            if not (child / "__init__.py").exists():
                continue
            seen_names.add(child.name)
            yield child, is_bundled


def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider directory by name, preferring bundled plugins."""
    for provider_dir, _is_bundled in _iter_provider_dirs():
        if provider_dir.name == name:
            return provider_dir
    return None


def _looks_like_memory_provider(provider_dir: Path) -> bool:
    """Return True when a directory appears to define a memory provider.

    This lightweight check keeps unrelated general plugins under
    ``$HERMES_HOME/plugins`` out of the memory provider list while still
    surfacing installed providers whose imports currently fail because
    dependencies are missing.
    """
    init_file = provider_dir / "__init__.py"
    try:
        contents = init_file.read_text()
    except OSError:
        return False
    return "register_memory_provider" in contents


# Backward-compatible alias for internal callers updated incrementally.
_find_provider_dir = find_provider_dir


def discover_memory_providers() -> List[Tuple[str, str, bool]]:
    """Scan bundled and user-installed directories for available providers.

    Returns list of ``(name, description, is_available)`` tuples. Bundled
    providers are always listed. User-installed providers are listed only
    when they can be resolved as memory providers, which avoids polluting
    the memory picker with general-purpose plugins under ``$HERMES_HOME/plugins``.
    """
    results = []

    for child, is_bundled in _iter_provider_dirs():
        # Read description from plugin.yaml if available
        desc = ""
        yaml_file = child / "plugin.yaml"
        if yaml_file.exists():
            try:
                import yaml
                with open(yaml_file) as f:
                    meta = yaml.safe_load(f) or {}
                desc = meta.get("description", "")
            except Exception:
                pass

        # Quick availability check — try loading and calling is_available()
        provider = None
        available = True
        try:
            provider = _load_provider_from_dir(child)
            if provider:
                available = provider.is_available()
            else:
                available = False
        except Exception:
            available = False

        if provider is None and not is_bundled and not _looks_like_memory_provider(child):
            continue

        results.append((child.name, desc, available))

    return results


def load_memory_provider(name: str) -> Optional["MemoryProvider"]:
    """Load and return a MemoryProvider instance by name.

    Returns None if the provider is not found or fails to load.
    Bundled providers take precedence over user-installed providers.
    """
    provider_dir = find_provider_dir(name)
    if provider_dir is None:
        logger.debug("Memory provider '%s' not found in bundled or user plugin dirs", name)
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
    module_name = f"plugins.memory.{name}"
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
        return config.get("memory", {}).get("provider") or None
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

    active_provider = _get_active_memory_provider()
    if not active_provider:
        return results

    # Only look at the active provider's directory
    plugin_dir = find_provider_dir(active_provider)
    if plugin_dir is None:
        return results

    cli_file = plugin_dir / "cli.py"
    if not cli_file.exists():
        return results

    module_name = f"plugins.memory.{active_provider}.cli"
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
                with open(yaml_file) as f:
                    meta = yaml.safe_load(f) or {}
                desc = meta.get("description", "")
                if desc:
                    help_text = desc
                    description = desc
            except Exception:
                pass

        handler_fn = getattr(cli_mod, f"{active_provider}_command", None) or                      getattr(cli_mod, "honcho_command", None)

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
