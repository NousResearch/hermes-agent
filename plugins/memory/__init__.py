"""Memory provider plugin discovery.

Scans ``plugins/memory/<name>/`` directories for memory provider plugins.
Each subdirectory must contain ``__init__.py`` with a class implementing
the MemoryProvider ABC.

Memory providers are separate from the general plugin system — they live
in the repo and are always available without user installation. Only ONE
can be active at a time, selected via ``memory.provider`` in config.yaml.

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
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_MEMORY_PLUGINS_DIR = Path(__file__).parent


def discover_memory_providers() -> List[Tuple[str, str, bool]]:
    """Scan plugins/memory/ for available providers.

    Returns list of (name, description, is_available) tuples.
    Does NOT import the providers — just reads plugin.yaml for metadata
    and does a lightweight availability check.
    """
    results = []
    if not _MEMORY_PLUGINS_DIR.is_dir():
        return results

    for child in sorted(_MEMORY_PLUGINS_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        init_file = child / "__init__.py"
        if not init_file.exists():
            continue

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
        available = True
        try:
            provider = _load_provider_from_dir(child)
            if provider:
                available = provider.is_available()
            else:
                available = False
        except Exception:
            available = False

        results.append((child.name, desc, available))

    return results


def load_memory_provider(name: str) -> Optional["MemoryProvider"]:
    """Load and return a MemoryProvider instance by name.

    Returns None if the provider is not found or fails to load.
    """
    provider_dir = _MEMORY_PLUGINS_DIR / name
    if not provider_dir.is_dir():
        logger.debug("Memory provider '%s' not found in %s", name, _MEMORY_PLUGINS_DIR)
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


def discover_plugin_cli_commands() -> List[dict]:
    """Scan memory plugin directories for CLI command registrations.

    Looks for a ``register_cli(subparser)`` function in each plugin's
    ``cli.py``.  Returns a list of dicts with keys:
    ``name``, ``help``, ``description``, ``setup_fn``, ``handler_fn``.

    This is a lightweight scan — it only imports ``cli.py``, not the
    full plugin module.  Safe to call during argparse setup before
    any provider is loaded.
    """
    results: List[dict] = []
    if not _MEMORY_PLUGINS_DIR.is_dir():
        return results

    for child in sorted(_MEMORY_PLUGINS_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        cli_file = child / "cli.py"
        if not cli_file.exists():
            continue

        module_name = f"plugins.memory.{child.name}.cli"
        try:
            # Import the CLI module (lightweight — no SDK needed)
            if module_name in sys.modules:
                cli_mod = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(
                    module_name, str(cli_file)
                )
                if not spec or not spec.loader:
                    continue
                cli_mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = cli_mod
                spec.loader.exec_module(cli_mod)

            register_cli = getattr(cli_mod, "register_cli", None)
            if not callable(register_cli):
                continue

            # Read metadata from plugin.yaml if available
            help_text = f"Manage {child.name} memory plugin"
            description = ""
            yaml_file = child / "plugin.yaml"
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

            handler_fn = getattr(cli_mod, "honcho_command", None) or \
                         getattr(cli_mod, f"{child.name}_command", None)

            results.append({
                "name": child.name,
                "help": help_text,
                "description": description,
                "setup_fn": register_cli,
                "handler_fn": handler_fn,
                "plugin": child.name,
            })
        except Exception as e:
            logger.debug("Failed to scan CLI for memory plugin '%s': %s", child.name, e)

    return results
