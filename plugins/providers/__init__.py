"""Provider plugin discovery.

Scans ``plugins/providers/<name>/`` directories for provider plugins.
Each subdirectory must contain ``__init__.py`` with a ``resolve()``
function that returns a runtime provider dict (same shape as
``resolve_runtime_provider()`` in ``hermes_cli/runtime_provider.py``).

Provider plugins are separate from the general plugin system — they live
in the repo and are always available. They are checked BEFORE the static
PROVIDER_REGISTRY in auth.py, so they can handle provider names that
would otherwise raise AuthError.

Usage:
    from plugins.providers import get_provider_plugin

    resolver = get_provider_plugin("blockrun")
    if resolver:
        runtime = resolver(explicit_api_key=None, explicit_base_url=None)
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROVIDERS_DIR = Path(__file__).parent

# Cache: provider_name -> resolve function
_provider_cache: Dict[str, Callable] = {}
_discovered: bool = False


def discover_provider_plugins() -> List[Tuple[str, str, bool]]:
    """Scan plugins/providers/ for available provider plugins.

    Returns list of (name, description, is_available) tuples.
    """
    results = []
    if not _PROVIDERS_DIR.is_dir():
        return results

    for child in sorted(_PROVIDERS_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        init_file = child / "__init__.py"
        if not init_file.exists():
            continue

        # Read description from plugin.yaml if present
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

        # Check if the module has a resolve() function
        available = False
        try:
            mod = _load_module(child)
            if mod and hasattr(mod, "resolve"):
                available = True
        except Exception:
            pass

        results.append((child.name, desc, available))

    return results


def get_provider_plugin(name: str) -> Optional[Callable[..., Dict[str, Any]]]:
    """Return the resolve() function for a provider plugin, or None.

    Also checks aliases defined in the plugin's ALIASES list.
    """
    global _discovered
    if not _discovered:
        _discover_all()
        _discovered = True

    return _provider_cache.get(name.lower())


def _discover_all() -> None:
    """Load all provider plugins and cache their resolve functions."""
    if not _PROVIDERS_DIR.is_dir():
        return

    for child in sorted(_PROVIDERS_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        init_file = child / "__init__.py"
        if not init_file.exists():
            continue

        try:
            mod = _load_module(child)
            if mod and hasattr(mod, "resolve"):
                _provider_cache[child.name.lower()] = mod.resolve
                # Register aliases if defined
                aliases = getattr(mod, "ALIASES", [])
                for alias in aliases:
                    _provider_cache[alias.lower()] = mod.resolve
                logger.debug(
                    "Provider plugin '%s' loaded (aliases: %s)",
                    child.name, aliases or "none",
                )
        except Exception as exc:
            logger.debug("Failed to load provider plugin '%s': %s", child.name, exc)


def _load_module(provider_dir: Path):
    """Import a provider plugin module with submodule support for relative imports.

    Follows the same pattern as plugins/memory/__init__.py — registers parent
    packages, pre-loads submodules, then executes the __init__.py.
    """
    name = provider_dir.name
    module_name = f"plugins.providers.{name}"

    if module_name in sys.modules:
        return sys.modules[module_name]

    init_file = provider_dir / "__init__.py"
    if not init_file.exists():
        return None

    # Ensure parent packages are registered
    for parent in ("plugins", "plugins.providers"):
        if parent not in sys.modules:
            parent_path = _PROVIDERS_DIR.parent if parent == "plugins" else _PROVIDERS_DIR
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
    # e.g., "from .provider import resolve_blockrun_provider" in blockrun plugin
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

    return mod
