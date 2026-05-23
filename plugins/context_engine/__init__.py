"""Context engine plugin discovery.

Scans two directories for context engine plugins:

1. Bundled engines: ``plugins/context_engine/<name>/`` (shipped with hermes-agent)
2. User-installed engines: ``$HERMES_HOME/plugins/<name>/``

Each subdirectory must contain ``__init__.py`` with a class implementing
the ContextEngine ABC.  Only ONE can be active at a time, selected via
``context.engine`` in config.yaml.  The default engine is ``"compressor"``
(the built-in ContextCompressor).  Bundled engines take precedence on
name collisions.

Usage:
    from plugins.context_engine import discover_context_engines, load_context_engine

    available = discover_context_engines()   # [(name, desc, available), ...]
    engine = load_context_engine("lcm")      # ContextEngine instance
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_CONTEXT_ENGINE_PLUGINS_DIR = Path(__file__).parent


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


def _is_context_engine_dir(path: Path) -> bool:
    """Heuristic: does *path* look like a context engine plugin?

    Checks for ``ContextEngine`` or ``register_context_engine`` in the
    ``__init__.py`` source.  Cheap text scan — no import needed.
    """
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(errors="replace")[:8192]
        return "ContextEngine" in source or "register_context_engine" in source
    except Exception:
        return False


def _find_engine_dir(name: str) -> Optional[Path]:
    """Resolve a context engine name to its directory.

    Checks bundled first, then user-installed.
    """
    # Bundled
    bundled = _CONTEXT_ENGINE_PLUGINS_DIR / name
    if bundled.is_dir() and (bundled / "__init__.py").exists():
        return bundled
    # User-installed
    user_dir = _get_user_plugins_dir()
    if user_dir:
        user = user_dir / name
        if user.is_dir() and _is_context_engine_dir(user):
            return user
    return None


def discover_context_engines() -> List[Tuple[str, str, bool]]:
    """Scan bundled and user-installed directories for available engines.

    Returns list of (name, description, is_available) tuples.
    Does NOT import the engines — just reads plugin.yaml for metadata
    and does a lightweight availability check.
    Bundled engines take precedence on name collisions.
    """
    results: List[Tuple[str, str, bool]] = []
    seen: set = set()

    def _scan_dir(scan_dir: Path) -> None:
        if not scan_dir.is_dir():
            return
        for child in sorted(scan_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if child.name in seen:
                continue  # bundled takes precedence
            init_file = child / "__init__.py"
            if not init_file.exists():
                continue

            # For user plugins, filter non-context-engine dirs
            if scan_dir != _CONTEXT_ENGINE_PLUGINS_DIR:
                if not _is_context_engine_dir(child):
                    continue

            seen.add(child.name)

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

            # Quick availability check
            available = True
            try:
                engine = _load_engine_from_dir(child)
                if engine is None:
                    available = False
                elif hasattr(engine, "is_available"):
                    available = engine.is_available()
            except Exception:
                available = False

            results.append((child.name, desc, available))

    # 1. Bundled engines (plugins/context_engine/<name>/)
    _scan_dir(_CONTEXT_ENGINE_PLUGINS_DIR)

    # 2. User-installed engines ($HERMES_HOME/plugins/<name>/)
    user_dir = _get_user_plugins_dir()
    if user_dir:
        _scan_dir(user_dir)

    return results


def load_context_engine(name: str) -> Optional["ContextEngine"]:
    """Load and return a ContextEngine instance by name.

    Checks both bundled (``plugins/context_engine/<name>/``) and
    user-installed (``$HERMES_HOME/plugins/<name>/``) directories.
    Bundled takes precedence on name collisions.

    Returns None if the engine is not found or fails to load.
    """
    engine_dir = _find_engine_dir(name)
    if not engine_dir:
        logger.debug(
            "Context engine '%s' not found in bundled or user plugins", name
        )
        return None

    try:
        engine = _load_engine_from_dir(engine_dir)
        if engine:
            return engine
        logger.warning(
            "Context engine '%s' loaded but no engine instance found", name
        )
        return None
    except Exception as e:
        logger.warning("Failed to load context engine '%s': %s", name, e)
        return None


def _load_engine_from_dir(engine_dir: Path) -> Optional["ContextEngine"]:
    """Import an engine module and extract the ContextEngine instance.

    The module must have either:
    - A register(ctx) function (plugin-style) — we simulate a ctx
    - A top-level class that extends ContextEngine — we instantiate it
    """
    name = engine_dir.name
    # Use a separate namespace for user-installed plugins so they don't
    # collide with bundled engines in sys.modules.
    _is_bundled = (
        _CONTEXT_ENGINE_PLUGINS_DIR in engine_dir.parents
        or engine_dir.parent == _CONTEXT_ENGINE_PLUGINS_DIR
    )
    module_name = (
        f"plugins.context_engine.{name}"
        if _is_bundled
        else f"_hermes_user_context_engine.{name}"
    )
    init_file = engine_dir / "__init__.py"

    if not init_file.exists():
        return None

    # Check if already loaded
    if module_name in sys.modules:
        mod = sys.modules[module_name]
    else:
        # Handle relative imports within the plugin
        # First ensure the parent packages are registered
        for parent in ("plugins", "plugins.context_engine"):
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

        # Now load the engine module
        spec = importlib.util.spec_from_file_location(
            module_name, str(init_file),
            submodule_search_locations=[str(engine_dir)]
        )
        if not spec:
            return None

        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod

        # Register submodules so relative imports work
        for sub_file in engine_dir.glob("*.py"):
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

    # Try register(ctx) pattern first (how plugins are written)
    if hasattr(mod, "register"):
        collector = _EngineCollector()
        try:
            mod.register(collector)
            if collector.engine:
                return collector.engine
        except Exception as e:
            logger.debug("register() failed for %s: %s", name, e)

    # Fallback: find a ContextEngine subclass and instantiate it
    from agent.context_engine import ContextEngine
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if (isinstance(attr, type) and issubclass(attr, ContextEngine)
                and attr is not ContextEngine):
            try:
                return attr()
            except Exception:
                pass

    return None


class _EngineCollector:
    """Fake plugin context that captures register_context_engine calls."""

    def __init__(self):
        self.engine = None

    def register_context_engine(self, engine):
        self.engine = engine

    # No-op for other registration methods
    def register_tool(self, *args, **kwargs):
        pass

    def register_hook(self, *args, **kwargs):
        pass

    def register_cli_command(self, *args, **kwargs):
        pass

    def register_memory_provider(self, *args, **kwargs):
        pass
