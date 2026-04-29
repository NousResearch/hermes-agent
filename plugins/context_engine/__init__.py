"""Context engine plugin discovery.

Scans ``plugins/context_engine/<name>/`` directories for context engine
plugins.  Each subdirectory must contain ``__init__.py`` with a class
implementing the ContextEngine ABC.

Context engines are separate from the general plugin system — they live
in the repo and are always available without user installation.  Only ONE
can be active at a time, selected via ``context.engine`` in config.yaml.
The default engine is ``"compressor"`` (the built-in ContextCompressor).

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


def _selected_context_engine_name() -> str:
    """Return the configured context engine name."""
    try:
        from hermes_cli.config import cfg_get, load_config
        config = load_config()
        return cfg_get(config, "context", "engine", default="compressor") or "compressor"
    except Exception:
        return "compressor"


def mark_context_engine_commands_synced(engine_name: str) -> None:
    try:
        from hermes_cli.plugins import get_plugin_manager
        setattr(get_plugin_manager(), "_context_engine_commands_synced_for", engine_name)
    except Exception:
        pass


def sync_configured_context_engine_commands() -> None:
    """Ensure slash commands for the configured context engine are registered.

    Gateway help/menu/dispatch can run before the first AIAgent instance is
    created.  In that path no one has loaded the configured context engine yet,
    so command enumeration must explicitly preload its command surface.
    """
    engine_name = _selected_context_engine_name()
    try:
        from hermes_cli.plugins import get_plugin_manager
        manager = get_plugin_manager()
    except Exception:
        manager = None
    if manager is not None and getattr(manager, "_context_engine_commands_synced_for", None) == engine_name:
        return
    if engine_name == "compressor":
        clear_context_engine_commands()
        mark_context_engine_commands_synced(engine_name)
        return
    load_context_engine(engine_name)


def discover_context_engines() -> List[Tuple[str, str, bool]]:
    """Scan plugins/context_engine/ for available engines.

    Returns list of (name, description, is_available) tuples.
    Does NOT import the engines — just reads plugin.yaml for metadata
    and does a lightweight availability check.
    """
    results = []
    if not _CONTEXT_ENGINE_PLUGINS_DIR.is_dir():
        return results

    for child in sorted(_CONTEXT_ENGINE_PLUGINS_DIR.iterdir()):
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
                with open(yaml_file, encoding="utf-8-sig") as f:
                    meta = yaml.safe_load(f) or {}
                desc = meta.get("description", "")
            except Exception:
                pass

        # Quick availability check — try loading and calling is_available()
        available = True
        try:
            engine = _load_engine_from_dir(child, register_commands=False)
            if engine is None:
                available = False
            elif hasattr(engine, "is_available"):
                available = engine.is_available()
        except Exception:
            available = False

        results.append((child.name, desc, available))

    return results


def load_context_engine(name: str) -> Optional["ContextEngine"]:
    """Load and return a ContextEngine instance by name.

    Returns None if the engine is not found or fails to load.
    """
    engine_dir = _CONTEXT_ENGINE_PLUGINS_DIR / name
    clear_context_engine_commands()
    if not engine_dir.is_dir():
        logger.debug("Context engine '%s' not found in %s", name, _CONTEXT_ENGINE_PLUGINS_DIR)
        mark_context_engine_commands_synced(name)
        return None

    try:
        engine = _load_engine_from_dir(engine_dir, register_commands=True)
        if engine:
            mark_context_engine_commands_synced(name)
            return engine
        logger.warning("Context engine '%s' loaded but no engine instance found", name)
        mark_context_engine_commands_synced(name)
        return None
    except Exception as e:
        logger.warning("Failed to load context engine '%s': %s", name, e)
        mark_context_engine_commands_synced(name)
        return None


def clear_context_engine_commands() -> None:
    """Remove slash commands registered by a previously loaded context engine."""
    try:
        from hermes_cli.plugins import get_plugin_manager
    except Exception:
        return
    try:
        commands = get_plugin_manager()._plugin_commands
    except Exception:
        return
    for command_name, meta in list(commands.items()):
        plugin_name = meta.get("plugin") if isinstance(meta, dict) else None
        if isinstance(plugin_name, str) and plugin_name.startswith("context_engine:"):
            commands.pop(command_name, None)


def _load_engine_from_dir(engine_dir: Path, *, register_commands: bool = False) -> Optional["ContextEngine"]:
    """Import an engine module and extract the ContextEngine instance.

    The module must have either:
    - A register(ctx) function (plugin-style) — we simulate a ctx
    - A top-level class that extends ContextEngine — we instantiate it
    """
    name = engine_dir.name
    module_name = f"plugins.context_engine.{name}"
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
        collector = _EngineCollector(name, register_commands=register_commands)
        try:
            mod.register(collector)
            if collector.engine:
                collector.commit_commands()
                return collector.engine
        except Exception as e:
            if register_commands:
                clear_context_engine_commands()
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
    """Plugin-like context that captures context-engine registrations."""

    def __init__(self, engine_name: str, *, register_commands: bool = False):
        self.engine_name = engine_name
        self.register_commands = register_commands
        self.engine = None
        self._pending_commands = []

    def register_context_engine(self, engine):
        self.engine = engine

    def register_command(self, name, handler, description="", args_hint=""):
        """Register an in-session slash command for the active context engine."""
        if not self.register_commands:
            return
        clean = str(name).lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning(
                "Context engine '%s' tried to register a command with an empty name.",
                self.engine_name,
            )
            return
        try:
            from hermes_cli.commands import resolve_command
            if resolve_command(clean) is not None:
                logger.warning(
                    "Context engine '%s' tried to register command '/%s' which conflicts "
                    "with a built-in command. Skipping.",
                    self.engine_name, clean,
                )
                return
        except Exception:
            pass
        self._pending_commands.append((clean, handler, description, str(args_hint or "").strip()))

    def commit_commands(self):
        """Commit buffered slash commands after the engine registered successfully."""
        if not self.register_commands or not self._pending_commands:
            return
        try:
            from hermes_cli.plugins import get_plugin_manager
            commands = get_plugin_manager()._plugin_commands
        except Exception as exc:
            logger.debug(
                "Context engine '%s' failed to register commands: %s",
                self.engine_name, exc,
            )
            return
        for clean, handler, description, args_hint in self._pending_commands:
            existing = commands.get(clean)
            existing_plugin = existing.get("plugin") if isinstance(existing, dict) else None
            if existing is not None and existing_plugin != f"context_engine:{self.engine_name}":
                logger.warning(
                    "Context engine '%s' tried to register command '/%s' which is "
                    "already registered by plugin '%s'. Skipping.",
                    self.engine_name, clean, existing_plugin or "unknown",
                )
                continue
            commands[clean] = {
                "handler": handler,
                "description": description or "Plugin command",
                "plugin": f"context_engine:{self.engine_name}",
                "args_hint": args_hint,
            }

    # No-op for other registration methods
    def register_tool(self, *args, **kwargs):
        pass

    def register_hook(self, *args, **kwargs):
        pass

    def register_cli_command(self, *args, **kwargs):
        pass

    def register_memory_provider(self, *args, **kwargs):
        pass
