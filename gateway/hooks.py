"""
Event Hook System

A lightweight event-driven system that fires handlers at key lifecycle points.
Hooks are discovered from ~/.hermes/hooks/ directories, each containing:
  - HOOK.yaml  (metadata: name, description, events list)
  - handler.py (Python handler with async def handle(event_type, context))

Events:
  - gateway:startup     -- Gateway process starts
  - session:start       -- New session created (first message of a new session)
  - session:end         -- Session ends (user ran /new or /reset)
  - session:reset       -- Session reset completed (new session entry created)
  - agent:start         -- Agent begins processing a message
  - agent:step          -- Each turn in the tool-calling loop
  - agent:end           -- Agent finishes processing
  - command:*           -- Any slash command executed (wildcard match)

Errors in hooks are caught and logged but never block the main pipeline.
"""

import asyncio
import importlib.util
from typing import Any, Callable, Dict, List, Optional

import logging
import yaml

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

HOOKS_DIR = get_hermes_home() / "hooks"


class HookRegistry:
    """
    Discovers, loads, and fires event hooks.

    Usage:
        registry = HookRegistry()
        registry.discover_and_load()
        await registry.emit("agent:start", {"platform": "telegram", ...})
    """

    def __init__(self):
        # event_type -> [handler_fn, ...]
        self._handlers: Dict[str, List[Callable]] = {}
        self._loaded_hooks: List[dict] = []  # metadata for listing

    @property
    def loaded_hooks(self) -> List[dict]:
        """Return metadata about all loaded hooks."""
        return list(self._loaded_hooks)

    def discover_and_load(self) -> None:
        """
        Scan the hooks directory for hook directories and load their handlers.

        Each hook directory must contain:
          - HOOK.yaml with at least 'name' and 'events' keys
          - handler.py with a top-level 'handle' function (sync or async)
        """
        if not HOOKS_DIR.exists():
            return

        for hook_dir in sorted(HOOKS_DIR.iterdir()):
            if not hook_dir.is_dir():
                continue

            manifest_path = hook_dir / "HOOK.yaml"
            handler_path = hook_dir / "handler.py"

            if not manifest_path.exists() or not handler_path.exists():
                continue

            try:
                manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                if not manifest or not isinstance(manifest, dict):
                    logger.warning("[hooks] Skipping %s: invalid HOOK.yaml", hook_dir.name)
                    continue

                hook_name = manifest.get("name", hook_dir.name)
                events = manifest.get("events", [])
                if not events:
                    logger.warning("[hooks] Skipping %s: no events declared", hook_name)
                    continue

                # Dynamically load the handler module
                spec = importlib.util.spec_from_file_location(
                    f"hermes_hook_{hook_name}", handler_path
                )
                if spec is None or spec.loader is None:
                    logger.warning("[hooks] Skipping %s: could not load handler.py", hook_name)
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                handle_fn = getattr(module, "handle", None)
                if handle_fn is None:
                    logger.warning("[hooks] Skipping %s: no 'handle' function found", hook_name)
                    continue

                # Register the handler for each declared event
                for event in events:
                    self._handlers.setdefault(event, []).append(handle_fn)

                self._loaded_hooks.append({
                    "name": hook_name,
                    "description": manifest.get("description", ""),
                    "events": events,
                    "path": str(hook_dir),
                })

                logger.info("[hooks] Loaded hook '%s' for events: %s", hook_name, events)

            except Exception as e:
                logger.error("[hooks] Error loading hook %s: %s", hook_dir.name, e)

    async def emit(self, event_type: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Fire all handlers registered for an event.

        Supports wildcard matching: handlers registered for "command:*" will
        fire for any "command:..." event. Handlers registered for a base type
        like "agent" won't fire for "agent:start" -- only exact matches and
        explicit wildcards.

        Args:
            event_type: The event identifier (e.g. "agent:start").
            context:    Optional dict with event-specific data.
        """
        if context is None:
            context = {}

        # Collect handlers: exact match + wildcard match
        handlers = list(self._handlers.get(event_type, []))

        # Check for wildcard patterns (e.g., "command:*" matches "command:reset")
        if ":" in event_type:
            base = event_type.split(":")[0]
            wildcard_key = f"{base}:*"
            handlers.extend(self._handlers.get(wildcard_key, []))

        for fn in handlers:
            try:
                result = fn(event_type, context)
                # Support both sync and async handlers
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("[hooks] Error in handler for '%s': %s", event_type, e)
