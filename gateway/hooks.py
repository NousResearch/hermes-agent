"""
Event Hook System

A lightweight event-driven system that fires handlers at key lifecycle points.
Hooks are discovered from ~/.hermes/hooks/ directories, each containing:
  - HOOK.yaml  (metadata: name, description, events list)
  - handler.py (Python handler with async def handle(event_type, context))

Events:
  - gateway:startup      -- Gateway process starts
  - session:start        -- New session created (first message of a new session)
  - session:end          -- Session ends (user ran /new or /reset)
  - session:reset        -- Session reset completed (new session entry created)
  - session:compact      -- Session context was compacted/summarized
  - agent:start          -- Agent begins processing a message
  - agent:bootstrap      -- Agent context files loaded (payload: platform, user_id, files[])
  - agent:step           -- Each turn in the tool-calling loop
  - agent:end            -- Agent finishes processing
  - message:received     -- Inbound message received (payload: platform, user_id, text, attachments[])
  - message:sent         -- Outbound message delivered (payload: platform, chat_id, text)
  - message:transcribed  -- Voice/audio transcribed to text (payload: platform, user_id, text, audio_path)
  - message:preprocessed -- Message after pre-processing transforms (payload: platform, user_id, text)
  - command:*            -- Any slash command executed (wildcard match)

Errors in hooks are caught and logged but never block the main pipeline.

Hook handlers can push reply strings into context["messages"] to send
additional messages back to the user after the hook fires.
"""

import asyncio
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from hermes_cli.config import get_hermes_home


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
                    print(f"[hooks] Skipping {hook_dir.name}: invalid HOOK.yaml", flush=True)
                    continue

                hook_name = manifest.get("name", hook_dir.name)
                events = manifest.get("events", [])
                if not events:
                    print(f"[hooks] Skipping {hook_name}: no events declared", flush=True)
                    continue

                # Dynamically load the handler module
                spec = importlib.util.spec_from_file_location(
                    f"hermes_hook_{hook_name}", handler_path
                )
                if spec is None or spec.loader is None:
                    print(f"[hooks] Skipping {hook_name}: could not load handler.py", flush=True)
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                handle_fn = getattr(module, "handle", None)
                if handle_fn is None:
                    print(f"[hooks] Skipping {hook_name}: no 'handle' function found", flush=True)
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

                print(f"[hooks] Loaded hook '{hook_name}' for events: {events}", flush=True)

            except Exception as e:
                print(f"[hooks] Error loading hook {hook_dir.name}: {e}", flush=True)

    # Events that fire all handlers in parallel (fire-and-forget style)
    PARALLEL_EVENTS = {
        "gateway:startup",
        "session:start", "session:end", "session:reset", "session:compact",
        "agent:start", "agent:bootstrap", "agent:step", "agent:end",
        "message:received", "message:sent", "message:transcribed", "message:preprocessed",
        "command:*",
    }

    async def emit(
        self,
        event_type: str,
        context: Optional[Dict[str, Any]] = None,
        parallel: Optional[bool] = None,
    ) -> List[str]:
        """
        Fire all handlers registered for an event.

        Supports wildcard matching: handlers registered for "command:*" will
        fire for any "command:..." event.

        Handlers can append strings to context["messages"] to send replies
        back to the user after the hook fires.

        Args:
            event_type: The event identifier (e.g. "agent:start").
            context:    Optional dict with event-specific data.
            parallel:   If True, run handlers concurrently. Defaults to True
                        for fire-and-forget events, False for modifying events.

        Returns:
            List of reply messages pushed by handlers via context["messages"].
        """
        if context is None:
            context = {}

        # Ensure messages list exists for handlers to push replies into
        if "messages" not in context:
            context["messages"] = []

        # Collect handlers: exact match + wildcard match
        handlers = list(self._handlers.get(event_type, []))

        # Check for wildcard patterns (e.g., "command:*" matches "command:reset")
        if ":" in event_type:
            base = event_type.split(":")[0]
            wildcard_key = f"{base}:*"
            handlers.extend(self._handlers.get(wildcard_key, []))

        if not handlers:
            return []

        # Determine execution mode
        use_parallel = parallel
        if use_parallel is None:
            base_event = event_type.split(":")[0] + ":*"
            use_parallel = event_type in self.PARALLEL_EVENTS or base_event in self.PARALLEL_EVENTS

        async def _run_one(fn):
            try:
                result = fn(event_type, context)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"[hooks] Error in handler for '{event_type}': {e}", flush=True)

        if use_parallel:
            await asyncio.gather(*[_run_one(fn) for fn in handlers], return_exceptions=True)
        else:
            for fn in handlers:
                await _run_one(fn)

        return list(context.get("messages", []))
