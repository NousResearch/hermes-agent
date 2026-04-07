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

    def _register_builtin_hooks(self) -> None:
        """Register built-in hooks that are always active."""
        try:
            from gateway.builtin_hooks.boot_md import handle as boot_md_handle

            self._handlers.setdefault("gateway:startup", []).append(boot_md_handle)
            self._loaded_hooks.append({
                "name": "boot-md",
                "description": "Run ~/.hermes/BOOT.md on gateway startup",
                "events": ["gateway:startup"],
                "path": "(builtin)",
            })
        except Exception as e:
            print(f"[hooks] Could not load built-in boot-md hook: {e}", flush=True)

        # Finetune feedback: record thumbs up/down reactions as quality signals.
        # Gated by finetune.feedback.gateway_reactions config flag.
        try:
            from hermes_cli.config import load_config
            ft_cfg = load_config().get("finetune", {})
            if ft_cfg.get("feedback", {}).get("gateway_reactions"):
                self._handlers.setdefault("reaction:add", []).append(
                    _finetune_reaction_handler
                )
                self._loaded_hooks.append({
                    "name": "finetune-feedback",
                    "description": "Record emoji reactions as finetune quality signals",
                    "events": ["reaction:add"],
                    "path": "(builtin)",
                })
        except Exception:
            pass  # finetune not configured — skip silently

    def discover_and_load(self) -> None:
        """
        Scan the hooks directory for hook directories and load their handlers.

        Also registers built-in hooks that are always active.

        Each hook directory must contain:
          - HOOK.yaml with at least 'name' and 'events' keys
          - handler.py with a top-level 'handle' function (sync or async)
        """
        self._register_builtin_hooks()

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
                print(f"[hooks] Error in handler for '{event_type}': {e}", flush=True)


def _finetune_reaction_handler(event_type: str, context: Dict[str, Any]) -> None:
    """Record emoji reactions as finetune quality feedback.

    Expected context keys:
        emoji: str — the reaction emoji (e.g. "👍", "thumbsup", "thumbsdown")
        session_id: str — the session the reacted message belongs to
        message_id: str — optional, the specific message
        platform: str — telegram, discord, slack, etc.
    """
    import json
    from datetime import datetime

    emoji = context.get("emoji", "")
    session_id = context.get("session_id", "")

    # Map common emoji names/chars to scores
    positive = {"👍", "thumbsup", "thumbs_up", "+1", "heart", "❤️", "⭐", "star"}
    negative = {"👎", "thumbsdown", "thumbs_down", "-1", "💩", "poop"}

    emoji_lower = emoji.lower().strip(":")
    if emoji_lower in positive or emoji == "👍":
        score = 1.0
        signal = "thumbs_up"
    elif emoji_lower in negative or emoji == "👎":
        score = 0.0
        signal = "thumbs_down"
    else:
        return  # Ignore non-feedback reactions

    feedback_path = get_hermes_home() / "finetune" / "feedback.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "session_id": session_id,
        "score": score,
        "signal": signal,
        "platform": context.get("platform", ""),
        "timestamp": datetime.now().isoformat(),
    }
    try:
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
