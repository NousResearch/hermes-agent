"""Event hook system: fires handlers at gateway lifecycle points.

Hooks live in ~/.hermes/hooks/<name>/ with HOOK.yaml (name, description, events) and
handler.py (``def handle(event_type, context)``, sync or async); errors never block
the pipeline.  Events: gateway:startup, session:start/end/reset, agent:start,
agent:step (each tool-loop turn), agent:end, command:* (wildcard).  agent:* context:
platform, user_id, chat_id, thread_id ("" outside a thread), chat_type
("dm"|"group"|"forum"|""), session_id, message (500 chars); agent:end adds response,
model, provider.  Forum follow-ups pass ``message_thread_id=int(thread_id)``.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from hermes_cli.config import get_hermes_home


HOOKS_DIR = get_hermes_home() / "hooks"


def _skip(name: str, reason: str) -> None:
    print(f"[hooks] Skipping {name}: {reason}", flush=True)


def _load_hook_dir(hook_dir: Path) -> Optional[tuple]:
    """``(name, events, handle_fn, description)`` for a valid hook dir, else None (reason printed)."""
    manifest_path, handler_path = hook_dir / "HOOK.yaml", hook_dir / "handler.py"
    if not manifest_path.exists() or not handler_path.exists():
        return None
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not manifest or not isinstance(manifest, dict):
        return _skip(hook_dir.name, "invalid HOOK.yaml")
    hook_name = manifest.get("name", hook_dir.name)
    events = manifest.get("events", [])
    if not events:
        return _skip(hook_name, "no events declared")
    # Register in sys.modules BEFORE exec_module so Pydantic/dataclass forward references
    # (``from __future__ import annotations``) resolve; otherwise a handler declaring a
    # BaseModel fails at first dispatch with "TypeAdapter ... is not fully defined".
    module_name = f"hermes_hook_{hook_name}"
    spec = importlib.util.spec_from_file_location(module_name, handler_path)
    if spec is None or spec.loader is None:
        return _skip(hook_name, "could not load handler.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    handle_fn = getattr(module, "handle", None)
    if handle_fn is None:
        return _skip(hook_name, "no 'handle' function found")
    return hook_name, events, handle_fn, manifest.get("description", "")


class HookRegistry:
    """Discovers, loads, and fires event hooks."""

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}  # event_type -> handlers
        self._loaded_hooks: List[dict] = []  # metadata for listing

    @property
    def loaded_hooks(self) -> List[dict]:
        return list(self._loaded_hooks)

    def _register_builtin_hooks(self) -> None:
        """Extension point for always-on built-in hooks; currently none shipped."""

    def discover_and_load(self) -> None:
        """Register built-in hooks, then load every valid hook dir under HOOKS_DIR."""
        self._register_builtin_hooks()
        if not HOOKS_DIR.exists():
            return
        for hook_dir in sorted(HOOKS_DIR.iterdir()):
            if not hook_dir.is_dir():
                continue
            try:
                loaded = _load_hook_dir(hook_dir)
            except Exception as e:
                print(f"[hooks] Error loading hook {hook_dir.name}: {e}", flush=True)
                continue
            if loaded is None:
                continue
            hook_name, events, handle_fn, description = loaded
            for event in events:
                self._handlers.setdefault(event, []).append(handle_fn)
            self._loaded_hooks.append(
                {"name": hook_name, "description": description, "events": events, "path": str(hook_dir)}
            )
            print(f"[hooks] Loaded hook '{hook_name}' for events: {events}", flush=True)

    def _resolve_handlers(self, event_type: str) -> List[Callable]:
        """Exact-match handlers first, then ``<base>:*`` wildcards.  A bare base type
        ("agent") does NOT fire for "agent:start" — only exact matches and explicit wildcards."""
        handlers = list(self._handlers.get(event_type, []))
        if ":" in event_type:
            handlers.extend(self._handlers.get(f"{event_type.split(':')[0]}:*", []))
        return handlers

    async def emit(self, event_type: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Fire all handlers for an event, discarding return values."""
        await self.emit_collect(event_type, context)

    async def emit_collect(self, event_type: str, context: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Fire handlers and return their non-None return values in order (decision-style
        hooks, e.g. ``command:<name>`` policies).  A failing handler is logged, not fatal."""
        if context is None:
            context = {}
        results: List[Any] = []
        for fn in self._resolve_handlers(event_type):
            try:
                result = fn(event_type, context)
                result = await result if asyncio.iscoroutine(result) else result  # sync or async handlers
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"[hooks] Error in handler for '{event_type}': {e}", flush=True)
        self._publish_agent_lifecycle_to_plugins(event_type, context)
        return results

    @staticmethod
    def _publish_agent_lifecycle_to_plugins(event_type: str, context: Dict[str, Any]) -> None:
        """Queue the stable, adapter-free lifecycle envelope for standalone plugins.

        Gateway hooks may keep their richer local context. Plugin subscribers receive only this documented
        primitive-data contract and run on the plugin event worker, so lifecycle decoration cannot block a
        gateway turn.
        """
        if event_type not in {"agent:start", "agent:step", "agent:end"}:
            return
        payload: Dict[str, Any] = {
            "lifecycle_schema_version": "hermes.gateway_agent_lifecycle.v1",
            "event_type": event_type,
            "platform": str(context.get("platform") or ""),
            "user_id": str(context.get("user_id") or ""),
            "chat_id": str(context.get("chat_id") or ""),
            "thread_id": str(context.get("thread_id") or ""),
            "chat_type": str(context.get("chat_type") or ""),
            "session_id": str(context.get("session_id") or ""),
            "message": str(context.get("message") or "")[:500],
        }
        if event_type == "agent:step":
            payload["iteration"] = context.get("iteration")
            payload["tool_names"] = list(context.get("tool_names") or ())
        elif event_type == "agent:end":
            payload["response"] = str(context.get("response") or "")[:500]
            payload["model"] = str(context.get("model") or "")
            payload["provider"] = str(context.get("provider") or "")
            payload["failed"] = bool(context.get("failed"))
        try:
            from hermes_cli.plugins import emit_core_event

            emit_core_event("gateway_agent_lifecycle", payload)
        except Exception as exc:
            print(f"[hooks] Plugin lifecycle delivery error: {exc}", flush=True)
