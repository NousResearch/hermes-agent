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

Context dict passed to ``agent:start`` / ``agent:end`` handlers:
  platform     -- source platform name (e.g. "telegram", "matrix", "slack")
  user_id      -- platform user id of the sender
  chat_id      -- platform chat id (group/DM identifier)
  thread_id    -- Telegram forum-topic id / thread root id (string; empty
                  when not in a thread / topic)
  chat_type    -- "dm" | "group" | "forum" (empty if unknown)
  session_id   -- Hermes session id
  message      -- inbound message text (truncated to 500 chars)

``agent:end`` adds:
  response     -- agent response text (truncated to 500 chars)
  model        -- model name that handled the turn
  provider     -- provider that handled the turn

Handlers posting a follow-up into the same Telegram forum-topic should
include ``message_thread_id=int(thread_id)`` when ``chat_type == "forum"``
and ``thread_id`` is non-empty.
"""

import asyncio
import importlib.util
import inspect
import sys
from typing import Any, Callable, Dict, List, Optional

import yaml

from hermes_cli.config import get_hermes_home


HOOKS_DIR = get_hermes_home() / "hooks"

# Sentinel for "no replacement value supplied" in emit_waterfall's next_fn.
_MISSING = object()


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
        """Register built-in hooks that are always active.

        Currently empty — no shipped built-in hooks. Kept as the extension
        point for future always-on gateway hooks so they drop in without
        re-plumbing discover_and_load().
        """
        return

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

                # Dynamically load the handler module.
                # Register in sys.modules BEFORE exec_module so Pydantic /
                # dataclasses / typing introspection can resolve forward
                # references (triggered by `from __future__ import annotations`
                # in the handler). Without this, a handler that declares a
                # Pydantic BaseModel for webhook/event payloads fails at first
                # dispatch with "TypeAdapter ... is not fully defined".
                module_name = f"hermes_hook_{hook_name}"
                spec = importlib.util.spec_from_file_location(
                    module_name, handler_path
                )
                if spec is None or spec.loader is None:
                    print(f"[hooks] Skipping {hook_name}: could not load handler.py", flush=True)
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(module_name, None)
                    raise

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

    def _resolve_handlers(self, event_type: str) -> List[Callable]:
        """Return all handlers that should fire for ``event_type``.

        Exact matches fire first, followed by wildcard matches (e.g.
        ``command:*`` matches ``command:reset``).
        """
        handlers = list(self._handlers.get(event_type, []))
        if ":" in event_type:
            base = event_type.split(":")[0]
            wildcard_key = f"{base}:*"
            handlers.extend(self._handlers.get(wildcard_key, []))
        return handlers

    async def emit(self, event_type: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Fire all handlers registered for an event, discarding return values.

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

        for fn in self._resolve_handlers(event_type):
            try:
                result = fn(event_type, context)
                # Support both sync and async handlers
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                print(f"[hooks] Error in handler for '{event_type}': {e}", flush=True)

    async def emit_collect(
        self,
        event_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Fire handlers and return their non-None return values in order.

        Like :meth:`emit` but captures each handler's return value. Used for
        decision-style hooks (e.g. ``command:<name>`` policies that want to
        allow/deny/rewrite the command before normal dispatch).

        Exceptions from individual handlers are logged but do not abort the
        remaining handlers.
        """
        if context is None:
            context = {}

        results: List[Any] = []
        for fn in self._resolve_handlers(event_type):
            try:
                result = fn(event_type, context)
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"[hooks] Error in handler for '{event_type}': {e}", flush=True)
        return results

    async def emit_waterfall(
        self,
        event_type: str,
        value: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Fire handlers in Cordis-style waterfall (around-middleware) mode.

        Each waterfall handler receives
        ``(event_type, value, context, next_fn)``:

        - call ``await next_fn(new_value=...)`` to delegate to the next
          handler, optionally replacing the current value for downstream
          listeners;
        - return without calling ``next_fn`` to **short-circuit** the chain
          — later handlers do not run and the current value is the result.

        This mirrors the Cordis waterfall dispatch used by DeepSeek Harness
        for ``tools/pre-execute`` / ``tools/execute`` / ``tools/post-execute``:
        cooperative listeners mutate a shared request or decision object and
        delegate, while a policy listener that owns a decision returns
        without delegating. ``prepend: true`` is not needed — registration
        order is the chain order.

        Legacy two-argument handlers (``handle(event_type, context)``) are
        invoked as observers in registration order: they cannot rewrite or
        short-circuit the chain, their return values are ignored, and a
        throwing observer does not abort the remaining handlers (same
        containment as :meth:`emit`). A handler whose signature accepts
        ``next_fn`` is treated as a waterfall participant: it may rewrite,
        delegate, or short-circuit.

        Exceptions from waterfall participants are logged and short-circuit
        the chain (fail-closed — a policy handler that crashed did not
        delegate, so continuing would skip the remaining policy).
        """
        if context is None:
            context = {}

        handlers = self._resolve_handlers(event_type)
        current = value
        index = 0

        async def _delegate(new_value: Any = _MISSING) -> Any:
            nonlocal current, index
            if new_value is not _MISSING:
                current = new_value
            index += 1
            return await _run_from(index)

        async def _run_from(start: int) -> Any:
            nonlocal current, index
            index = start
            while index < len(handlers):
                fn = handlers[index]
                try:
                    sig = inspect.signature(fn)
                except (TypeError, ValueError):
                    sig = None

                # A handler that accepts four positional parameters is a
                # waterfall participant; anything else is an observer.
                waterfall_participant = bool(
                    sig
                    and any(
                        p.kind
                        in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                        for p in list(sig.parameters.values())[3:4]
                    )
                )

                if not waterfall_participant:
                    # Legacy observer: run, ignore return value, keep the
                    # chain alive exactly like emit() containment.
                    try:
                        result = fn(event_type, context)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        print(
                            f"[hooks] Error in observer for '{event_type}': {e}",
                            flush=True,
                        )
                    index += 1
                    continue

                delegated = False

                async def _local_next(new_value: Any = _MISSING) -> Any:
                    nonlocal delegated
                    delegated = True
                    return await _delegate(new_value)

                try:
                    result = fn(event_type, current, context, _local_next)
                    if asyncio.iscoroutine(result):
                        result = await result
                except Exception as e:
                    print(
                        f"[hooks] Error in waterfall handler for '{event_type}': {e}",
                        flush=True,
                    )
                    # Fail-closed: a crashed participant did not delegate.
                    # Advance past the end so no delegating ancestor frame's
                    # while-loop resumes and re-invokes this handler (a
                    # throwing participant was already executed — running it
                    # again would double-fire and see a stale input).
                    index = len(handlers)
                    break
                if not delegated:
                    # Short-circuit: the participant owns the decision and its
                    # return value IS the waterfall result (Cordis semantics).
                    if result is not None:
                        current = result
                    # Advance past the end: this participant was already
                    # executed. Without this, every delegating ancestor frame
                    # resumes its while-loop at the SAME un-advanced index and
                    # re-invokes the short-circuiting handler (dispatch order
                    # [A,B,C] becomes [A,B,C,C,C], and the repeat runs see the
                    # prior run's return value as their input).
                    index = len(handlers)
                    break
            return current

        return await _run_from(0)
