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
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from hermes_cli.config import get_hermes_home


HOOKS_DIR = get_hermes_home() / "hooks"

# Sentinel for "no replacement value supplied" in emit_waterfall's next_fn.
_MISSING = object()


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

        ``next_fn()`` is **one-shot**: delegate at most once per handler and
        always **await** it. A sync handler must ``return next_fn()`` — the
        dispatcher awaits the returned coroutine. Calling ``next_fn()``
        without awaiting (or without returning it from a sync handler) does
        NOT delegate: the handler is then treated as a short-circuit and its
        return value becomes the result, while the un-awaited coroutine is
        dropped. A second ``next_fn()`` call is ignored with a warning
        instead of silently advancing past a downstream handler.

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
                next_called = False

                async def _local_next(new_value: Any = _MISSING) -> Any:
                    nonlocal delegated, next_called
                    if next_called:
                        # Contract violation: delegate at most once. A second
                        # call would otherwise advance the shared index past
                        # a downstream handler (silent skip) — warn and
                        # return the current value instead.
                        print(
                            f"[hooks] Waterfall handler for '{event_type}' called "
                            "next_fn() more than once — subsequent calls are "
                            "ignored (delegate at most once).",
                            flush=True,
                        )
                        return current
                    next_called = True
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
