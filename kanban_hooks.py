"""Kanban event hook loader and async dispatcher.

Mirrors the gateway hook pattern but is scoped to kanban lifecycle events.
Kanban DB mutations call ``emit_kanban_event()`` from synchronous code paths,
so dispatch is offloaded onto a dedicated background event loop thread.

Operational notes:
- hooks are discovered when the singleton registry is first created; new or
  changed hooks are picked up on the next Hermes process start, not hot-reloaded
  into an already-running process.
- the dispatcher thread is daemonized on purpose; it lives for the lifetime of
  the current Hermes process and is torn down explicitly only in tests.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from hermes_constants import get_hermes_home


_log = logging.getLogger(__name__)

KANBAN_HOOKS_DIR = get_hermes_home() / "kanban-hooks"
SUPPORTED_EVENTS = frozenset(
    {
        "task:blocked",
        "task:unblocked",
        "task:completed",
        "task:created",
    }
)


class HookRegistry:
    """Singleton kanban hook registry."""

    _instance: "HookRegistry | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable[..., Any]]] = {}
        self._loaded_hooks: List[dict] = []
        self.discover_and_load()

    @property
    def loaded_hooks(self) -> List[dict]:
        return list(self._loaded_hooks)

    @classmethod
    def instance(cls) -> "HookRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        _shutdown_dispatcher_for_tests()
        with cls._instance_lock:
            cls._instance = None

    def discover_and_load(self) -> None:
        hooks_dir = get_hermes_home() / "kanban-hooks"
        if not hooks_dir.exists():
            return

        for hook_dir in sorted(hooks_dir.iterdir()):
            if not hook_dir.is_dir():
                continue

            manifest_path = hook_dir / "HOOK.yaml"
            handler_path = hook_dir / "handler.py"
            if not manifest_path.exists() or not handler_path.exists():
                continue

            try:
                manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    _log.warning("[kanban-hooks] Skipping %s: invalid HOOK.yaml", hook_dir.name)
                    continue

                hook_name = str(manifest.get("name") or hook_dir.name)
                events = manifest.get("events") or []
                if not isinstance(events, list):
                    _log.warning("[kanban-hooks] Skipping %s: events must be a list", hook_name)
                    continue
                filtered_events = [
                    str(event).strip() for event in events
                    if str(event).strip() in SUPPORTED_EVENTS or str(event).strip() == "task:*"
                ]
                if not filtered_events:
                    _log.warning("[kanban-hooks] Skipping %s: no supported events declared", hook_name)
                    continue

                module_name = f"hermes_kanban_hook_{hook_name.replace('-', '_')}"
                spec = importlib.util.spec_from_file_location(module_name, handler_path)
                if spec is None or spec.loader is None:
                    _log.warning("[kanban-hooks] Skipping %s: could not load handler.py", hook_name)
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
                    _log.warning("[kanban-hooks] Skipping %s: no 'handle' function found", hook_name)
                    continue

                for event in filtered_events:
                    self._handlers.setdefault(event, []).append(handle_fn)

                self._loaded_hooks.append(
                    {
                        "name": hook_name,
                        "description": str(manifest.get("description") or ""),
                        "events": filtered_events,
                        "path": str(hook_dir),
                    }
                )
            except Exception:
                _log.exception("[kanban-hooks] Error loading hook %s", hook_dir.name)

    def _resolve_handlers(self, event_type: str) -> List[Callable[..., Any]]:
        handlers = list(self._handlers.get(event_type, []))
        if ":" in event_type:
            wildcard_key = f"{event_type.split(':', 1)[0]}:*"
            handlers.extend(self._handlers.get(wildcard_key, []))
        return handlers

    async def emit(self, event_type: str, context: Optional[Dict[str, Any]] = None) -> None:
        if context is None:
            context = {}
        for fn in self._resolve_handlers(event_type):
            try:
                result = fn(event_type, context)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                _log.exception("[kanban-hooks] Error in handler for %s", event_type)


_dispatcher_lock = threading.Lock()
_dispatcher_loop: asyncio.AbstractEventLoop | None = None
_dispatcher_thread: threading.Thread | None = None
_dispatcher_ready: threading.Event | None = None
_pending_futures: set[Any] = set()


def _dispatcher_main(ready: threading.Event) -> None:
    global _dispatcher_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _dispatcher_loop = loop
    ready.set()
    try:
        loop.run_forever()
    finally:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        _dispatcher_loop = None


def _ensure_dispatcher_loop() -> asyncio.AbstractEventLoop:
    global _dispatcher_thread, _dispatcher_ready
    with _dispatcher_lock:
        loop = _dispatcher_loop
        if loop is not None and loop.is_running():
            return loop
        ready = threading.Event()
        thread = threading.Thread(
            target=_dispatcher_main,
            args=(ready,),
            name="kanban-hooks-dispatcher",
            daemon=True,
        )
        _dispatcher_ready = ready
        _dispatcher_thread = thread
        thread.start()
    ready.wait(timeout=2.0)
    loop = _dispatcher_loop
    if loop is None:
        raise RuntimeError("kanban hook dispatcher failed to start")
    return loop


def _track_future(future: Any) -> None:
    _pending_futures.add(future)

    def _done(done_future: Any) -> None:
        _pending_futures.discard(done_future)
        try:
            done_future.result()
        except Exception:
            _log.exception("[kanban-hooks] Background dispatch failed")

    future.add_done_callback(_done)


def emit_kanban_event(event_type: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Queue a kanban hook event for background delivery.

    Safe to call from synchronous DB mutation paths. Unsupported events are ignored.
    """
    if event_type not in SUPPORTED_EVENTS:
        return
    registry = HookRegistry.instance()
    if not registry._resolve_handlers(event_type):
        return
    loop = _ensure_dispatcher_loop()
    future = asyncio.run_coroutine_threadsafe(registry.emit(event_type, dict(context or {})), loop)
    _track_future(future)


def wait_for_pending_events(timeout: float = 2.0) -> None:
    """Test helper: wait until the current pending hook queue drains."""
    deadline = time.time() + max(0.0, float(timeout))
    while True:
        pending = list(_pending_futures)
        if not pending:
            return
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for kanban hook dispatch")
        pending[0].result(timeout=remaining)


def _shutdown_dispatcher_for_tests() -> None:
    global _dispatcher_thread, _dispatcher_ready
    with _dispatcher_lock:
        loop = _dispatcher_loop
        thread = _dispatcher_thread
        _dispatcher_thread = None
        _dispatcher_ready = None
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout=2.0)
    _pending_futures.clear()
