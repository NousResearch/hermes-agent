"""Host-owned cooperative thread lifetimes for plugin reloads.

Workers are trusted plugin code, not a sandbox. A stop timeout fences activation;
Python threads cannot be forcibly killed. This module belongs to the host, not a
reloadable plugin namespace, so old closures read the same current-owner slot.
"""
from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from uuid import uuid4

from agent.memory_provider import spawn_context_thread
from hermes_cli.plugins_loader import _plugin_home_scope


@dataclass
class _Slot:
    transition: threading.Lock = field(default_factory=threading.Lock)
    lock: threading.RLock = field(default_factory=threading.RLock)
    owner: PluginWorkerScope | None = None


_slots: dict[tuple[str, str], _Slot] = {}
_slots_lock = threading.Lock()


class PluginWorkerScope:
    """One plugin load's threads and revocable readiness token.

    ``start_thread`` passes a shared stop Event to the target. Complete durable
    initialization before ``activate``. Each send must check ``is_active`` just
    before network admission. Already admitted I/O may finish during shutdown.
    """

    def __init__(self, slot: _Slot, home: Path, timeout: float):
        self._slot = slot
        self._home = home
        self._timeout = timeout
        self._ready = False
        self._closed = False
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.instance_id = uuid4().hex

    @property
    def stop_event(self) -> threading.Event:
        return self._stop

    def is_active(self) -> bool:
        with self._slot.lock:
            return self._slot.owner is self and self._ready and not self._closed and not self._stop.is_set()

    def activate(self) -> bool:
        with self._slot.lock:
            if self._slot.owner is not self or self._closed or self._stop.is_set():
                return False
            self._ready = True
            return True

    def start_thread(self, target: Callable[[threading.Event], None], *, name: str) -> threading.Thread:
        """Start a profile-bound thread only while this load owns the slot."""
        with self._slot.lock:
            if self._slot.owner is not self or self._closed or self._stop.is_set():
                raise RuntimeError("plugin worker scope is closed")
            with _plugin_home_scope(self._home):
                thread = spawn_context_thread(target, name=name, args=(self._stop,))
            thread.start()
            self._threads.append(thread)
            return thread

    def close(self) -> bool:
        """Revoke before waiting; return true only after every managed thread exits."""
        with self._slot.lock:
            self._ready = False
            self._closed = True
            self._stop.set()
            threads = tuple(self._threads)
        deadline = time.monotonic() + self._timeout
        for thread in threads:
            if thread is threading.current_thread():
                return False
            thread.join(max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in threads)


def acquire_worker_scope(home: Path, plugin_id: str, *, timeout: float = 5.0) -> PluginWorkerScope:
    """Fence and stop a preceding load, including one whose unload never ran."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout):
        raise ValueError("worker shutdown timeout must be finite")
    if not 0 < timeout <= 30:
        raise ValueError("worker shutdown timeout must be in (0, 30]")
    key = (str(home), plugin_id)
    with _slots_lock:
        slot = _slots.setdefault(key, _Slot())
    with slot.transition:
        with slot.lock:
            old = slot.owner
        if old is not None and not old.close():
            raise RuntimeError("previous plugin workers have not stopped")
        scope = PluginWorkerScope(slot, home, float(timeout))
        with slot.lock:
            slot.owner = scope
        return scope


def context_worker_scope(ctx, home: Path, *, timeout: float = 5.0) -> PluginWorkerScope:
    """Keep one scope per context and attach it to the ordinary unload ledger."""
    with ctx._worker_scope_lock:
        scope = getattr(ctx, "_worker_scope", None)
        if scope is None:
            scope = acquire_worker_scope(home, ctx.plugin_id, timeout=timeout)
            ctx._worker_scope = scope
            ctx.on_unload(scope.close)
        return scope


def spawn_task(ctx, coro, *, name: str | None = None) -> asyncio.Task:
    """Preserve the existing asyncio cancellation-only API; do not change its contract."""
    if not asyncio.iscoroutine(coro):
        raise TypeError("spawn_task expects a coroutine")
    task_name = name or f"plugin:{ctx.plugin_id}:task"
    task = asyncio.get_running_loop().create_task(coro, name=task_name)
    handle = ctx._track("background_task", task_name, lambda: task.done() or task.cancel())
    task.add_done_callback(lambda _t: handle.dispose())
    return task
