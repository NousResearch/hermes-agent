"""Asynchronous per-consumer plugin observers for streaming and memory events.

Each registered hook callback gets its own bounded queue + daemon worker thread
so plugin code never runs inline on the token path. Queues drop the oldest
pending event when full; dispatchers for callbacks that are no longer
registered are stopped lazily on the next lookup or at manager unload.
"""

from __future__ import annotations

import contextvars
import logging
import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION

logger = logging.getLogger(__name__)

# One bounded FIFO is owned by each (hook, callback) consumer. Producers never
# wait for plugin code: when full, enqueue drops the oldest pending event.
_QUEUE_SIZE = 1024
_STOP = object()
_FALLBACK_SCOPE = object()


@dataclass
class _ConsumerDispatcher:
    scope_key: object
    hook_name: str
    callback: Callable[..., Any]
    events: "queue.Queue[_QueuedObserverEvent | object]"
    thread: threading.Thread | None = None


@dataclass(frozen=True)
class _QueuedObserverEvent:
    payload: dict[str, Any]
    context: contextvars.Context


_dispatcher_lock = threading.Lock()
_dispatchers: dict[tuple[object, str, int], _ConsumerDispatcher] = {}


def _callback_name(callback: Callable[..., Any]) -> str:
    return getattr(callback, "__name__", repr(callback))


def _worker(dispatcher: _ConsumerDispatcher) -> None:
    while True:
        item = dispatcher.events.get()
        try:
            if item is _STOP:
                return
            if not isinstance(item, _QueuedObserverEvent):
                continue
            payload = dict(item.payload)
            payload.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
            try:
                item.context.run(dispatcher.callback, **payload)
            except Exception as exc:
                # Fires once per streaming delta: a mis-declared callback fails identically every
                # time, so it goes through the manager's warn-once reporter (#111922).
                from hermes_cli.plugins import get_plugin_manager

                # Context.run restores the worker's context when the callback raises.
                # Resolve the reporter inside the event's owning profile as well.
                manager = item.context.run(get_plugin_manager)
                item.context.run(
                    manager._report_hook_failure,
                    dispatcher.hook_name, dispatcher.callback, payload, exc,
                )
        finally:
            dispatcher.events.task_done()


def _registered_callbacks(hook_name: str) -> tuple[Callable[..., Any], ...]:
    try:
        from hermes_cli import plugins

        callbacks = plugins.iter_hook_callbacks(hook_name)
        if callbacks:
            return callbacks
        # ``iter_hook_callbacks`` is also used by test doubles and older
        # embedders that expose only the snapshot method. The public
        # ``has_hook`` gate is the established lazy-discovery contract; use it
        # when an undiscovered manager returned an empty snapshot, then retry
        # the snapshot after discovery.
        if not plugins.has_hook(hook_name):
            return ()
        return plugins.iter_hook_callbacks(hook_name)
    except Exception:
        logger.debug("plugin stream hook callback lookup failed: %s", hook_name, exc_info=True)
        return ()


def _active_plugin_manager():
    """Return the manager for this context, or None when plugin lookup fails."""
    try:
        from hermes_cli import plugins

        return plugins.get_plugin_manager()
    except Exception:
        return None


def _active_manager_scope(manager: Any) -> object:
    """Return the manager's unique observer lifetime token, not its recyclable ``id()``."""
    if manager is None:
        return _FALLBACK_SCOPE
    # PluginManager rotates this opaque token at unload-all. The manager object
    # fallback keeps older embedders/test doubles scoped without using id().
    return getattr(manager, "_observer_dispatcher_scope", manager)


def _same_callbacks(left: tuple[Callable[..., Any], ...], right: tuple[Callable[..., Any], ...]) -> bool:
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


@contextmanager
def _registered_dispatch_scope(hook_name: str):
    """Snapshot callbacks before taking the manager lock, then validate while holding it.

    This preserves the existing lazy callback lookup without reversing the manager-registry lock
    order. The lock remains held through dispatcher creation and enqueue, so unload either follows
    a completed enqueue and retires it, or changes the lifetime token first and makes the stale
    enqueue a no-op.
    """
    manager = _active_plugin_manager()
    scope_key = _active_manager_scope(manager)
    manager_callbacks = None
    manager_hooks = getattr(manager, "_hooks", None)
    if isinstance(manager_hooks, dict):
        manager_callbacks = tuple(manager_hooks.get(hook_name, ()))
    callbacks = _registered_callbacks(hook_name)
    callbacks_from_manager = manager_callbacks is not None and _same_callbacks(
        callbacks, manager_callbacks
    )
    manager_lock = getattr(manager, "_discovery_lock", None)
    if manager_lock is None:
        yield scope_key, callbacks, True
        return

    # Never make the observer producer wait behind a discovery/reload transaction. It is safe to
    # drop this observation while the manager is busy; lifecycle teardown will retire the old queue.
    if not manager_lock.acquire(blocking=False):
        yield scope_key, callbacks, False
        return
    try:
        still_current = _active_manager_scope(manager) is scope_key
        if callbacks_from_manager:
            current_hooks = getattr(manager, "_hooks", {})
            still_current = still_current and _same_callbacks(
                callbacks, tuple(current_hooks.get(hook_name, ()))
            )
        yield scope_key, callbacks, still_current
    finally:
        manager_lock.release()


def _stop_dispatcher(
    dispatcher: _ConsumerDispatcher, timeout: float = 1.0, *, discard_pending: bool = False
) -> None:
    if discard_pending:
        while True:
            try:
                dispatcher.events.get_nowait()
                dispatcher.events.task_done()
            except queue.Empty:
                break
    try:
        dispatcher.events.put_nowait(_STOP)
    except queue.Full:
        try:
            dispatcher.events.get_nowait()
            dispatcher.events.task_done()
        except queue.Empty:
            pass
        try:
            dispatcher.events.put_nowait(_STOP)
        except queue.Full:
            pass
    if dispatcher.thread is not None and dispatcher.thread is not threading.current_thread():
        dispatcher.thread.join(timeout=timeout)


def _dispatchers_for_scope(
    scope_key: object, hook_name: str, callbacks: tuple[Callable[..., Any], ...]
) -> tuple[list[_ConsumerDispatcher], list[_ConsumerDispatcher]]:
    callback_ids = {id(callback) for callback in callbacks}
    stale: list[_ConsumerDispatcher] = []
    ready: list[_ConsumerDispatcher] = []
    with _dispatcher_lock:
        for key, dispatcher in list(_dispatchers.items()):
            key_scope, key_hook_name, callback_id = key
            if (
                key_scope is scope_key
                and key_hook_name == hook_name
                and callback_id not in callback_ids
            ):
                stale.append(_dispatchers.pop(key))

        for callback in callbacks:
            key = (scope_key, hook_name, id(callback))
            dispatcher = _dispatchers.get(key)
            if dispatcher is None or dispatcher.thread is None or not dispatcher.thread.is_alive():
                events: "queue.Queue[_QueuedObserverEvent | object]" = queue.Queue(
                    maxsize=_QUEUE_SIZE
                )
                dispatcher = _ConsumerDispatcher(
                    scope_key=scope_key,
                    hook_name=hook_name,
                    callback=callback,
                    events=events,
                )
                dispatcher.thread = threading.Thread(
                    target=_worker,
                    args=(dispatcher,),
                    daemon=True,
                    name=f"plugin-stream-hook:{hook_name}",
                )
                dispatcher.thread.start()
                _dispatchers[key] = dispatcher
            ready.append(dispatcher)

    return ready, stale


def _dispatchers_for(hook_name: str) -> list[_ConsumerDispatcher]:
    with _registered_dispatch_scope(hook_name) as (scope_key, callbacks, still_current):
        if not still_current:
            return []
        ready, stale = _dispatchers_for_scope(scope_key, hook_name, callbacks)

    for dispatcher in stale:
        _stop_dispatcher(dispatcher, timeout=0.2)
    return ready


def enqueue_plugin_observer_hook(hook_name: str, **payload: Any) -> bool:
    """Queue an observer hook without running plugin code on the caller.

    The shared dispatcher keeps one daemon worker and bounded FIFO per
    registered callback. ``put_nowait`` makes the producer non-blocking; a
    full queue drops its oldest pending event so a slow consumer cannot grow
    memory or delay the agent. Callback exceptions are isolated in the worker.
    """
    queued = False
    event_payload = dict(payload)
    event_context = contextvars.copy_context()
    with _registered_dispatch_scope(hook_name) as (scope_key, callbacks, still_current):
        if not still_current:
            return False
        dispatchers, stale = _dispatchers_for_scope(scope_key, hook_name, callbacks)
        for dispatcher in dispatchers:
            # A Context cannot be entered concurrently by two workers. Each
            # consumer gets an independent copy of the originating enqueue
            # context, while retaining the same event payload.
            item = _QueuedObserverEvent(
                payload=event_payload,
                context=event_context.copy(),
            )
            try:
                dispatcher.events.put_nowait(item)
                queued = True
                continue
            except queue.Full:
                try:
                    dispatcher.events.get_nowait()
                    dispatcher.events.task_done()
                except queue.Empty:
                    pass
            try:
                dispatcher.events.put_nowait(item)
                queued = True
            except queue.Full:
                logger.debug(
                    "plugin stream hook queue full after drop-oldest: %s callback=%s",
                    hook_name,
                    _callback_name(dispatcher.callback),
                )

    for dispatcher in stale:
        _stop_dispatcher(dispatcher, timeout=0.2)
    return queued


def retire_plugin_observer_dispatchers(
    manager: Any, *, unload_all: bool
) -> list[_ConsumerDispatcher]:
    """Retire dispatchers owned by a manager unload while its discovery lock is held.

    Unload-all rotates the opaque lifetime token so an enqueue that captured the old generation
    cannot attach to a reloaded manager. Targeted unload only retires callbacks no longer present.
    This touches only the selected manager's scope; cached sibling profiles remain active.
    """
    scope_key = _active_manager_scope(manager)
    manager_hooks = getattr(manager, "_hooks", {})
    live_callback_ids = {
        hook_name: {id(callback) for callback in callbacks}
        for hook_name, callbacks in manager_hooks.items()
    }
    if unload_all:
        manager._observer_dispatcher_scope = object()

    stale: list[_ConsumerDispatcher] = []
    with _dispatcher_lock:
        for key, dispatcher in list(_dispatchers.items()):
            key_scope, hook_name, callback_id = key
            if key_scope is not scope_key:
                continue
            if unload_all or callback_id not in live_callback_ids.get(hook_name, set()):
                stale.append(_dispatchers.pop(key))
    return stale


def enqueue_plugin_stream_hook(hook_name: str, **payload: Any) -> bool:
    """Backward-compatible name for the shared observer dispatcher."""
    return enqueue_plugin_observer_hook(hook_name, **payload)


def has_stream_observer_hooks() -> bool:
    return any(_registered_callbacks(name) for name in ("on_stream_start", "on_stream_delta", "on_stream_end"))


def has_reasoning_stream_observer_hooks() -> bool:
    return stream_reasoning_deltas_enabled() and bool(_registered_callbacks("on_stream_delta"))


def stream_reasoning_deltas_enabled() -> bool:
    """Return True only when the user opted plugins into reasoning deltas.

    Read-only scalar lookup: skips ``load_config()``'s deepcopy. Callers on the token path
    should still cache the result per stream (``_fire_reasoning_delta`` does)."""
    try:
        from hermes_cli import config as config_mod
        config = config_mod.load_config_readonly()
        return bool(config_mod.cfg_get(config, "plugins", "stream_reasoning_deltas", default=False))
    except Exception:
        logger.debug("failed to read plugins.stream_reasoning_deltas", exc_info=True)
        return False


def shutdown_plugin_observer_dispatcher(timeout: float = 1.0) -> None:
    """Stop observer workers with a bounded drain, used by tests/shutdown.

    A stop sentinel is placed after pending events when possible, allowing a
    short FIFO drain. The worker join is bounded by ``timeout``; a blocked
    callback remains on its daemon worker and is never allowed to hold process
    teardown indefinitely.
    """
    global _dispatchers
    with _dispatcher_lock:
        dispatchers = list(_dispatchers.values())
        _dispatchers = {}
    for dispatcher in dispatchers:
        _stop_dispatcher(dispatcher, timeout=timeout)


def shutdown_plugin_stream_hook_dispatcher(timeout: float = 1.0) -> None:
    """Backward-compatible name for the shared observer dispatcher shutdown."""
    shutdown_plugin_observer_dispatcher(timeout=timeout)
