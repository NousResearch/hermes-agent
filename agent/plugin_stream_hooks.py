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
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

from agent.memory_provider import MemoryObservation, spawn_context_thread
from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION

logger = logging.getLogger(__name__)

# One bounded FIFO is owned by each (hook, callback) consumer. Producers never
# wait for plugin code: when full, enqueue drops the oldest pending event.
_QUEUE_SIZE = 1024
_MAX_OBSERVER_EVENT_BYTES = 128 * 1024
_MAX_OBSERVER_RETAINED_BYTES = 16 * 1024 * 1024
_STOP = object()
_FALLBACK_SCOPE = object()

_discovery_scheduling_lock = threading.Lock()
_scheduled_discovery: set[int] = set()
_observer_retained_lock = threading.Lock()
_observer_retained_bytes = 0


@dataclass
class _RetainedObserverText:
    text: str
    size: int
    references: int

    def release(self) -> None:
        global _observer_retained_bytes
        with _observer_retained_lock:
            if self.references <= 0:
                return
            self.references -= 1
            if self.references == 0:
                _observer_retained_bytes -= self.size
                self.text = ""


@dataclass
class _ConsumerDispatcher:
    scope_key: object
    hook_name: str
    callback: Callable[..., Any]
    events: queue.Queue[_QueuedObserverEvent | object]
    thread: threading.Thread | None = None
    retired: bool = False


@dataclass(frozen=True)
class _QueuedObserverEvent:
    payload: dict[str, Any]
    context: contextvars.Context
    retained_text: _RetainedObserverText | None = None


_dispatcher_lock = threading.Lock()
_dispatchers: dict[tuple[object, str, int], _ConsumerDispatcher] = {}


def _retain_observer_text(text: str, references: int) -> _RetainedObserverText | None:
    global _observer_retained_bytes
    size = sys.getsizeof(text)
    with _observer_retained_lock:
        if (
            references < 1
            or size > _MAX_OBSERVER_RETAINED_BYTES
            or _observer_retained_bytes + size > _MAX_OBSERVER_RETAINED_BYTES
        ):
            return None
        _observer_retained_bytes += size
    return _RetainedObserverText(text, size, references)


def _release_queued_text(item: Any) -> None:
    if isinstance(item, _QueuedObserverEvent) and item.retained_text is not None:
        item.retained_text.release()


def _callback_name(callback: Callable[..., Any]) -> str:
    return getattr(callback, "__name__", repr(callback))


def _put_drop_oldest(events: queue.Queue[Any], item: Any) -> bool:
    """Put without waiting; evict the oldest pending event once when full."""
    try:
        events.put_nowait(item)
        return True
    except queue.Full:
        try:
            evicted = events.get_nowait()
            events.task_done()
            _release_queued_text(evicted)
        except queue.Empty:
            pass
    try:
        events.put_nowait(item)
        return True
    except queue.Full:
        return False


def _worker(dispatcher: _ConsumerDispatcher) -> None:
    while True:
        item = dispatcher.events.get()
        payload: dict[str, Any] | None = None
        try:
            if item is _STOP:
                return
            if not isinstance(item, _QueuedObserverEvent):
                continue
            # Retirement and this gate share a lock. Work that passed the gate is
            # already in flight; queued work not yet admitted is never invoked.
            with _dispatcher_lock:
                if dispatcher.retired:
                    continue
            payload = dict(item.payload)
            if item.retained_text is not None:
                payload["final_text"] = item.retained_text.text
            payload.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
            try:
                from hermes_cli.plugins_dispatch import PluginDispatchMixin

                # Reuse the normal hook signature filtering and coroutine resolution
                # while keeping the callback on this off-turn worker.
                item.context.run(
                    PluginDispatchMixin._invoke_hook_callback,
                    dispatcher.callback,
                    payload,
                )
            # Plugin SystemExit is isolated like other callback failures, rather than
            # terminating this worker and stranding later retained payloads.
            except (Exception, SystemExit) as exc:
                if isinstance(exc, SystemExit):
                    logger.debug(
                        "plugin observer callback exited; continuing dispatcher",
                        exc_info=True,
                    )
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
            if payload is not None:
                payload.clear()
            _release_queued_text(item)
            dispatcher.events.task_done()


def _manager_hook_callbacks(manager: Any, hook_name: str) -> tuple[Callable[..., Any], ...]:
    iterator = getattr(manager, "iter_hook_callbacks", None)
    if callable(iterator):
        callbacks = iterator(hook_name)
    else:
        callbacks = getattr(manager, "_hooks", {}).get(hook_name, ())
    return tuple(callbacks) if isinstance(callbacks, (tuple, list)) else ()


def _schedule_lazy_discovery(manager: Any, discover: Callable[[], Any]) -> None:
    manager_id = id(manager)
    with _discovery_scheduling_lock:
        if manager_id in _scheduled_discovery:
            return
        _scheduled_discovery.add(manager_id)

    def run_discovery() -> None:
        try:
            lock = getattr(manager, "_discovery_lock", None)
            if lock is None:
                if not getattr(manager, "_retired", False) and not getattr(manager, "_discovered", True):
                    discover()
            else:
                with lock:
                    if not getattr(manager, "_retired", False) and not getattr(manager, "_discovered", True):
                        discover()
        except Exception:
            logger.warning("lazy plugin observer discovery failed", exc_info=True)
        finally:
            with _discovery_scheduling_lock:
                _scheduled_discovery.discard(manager_id)

    try:
        thread = spawn_context_thread(
            run_discovery, name="plugin-observer-discovery", daemon=True
        )
        thread.start()
    except Exception:
        with _discovery_scheduling_lock:
            _scheduled_discovery.discard(manager_id)
        logger.debug("could not schedule lazy plugin observer discovery", exc_info=True)


def _registered_callbacks(
    hook_name: str, manager: Any = None, *, allow_discovery: bool = True
) -> tuple[Callable[..., Any], ...]:
    """Snapshot callbacks, discovering lazily unless a capability probe opts out."""
    lock = None
    if manager is None:
        manager = _active_plugin_manager()
        if manager is not None:
            lock = getattr(manager, "_discovery_lock", None)
    if manager is None:
        return ()

    if lock is not None and not lock.acquire(blocking=False):
        return ()
    try:
        if not getattr(manager, "_discovered", True):
            if not allow_discovery:
                return ()
            discover = getattr(manager, "discover_and_load", None)
            if callable(discover):
                _schedule_lazy_discovery(manager, discover)
            return ()
        return _manager_hook_callbacks(manager, hook_name)
    except Exception:
        logger.debug("plugin stream hook callback lookup failed: %s", hook_name, exc_info=True)
        return ()
    finally:
        if lock is not None:
            lock.release()


def _active_plugin_manager():
    """Return the manager for this context, or None when lookup must fail open."""
    try:
        from hermes_cli import plugins

        getter = plugins.get_plugin_manager
        if getattr(getter, "__module__", None) != getattr(plugins, "__name__", None):
            return getter()
        condition = getattr(plugins, "_plugin_manager_teardown_condition", None)
        home_key = getattr(plugins, "_plugin_home_key", None)
        teardown_owners = getattr(plugins, "_plugin_manager_teardown_owners", None)
        if condition is None or not callable(home_key) or teardown_owners is None:
            return getter()
        if not condition.acquire(blocking=False):
            return None
        try:
            if home_key() in teardown_owners:
                return None
            # Don't take host-publication locks while holding the manager registry lock.
            return getter(_attach_hosts=False)
        finally:
            condition.release()
    except Exception:  # health: allow BLE001 -- observer lookup must fail open without blocking the token path
        logger.debug("plugin stream hook manager lookup failed", exc_info=True)
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


def _observer_payload_size(payload: dict[str, Any]) -> int | None:
    """Approximate the retained size of a JSON-shaped observer payload; None for unsupported values."""
    size = 0
    pending = [payload]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, MemoryObservation):
            children = (vars(value),)
        elif isinstance(value, _RetainedObserverText):
            children = ()
        elif isinstance(value, dict):
            children = (child for pair in dict.items(value) for child in pair)
        elif type(value) in (list, tuple, set, frozenset):
            children = iter(value)
        elif value is None or type(value) in (bool, int, float, str, bytes):
            children = ()
        else:
            return None
        size += sys.getsizeof(value)
        if size > _MAX_OBSERVER_EVENT_BYTES:
            return size
        pending.extend(children)
    return size


@contextmanager
def _registered_dispatch_scope(hook_name: str):
    """Snapshot callbacks under a nonblocking manager lock and validate the scope.

    Discovery holds only this captured manager's lock; no global manager lookup can race a
    reservation. The lock remains held through dispatcher creation and enqueue, so unload either
    follows a completed enqueue and retires it, or wins the lock and makes the stale enqueue a no-op.
    """
    manager = _active_plugin_manager()
    if manager is None:
        yield _FALLBACK_SCOPE, (), False
        return
    scope_key = _active_manager_scope(manager)
    try:
        from hermes_cli.plugins import PluginManager

        is_plugin_manager = isinstance(manager, PluginManager)
    except ImportError:
        is_plugin_manager = False
    manager_lock = getattr(manager, "_discovery_lock", None)
    # Never make the observer producer wait behind a discovery/reload transaction. It is safe to
    # drop this observation while the manager is busy; lifecycle teardown will retire the old queue.
    if manager_lock is not None and not manager_lock.acquire(blocking=False):
        yield scope_key, (), False
        return
    try:
        callbacks = _registered_callbacks(hook_name, manager)
        still_current = _active_manager_scope(manager) is scope_key
        if is_plugin_manager:
            current_hooks = getattr(manager, "_hooks", {})
            still_current = still_current and _same_callbacks(
                callbacks, tuple(current_hooks.get(hook_name, ()))
            )
        yield scope_key, callbacks, still_current
    finally:
        if manager_lock is not None:
            manager_lock.release()


def _stop_dispatcher(
    dispatcher: _ConsumerDispatcher, timeout: float = 1.0, *, discard_pending: bool = False
) -> None:
    if discard_pending:
        while True:
            try:
                discarded = dispatcher.events.get_nowait()
                dispatcher.events.task_done()
                _release_queued_text(discarded)
            except queue.Empty:
                break
    _put_drop_oldest(dispatcher.events, _STOP)
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
                dispatcher = _dispatchers.pop(key)
                dispatcher.retired = True
                stale.append(dispatcher)

        for callback in callbacks:
            key = (scope_key, hook_name, id(callback))
            dispatcher = _dispatchers.get(key)
            if dispatcher is None or dispatcher.thread is None or not dispatcher.thread.is_alive():
                events: queue.Queue[_QueuedObserverEvent | object] = queue.Queue(
                    maxsize=_QUEUE_SIZE
                )
                dispatcher = _ConsumerDispatcher(
                    scope_key=scope_key,
                    hook_name=hook_name,
                    callback=callback,
                    events=events,
                )
                dispatcher.thread = spawn_context_thread(
                    _worker,
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


def _enqueue_plugin_observer_hook(
    hook_name: str,
    *,
    payload: dict[str, Any] | None = None,
    payload_factory: Callable[[], dict[str, Any]] | None = None,
) -> bool:
    event_payload: dict[str, Any] = {}
    payload_size: int | None = None
    if payload_factory is None:
        event_payload = dict(payload if payload is not None else {})
        payload_size = _observer_payload_size(event_payload)
        if payload_size is None or (
            payload_size > _MAX_OBSERVER_EVENT_BYTES and hook_name != "on_stream_end"
        ):
            logger.debug("plugin observer event dropped: payload exceeds size bound: %s", hook_name)
            return False
    queued = False
    event_context = contextvars.copy_context()
    with _registered_dispatch_scope(hook_name) as (scope_key, callbacks, still_current):
        if not still_current:
            return False
        if payload_factory is not None:
            if not callbacks:
                return False
            event_payload = dict(payload_factory())
            payload_size = _observer_payload_size(event_payload)
            if payload_size is None or (
                payload_size > _MAX_OBSERVER_EVENT_BYTES and hook_name != "on_stream_end"
            ):
                logger.debug("plugin observer event dropped: payload exceeds size bound: %s", hook_name)
                return False
        if payload_size is None:
            return False
        dispatchers, stale = _dispatchers_for_scope(scope_key, hook_name, callbacks)
        retained_text = None
        if payload_size > _MAX_OBSERVER_EVENT_BYTES:
            final_text = event_payload.get("final_text")
            if hook_name == "on_stream_end" and isinstance(final_text, str):
                retained_text = _retain_observer_text(final_text, len(dispatchers))
                if retained_text is not None:
                    event_payload = dict(event_payload)
                    event_payload["final_text"] = retained_text
                    payload_size = _observer_payload_size(event_payload)
            if retained_text is None or payload_size is None or payload_size > _MAX_OBSERVER_EVENT_BYTES:
                if retained_text is not None:
                    for _ in dispatchers:
                        retained_text.release()
                logger.debug("plugin observer event dropped: payload exceeds size bound: %s", hook_name)
                dispatchers = []
        for dispatcher in dispatchers:
            # A Context cannot be entered concurrently by two workers. Each
            # consumer gets an independent copy of the originating enqueue
            # context, while retaining the same event payload.
            item = _QueuedObserverEvent(
                payload=event_payload,
                context=event_context.copy(),
                retained_text=retained_text,
            )
            if _put_drop_oldest(dispatcher.events, item):
                queued = True
            else:
                _release_queued_text(item)
                logger.debug(
                    "plugin stream hook queue full after drop-oldest: %s callback=%s",
                    hook_name,
                    _callback_name(dispatcher.callback),
                )

    for dispatcher in stale:
        _stop_dispatcher(dispatcher, timeout=0.2)
    return queued


def _enqueue_plugin_observer_hook_with_payload_factory(
    hook_name: str, payload_factory: Callable[[], dict[str, Any]]
) -> bool:
    """Build and validate payload only after nonblocking callback discovery succeeds."""
    return _enqueue_plugin_observer_hook(hook_name, payload_factory=payload_factory)


def enqueue_plugin_observer_hook(hook_name: str, **payload: Any) -> bool:
    """Queue an observer hook without running plugin code on the caller.

    The shared dispatcher keeps one daemon worker and bounded FIFO per
    registered callback. Ordinary payload graphs over
    ``_MAX_OBSERVER_EVENT_BYTES`` are dropped whole; oversized
    ``on_stream_end.final_text`` is retained by reference under a shared memory
    budget and restored in full. ``put_nowait`` never waits for queue capacity;
    a full queue drops its oldest pending event so a slow consumer cannot delay
    the agent. Callback exceptions are isolated in the worker.
    """
    return _enqueue_plugin_observer_hook(hook_name, payload=payload)


def retire_plugin_observer_dispatchers(
    manager: Any,
    *,
    unload_all: bool,
    callbacks_to_retire: set[tuple[str, int]] | None = None,
) -> list[_ConsumerDispatcher]:
    """Retire dispatchers owned by a manager before plugin registrations are released.

    Unload-all rotates the opaque lifetime token so an enqueue that captured the old generation
    cannot attach to a reloaded manager. Targeted unload retires every dispatcher whose callback
    registration is removed, even when another plugin still registers the same callable. Retirement
    is marked before releasing the discovery lock; a worker that already passed its gate may finish,
    but queued work cannot begin after that boundary. Queue discard and bounded joining happen after
    the manager lock is released. Cached sibling profiles remain active.
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
            should_retire = unload_all or (
                (hook_name, callback_id) in callbacks_to_retire
                if callbacks_to_retire is not None
                else callback_id not in live_callback_ids.get(hook_name, set())
            )
            if should_retire:
                dispatcher = _dispatchers.pop(key)
                dispatcher.retired = True
                stale.append(dispatcher)
    return stale


def enqueue_plugin_stream_hook(hook_name: str, **payload: Any) -> bool:
    """Backward-compatible name for the shared observer dispatcher."""
    return enqueue_plugin_observer_hook(hook_name, **payload)


def has_stream_observer_hooks() -> bool:
    return any(
        _registered_callbacks(name, allow_discovery=False)
        for name in ("on_stream_start", "on_stream_delta", "on_stream_end")
    )


def has_reasoning_stream_observer_hooks() -> bool:
    return stream_reasoning_deltas_enabled() and bool(
        _registered_callbacks("on_stream_delta", allow_discovery=False)
    )


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
