"""Asynchronous per-consumer plugin observers for streaming and memory events."""

from __future__ import annotations

import contextvars
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION

logger = logging.getLogger(__name__)

# One bounded FIFO is owned by each (hook, callback) consumer. Producers never
# wait for plugin code: when full, enqueue drops the oldest pending event.
_QUEUE_SIZE = 1024
_STOP = object()


@dataclass
class _ConsumerDispatcher:
    scope_key: int
    hook_name: str
    callback: Callable[..., Any]
    events: "queue.Queue[_QueuedObserverEvent | object]"
    thread: threading.Thread | None = None


@dataclass(frozen=True)
class _QueuedObserverEvent:
    payload: dict[str, Any]
    context: contextvars.Context


_dispatcher_lock = threading.Lock()
_dispatchers: dict[tuple[int, str, int], _ConsumerDispatcher] = {}


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
                logger.warning(
                    "Hook '%s' callback %s raised: %s",
                    dispatcher.hook_name,
                    _callback_name(dispatcher.callback),
                    exc,
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


def _active_manager_scope() -> int:
    """Identify the active profile/plugin manager for dispatcher state."""
    try:
        from hermes_cli import plugins

        return id(plugins.get_plugin_manager())
    except Exception:
        # Discovery failures are fail-open for observers. A single fallback
        # scope still keeps cleanup deterministic for callers without a plugin
        # manager, while normal profile-aware paths use the manager identity.
        return 0


def _stop_dispatcher(dispatcher: _ConsumerDispatcher, timeout: float = 1.0) -> None:
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
    if dispatcher.thread is not None:
        dispatcher.thread.join(timeout=timeout)


def _dispatchers_for(hook_name: str) -> list[_ConsumerDispatcher]:
    scope_key = _active_manager_scope()
    callbacks = _registered_callbacks(hook_name)
    callback_ids = {id(callback) for callback in callbacks}
    stale: list[_ConsumerDispatcher] = []
    ready: list[_ConsumerDispatcher] = []
    with _dispatcher_lock:
        for key, dispatcher in list(_dispatchers.items()):
            key_scope, key_hook_name, callback_id = key
            if (
                key_scope == scope_key
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
    for dispatcher in _dispatchers_for(hook_name):
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
    return queued


def enqueue_plugin_stream_hook(hook_name: str, **payload: Any) -> bool:
    """Backward-compatible name for the shared observer dispatcher."""
    return enqueue_plugin_observer_hook(hook_name, **payload)


def has_stream_observer_hooks() -> bool:
    return any(_registered_callbacks(name) for name in ("on_stream_start", "on_stream_delta", "on_stream_end"))


def has_reasoning_stream_observer_hooks() -> bool:
    return stream_reasoning_deltas_enabled() and bool(_registered_callbacks("on_stream_delta"))


def stream_reasoning_deltas_enabled() -> bool:
    """Return True only when the user opted plugins into reasoning deltas."""
    try:
        from hermes_cli import config as config_mod

        config = config_mod.load_config()
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
