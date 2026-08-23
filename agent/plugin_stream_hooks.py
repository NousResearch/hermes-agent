"""Asynchronous per-consumer plugin observers for streaming and memory events.

Each registered hook callback gets its own bounded queue + daemon worker thread
so plugin code never runs inline on the token path. Queues drop the oldest
pending event when full; dispatchers for callbacks that are no longer
registered are stopped lazily on the next lookup.
"""

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
    hook_name: str
    callback: Callable[..., Any]
    events: "queue.Queue[tuple[contextvars.Context, dict[str, Any]] | object]"
    thread: threading.Thread | None = None


_dispatcher_lock = threading.Lock()
_dispatchers: dict[tuple[str, int], _ConsumerDispatcher] = {}


def _callback_name(callback: Callable[..., Any]) -> str:
    return getattr(callback, "__name__", repr(callback))


def _put_drop_oldest(events: "queue.Queue[Any]", item: Any) -> bool:
    """put_nowait; on a full queue evict the oldest pending event and retry once."""
    try:
        events.put_nowait(item)
        return True
    except queue.Full:
        try:
            events.get_nowait()
            events.task_done()
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
        try:
            if item is _STOP:
                return
            context, payload = item
            payload = dict(payload)
            payload.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
            try:
                # The worker outlives every turn and serves every profile; run the callback in the
                # enqueuing turn's contextvars so it sees that turn's profile scope (home override,
                # secrets), not an unbound context that fail-closed plugin bindings refuse (#118538).
                # ``copy()``: one snapshot fans out to N consumer threads and a Context can only be
                # entered by one thread at a time.
                context.copy().run(dispatcher.callback, **payload)
            except Exception as exc:
                # Fires once per streaming delta: a mis-declared callback fails identically every
                # time, so it goes through the manager's warn-once reporter (#111922).
                from hermes_cli.plugins import get_plugin_manager

                get_plugin_manager()._report_hook_failure(dispatcher.hook_name, dispatcher.callback, payload, exc)
        finally:
            dispatcher.events.task_done()


def _registered_callbacks(hook_name: str) -> tuple[Callable[..., Any], ...]:
    try:
        from hermes_cli import plugins
        return plugins.iter_hook_callbacks(hook_name)
    except Exception:
        logger.debug("plugin stream hook callback lookup failed: %s", hook_name, exc_info=True)
        return ()


def _stop_dispatcher(dispatcher: _ConsumerDispatcher, timeout: float = 1.0) -> None:
    _put_drop_oldest(dispatcher.events, _STOP)
    if dispatcher.thread is not None:
        dispatcher.thread.join(timeout=timeout)


def _start_dispatcher(hook_name: str, callback: Callable[..., Any]) -> _ConsumerDispatcher:
    dispatcher = _ConsumerDispatcher(hook_name=hook_name, callback=callback, events=queue.Queue(maxsize=_QUEUE_SIZE))
    dispatcher.thread = threading.Thread(
        target=_worker, args=(dispatcher,), daemon=True, name=f"plugin-stream-hook:{hook_name}"
    )
    dispatcher.thread.start()
    return dispatcher


def _dispatchers_for(hook_name: str) -> list[_ConsumerDispatcher]:
    """Live dispatcher per registered callback (restarting dead workers); stale
    ones for unregistered callbacks are stopped outside the lock."""
    callbacks = _registered_callbacks(hook_name)
    if not callbacks:
        return []

    callback_ids = {id(callback) for callback in callbacks}
    ready: list[_ConsumerDispatcher] = []
    with _dispatcher_lock:
        stale = [_dispatchers.pop(key) for key in list(_dispatchers) if key[0] == hook_name and key[1] not in callback_ids]
        for callback in callbacks:
            key = (hook_name, id(callback))
            dispatcher = _dispatchers.get(key)
            if dispatcher is None or dispatcher.thread is None or not dispatcher.thread.is_alive():
                dispatcher = _dispatchers[key] = _start_dispatcher(hook_name, callback)
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
    item = (contextvars.copy_context(), dict(payload))
    for dispatcher in _dispatchers_for(hook_name):
        if _put_drop_oldest(dispatcher.events, item):
            queued = True
        else:
            logger.debug(
                "plugin stream hook queue full after drop-oldest: %s callback=%s",
                hook_name, _callback_name(dispatcher.callback),
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
