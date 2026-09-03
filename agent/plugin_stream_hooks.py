"""Asynchronous per-consumer plugin observers for streaming LLM output."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION

logger = logging.getLogger(__name__)

_QUEUE_SIZE = 1024
_STOP = object()


@dataclass
class _ConsumerDispatcher:
    hook_name: str
    callback: Callable[..., Any]
    events: "queue.Queue[dict[str, Any] | object]"
    thread: threading.Thread | None = None


_dispatcher_lock = threading.Lock()
_dispatchers: dict[tuple[str, int], _ConsumerDispatcher] = {}


def _callback_name(callback: Callable[..., Any]) -> str:
    return getattr(callback, "__name__", repr(callback))


def _worker(dispatcher: _ConsumerDispatcher) -> None:
    while True:
        item = dispatcher.events.get()
        try:
            if item is _STOP:
                return
            payload = dict(item)
            payload.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
            try:
                dispatcher.callback(**payload)
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

        return plugins.iter_hook_callbacks(hook_name)
    except Exception:
        logger.debug("plugin stream hook callback lookup failed: %s", hook_name, exc_info=True)
        return ()


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
    callbacks = _registered_callbacks(hook_name)
    if not callbacks:
        return []

    callback_ids = {id(callback) for callback in callbacks}
    stale: list[_ConsumerDispatcher] = []
    ready: list[_ConsumerDispatcher] = []
    with _dispatcher_lock:
        for key, dispatcher in list(_dispatchers.items()):
            key_hook_name, callback_id = key
            if key_hook_name == hook_name and callback_id not in callback_ids:
                stale.append(_dispatchers.pop(key))

        for callback in callbacks:
            key = (hook_name, id(callback))
            dispatcher = _dispatchers.get(key)
            if dispatcher is None or dispatcher.thread is None or not dispatcher.thread.is_alive():
                events: "queue.Queue[dict[str, Any] | object]" = queue.Queue(maxsize=_QUEUE_SIZE)
                dispatcher = _ConsumerDispatcher(
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


def enqueue_plugin_stream_hook(hook_name: str, **payload: Any) -> bool:
    """Queue an observer hook for each consumer without running plugin code inline."""
    queued = False
    item = dict(payload)
    for dispatcher in _dispatchers_for(hook_name):
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


def shutdown_plugin_stream_hook_dispatcher(timeout: float = 1.0) -> None:
    """Stop background stream hook dispatchers; used by tests and clean shutdown paths."""
    global _dispatchers
    with _dispatcher_lock:
        dispatchers = list(_dispatchers.values())
        _dispatchers = {}
    for dispatcher in dispatchers:
        _stop_dispatcher(dispatcher, timeout=timeout)
