"""Asynchronous plugin observer delivery for streaming LLM output."""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE_SIZE = 1024
_STOP = object()

_dispatcher_lock = threading.Lock()
_dispatcher_queue: "queue.Queue[tuple[str, dict[str, Any]] | object] | None" = None
_dispatcher_thread: threading.Thread | None = None


def _queue_size() -> int:
    raw = os.getenv("HERMES_PLUGIN_STREAM_HOOK_QUEUE_SIZE", "").strip()
    if not raw:
        return _DEFAULT_QUEUE_SIZE
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid HERMES_PLUGIN_STREAM_HOOK_QUEUE_SIZE=%r; using %s", raw, _DEFAULT_QUEUE_SIZE)
        return _DEFAULT_QUEUE_SIZE


def _worker(events: "queue.Queue[tuple[str, dict[str, Any]] | object]") -> None:
    while True:
        item = events.get()
        try:
            if item is _STOP:
                return
            hook_name, payload = item
            try:
                from hermes_cli import plugins

                plugins.invoke_hook(hook_name, **payload)
            except Exception:
                logger.debug("plugin stream hook delivery failed: %s", hook_name, exc_info=True)
        finally:
            events.task_done()


def _ensure_dispatcher() -> "queue.Queue[tuple[str, dict[str, Any]] | object]":
    global _dispatcher_queue, _dispatcher_thread
    with _dispatcher_lock:
        if _dispatcher_thread is not None and _dispatcher_thread.is_alive() and _dispatcher_queue is not None:
            return _dispatcher_queue
        _dispatcher_queue = queue.Queue(maxsize=_queue_size())
        _dispatcher_thread = threading.Thread(
            target=_worker,
            args=(_dispatcher_queue,),
            daemon=True,
            name="plugin-stream-hooks",
        )
        _dispatcher_thread.start()
        return _dispatcher_queue


def _has_hook(hook_name: str) -> bool:
    try:
        from hermes_cli import plugins

        return plugins.has_hook(hook_name)
    except Exception:
        logger.debug("plugin stream hook availability check failed: %s", hook_name, exc_info=True)
        return False


def enqueue_plugin_stream_hook(hook_name: str, **payload: Any) -> bool:
    """Queue an observer hook without running plugin code on the token path."""
    if not _has_hook(hook_name):
        return False

    events = _ensure_dispatcher()
    item = (hook_name, dict(payload))
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
            logger.debug("plugin stream hook queue full after drop-oldest: %s", hook_name)
            return False


def has_stream_observer_hooks() -> bool:
    return any(
        _has_hook(name)
        for name in ("on_stream_start", "on_stream_delta", "on_stream_end", "on_interim_message")
    )


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
    """Stop the background dispatcher; used by tests and clean shutdown paths."""
    global _dispatcher_queue, _dispatcher_thread
    with _dispatcher_lock:
        events = _dispatcher_queue
        thread = _dispatcher_thread
        _dispatcher_queue = None
        _dispatcher_thread = None
    if events is None or thread is None:
        return
    try:
        events.put_nowait(_STOP)
    except queue.Full:
        try:
            events.get_nowait()
            events.task_done()
        except queue.Empty:
            pass
        try:
            events.put_nowait(_STOP)
        except queue.Full:
            pass
    thread.join(timeout=timeout)
