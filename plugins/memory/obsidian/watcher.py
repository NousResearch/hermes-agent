"""Filesystem watcher for live vault index updates.

Uses ``watchdog`` if installed; falls back to a no-op stub so the plugin
works without the optional dependency. The watcher calls back into
VaultIndex when notes are created, modified, or deleted.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from plugins.memory.obsidian.index import VaultIndex


class _NoOpWatcher:
    """Stub used when watchdog is not installed."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _try_import_watchdog():
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
        return Observer, FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
    except ImportError:
        return None


def make_watcher(vault_path: Path, index: "VaultIndex") -> _NoOpWatcher:
    """Build a watchdog observer for `vault_path` that updates `index` on changes.

    Returns a no-op stub if watchdog is not installed.
    """
    result = _try_import_watchdog()
    if result is None:
        logger.debug("watchdog not installed — vault watcher disabled")
        return _NoOpWatcher()

    Observer, FileSystemEventHandler, *_ = result

    class _Handler(FileSystemEventHandler):
        def __init__(self, idx: "VaultIndex") -> None:
            self._idx = idx
            self._debounce: dict[str, threading.Timer] = {}
            self._lock = threading.Lock()

        def _schedule(self, path: str, fn: Callable) -> None:
            with self._lock:
                if path in self._debounce:
                    self._debounce[path].cancel()
                t = threading.Timer(0.5, fn)
                self._debounce[path] = t
                t.start()

        def on_modified(self, event):
            if event.is_directory or not event.src_path.endswith(".md"):
                return
            self._schedule(event.src_path, lambda: self._idx.update_note(Path(event.src_path)))

        def on_created(self, event):
            if event.is_directory or not event.src_path.endswith(".md"):
                return
            self._schedule(event.src_path, lambda: self._idx.update_note(Path(event.src_path)))

        def on_deleted(self, event):
            if event.is_directory or not event.src_path.endswith(".md"):
                return
            self._idx.remove_note(Path(event.src_path))

        def on_moved(self, event):
            if not event.dest_path.endswith(".md"):
                return
            self._idx.remove_note(Path(event.src_path))
            self._schedule(event.dest_path, lambda: self._idx.update_note(Path(event.dest_path)))

    observer = Observer()
    handler = _Handler(index)
    observer.schedule(handler, str(vault_path), recursive=True)
    return observer  # type: ignore[return-value]
