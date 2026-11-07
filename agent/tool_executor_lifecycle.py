"""Order tool UI callbacks across worker abandonment."""

from __future__ import annotations

import threading
from typing import Callable


class _ToolLifecycle:
    """Order stable-ID callbacks without making the owner wait for a blocked start.

    A UI start can block before publishing (e.g. reading an edit snapshot). If
    the owner settles meanwhile, that callback must finish before completion
    cleans up its state. Completion also prevents a not-yet-entered late start.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._starting = False
        self._completed = False
        self._pending_complete: Callable[[], None] | None = None

    def start(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._completed:
                return
            self._starting = True
        try:
            callback()
        finally:
            with self._lock:
                self._starting = False
                complete, self._pending_complete = self._pending_complete, None
            if complete is not None:
                complete()

    def complete(self, callback: Callable[[], None]) -> None:
        with self._lock:
            # One start/completion pair per call: the first completion owns cleanup.
            if self._completed:
                return
            self._completed = True
            if self._starting:
                self._pending_complete = callback
                return
        callback()
