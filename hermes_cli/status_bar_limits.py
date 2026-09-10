"""Subscription usage read-out for the status bar (``limits`` in ``display.status_bar.fields``).

Polls ``agent.account_usage`` for the Codex and Claude Code subscriptions on a daemon thread and
keeps the 5-hour session window per provider - the one that moves while you work; the weekly
windows stay in ``/usage``. The render path only reads the cached snapshot, so a slow or failed
fetch never stalls a repaint.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# provider id -> short bar label. The order is the bar order.
PROVIDERS: tuple[tuple[str, str], ...] = (("openai-codex", "cx"), ("anthropic", "cc"))

DEFAULT_INTERVAL_S = 60.0


def session_window_percent(snapshot) -> Optional[float]:
    """``used_percent`` of the 5-hour window; None when the snapshot has none.

    Both providers label that window "Session" / "Current session" (``agent.account_usage``),
    while the weekly and per-model windows never carry the word.
    """
    if snapshot is None:
        return None
    for window in getattr(snapshot, "windows", ()):
        if "session" in window.label.lower() and window.used_percent is not None:
            return max(0.0, min(100.0, float(window.used_percent)))
    return None


def format_limit(label: str, used_percent: float) -> str:
    """``cx 100%`` / ``cc  27%`` - the percent is right-aligned to three cells so the bar never shifts."""
    return f"{label} {round(used_percent):>3d}%"


class AccountLimitsPoller:
    """Background refresh of per-provider session-window usage. ``read()`` is lock-free and cheap."""

    def __init__(self, fetch: Optional[Callable] = None, interval_s: float = DEFAULT_INTERVAL_S,
                 on_change: Optional[Callable[[], None]] = None):
        self._fetch = fetch
        self._interval_s = max(5.0, float(interval_s))
        self._on_change = on_change
        self._limits: dict[str, float] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._loop, name="status-bar-limits", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def read(self) -> list[tuple[str, float]]:
        """``[(label, used_percent), ...]`` in bar order; providers without data are omitted."""
        limits = self._limits
        return [(label, limits[provider]) for provider, label in PROVIDERS if provider in limits]

    def refresh_once(self) -> None:
        fetch = self._fetch
        if fetch is None:
            from agent.account_usage import fetch_account_usage
            fetch = fetch_account_usage
        fresh = dict(self._limits)
        for provider, _label in PROVIDERS:
            try:
                used = session_window_percent(fetch(provider))
            except Exception:
                logger.debug("status bar limits: %s fetch failed", provider, exc_info=True)
                continue  # keep the last good reading rather than blanking the segment
            if used is None:
                fresh.pop(provider, None)
            else:
                fresh[provider] = used
        changed = fresh != self._limits
        self._limits = fresh  # single reference swap: readers never see a half-built dict
        if changed and self._on_change is not None:
            self._on_change()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.refresh_once()
            self._stop.wait(self._interval_s)
