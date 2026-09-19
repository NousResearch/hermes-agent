"""Subscription usage read-out for the status bar (``limits`` in ``display.status_bar.fields``).

Polls ``agent.account_usage`` for the Codex and Claude Code subscriptions on a daemon thread. Per
provider it keeps the shortest window the plan reports (the 5-hour session window; weekly only on
plans without one) plus any per-model weekly cap (``fable 57%``), since a model can be out for the
week while the session window is free. The render path only reads the cached list, so a slow or
failed fetch never stalls a repaint.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# provider id -> short bar label. The order is the bar order.
PROVIDERS: tuple[tuple[str, str], ...] = (("openai-codex", "cx"), ("anthropic", "cc"))

DEFAULT_INTERVAL_S = 60.0


def _clamp(value) -> float:
    return max(0.0, min(100.0, float(value)))


def bar_readings(snapshot, label: str) -> list[tuple[str, float]]:
    """``[(label, used_percent), ...]`` for one provider, or ``[]`` when nothing is reported.

    First the shortest window the plan reports: the one labelled "Session" / "Current session"
    (``agent.account_usage`` names both providers' 5-hour windows that way), else the first
    account-wide weekly window (plans that only meter weekly). Then one reading per per-model
    weekly cap (labels ending in " week" other than the account-wide one), keyed by the model name
    in lower case - a model can be out for the week while the session window is free.
    """
    if snapshot is None:
        return []
    windows = [w for w in getattr(snapshot, "windows", ()) if w.used_percent is not None]
    session = next((w for w in windows if "session" in w.label.lower()), None)
    account_week = next((w for w in windows if w.label.lower() in ("weekly", "current week")), None)
    readings: list[tuple[str, float]] = []
    shortest = session or account_week
    if shortest is not None:
        readings.append((label, _clamp(shortest.used_percent)))
    for w in windows:
        name = w.label.lower()
        if name.endswith(" week") and w is not account_week and name != "current week":
            readings.append((name[: -len(" week")], _clamp(w.used_percent)))
    return readings


def format_limit(label: str, used_percent: float) -> str:
    """``cx 100%`` / ``cc  27%`` - the percent is right-aligned to three cells so the bar never shifts."""
    return f"{label} {round(used_percent):>3d}%"


class AccountLimitsPoller:
    """Background refresh of per-provider bar readings. ``read()`` is lock-free and cheap."""

    def __init__(self, fetch: Optional[Callable] = None, interval_s: float = DEFAULT_INTERVAL_S,
                 on_change: Optional[Callable[[], None]] = None):
        self._fetch = fetch
        self._interval_s = max(5.0, float(interval_s))
        self._on_change = on_change
        self._limits: dict[str, tuple[tuple[str, float], ...]] = {}
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
        return [reading for provider, _label in PROVIDERS for reading in limits.get(provider, ())]

    def refresh_once(self) -> None:
        fetch = self._fetch
        if fetch is None:
            from agent.account_usage import fetch_account_usage
            fetch = fetch_account_usage
        fresh = dict(self._limits)
        for provider, label in PROVIDERS:
            try:
                readings = bar_readings(fetch(provider), label)
            except Exception:
                logger.debug("status bar limits: %s fetch failed", provider, exc_info=True)
                continue  # keep the last good reading rather than blanking the segment
            if readings:
                fresh[provider] = tuple(readings)
            else:
                fresh.pop(provider, None)
        changed = fresh != self._limits
        self._limits = fresh  # single reference swap: readers never see a half-built dict
        if changed and self._on_change is not None:
            self._on_change()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.refresh_once()
            self._stop.wait(self._interval_s)
