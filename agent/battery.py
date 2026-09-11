"""System-battery read-out for the CLI/TUI status bar.

Reads the host battery through ``psutil`` and exposes a compact, colour-coded label. Everything
degrades to "unavailable" (no battery / read failure) so callers can render unconditionally. The
status bar repaints on every keystroke, so :func:`read_battery` memoises the reading for a few seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BatteryStatus:
    """One reading: ``percent`` clamped 0-100; ``plugged`` None when the platform can't tell."""

    available: bool
    percent: Optional[int] = None
    plugged: Optional[bool] = None

    @property
    def charging(self) -> bool:
        return bool(self.plugged)


UNAVAILABLE = BatteryStatus(available=False)

# Colour buckets, mirroring the status-bar context styles but inverted (full battery = "good").
CATEGORY_GOOD = "good"
CATEGORY_WARN = "warn"
CATEGORY_BAD = "bad"
CATEGORY_CRITICAL = "critical"
CATEGORY_DIM = "dim"

# (upper bound inclusive, category) for a discharging battery; first match wins.
_LEVEL_CATEGORIES = ((10, CATEGORY_CRITICAL), (20, CATEGORY_BAD), (50, CATEGORY_WARN))

_CACHE_TTL_SECONDS = 8.0
_cache: Optional[tuple[float, BatteryStatus]] = None


def _read_battery_uncached() -> BatteryStatus:
    try:
        import psutil

        # ``sensors_battery`` is missing on some platforms/builds of psutil.
        batt = getattr(psutil, "sensors_battery")()
    except Exception:
        return UNAVAILABLE
    if batt is None:
        return UNAVAILABLE
    percent: Optional[int] = None
    raw_percent = getattr(batt, "percent", None)
    if raw_percent is not None:
        try:
            percent = max(0, min(100, int(round(float(raw_percent)))))
        except (TypeError, ValueError):
            percent = None
    plugged = getattr(batt, "power_plugged", None)
    return BatteryStatus(available=True, percent=percent, plugged=None if plugged is None else bool(plugged))


def read_battery(use_cache: bool = True) -> BatteryStatus:
    """Return the current battery status (cached for a few seconds)."""
    global _cache
    if use_cache and _cache is not None and time.monotonic() - _cache[0] < _CACHE_TTL_SECONDS:
        return _cache[1]
    status = _read_battery_uncached()
    _cache = (time.monotonic(), status)
    return status


def clear_cache() -> None:
    """Drop the memoised reading (used by tests)."""
    global _cache
    _cache = None


def battery_category(status: BatteryStatus) -> str:
    """Bucket a reading into a colour category: good/warn/bad/critical/dim."""
    if not status.available or status.percent is None:
        return CATEGORY_DIM
    if status.charging:  # on AC power the level isn't a concern
        return CATEGORY_GOOD
    for bound, category in _LEVEL_CATEGORIES:
        if status.percent <= bound:
            return category
    return CATEGORY_GOOD


def battery_glyph(status: BatteryStatus) -> str:
    """Leading glyph: a bolt while charging, else a battery."""
    return "\u26a1" if status.charging else "\U0001f50b"  # ⚡ / 🔋


def format_battery(status: BatteryStatus) -> str:
    """Compact label like ``🔋 82%`` / ``⚡ 82%`` (empty if N/A)."""
    if not status.available or status.percent is None:
        return ""
    return f"{battery_glyph(status)} {status.percent}%"
