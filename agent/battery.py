"""System-battery read-out for the CLI/TUI status bar.

Reads the host battery through ``psutil`` (already a Hermes dependency) and
exposes a compact, colour-coded label.  Everything degrades to "unavailable"
when there is no battery (desktops, servers, VMs) or when the read fails, so
callers can render the result unconditionally and simply show nothing.

The status bar repaints often (every keystroke and on a ~1s idle refresh), so
:func:`read_battery` memoises the last reading for a few seconds instead of
hitting ``psutil`` on every frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BatteryStatus:
    """A single battery reading.

    ``available`` is False on machines without a battery (or when the read
    failed).  ``percent`` is clamped to 0-100.  ``plugged`` is True when on AC
    power, False on battery, and None when the platform can't tell.
    """

    available: bool
    percent: Optional[int] = None
    plugged: Optional[bool] = None

    @property
    def charging(self) -> bool:
        return bool(self.plugged)


UNAVAILABLE = BatteryStatus(available=False)

# Colour buckets, mirroring the status-bar context styles but inverted (a full
# battery is "good", an empty one is "critical").
CATEGORY_GOOD = "good"
CATEGORY_WARN = "warn"
CATEGORY_BAD = "bad"
CATEGORY_CRITICAL = "critical"
CATEGORY_DIM = "dim"

_CACHE_TTL_SECONDS = 8.0
_cache: Optional[tuple[float, BatteryStatus]] = None


def _read_battery_uncached() -> BatteryStatus:
    try:
        import psutil
    except Exception:
        return UNAVAILABLE

    # ``sensors_battery`` is missing on some platforms/builds of psutil.
    reader = getattr(psutil, "sensors_battery", None)
    if reader is None:
        return UNAVAILABLE

    try:
        batt = reader()
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
    if plugged is not None:
        plugged = bool(plugged)

    return BatteryStatus(available=True, percent=percent, plugged=plugged)


def read_battery(use_cache: bool = True) -> BatteryStatus:
    """Return the current battery status (cached for a few seconds)."""
    global _cache
    if use_cache and _cache is not None:
        ts, cached = _cache
        if time.monotonic() - ts < _CACHE_TTL_SECONDS:
            return cached

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
    # On AC power the level isn't a concern — always read as healthy.
    if status.charging:
        return CATEGORY_GOOD
    pct = status.percent
    if pct <= 10:
        return CATEGORY_CRITICAL
    if pct <= 20:
        return CATEGORY_BAD
    if pct <= 50:
        return CATEGORY_WARN
    return CATEGORY_GOOD


def battery_glyph(status: BatteryStatus) -> str:
    """Return the leading glyph: a bolt while charging, else a battery."""
    return "\u26a1" if status.charging else "\U0001f50b"  # ⚡ / 🔋


def format_battery(status: BatteryStatus) -> str:
    """Return a compact label like ``🔋 82%`` / ``⚡ 82%`` (empty if N/A)."""
    if not status.available or status.percent is None:
        return ""
    return f"{battery_glyph(status)} {status.percent}%"
