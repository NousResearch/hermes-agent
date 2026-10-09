"""Disk-usage rollup for ``/api/status``.

Companion to :mod:`gateway.memory_status`: a hosted agent can fill its data
volume (SQLite writes failing, config saves lost) while its dashboard looks
healthy.  Sampled live via one ``statvfs`` call, so there is no ``sampled_at``.
``/api/status`` is unauthenticated: only coarse numbers (MB, one-decimal
percent) and an enum.  Best-effort: an unreadable filesystem degrades to
``pressure="unknown"`` rather than raising into the status endpoint.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.memory_status import _nonneg_int

# Percent alone misleads both ways: 90% used on 100 GB leaves 10 GB, while 50% on a tiny
# volume is one download from write failures.  So percent triggers are gated on absolute
# headroom also being low, and a hard floor applies regardless of size (below it SQLite
# journaling / config writes are at risk).  Rows: (level, free floor, used-%, headroom); worst first.
_PRESSURE_TIERS = (
    ("critical", 256, 95.0, 1024),  # < 256 MB free, or >= 95% used AND < 1 GB free
    ("elevated", 512, 85.0, 4096),  # < 512 MB free, or >= 85% used AND < 4 GB free
)
_BYTES_PER_MB = 1024 * 1024


def classify_disk_pressure(free_mb: Any, total_mb: Any) -> str:
    """``ok``/``elevated``/``critical`` from free/total MB; ``unknown`` when the sample
    is missing/malformed — "could not read it" must never read as "fine"."""
    free, total = _nonneg_int(free_mb), _nonneg_int(total_mb)
    if free is None or not total:
        return "unknown"
    used_percent = (1 - free / total) * 100.0
    for level, free_floor, percent_floor, headroom in _PRESSURE_TIERS:
        if free < free_floor or (used_percent >= percent_floor and free < headroom):
            return level
    return "ok"


def collect_disk_status(home: Optional[Path] = None) -> Dict[str, Any]:
    """``disk`` block for ``/api/status`` (same ``home`` contract as ``memory``).
    Never raises — an unreadable/unmounted filesystem yields ``pressure="unknown"``."""
    status: Dict[str, Any] = {"pressure": "unknown", "total_mb": None, "free_mb": None, "used_percent": None}
    try:
        if home is None:
            from hermes_constants import get_hermes_home

            home = get_hermes_home()
        usage = shutil.disk_usage(home)
    except Exception:
        return status
    if usage.total <= 0:
        return status
    total_mb, free_mb = usage.total // _BYTES_PER_MB, usage.free // _BYTES_PER_MB
    status.update(total_mb=total_mb, free_mb=free_mb, used_percent=round((usage.used / usage.total) * 100, 1),
                  pressure=classify_disk_pressure(free_mb, total_mb))
    return status
