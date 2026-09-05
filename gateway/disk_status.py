"""Disk-usage rollup for ``/api/status`` (NS-656).

Companion to :mod:`gateway.memory_status`, closing the same class of gap
for storage: a hosted agent can fill its data volume completely — SQLite
writes failing, session persistence dead, config saves lost — while its
dashboard and the NAS agent card both look perfectly healthy.  Fleet
incidents OOF-2 (unrecoverable disk-full) and OOF-107 (fleet-wide disk
exhaustion, remediated by hand) are exactly this failure mode.

The readiness endpoint already probes disk (``gateway/readiness.py::
_probe_disk``), but readiness is a component verdict, not user-facing
telemetry — nothing renders it.  This module produces the public block
the dashboard SPA and the NAS availability sweep actually consume.

Unlike the memory block (which distills already-persisted heartbeat
files), disk is sampled live via :func:`shutil.disk_usage` — a single
``statvfs`` call, the same thing the readiness probe does per request.
There is no meaningful "staleness" dimension, so no ``sampled_at``.

Public-safety note: ``/api/status`` is an unauthenticated liveness probe
(``PUBLIC_API_PATHS``).  This block carries only coarse numbers (MB
granularity, whole-percent usage) and an enum — the same disclosure
class as the ``memory`` block.

Everything is best-effort and read-only: an unreadable filesystem
degrades to ``pressure="unknown"`` rather than raising into the status
endpoint.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Disk-pressure thresholds. Percent alone misleads in both directions:
# 90% used on a 100 GB volume leaves a comfortable 10 GB, while 50% used
# on a tiny volume can be one image download from write failures. So the
# percent triggers are gated on absolute headroom also being low, and a
# hard absolute floor applies regardless of size — below it, SQLite
# journaling and config writes are at genuine risk on any volume.
_CRITICAL_FREE_MB = 256  # < 256 MB free: critical on any volume
_CRITICAL_PERCENT = 95.0  # >= 95% used AND < 1 GB free: critical
_CRITICAL_HEADROOM_MB = 1024
_ELEVATED_FREE_MB = 512  # < 512 MB free: elevated on any volume
_ELEVATED_PERCENT = 85.0  # >= 85% used AND < 4 GB free: elevated
_ELEVATED_HEADROOM_MB = 4096

_BYTES_PER_MB = 1024 * 1024


def _coerce_mb(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def classify_disk_pressure(free_mb: Any, total_mb: Any) -> str:
    """Map free/total MB to ``ok``/``elevated``/``critical``.

    ``unknown`` when the sample is missing or malformed — the caller must
    not treat "we could not read it" as "disk is fine".
    """
    free = _coerce_mb(free_mb)
    total = _coerce_mb(total_mb)
    if free is None or total is None or total <= 0:
        return "unknown"
    used_percent = (1 - free / total) * 100.0
    if free < _CRITICAL_FREE_MB or (
        used_percent >= _CRITICAL_PERCENT and free < _CRITICAL_HEADROOM_MB
    ):
        return "critical"
    if free < _ELEVATED_FREE_MB or (
        used_percent >= _ELEVATED_PERCENT and free < _ELEVATED_HEADROOM_MB
    ):
        return "elevated"
    return "ok"


def collect_disk_status(home: Optional[Path] = None) -> Dict[str, Any]:
    """Build the ``disk`` block for ``/api/status``.

    ``home`` scopes the sample to a profile's HERMES_HOME (the status
    endpoint's ``?profile=`` handling passes it through); on hosted
    images every profile shares the ``/opt/data`` volume, so the answer
    is the same — but scoping keeps the contract identical to the
    ``memory`` block's.

    Always returns a dict — an unreadable/unmounted filesystem yields
    ``{"pressure": "unknown", ...}``.  Never raises.
    """
    status: Dict[str, Any] = {
        "pressure": "unknown",
        "total_mb": None,
        "free_mb": None,
        "used_percent": None,
    }
    try:
        if home is None:
            from hermes_constants import get_hermes_home

            home = get_hermes_home()
        usage = shutil.disk_usage(home)
    except Exception:
        return status
    if usage.total <= 0:
        return status
    total_mb = usage.total // _BYTES_PER_MB
    free_mb = usage.free // _BYTES_PER_MB
    status["total_mb"] = total_mb
    status["free_mb"] = free_mb
    status["used_percent"] = round((usage.used / usage.total) * 100, 1)
    status["pressure"] = classify_disk_pressure(free_mb, total_mb)
    return status
