"""Host CPU pressure classification for the kanban dispatch admission guard.

Mirrors :mod:`gateway.memory_status`'s ``classify_pressure`` pattern (t_4008d306
/ Astra 2026-09-08 control-plane review P0): the dispatcher's memory guard
already stops new worker spawns under critical *memory* pressure, but nothing
equivalent existed for CPU. A saturated host (load average >> core count, or
Linux PSI ``some`` CPU stall high) starves every board's workers the same way
OOM does, so the same "critical -> spawn nothing, elevated -> at most one,
unknown -> no restriction" contract applies here.

Two independent signals are read, worst-of wins:

- ``os.getloadavg()[0]`` (1-minute load) normalized by CPU count. Available on
  every POSIX host, but blind to *why* the CPU is busy (a single pinned core
  vs genuine host-wide contention) and smooths over short spikes.
- Linux PSI ``/proc/pressure/cpu`` ``some avg60`` — the percentage of the last
  60s some task was stalled waiting for CPU. This is the signal the 2026-09-08
  probe used (avg300=76.64) and directly captures contention from cron jobs,
  CI runs and local inference processes that never touch the kanban DB and so
  are invisible to the dispatcher's own ``count_running_tasks*`` bookkeeping.
  Unavailable in containers/non-Linux/cgroup v1-without-PSI hosts.

Neither signal is authoritative alone: PSI is more precise but not always
present; load average is universal but coarser. Reading both and taking the
worse classification means a host with PSI available gets the sharper signal,
and one without it still gets a coverage-limited but real admission decision
rather than "unknown" whenever a single core is transiently pinned.

Known coverage gap (honestly stated, not silently assumed complete): this
module only classifies pressure *already on the host* — it does not itself
enumerate or attribute load to cron, CI or inference processes. It cannot
distinguish "the host is busy with legitimate reserved-capacity work" from
"the host is busy with runaway dispatcher fan-out"; it only says "busy", which
is exactly the level the memory guard operates at too.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

# Thresholds on normalized load (load1 / cpu_count). A value of 1.0 means
# "as many runnable processes as cores" — the classic saturation line.
_CRITICAL_NORMALIZED_LOAD = 2.0
_ELEVATED_NORMALIZED_LOAD = 1.0

# Thresholds on Linux PSI "some avg60" (percent of the last 60s with at least
# one runnable task stalled on CPU). 300s-window figures from a point-in-time
# probe (e.g. avg300=76.64) are worse-is-later signals of sustained pressure;
# avg60 is used here for a tighter dispatch-tick reaction time.
_CRITICAL_PSI_SOME_AVG60 = 60.0
_ELEVATED_PSI_SOME_AVG60 = 20.0

_PROC_PRESSURE_CPU_PATH = "/proc/pressure/cpu"

_LEVEL_RANK = {"unknown": -1, "ok": 0, "elevated": 1, "critical": 2}


def _nonneg_number(value: Any) -> Optional[float]:
    """Return *value* as a float if it is a non-negative, non-bool number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return float(value)
    return None


def _worse(a: str, b: str) -> str:
    """The more severe of two pressure levels; ``unknown`` loses to any known level."""
    ra, rb = _LEVEL_RANK.get(a, -1), _LEVEL_RANK.get(b, -1)
    if ra < 0:
        return b
    if rb < 0:
        return a
    return a if ra >= rb else b


def classify_load_pressure(load1: Any, cpu_count: Any) -> str:
    """``ok``/``elevated``/``critical`` from 1-minute load average normalized
    by CPU count; ``unknown`` when either input is missing/malformed."""
    load = _nonneg_number(load1)
    cores = _nonneg_number(cpu_count)
    if load is None or cores is None or cores <= 0:
        return "unknown"
    normalized = load / cores
    if normalized >= _CRITICAL_NORMALIZED_LOAD:
        return "critical"
    if normalized >= _ELEVATED_NORMALIZED_LOAD:
        return "elevated"
    return "ok"


def classify_psi_pressure(psi_some_avg60: Any) -> str:
    """``ok``/``elevated``/``critical`` from Linux PSI CPU ``some avg60``
    (percent); ``unknown`` when the sample is missing/malformed."""
    psi = _nonneg_number(psi_some_avg60)
    if psi is None:
        return "unknown"
    if psi >= _CRITICAL_PSI_SOME_AVG60:
        return "critical"
    if psi >= _ELEVATED_PSI_SOME_AVG60:
        return "elevated"
    return "ok"


def classify_cpu_pressure(
    load1: Any = None,
    cpu_count: Any = None,
    psi_some_avg60: Any = None,
) -> str:
    """Worst-of load-average and PSI classification; ``unknown`` only when
    BOTH signals are unavailable — "could not read either" must never read as
    "fine", matching :func:`gateway.memory_status.classify_pressure`."""
    return _worse(
        classify_load_pressure(load1, cpu_count),
        classify_psi_pressure(psi_some_avg60),
    )


def _read_psi_some_avg60(path: str = _PROC_PRESSURE_CPU_PATH) -> Optional[float]:
    """Parse ``some avg60=<value>`` from ``/proc/pressure/cpu``; ``None`` when
    the file is absent (non-Linux, containers without PSI, cgroup v1)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("some "):
                    continue
                for field in line.split():
                    if field.startswith("avg60="):
                        return float(field.split("=", 1)[1])
    except (OSError, ValueError):
        return None
    return None


def sample_cpu() -> Dict[str, Any]:
    """Best-effort host CPU snapshot: ``{}`` when nothing could be read.

    Never raises — a missing ``/proc/pressure/cpu`` (non-Linux, containers) or
    a ``getloadavg`` failure degrades to partial or empty results, letting
    :func:`classify_cpu_pressure` fall back to ``unknown``/worse-of rather
    than the caller crashing.
    """
    sample: Dict[str, Any] = {}
    try:
        load1, _load5, _load15 = os.getloadavg()
        sample["load1"] = load1
    except (OSError, AttributeError):
        pass
    try:
        cpu_count = os.cpu_count()
        if cpu_count:
            sample["cpu_count"] = cpu_count
    except Exception:
        pass
    psi = _read_psi_some_avg60()
    if psi is not None:
        sample["psi_some_avg60"] = psi
    return sample
