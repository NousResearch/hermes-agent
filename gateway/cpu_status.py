"""CPU pressure classifier for host-level admission guards.

The mirror image of :mod:`gateway.memory_status`: the kanban dispatcher's
memory-pressure guard can see only that the HOST is out of memory, never that
the host is out of CPU while the board's own tasks sit idle behind static
concurrency caps. Two signals, both cheap and both host-wide, worst one wins:

* ``load1_per_core`` — 1-minute load average over ``os.cpu_count()``. Covers
  every process on the machine, including the ones outside the kanban DB.
* ``psi_some_avg60_pct`` — Linux PSI ``some avg60`` from ``/proc/pressure/cpu``.
  Saturates earlier than loadavg and reports the STALL share directly.

``unknown`` is returned only when BOTH signals are unreadable (non-Linux, a
container with no ``/proc/pressure``, no loadavg). It fails open, exactly like
:func:`gateway.memory_status.classify_pressure`: an unreadable signal must
never read as "fine" AND must never brick dispatch either.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

# Thresholds on load per core. Conservative: 1.0 means every core already has a
# runnable/running task queued, 2.0 means the scheduler is oversubscribed 2x.
LOAD_ELEVATED_PER_CORE = 1.0
LOAD_CRITICAL_PER_CORE = 2.0

# Thresholds on Linux PSI ``some avg60`` (percent of the last 60s in which at
# least one task was stalled on CPU). Deliberately conservative: 20% stall
# already means workers are visibly slow, 60% means admission is making it
# worse.
PSI_ELEVATED_PCT = 20.0
PSI_CRITICAL_PCT = 60.0

_PRESSURE_TIERS = (  # order-sensitive: worst first
    ("critical", LOAD_CRITICAL_PER_CORE, PSI_CRITICAL_PCT),
    ("elevated", LOAD_ELEVATED_PER_CORE, PSI_ELEVATED_PCT),
)

_CPU_COUNT_FLOOR = 1


def _nonneg_number(value: Any) -> Optional[float]:
    """Return *value* as a finite non-negative float, or None (bools rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number < 0:
        return None
    return number


def classify_cpu_pressure(
    load1_per_core: Any = None,
    psi_some_avg60_pct: Any = None,
    *,
    elevated: float = LOAD_ELEVATED_PER_CORE,
    critical: float = LOAD_CRITICAL_PER_CORE,
    psi_elevated: float = PSI_ELEVATED_PCT,
    psi_critical: float = PSI_CRITICAL_PCT,
) -> str:
    """``ok``/``elevated``/``critical``/``unknown`` from whichever signals read.

    A tier trips when EITHER signal reaches it, so a calm CPU can never mask a
    loadavg spike and vice versa. Malformed or missing values count as
    unreadable; ``unknown`` only when no signal at all could be read.
    """
    load = _nonneg_number(load1_per_core)
    psi = _nonneg_number(psi_some_avg60_pct)
    if load is None and psi is None:
        return "unknown"
    thresholds = (
        ("critical", critical, psi_critical),
        ("elevated", elevated, psi_elevated),
    )
    for level, load_tier, psi_tier in thresholds:
        if (load is not None and load >= load_tier) or (psi is not None and psi >= psi_tier):
            return level
    return "ok"


def _load1_per_core() -> Optional[float]:
    """1-minute load average divided by the CPU count, or None when unreadable.

    ``ponytail:`` loadavg is a coarse, already-smoothed 1-minute sample, not an
    instantaneous reading, and ``os.cpu_count()`` ignores cgroup CPU quota --
    a container limited to 1 of 8 cores reports 8. Both err toward a HIGHER
    load ratio, so the guard can trip early on a quota-limited container; that
    is the conservative direction for an admission guard.
    """
    try:
        getloadavg = os.getloadavg
    except AttributeError:
        # Windows has no loadavg at all.
        return None
    if getloadavg is None:  # pragma: no cover - defensive against a stubbed os
        return None
    try:
        load1 = getloadavg()[0]
        count = os.cpu_count() or _CPU_COUNT_FLOOR
    except (OSError, TypeError, ValueError, IndexError):
        return None
    load = _nonneg_number(load1)
    return None if load is None else load / max(count, _CPU_COUNT_FLOOR)


PSI_CPU_PATH = "/proc/pressure/cpu"


def _psi_some_avg60_pct(path: Any = PSI_CPU_PATH) -> Optional[float]:
    """Linux PSI ``some avg60`` in percent, or None when the file is absent.

    ``ponytail:`` a single point read of a 60-second decaying average. It says
    nothing about right now (a burst that ended 10s ago still reads high) and
    nothing about which task stalled. It is read because it saturates EARLIER
    than loadavg does, which is what the admission guard needs; the smoothing
    is why the thresholds above are conservative rather than tight.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in raw.splitlines():
        if not line.startswith("some "):
            continue
        for field in line.split()[1:]:
            key, _, value = field.partition("=")
            if key == "avg60":
                try:
                    return _nonneg_number(float(value))
                except ValueError:
                    return None
    return None


def sample_cpu_pressure() -> Dict[str, Any]:
    """Best-effort CPU snapshot; ``{}`` when every signal is unreadable.

    Never raises. Callers treat ``{}`` as "unknown" and fail open.
    """
    sample: Dict[str, Any] = {}
    load = _load1_per_core()
    if load is not None:
        sample["load1_per_core"] = load
    psi = _psi_some_avg60_pct()
    if psi is not None:
        sample["psi_some_avg60_pct"] = psi
    return sample
