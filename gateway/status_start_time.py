"""Start-time fingerprints backing the PID-reuse guard, split from ``gateway.status`` so the
facade can keep shrinking (FILE_LINES ratchet; the moved code keeps its cap).

A ``(pid, start_time)`` pair uniquely identifies a process incarnation: a recycled PID (same
number, different process) yields a different fingerprint and is never mistaken for the
original.  The quantization rule of the fingerprint itself is a public contract — see
:func:`get_process_start_time`."""

from pathlib import Path
from typing import Any, Optional


def _start_times_agree(current: Any, *recorded: Any) -> bool:
    """Same process object: all fingerprints > 0 and within 1ms of ``current``; raises on junk."""
    cur = float(current)
    return cur > 0 and all(r > 0 and abs(r - cur) <= 0.001 for r in map(float, recorded))


# Same-host start-time readings can drift by ~1 s between the claim-time and a later liveness read
# (macOS ``kern.boottime`` adjustment, #117505). Both fingerprint scales are ×100 (Linux /proc ticks,
# psutil centiseconds), so 200 means 2 s on either platform — a recycled PID is essentially never
# that close to the original's start time.
START_TIME_DRIFT_TOLERANCE = 200


def start_time_fingerprints_match(recorded: Any, current: Any, tolerance: int = START_TIME_DRIFT_TOLERANCE) -> bool:
    """Liveness-reconciliation comparator for :func:`get_process_start_time` fingerprints: the
    recorded owner and the current reading are the same incarnation when they agree within
    ``tolerance``. Raises on junk; callers decide what an unreadable (``None``) side means."""
    return abs(int(current) - int(recorded)) <= tolerance


def _get_process_start_time(pid: int) -> Optional[int]:
    """Return a stable per-process start-time fingerprint, or None.

    On Linux this is field 22 of ``/proc/<pid>/stat`` (start time in clock
    ticks since boot, an int).  On platforms without ``/proc`` (macOS, Windows)
    we fall back to ``psutil.Process(pid).create_time()`` — a float epoch
    timestamp — rounded to epoch centiseconds (``round(ct * 100)``,
    round-half-even; **not** truncated) for stable equality.

    The two sources are never mixed on a single platform: ``/proc`` always
    succeeds first on Linux, and always fails on macOS/Windows so psutil is
    always used there.  Because the guard only compares the value recorded at
    spawn against the live value *on the same host*, the differing units across
    platforms are irrelevant — only same-source equality matters.
    """
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        # Field 22 in /proc/<pid>/stat is process start time (clock ticks).
        return int(stat_path.read_text(encoding="utf-8").split()[21])  # windows-footgun: ok (/proc is BOM-free)
    except (FileNotFoundError, IndexError, PermissionError, ValueError, OSError):
        pass

    # No /proc (macOS / Windows): psutil is a hard dependency and exposes a
    # cross-platform creation time.  Round to centiseconds (round-half-even,
    # NOT truncation — int(ct * 100) disagrees ~50% of the time, see #135536) so
    # repeated reads of the same process compare equal without float-precision
    # fragility.
    try:
        import psutil  # type: ignore
        return round(psutil.Process(pid).create_time() * 100)
    except Exception:
        return None
