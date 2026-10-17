"""Process start-time fingerprints read through psutil, for platforms without ``/proc``.

``gateway.status.get_process_start_time`` reads Linux ``/proc/<pid>/stat`` first; macOS and
Windows land here.
"""

from __future__ import annotations

from typing import Any, Optional


def psutil_start_time(pid: int) -> Optional[int]:
    """``pid``'s creation time in centiseconds (stable equality without float fragility), or None."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    try:
        return round(psutil_create_time(psutil.Process(pid)) * 100)
    except (psutil.Error, OSError, ValueError, TypeError, OverflowError):
        # Gone, denied, a zombie, or a malformed PID: no fingerprint, the caller decides.
        return None


def psutil_create_time(proc: Any) -> float:
    """``proc``'s creation time as the kernel reports it, the same value in every process.

    On macOS (and the BSDs) psutil's public ``create_time()`` adds ``INIT_BOOT_TIME - boot_time()``,
    where ``INIT_BOOT_TIME`` is captured when the *reading* process first imported psutil. macOS
    steps ``kern.boottime`` by seconds over time, so a gateway that has run for days reads every
    other process's start time shifted by the accumulated drift while a fresh CLI reads it
    unshifted: their fingerprints for one live process stop agreeing, and the cron reaper marks a
    running ``hermes cron run`` execution ``unknown`` (#138216). Fingerprints are compared across
    processes of different ages, so read the unadjusted kernel value (``monotonic=True``). A psutil
    backend whose ``create_time`` takes no such argument (Windows, SunOS, AIX) applies no
    reader-dependent correction, so its public reading is used as before.
    """
    try:
        return float(proc._proc.create_time(monotonic=True))
    except (AttributeError, TypeError):
        return float(proc.create_time())
