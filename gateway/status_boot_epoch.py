"""Records left behind by a previous boot: why they can name a live PID that is not a gateway.

PIDs are per-boot, and Windows recycles the dead gateway's number onto whatever comes next —
usually a protected process (a security agent, a session-0 service). Every probe then disagrees
with every other: ``OpenProcess`` fails with ACCESS_DENIED, which ``win32_pid_exists`` reports as
"alive", while psutil can read neither the start time nor the command line. ``gateway.pid``,
``gateway.lock`` and the machine-global ``telegram-bot-token-*`` lock therefore read as held by a
live gateway this process merely cannot inspect: "another gateway instance is already running"
(exit 1) on the pid record and "telegram bot token already in use" (exit 78, parked with no retry)
on the token lock — on every start, until the file is deleted by hand.

A record's own timestamps settle it: a holder whose start fingerprint or record stamp predates
this machine's boot is not alive. The boot epoch is only ever used to declare a record STALE, and
only for a holder no probe can identify, so a wall-clock step can never reclaim a lock that is
actually held.
"""

from __future__ import annotations

import contextlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# A clock step must not read as a new boot: boot time is wall clock minus uptime.
_EPOCH_SLACK_S = 300

_UNSET = object()
_boot_epoch_cache: Any = _UNSET


def current_boot_epoch() -> Optional[int]:
    """Epoch seconds this machine booted at, or None when it cannot be read."""
    global _boot_epoch_cache
    if _boot_epoch_cache is not _UNSET:
        return _boot_epoch_cache
    value: Optional[int] = None
    with contextlib.suppress(Exception):
        import psutil  # type: ignore

        value = int(psutil.boot_time()) or None
    if value is None:
        # No psutil (macOS/Linux without it): /proc, then the Windows uptime counter.
        with contextlib.suppress(OSError, ValueError, IndexError):
            for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
                if line.startswith("btime "):
                    value = int(line.split()[1]) or None
                    break
    if value is None:
        with contextlib.suppress(Exception):
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            kernel32.GetTickCount64.restype = ctypes.c_ulonglong
            value = int(time.time() - kernel32.GetTickCount64() / 1000.0)
    _boot_epoch_cache = value
    return value


def pid_identity_unreadable(pid: Optional[int]) -> bool:
    """True when a PID that answers the liveness probe yields NO identity at all.

    That is the recycled number: a live same-user gateway always exposes a start time and a
    command line, so only a foreign or protected owner leaves both empty.
    """
    if pid is None:
        return False
    from gateway import status

    return (status._get_process_start_time(pid) is None
            and not status._read_process_cmdline(pid))


def _start_epoch_seconds(record: Optional[dict[str, Any]]) -> Optional[float]:
    """The record's start fingerprint as epoch seconds, when it IS a wall-clock fingerprint.

    Windows stamps ``create_time * 100`` (a centisecond epoch). A Linux ``/proc`` tick count is
    boot-relative, so it is rejected rather than read as a date in 1970 — platforms without the
    recycled-PID problem do not need this guard anyway, and their records fall through to
    ``updated_at``.
    """
    from gateway.status import _EPOCH_MIN_PLAUSIBLE

    raw = (record or {}).get("start_time")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    if raw <= _EPOCH_MIN_PLAUSIBLE * 100:
        return None
    return raw / 100.0


def _updated_at_seconds(record: Optional[dict[str, Any]]) -> Optional[float]:
    raw = (record or {}).get("updated_at")
    if not isinstance(raw, str):
        return None
    from gateway.status import normalize_updated_at

    # Persisted timestamps without an offset are UTC, not the caller's timezone.
    normalized = normalize_updated_at(raw)
    if normalized is None:
        return None
    return datetime.fromisoformat(normalized).timestamp()


def record_predates_boot(record: Optional[dict[str, Any]]) -> bool:
    """True when the record's own start fingerprint or write stamp predates this boot."""
    boot = current_boot_epoch()
    if boot is None:
        return False
    cutoff = boot - _EPOCH_SLACK_S
    started = _start_epoch_seconds(record)
    if started is not None:
        return started < cutoff
    stamped = _updated_at_seconds(record)
    return stamped is not None and stamped < cutoff


def live_pid_from_record(record: Optional[dict[str, Any]]) -> Optional[int]:
    """The record's PID when it is alive and provably the process the record describes.

    ``status._live_pid_from_record`` is the facade name every pid/lock reader imports; the guards
    that need the boot epoch live here so ``status.py`` stays within its line budget.
    """
    from gateway import status

    record = record or {}
    pid = status._pid_from_record(record)
    if pid is None or not status._pid_exists(pid):
        return None
    if status._start_times_conflict(record.get("start_time"), status._get_process_start_time(pid)):
        return None
    if pid_identity_unreadable(pid) and record_predates_boot(record):
        return None
    return pid
