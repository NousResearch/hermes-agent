"""A clock that keeps counting while the computer sleeps, for Group Chat host leases.

A lease is a promise about elapsed time: "I won't follow anyone else for the next 20 seconds".
Python's ``time.monotonic()`` stops while a Mac (and some PCs) sleeps, so a lease measured with
it would seem to last longer than it really did. ``now()`` keeps counting through sleep:

- Linux: ``CLOCK_BOOTTIME``;
- macOS: ``CLOCK_MONOTONIC``, which, unlike ``time.monotonic()``, advances during sleep;
- Windows: ``QueryInterruptTime``, which includes sleep and hibernation;
- anything else: the wall clock, whose jumps ``SuspendDetector`` treats as sleep.

Each of these starts near zero when the computer boots, so a value saved before a reboot is
never compared as if no reboot happened: ``boot_id()`` names the current boot as the operating
system reports it (Linux ``/proc/sys/kernel/random/boot_id``, macOS ``kern.bootsessionuuid``,
Windows ``BootId``), never derived from the wall clock, and callers bound what they saved by the
time since this boot. ``SuspendDetector`` compares ``now()`` with a clock that stops during sleep:
when ``now()`` ran further ahead than the tolerance, the computer slept in between.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# How much further now() may run than the awake clock before the gap counts as sleep.
SUSPEND_TOLERANCE_SECONDS = 2.0


def _windows_clocks():
    try:
        import ctypes
        kernel = ctypes.windll.kernelbase  # type: ignore[attr-defined]
        with_sleep, awake = kernel.QueryInterruptTime, kernel.QueryUnbiasedInterruptTime
    except (AttributeError, OSError, ImportError):
        return None

    def reader(query):
        value = ctypes.c_ulonglong()

        def read() -> float:
            query(ctypes.byref(value))
            return value.value / 10_000_000
        return read
    return reader(with_sleep), reader(awake)


def _pick():
    """``(clock counting sleep, clock stopping during sleep, whether the first one is exact)``."""
    if sys.platform.startswith("linux") and hasattr(time, "CLOCK_BOOTTIME"):
        return (lambda: time.clock_gettime(time.CLOCK_BOOTTIME)), time.monotonic, True
    if sys.platform == "darwin" and hasattr(time, "CLOCK_MONOTONIC"):
        return (lambda: time.clock_gettime(time.CLOCK_MONOTONIC)), time.monotonic, True
    if os.name == "nt":
        clocks = _windows_clocks()
        if clocks is not None:
            return clocks[0], clocks[1], True
    return time.time, time.monotonic, False


_NOW, _AWAKE, EXACT = _pick()


def now() -> float:
    """Seconds on a clock that keeps counting while the computer sleeps."""
    return float(_NOW())


def awake() -> float:
    """Seconds on a clock that stops while the computer sleeps."""
    return float(_AWAKE())


def _sysctl_text(name: bytes) -> str:
    import ctypes
    import ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    size = ctypes.c_size_t(0)
    if libc.sysctlbyname(name, None, ctypes.byref(size), None, 0) != 0 or not size.value:
        return ""
    buffer = ctypes.create_string_buffer(size.value)
    if libc.sysctlbyname(name, buffer, ctypes.byref(size), None, 0) != 0:
        return ""
    return buffer.value.decode("ascii", "replace").strip()


def _windows_boot_id() -> str:
    import winreg  # type: ignore[import-not-found]
    key = r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
        value, _ = winreg.QueryValueEx(handle, "BootId")
    return f"boot-{int(value)}"


def _read_boot_id() -> str:
    try:
        if sys.platform.startswith("linux"):
            return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        if sys.platform == "darwin":
            return _sysctl_text(b"kern.bootsessionuuid")
        if os.name == "nt":
            return _windows_boot_id()
    except (OSError, ValueError, TypeError, AttributeError, ImportError):
        pass
    return ""


_BOOT: str | None = None


def boot_id() -> str:
    """This boot's identity as the operating system reports it; ``""`` when it reports none."""
    global _BOOT
    if _BOOT is None:
        _BOOT = _read_boot_id()
    return _BOOT


class SuspendDetector:
    """Reports whether the computer slept since the last ``check()``."""

    def __init__(self):
        self._last = (now(), awake())

    def check(self) -> bool:
        current = (now(), awake())
        (previous_now, previous_awake), self._last = self._last, current
        return (current[0] - previous_now) - (current[1] - previous_awake) > SUSPEND_TOLERANCE_SECONDS
