"""This process's POSIX advisory locks on a file, read from ``/proc/self/fdinfo`` (Linux).

Not ``/proc/locks``: that table is host-wide and the kernel serves it over several ``read()``
calls, so lock churn in other processes (a parallel test run) shifts it mid-read and silently
skips entries — a lock this process still holds reads as cancelled. ``fdinfo`` lists this
process's locks per open file and is read in one call.
"""
import os
from pathlib import Path


def own_posix_locks(*paths) -> list[str]:
    """Sorted ``"POSIX  ADVISORY  <type> <pid> <dev>:<ino> <start> <end>"`` lines this process holds on *paths*."""
    targets = set()
    for path in paths:
        try:
            st = os.stat(path)
        except OSError:
            continue
        targets.add((st.st_dev, st.st_ino))
    found = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            st = os.stat(f"/proc/self/fd/{fd}")
            info = Path(f"/proc/self/fdinfo/{fd}").read_text(encoding="utf-8")
        except OSError:
            continue
        if (st.st_dev, st.st_ino) in targets:
            found.update(line.split(": ", 1)[1] for line in info.splitlines()
                         if line.startswith("lock:") and "POSIX  ADVISORY" in line)
    return sorted(found)
