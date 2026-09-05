"""SIGKILL any process left in this systemd unit's cgroup.

Runs as ``ExecStopPost=`` so it only fires after the gateway's main process
has exited. The gateway already reaps its own tool subprocesses on a clean
shutdown; this is the safety net for long-lived helpers it doesn't track
(``adb``, platform bridges, etc.) that would otherwise be orphaned in the
cgroup and block ``Restart=always`` — issue #37454.

We deliberately iterate ``cgroup.procs`` and send per-PID SIGKILLs instead
of writing ``1`` to ``cgroup.kill``: the original failure mode in #37454
was the kernel returning ``EINVAL`` on the cgroup-wide kill, while per-PID
signal delivery uses a separate code path that still works.
"""

from __future__ import annotations

import os
import re
import signal
import sys
from pathlib import Path


def _own_cgroup_path() -> str | None:
    """Return the cgroup v2 path for the calling process, or None."""
    try:
        text = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^0::(.+)$", text, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def _read_cgroup_pids(cgroup_path: str) -> list[int]:
    procs_file = Path(f"/sys/fs/cgroup{cgroup_path}/cgroup.procs")
    try:
        raw = procs_file.read_text(encoding="utf-8")
    except OSError:
        return []
    pids: list[int] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def reap_cgroup(cgroup_path: str | None = None) -> int:
    """SIGKILL every PID in the cgroup other than the caller. Returns the count killed."""
    if cgroup_path is None:
        cgroup_path = _own_cgroup_path()
    if not cgroup_path:
        return 0
    own = os.getpid()
    killed = 0
    for pid in _read_cgroup_pids(cgroup_path):
        if pid == own:
            continue
        try:
            os.kill(pid, signal.SIGKILL)  # windows-footgun: ok — Linux-only (reads /proc, /sys/fs/cgroup; runs from a systemd unit)
            killed += 1
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    return killed


def main() -> int:
    reap_cgroup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
