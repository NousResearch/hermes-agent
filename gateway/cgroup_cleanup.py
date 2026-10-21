"""SIGKILL any process left in this systemd unit's cgroup.

Runs as ``ExecStopPost=`` after the gateway's main process has exited: the
safety net for long-lived helpers the gateway doesn't track (``adb``, platform
bridges) that would otherwise be orphaned in the cgroup and block
``Restart=always``.  Per-PID SIGKILLs over ``cgroup.procs`` are used instead of
writing ``1`` to ``cgroup.kill``: the kernel has returned ``EINVAL`` on the
cgroup-wide kill while per-PID signal delivery still works.
"""

from __future__ import annotations

import contextlib
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
    return match.group(1).strip() if match else None


def _read_cgroup_pids(cgroup_path: str) -> list[int]:
    try:
        raw = Path(f"/sys/fs/cgroup{cgroup_path}/cgroup.procs").read_text(encoding="utf-8")
    except OSError:
        return []
    pids: list[int] = []
    for line in raw.splitlines():
        with contextlib.suppress(ValueError):
            pids.append(int(line.strip()))
    return pids


def _parent_is_systemd() -> bool:
    """True when this process was spawned by a systemd manager (ExecStopPost etc.).

    Any other parent (an agent terminal tool, a shell) shares a *live*
    process's cgroup, so reaping there would SIGKILL that process. No PID-1
    shortcut: in a container the gateway itself can be PID 1 (or init can be
    ``tini``), so PID 1 must present as systemd too; an unreadable
    ``/proc/<ppid>/comm`` fails closed.
    """
    ppid = os.getppid()
    try:
        return Path(f"/proc/{ppid}/comm").read_text(encoding="utf-8").strip() == "systemd"
    except OSError:
        return False


def _live_gateway_among(pids: list[int]) -> bool:
    """True when one of ``pids`` is a live Hermes gateway runtime.

    A PID whose command line can't be read has already exited (or is a
    zombie) — exactly what the reaper clears — so only a readable,
    gateway-shaped command line blocks. Uses the *runtime* matcher (``run``
    or ``restart``): without a service manager, ``gateway restart`` runs the
    gateway in-process and is itself the live runtime.
    """
    from gateway.status import _read_process_cmdline, looks_like_gateway_runtime_command_line

    for pid in pids:
        cmdline = _read_process_cmdline(pid)
        if cmdline and looks_like_gateway_runtime_command_line(cmdline):
            return True
    return False


def reap_cgroup(cgroup_path: str | None = None) -> int | None:
    """SIGKILL every PID in the cgroup other than the caller. Returns the count killed.

    Returns None (no signals) when a live gateway process is still in the
    cgroup — the reaper must never signal the live gateway, however invoked.
    """
    cgroup_path = _own_cgroup_path() if cgroup_path is None else cgroup_path
    if not cgroup_path:
        return 0
    me = os.getpid()
    others = [pid for pid in _read_cgroup_pids(cgroup_path) if pid != me]
    if not others:
        return 0
    if _live_gateway_among(others):
        print(
            "cgroup_cleanup: refusing — a live gateway process is still in the "
            "cgroup; reaping would SIGKILL it. Stop the service first (then "
            "ExecStopPost reaps its orphans), or call reap_cgroup(path) once "
            "the gateway process has exited.",
            file=sys.stderr,
        )
        return None
    killed = 0
    # Kill from a fresh read: the guard above can take seconds (status import,
    # per-PID cmdline/ps fallback), so `others` may hold exited/reused PIDs and
    # miss orphans spawned meanwhile.
    for pid in _read_cgroup_pids(cgroup_path):
        if pid == me:
            continue
        try:
            os.kill(pid, signal.SIGKILL)  # windows-footgun: ok — Linux-only (reads /proc, /sys/fs/cgroup; runs from a systemd unit)
            killed += 1
        except (ProcessLookupError, PermissionError):
            continue
    return killed


def main() -> int:
    if not _parent_is_systemd():
        print(
            "cgroup_cleanup: refusing — not spawned by systemd. Running this "
            "inside a live process's cgroup would SIGKILL that process. "
            "Run it via the systemd unit's ExecStopPost, or call "
            "reap_cgroup(cgroup_path) with an explicit stopped-service path.",
            file=sys.stderr,
        )
        return 1
    return 1 if reap_cgroup() is None else 0


if __name__ == "__main__":
    sys.exit(main())
