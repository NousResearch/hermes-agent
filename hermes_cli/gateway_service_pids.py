"""The gateway processes behind a service job's PID.

A wrapped job reports its launcher, not the gateway: launchd's job PID is osascript (osascript ->
stderr_timestamp -> gateway since #118128), systemd's can be ``doppler run`` (#66900). The service
manager's PID alone then leaves the real gateway looking like a manual process to the update sweep.
"""

from __future__ import annotations

import contextlib


def with_gateway_descendants(pids: set) -> set:
    """*pids* plus the ``gateway run`` descendants of each."""
    found = set(pids)
    for pid in pids:
        found |= gateway_descendants_of(pid)
    return found


def gateway_descendants_of(pid: int, *, psutil_module=None) -> set[int]:
    """``gateway run`` processes under a service job PID; empty when the process table can't be read.

    Filtered by the canonical matcher: ``find_gateway_pids`` targets service PIDs, and the gateway's
    own tool subprocesses must never become ``gateway stop`` targets.
    """
    from gateway.status import looks_like_gateway_command_line

    if psutil_module is None:
        try:
            import psutil as psutil_module  # type: ignore[no-redef]
        except ImportError:
            return set()
    try:
        children = psutil_module.Process(pid).children(recursive=True)
    except (psutil_module.Error, OSError):
        return set()
    found: set[int] = set()
    for child in children:
        with contextlib.suppress(psutil_module.Error, OSError):
            if looks_like_gateway_command_line(" ".join(child.cmdline())):
                found.add(int(child.pid))
    return found
