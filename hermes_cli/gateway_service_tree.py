"""Service descendants protected from the orphan reaper, not gateway stop targets."""

import contextlib


def exclusion_pids(service_pids: set[int]) -> set[int]:
    """Include children of registered launchers in the kill-exclusion set."""
    import psutil

    protected = set(service_pids)
    # launchd may supervise a launcher whose child runs the actual gateway.
    # Do not expand arbitrary PID-1 ancestry: real orphans also descend from init.
    for service_pid in service_pids:
        if service_pid <= 1:
            continue
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            protected.update(child.pid for child in psutil.Process(service_pid).children(recursive=True))
    return protected
