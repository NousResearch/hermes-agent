"""Find a process's children by their command line without racing their exit.

``[p for p in proc.children() if p.cmdline() ...]`` reads each child's ``/proc`` entry after the
list was taken: a short-lived sibling that exits in between is a zombie or gone, and ``cmdline()``
raises ``ZombieProcess`` / ``NoSuchProcess`` for a process the test never cared about. Such a
child cannot be the one being looked for, so it is skipped.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import psutil


def matching(procs: Iterable[psutil.Process], predicate: Callable[[psutil.Process], bool]) -> list[psutil.Process]:
    """The live processes in *procs* that *predicate* accepts; one that exits mid-read is skipped."""
    found = []
    for proc in procs:
        try:
            if predicate(proc):
                found.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):  # ZombieProcess is a NoSuchProcess
            continue
    return found


def children_matching(pid: int | psutil.Process, predicate: Callable[[psutil.Process], bool], *,
                      recursive: bool = False) -> list[psutil.Process]:
    """``matching`` over the children of *pid*."""
    proc = pid if isinstance(pid, psutil.Process) else psutil.Process(pid)
    return matching(proc.children(recursive=recursive), predicate)
