"""The ONE place the dashboard lane cleans up processes a test left behind.

Two problems, one mechanism:

* Detached children escape every handle the test holds. ``gateway/shutdown_forensics.py``
  runs a ``timeout 5 bash -c ...`` diagnostic in its own session on every gateway shutdown,
  ``ptyprocess`` makes each TUI a session leader, MCP stdio servers ``setsid``. When their parent
  dies they are reparented, by default to init.
* A process reparented to init is outside the pytest process subtree, so the live-system guards
  (``tests/conftest.py`` and the developer plugin) refuse to signal it, and
  ``@pytest.mark.live_system_guard_bypass`` cannot help: the plugin re-arms in
  ``pytest_runtest_teardown`` before fixture finalizers run, so a bypass never covers teardown.

``adopt_orphans()`` makes the pytest process the child subreaper (``PR_SET_CHILD_SUBREAPER``):
an orphaned descendant reparents to pytest instead of init and stays inside the subtree the
guards allow. Nothing is exempted from the guard; a process that did not descend from this pytest
process can never be adopted, so it stays refused.

``reap()`` then waits (with a deadline) for everything a test spawned to exit on its own. Only
processes still alive at the deadline are SIGKILLed, each checked against the (pid, kernel start
time) recorded when it was first seen, so a recycled pid is never hit. The killed ones come back as
the leak report.
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys
import time
from pathlib import Path
from typing import Callable

Identity = tuple[int, int]  # (pid, kernel start time in clock ticks)
Finder = Callable[[], dict[Identity, str]]

REAP_TIMEOUT = 30.0  # the gateway's shutdown diagnostic self-terminates within ``timeout 5``
_PR_SET_CHILD_SUBREAPER = 36
_adopters = 0


def _set_subreaper(on: bool) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, int(on), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER) failed")


def adopt_orphans() -> None:
    """Start adopting orphaned descendants (refcounted; pair with ``release_orphans``)."""
    global _adopters
    if sys.platform.startswith("linux") and _adopters == 0:
        _set_subreaper(True)
    _adopters += 1


def release_orphans() -> None:
    """Stop adopting once no sandbox needs it. Processes already adopted stay our children."""
    global _adopters
    _adopters = max(0, _adopters - 1)
    if sys.platform.startswith("linux") and _adopters == 0:
        _set_subreaper(False)


def stat(pid: int) -> list[str] | None:
    """``/proc/<pid>/stat`` fields after ``comm`` (state is index 0, ppid 1, sid 3, start 19)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None


def start_time(pid: int) -> int | None:
    """Kernel start time of a live, non-zombie process; None when it is gone or a zombie."""
    fields = stat(pid)
    return int(fields[19]) if fields and fields[0] not in ("Z", "X") else None


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return "?"


def all_stats() -> dict[int, list[str]]:
    """Every live, non-zombie process."""
    out = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit() and (fields := stat(int(entry))) is not None and fields[0] not in ("Z", "X"):
            out[int(entry)] = fields
    return out


def with_home(home: Path) -> Finder:
    """Every live process whose environment points HOME into ``home`` (the sandbox), except us."""
    needle = f"HOME={home}".encode()

    def find() -> dict[Identity, str]:
        found = {}
        for pid in all_stats():
            if pid == os.getpid():
                continue
            try:
                if needle not in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
                    continue
            except OSError:
                continue
            if (started := start_time(pid)) is not None:
                found[(pid, started)] = cmdline(pid)
        return found
    return find


def _collect_zombies(idents: dict[Identity, str]) -> None:
    """``waitpid`` the adopted ones that already exited, so no zombie outlives the test."""
    for pid, _ in idents:
        fields = stat(pid)
        if fields and fields[0] == "Z" and int(fields[1]) == os.getpid():
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass


def kill_identified(idents: dict[Identity, str]) -> list[str]:
    """SIGKILL exactly the recorded processes that are still the SAME process (pid and start time)."""
    killed = []
    for (pid, started), cmd in idents.items():
        if start_time(pid) != started:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        killed.append(f"{pid}: {cmd[:160]}")
    return killed


def reap(*finders: Finder, timeout: float = REAP_TIMEOUT) -> list[str]:
    """Wait up to ``timeout`` s for every process the finders report to exit by itself, then kill
    the survivors (identity-checked). Returns the killed ones: an empty list means nothing leaked."""
    seen: dict[Identity, str] = {}
    deadline = time.monotonic() + timeout
    while True:
        for find in finders:
            seen.update(find())
        _collect_zombies(seen)
        live = {ident: cmd for ident, cmd in seen.items() if start_time(ident[0]) == ident[1]}
        if not live:
            return []
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    killed = kill_identified(live)
    settle = time.monotonic() + 10
    while time.monotonic() < settle and any(start_time(pid) == started for pid, started in live):
        _collect_zombies(live)
        time.sleep(0.05)
    _collect_zombies(live)
    return killed
