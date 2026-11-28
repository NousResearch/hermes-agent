"""Whole-process-table reads for gateway/kanban process identity: the ``ps`` ``comm`` pre-filter
and the ``[(pid, cmdline), …]`` sweep the kanban dispatcher uses to find a process no DB-side
guard can see.

Sibling of :mod:`gateway.status`; the single-PID reader it degrades to lives there and is
late-imported inside the function, so this module pulls nothing but stdlib at import time.
"""

import os
import subprocess
from typing import Optional


# Cheap pre-filter for the ``ps`` fallback of :func:`iter_process_cmdlines`: whether a ``comm``
# field names a plausible Hermes entrypoint, decided on the BASENAME so a path never has to match
# exactly. Exact-token equality for everything except the interpreter family, which legitimately
# varies by version suffix (``python``, ``python3``, ``python3.12``, ``pythonw.exe``) and is
# therefore the single allowlisted prefix — a prefix test on the rest would let a ``sherlock``
# monitor or a ``shipmentd`` worker cost a per-PID ``ps`` fork forever (mirrors the token care of
# :func:`profile_flag_value`).
_HERMES_COMM_NAMES = ("hermes", "hermes.exe", "sh")
_HERMES_COMM_PREFIXES = ("python",)


def _comm_looks_like_hermes(comm: str) -> bool:
    """True when a ``ps`` ``comm`` value could be a Hermes CLI process (basename test, no substrings).

    Linux truncates ``comm`` at 15 bytes and macOS spells it as the full executable path, so the
    test runs on the normalized basename and survives both spellings."""
    base = (comm or "").strip().strip("\"'").replace("\\", "/").rsplit("/", 1)[-1].lower()
    if not base:
        return False
    return base in _HERMES_COMM_NAMES or base.startswith(_HERMES_COMM_PREFIXES)


def iter_process_cmdlines() -> list[tuple[int, str]]:
    """``[(pid, command line), …]`` for every visible process — never raises, may be partial.

    The whole-table counterpart of :func:`_read_process_cmdline`, for callers hunting an *unknown*
    PID: the kanban dispatcher looks for an external ``hermes -z`` oneshot carrying a task id on
    its argv, which no DB-side guard can see. Preferred path is ``psutil.process_iter``, which
    reads the table in one in-process sweep (a ``sysctl`` on macOS) instead of forking; without
    psutil — or when it raises — POSIX falls back to one ``ps -ax -o pid=,comm=`` listing plus a
    per-PID ``ps -o command=`` for only the entries whose ``comm`` basename looks like Hermes
    (see :func:`_comm_looks_like_hermes`): one fork per candidate, not one per process. Windows
    without psutil returns ``[]`` so the feature degrades to off.

    This process's own PID is skipped — a dispatcher importing this module is never the external
    worker it is looking for — and every per-process failure (zombie, ``EPERM``, a process that
    vanishes mid-scan) drops just that entry, so a scan never raises at the call site.
    """
    from gateway.status import _IS_WINDOWS, _read_process_cmdline
    own_pid = os.getpid()
    seen: set[int] = set()
    entries: list[tuple[int, str]] = []
    try:
        import psutil  # type: ignore
    except Exception:  # health: allow BLE001 -- a broken psutil install must fall through to the ps reader, never raise at the dispatch call site
        psutil = None  # type: ignore[assignment]
    if psutil is not None:
        try:
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    pid = int(proc.info.get("pid") or 0)
                    if pid <= 0 or pid == own_pid:
                        continue
                    cmdline = " ".join(proc.info.get("cmdline") or []).strip()
                    if not cmdline:
                        continue
                    seen.add(pid)
                    entries.append((pid, cmdline))
                except Exception:  # health: allow BLE001 -- a process can vanish between listing and reading; it is simply absent from the table
                    continue
            return entries
        except Exception:  # health: allow BLE001, S110 -- an unsupported or access-denied sweep must degrade to the ps reader, not raise
            pass  # the deliberate fallthrough to the ``ps`` reader below
    if _IS_WINDOWS:
        return entries  # no usable whole-table reader here
    try:
        result = subprocess.run(
            ["ps", "-ax", "-o", "pid=,comm="],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return entries
    if result.returncode != 0:
        return entries
    for line in result.stdout.splitlines():
        pid_text, _, comm = line.strip().partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == own_pid or pid in seen or not _comm_looks_like_hermes(comm):
            continue
        cmdline = _read_process_cmdline(pid)
        if not cmdline:
            continue
        seen.add(pid)
        entries.append((pid, cmdline))
    return entries
