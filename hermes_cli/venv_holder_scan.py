"""POSIX live-venv holder detection — stdlib only, no ``psutil``.

Why this exists: replacing a venv that running processes are executing from strands
those processes against a deleted interpreter tree (a long-lived gateway keeps an open
``SSL_CERT_FILE``/CA path into it and then fails every TLS call). The updater has a
holder signal for that, but off Windows it is always empty — two reasons:

* ``update_cmd_windows._detect_venv_python_processes`` early-returns ``[]`` unless
  ``_m()._is_windows()`` **and** ``psutil`` import.
* the same-named hook on ``hermes_cli.main`` is a retired no-work shim (``return []``).

So on Linux/macOS every holder-gated step silently sees zero holders. This module
restores the signal with ``/proc`` (Linux) and a best-effort fallback that never raises:
an unreadable or mid-exit process is skipped, not fatal.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "pids_holding_venv",
    "proc_cmdline",
    "proc_exe",
    "proc_name",
    "self_and_ancestor_pids",
]


def _norm(path: str | os.PathLike[str]) -> str:
    """Canonical form of a directory with no trailing separator (``""`` on failure)."""
    try:
        return os.path.realpath(str(path)).rstrip(os.sep)
    except (OSError, ValueError):
        return ""


def _read_bytes(path: str) -> bytes | None:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except (OSError, ValueError):
        return None


def proc_cmdline(pid: int) -> str:
    """NUL-joined argv of *pid* as one string; ``""`` when unreadable (kernel threads)."""
    raw = _read_bytes(f"/proc/{pid}/cmdline")
    if not raw:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def proc_exe(pid: int) -> str:
    """Resolved executable of *pid*; ``""`` when unreadable (already exited, no perms)."""
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except (OSError, ValueError):
        return ""


def proc_name(pid: int) -> str:
    """``argv[0]`` basename, else ``exe`` basename, else ``"pid <n>"``."""
    cmdline = proc_cmdline(pid)
    if cmdline:
        first = cmdline.split(" ", 1)[0]
        base = os.path.basename(first)
        if base:
            return base
    exe = proc_exe(pid)
    return os.path.basename(exe) if exe else f"pid {pid}"


def _ppid(pid: int) -> int:
    """Parent pid from ``/proc/<pid>/stat``; ``0`` when unreadable.

    ``stat``'s second field is the executable name in parentheses and may itself contain
    spaces and parentheses, so split after the *last* ``)``.
    """
    raw = _read_bytes(f"/proc/{pid}/stat")
    if not raw:
        return 0
    try:
        fields = raw.decode("utf-8", "replace").rsplit(")", 1)[1].split()
        return int(fields[1])
    except (IndexError, ValueError):
        return 0


def self_and_ancestor_pids(pid: int | None = None) -> set[int]:
    """``{pid}`` plus its parent chain — the updater's own tree, never a holder (#87594)."""
    chain: set[int] = set()
    current = os.getpid() if pid is None else int(pid)
    while current > 0 and current not in chain:
        chain.add(current)
        current = _ppid(current)
    return chain


def _candidate_paths(token: str) -> tuple[str, ...]:
    """A path token as written, plus its resolved form when they differ.

    Both matter: ``/venv/bin/python`` is usually a *symlink* to a shared interpreter, so
    resolving alone would collapse it outside the venv and miss a real holder.
    """
    if not token:
        return ()
    raw = os.path.normpath(token).rstrip(os.sep)
    resolved = _norm(token)
    return (raw,) if resolved in ("", raw) else (raw, resolved)


def _under(candidate: str, venv_norm: str) -> bool:
    return bool(candidate) and (
        candidate == venv_norm or candidate.startswith(venv_norm + os.sep)
    )


def _references_venv(pid: int, venv_norm: str) -> str | None:
    """Which part of *pid* points into *venv_norm*: ``"exe"``, ``"cmdline"``, ``"env"`` or ``None``."""
    for candidate in _candidate_paths(proc_exe(pid)):
        if _under(candidate, venv_norm):
            return "exe"

    cmdline = proc_cmdline(pid)
    if cmdline:
        for token in cmdline.split():
            for candidate in _candidate_paths(token):
                if _under(candidate, venv_norm):
                    return "cmdline"

    raw = _read_bytes(f"/proc/{pid}/environ")
    if raw:
        for entry in raw.split(b"\0"):
            if entry.startswith(b"VIRTUAL_ENV="):
                value = entry.split(b"=", 1)[1].decode("utf-8", "replace")
                if _norm(value) == venv_norm:
                    return "env"
            elif entry.startswith(b"PYTHONPATH="):
                for item in entry.split(b"=", 1)[1].decode("utf-8", "replace").split(os.pathsep):
                    if _norm(item) == venv_norm:
                        return "env"
    return None


def pids_holding_venv(
    venv_dir: str | os.PathLike[str],
    *,
    exclude_pids: set[int] | None = None,
    include_ancestors: bool = False,
) -> list[int]:
    """Pids executing from, or pointed at, *venv_dir* — ascending, never raises.

    ``exclude_pids`` is honoured verbatim. By default the caller's own process chain is
    excluded as well, because an updater naturally lives under the very venv it is about
    to replace; pass ``include_ancestors=True`` for a raw inventory (e.g. a --list style
    report that should show every holder).
    """
    venv_norm = _norm(venv_dir)
    if not venv_norm or not os.path.isdir("/proc"):
        return []
    skip = set(exclude_pids or ())
    if not include_ancestors:
        skip |= self_and_ancestor_pids()
    holders: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in skip:
            continue
        try:
            if _references_venv(pid, venv_norm):
                holders.append(pid)
        except Exception:  # health: allow BLE001 -- /proc races with process exit mid-scan; a holder must never abort the walk
            continue
    return sorted(holders)


def venv_dir_for_project(project_root: str | os.PathLike[str]) -> Path:
    """``<root>/.venv`` when present, else ``<root>/venv`` — the checkout's own venv."""
    root = Path(project_root)
    for name in (".venv", "venv"):
        candidate = root / name
        if (candidate / "pyvenv.cfg").is_file():
            return candidate
    return root / "venv"
