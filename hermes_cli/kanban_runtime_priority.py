"""Protect configured runtime capacity while background workers improve repositories."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
from hermes_cli.kanban_db_workspace import _git_common_dir


@dataclass(frozen=True)
class ProcessSnapshot:
    """Read-only process identity used by the priority-runtime guard."""

    pid: int
    argv: tuple[str, ...]
    cwd: Optional[str]


@dataclass(frozen=True)
class ProcessScan:
    """A process snapshot plus whether all relevant processes were readable."""

    snapshots: tuple[ProcessSnapshot, ...]
    complete: bool


def _process_name_can_hide_python_runtime(name: Any) -> bool:
    """Whether an unreadable process name could be the guarded Python owner.

    macOS exposes login-shell supervisor rows with an empty cmdline and no cwd.
    Those rows cannot execute a Python script themselves and must not make the
    entire scan ``unknown``.  An unreadable Python row remains fail-closed.
    """

    try:
        basename = Path(str(name or "")).name
    except (TypeError, ValueError):
        return True
    if not basename:
        return True
    return bool(re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", basename))


def _process_scan() -> ProcessScan:
    """Collect a portable, read-only process snapshot via the core psutil dep.

    Only processes owned by the current user can plausibly run an entrypoint
    from that user's configured project roots. Unreadable same-user rows make
    the scan incomplete; the caller then protects runtime capacity rather than
    assuming the priority process is absent.
    """
    snapshots: list[ProcessSnapshot] = []
    complete = True
    try:
        import psutil  # type: ignore

        current_pid = os.getpid()
        current_user = psutil.Process(current_pid).username()
        for proc in psutil.process_iter(["pid", "username", "name", "cmdline", "cwd"]):
            try:
                info = proc.info
                if int(info.get("pid") or 0) == current_pid:
                    continue
                username = info.get("username")
                if username is not None and username != current_user:
                    continue
                argv = tuple(str(arg) for arg in (info.get("cmdline") or ()))
                cwd = info.get("cwd")
                if username is None:
                    complete = False
                    continue
                if not argv:
                    if _process_name_can_hide_python_runtime(info.get("name")):
                        complete = False
                    continue
                if cwd is None:
                    complete = False
                    continue
                snapshots.append(
                    ProcessSnapshot(
                        pid=int(info["pid"]),
                        argv=argv,
                        cwd=str(cwd),
                    )
                )
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (psutil.AccessDenied, OSError, ValueError, TypeError):
                complete = False
    except Exception:
        return ProcessScan(snapshots=(), complete=False)
    return ProcessScan(snapshots=tuple(snapshots), complete=complete)


def _resolved_path(value: str, *, base: Optional[Path] = None) -> Optional[Path]:
    try:
        path = Path(value).expanduser()
        if not path.is_absolute():
            if base is None:
                return None
            path = base / path
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _snapshot_runs_target(
    snapshot: ProcessSnapshot,
    targets: set[Path],
    *,
    linked_worktree_common_dirs: Optional[set[Path]] = None,
    linked_worktree_entries: tuple[Path, ...] = (),
) -> bool:
    """Prove that argv executes one of ``targets`` as a Python script."""
    cwd = _resolved_path(snapshot.cwd) if snapshot.cwd else None
    if not snapshot.argv:
        return False
    executable = Path(snapshot.argv[0]).name
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable):
        index = 1
        while index < len(snapshot.argv):
            arg = snapshot.argv[index]
            if arg in {"-c", "-m", "-"}:
                return False
            if arg == "--":
                index += 1
                break
            if not arg.startswith("-"):
                break
            index += 2 if arg in {"-W", "-X"} else 1
        if index >= len(snapshot.argv):
            return False
    else:
        index = 0
    # Arguments passed to another script cannot identify the process owner.
    for arg in (snapshot.argv[index],):
        candidate = _resolved_path(arg, base=cwd)
        candidate_matches = candidate in targets
        if (
            not candidate_matches
            and candidate is not None
            and linked_worktree_common_dirs
        ):
            for entry in linked_worktree_entries:
                if len(candidate.parts) < len(entry.parts):
                    continue
                if candidate.parts[-len(entry.parts) :] != entry.parts:
                    continue
                candidate_root = candidate.parents[len(entry.parts) - 1]
                candidate_common_dir = _git_common_dir(candidate_root)
                if candidate_common_dir in linked_worktree_common_dirs:
                    candidate_matches = True
                    break
        if not candidate_matches:
            continue
        return True
    return False


def priority_runtime_state(
    guard: Optional[Mapping[str, Any]],
    *,
    process_scan: Optional[ProcessScan] = None,
) -> str:
    """Return ``active``, ``inactive``, or ``unknown`` for a guarded runtime.

    A match requires an exact configured project root and exact relative
    entrypoint. Merely containing ``main.py`` in a command string, or running
    an unrelated project's file with the same basename, never matches.
    """
    if not isinstance(guard, Mapping) or not bool(guard.get("enabled", False)):
        return "inactive"
    raw_roots = guard.get("project_roots")
    raw_entries = guard.get("entrypoints", ("main.py",))
    if not isinstance(raw_roots, (list, tuple)) or not raw_roots:
        return "inactive"
    if not isinstance(raw_entries, (list, tuple)) or not raw_entries:
        return "inactive"

    targets: set[Path] = set()
    roots: list[Path] = []
    entries: list[Path] = []
    for raw_root in raw_roots:
        root = _resolved_path(str(raw_root))
        if root is None:
            continue
        roots.append(root)
        for raw_entry in raw_entries:
            entry = Path(str(raw_entry))
            if entry.is_absolute() or ".." in entry.parts:
                continue
            if entry not in entries:
                entries.append(entry)
            target = _resolved_path(str(entry), base=root)
            if target is not None:
                targets.add(target)
    if not targets:
        return "inactive"

    linked_common_dirs: Optional[set[Path]] = None
    if bool(guard.get("include_linked_worktrees", False)):
        linked_common_dirs = {
            common_dir
            for root in roots
            if (common_dir := _git_common_dir(root)) is not None
        }
    scan = process_scan if process_scan is not None else _process_scan()
    for snapshot in scan.snapshots:
        if _snapshot_runs_target(
            snapshot,
            targets,
            linked_worktree_common_dirs=linked_common_dirs,
            linked_worktree_entries=tuple(entries),
        ):
            return "active"
    return "inactive" if scan.complete else "unknown"


def configured_priority_runtime_guard() -> Mapping[str, Any]:
    """Read the generic priority-runtime guard block for daemon dispatch."""
    try:
        from hermes_cli.config import load_config_readonly

        raw = (load_config_readonly() or {}).get("kanban", {}).get(
            "priority_runtime_guard", {}
        )
    except Exception:
        return {}
    return raw if isinstance(raw, Mapping) else {}
