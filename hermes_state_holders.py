"""Process and descriptor authority for state.db structural maintenance.

This module owns the proof that no foreign process still holds the active or
an unlinked SQLite DB/WAL/SHM generation.  ``hermes_state`` supplies only the
SQLite connection factory needed by the final lock probe.
"""

from __future__ import annotations

import errno
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set, Tuple

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"
_HERMES_EXECUTABLES = frozenset({"hermes", "hermes-agent", "hermes-acp"})
_HERMES_PYTHON_MODULES = frozenset({"acp_adapter", "hermes_cli.main"})
_HERMES_PYTHON_SCRIPTS = frozenset({"hermes_cli/main.py", "run_agent.py"})
_PYTHON_SHORT_OPTIONS_WITH_OPERANDS = frozenset({"Q", "W", "X"})
_PYTHON_LONG_OPTIONS_WITH_OPERANDS = frozenset(
    {"--check-hash-based-pycs", "--jit"}
)


def _read_proc_argv(pid: int) -> Optional[List[str]]:
    """Read /proc/<pid>/cmdline without losing argv boundaries."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
        if not raw:
            return None
        argv = raw.decode("utf-8", "replace").split("\x00")
        if argv[-1] == "":
            argv.pop()
        return argv or None
    except OSError:
        return None


def _looks_like_python_executable(program: str) -> bool:
    name = os.path.basename(program).lower().removesuffix(".exe")
    for prefix in ("python", "pypy"):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            return not suffix or all(char.isdigit() or char == "." for char in suffix)
    return False


def _python_execution_target(argv: Sequence[str]) -> Optional[Tuple[str, str]]:
    """Return the Python module or script selected by interpreter options."""
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            index += 1
            return ("script", argv[index]) if index < len(argv) else None
        if arg in _PYTHON_LONG_OPTIONS_WITH_OPERANDS:
            index += 2
            continue
        if arg.startswith("--check-hash-based-pycs=") or arg.startswith("--jit="):
            index += 1
            continue
        if arg.startswith("--"):
            index += 1
            continue
        if arg.startswith("-") and arg != "-":
            options = arg[1:]
            option_index = 0
            consumed_next = False
            while option_index < len(options):
                option = options[option_index]
                attached = options[option_index + 1 :]
                if option == "c":
                    return None
                if option == "m":
                    if attached:
                        return "module", attached
                    index += 1
                    return ("module", argv[index]) if index < len(argv) else None
                if option in _PYTHON_SHORT_OPTIONS_WITH_OPERANDS:
                    consumed_next = not attached
                    break
                option_index += 1
            index += 2 if consumed_next else 1
            continue
        return "script", arg
    return None


def _looks_like_hermes(argv: Sequence[str]) -> bool:
    """Return whether argv identifies a supported Hermes execution target."""
    if not argv:
        return False
    program = os.path.basename(argv[0]).lower().removesuffix(".exe")
    if program in _HERMES_EXECUTABLES:
        return True
    if not _looks_like_python_executable(program):
        return False
    target = _python_execution_target(argv)
    if target is None:
        return False
    kind, value = target
    normalized = value.lower().replace("\\", "/")
    if kind == "module":
        return normalized in _HERMES_PYTHON_MODULES
    return any(
        normalized == script or normalized.endswith(f"/{script}")
        for script in _HERMES_PYTHON_SCRIPTS
    )


def foreign_state_db_holders(db_path: Path) -> List[Tuple[int, str]]:
    """Return foreign holders of the DB or one of its WAL sidecars.

    A scan failure is represented as an unknown holder. Structural maintenance
    must not assume quiescence when an old, unlinked SQLite generation may
    still be open by another process.
    """
    if _IS_WINDOWS:
        return []

    def _canonical(path: str) -> str:
        clean = path.removesuffix(" (deleted)")
        return os.path.normcase(os.path.abspath(clean))

    db_path_str = os.path.abspath(os.fspath(db_path))
    watched = {
        _canonical(db_path_str),
        _canonical(db_path_str + "-wal"),
        _canonical(db_path_str + "-shm"),
    }
    holders: List[Tuple[int, str]] = []
    watched_ids: Set[Tuple[int, int]] = set()
    db_dev: Optional[int] = None
    for candidate in (db_path_str, db_path_str + "-wal", db_path_str + "-shm"):
        try:
            stat_result = os.stat(candidate)
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.ESRCH):
                holders.append(
                    (-1, f"watched-file stat failed: {candidate}: {exc}")
                )
            continue
        watched_ids.add((stat_result.st_dev, stat_result.st_ino))
        if candidate == db_path_str:
            db_dev = stat_result.st_dev

    if sys.platform.startswith("linux"):
        try:
            own_pid = os.getpid()
            for pid_str in os.listdir("/proc"):
                if not pid_str.isdigit():
                    continue
                pid = int(pid_str)
                if pid == own_pid:
                    continue
                fd_dir = f"/proc/{pid}/fd"
                try:
                    fds = os.listdir(fd_dir)
                except OSError:
                    argv = _read_proc_argv(pid)
                    if argv is not None and _looks_like_hermes(argv):
                        cmdline = " ".join(argv)
                        holders.append((pid, f"uninspectable holder: {cmdline[:80]}"))
                    continue
                for fd in fds:
                    fd_path = f"{fd_dir}/{fd}"
                    try:
                        target = os.readlink(fd_path)
                    except OSError as exc:
                        if exc.errno in (errno.ENOENT, errno.ESRCH):
                            continue
                        argv = _read_proc_argv(pid)
                        if argv is not None and _looks_like_hermes(argv):
                            holders.append(
                                (
                                    pid,
                                    f"uninspectable descriptor: {fd_path}: {exc}",
                                )
                            )
                        continue
                    target_is_watched = _canonical(target) in watched
                    try:
                        fd_stat = os.stat(fd_path)
                    except OSError as exc:
                        if exc.errno in (errno.ENOENT, errno.ESRCH):
                            continue
                        if target_is_watched:
                            holders.append(
                                (pid, f"uninspectable descriptor: {target}: {exc}")
                            )
                        else:
                            argv = _read_proc_argv(pid)
                            if argv is not None and _looks_like_hermes(argv):
                                holders.append(
                                    (
                                        pid,
                                        "uninspectable descriptor: "
                                        f"{target}: {exc}",
                                    )
                                )
                        continue
                    if (fd_stat.st_dev, fd_stat.st_ino) in watched_ids or (
                        target_is_watched
                        and target.endswith(" (deleted)")
                        and db_dev is not None
                        and fd_stat.st_dev == db_dev
                    ):
                        holders.append((pid, target))
        except Exception as exc:
            logger.warning(
                "Could not prove state.db has no foreign holders; "
                "deferring structural maintenance: %s",
                exc,
            )
            holders.append((-1, f"open-file scan failed: {exc}"))
        return holders

    if psutil is None:
        return [(-1, "open-file scan unavailable")]
    try:
        for process in psutil.process_iter(["pid", "open_files"]):
            info = process.info
            pid = int(info["pid"])
            if pid == os.getpid():
                continue
            for opened in info.get("open_files") or ():
                path = getattr(opened, "path", "")
                if path and _canonical(path) in watched:
                    holders.append((pid, path))
    except Exception as exc:
        logger.warning(
            "Could not prove state.db has no foreign holders; "
            "deferring structural maintenance: %s",
            exc,
        )
        holders.append((-1, f"open-file scan failed: {exc}"))
    return holders


def live_writer_holds_db(
    db_path: Path,
    *,
    connect_repair_durable: Callable[..., sqlite3.Connection],
) -> bool:
    """Return whether repair lacks proven exclusive ownership of ``db_path``."""
    foreign_holders = foreign_state_db_holders(db_path)
    if any(
        pid < 0
        or path.startswith("uninspectable holder:")
        or path.startswith("uninspectable descriptor:")
        or path.endswith(" (deleted)")
        for pid, path in foreign_holders
    ):
        return True

    probe = None
    try:
        probe = connect_repair_durable(db_path, timeout=0.0)
        probe.execute("PRAGMA locking_mode=EXCLUSIVE")
        probe.execute("BEGIN IMMEDIATE")
        probe.execute("ROLLBACK")
        return False
    except sqlite3.OperationalError as exc:
        lowered = str(exc).lower()
        return "locked" in lowered or "busy" in lowered
    except sqlite3.DatabaseError:
        return False
    except Exception:
        return False
    finally:
        if probe is not None:
            try:
                probe.execute("PRAGMA locking_mode=NORMAL")
            except Exception:
                pass
            try:
                probe.close()
            except Exception:
                pass
