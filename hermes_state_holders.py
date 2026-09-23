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

from hermes_state_errors import is_sqlite_lock_error

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]


def read_only_db_uri(db_path) -> str:
    """``file:`` URI for a ``mode=ro`` open. ``as_uri()`` percent-encodes ``?``/``#`` in the home
    path; a raw ``f"file:{path}?mode=ro"`` truncates there and opens the wrong (empty) database."""
    return Path(db_path).resolve().as_uri() + "?mode=ro"


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


def describe_holder_pid(pid: int) -> str:
    """``PID 123 (hermes gateway run)`` for operator-facing holder lists; /proc argv first, psutil elsewhere."""
    argv = _read_proc_argv(pid)
    if argv is None and psutil is not None:
        try:
            argv = psutil.Process(pid).cmdline() or None
        except Exception:
            argv = None
    who = " ".join(" ".join([os.path.basename(argv[0]), *argv[1:]]).split())[:80] if argv else "command line unavailable"
    return f"PID {pid} ({who})"


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


def canonical_sqlite_path(path: str) -> str:
    """Normalize a /proc fd target, stripping the Linux `` (deleted)`` suffix."""
    return os.path.normcase(os.path.abspath(path.removesuffix(" (deleted)")))


_HOME_FLAGS = ("--hermes-home",)
_PROFILE_FLAGS = ("--profile", "-p")
_STATE_DB_NAMES = ("state.db", "state.db-wal", "state.db-shm")


def _norm_path(value: str) -> str:
    return os.path.normcase(os.path.normpath(value))


def _argv_flag_value(argv: Sequence[str], flags: Sequence[str]) -> Optional[str]:
    """Last ``--flag X`` / ``--flag=X`` value, token-exact (``--profile timothy`` is not ``tim``)."""
    value: Optional[str] = None
    index, count = 0, len(argv)
    while index < count:
        token = argv[index]
        if isinstance(token, str):
            if token in flags and index + 1 < count and isinstance(argv[index + 1], str):
                value = argv[index + 1]
                index += 2
                continue
            for flag in flags:
                if token.startswith(flag + "="):
                    value = token[len(flag) + 1:]
                    break
        index += 1
    return value


def _argv_env_home(argv: Sequence[str]) -> Optional[str]:
    """``HERMES_HOME=<path>`` env-style assignment on the argv (``env HERMES_HOME=… hermes …``)."""
    for token in reversed(list(argv)):
        if isinstance(token, str) and token.startswith("HERMES_HOME="):
            return token[len("HERMES_HOME="):]
    return None


def _store_install_layout(this_home: str) -> Tuple[Optional[str], Optional[str]]:
    """``(<install root>, <our profile name>)`` for the home holding the store, else ``(None, None)``.

    Derived with the canonical ``named_profile_home`` predicate, never a ``basename == "profiles"``
    string test: an arbitrary ``<X>/profiles/<n>/`` tree is not a Hermes install, and promoting
    ``<X>`` to "ours" swallows an unrelated instance living under it — the literal two-instance
    shape of #92401. The root store (``~/.hermes/state.db``) is its own root with no profile name,
    so ANY named-profile selection contradicts it.
    """
    try:
        from hermes_constants import named_profile_home

        profile_home = named_profile_home(this_home)
        if profile_home is not None:
            return os.path.abspath(str(profile_home.parent.parent)), profile_home.name
        if os.path.basename(this_home) == ".hermes":
            return os.path.abspath(this_home), None
    except Exception:  # constants import/resolution must never break a holder scan
        logger.debug("Could not classify the install layout of %s", this_home, exc_info=True)
    return None, None


def _names_other_profile(normalized: str, install_root: Optional[str], our_profile: Optional[str]) -> bool:
    """True when the token is under ``<install root>/profiles/<name>`` for a name that is not ours."""
    if install_root is None:
        return False
    prefix = _norm_path(os.path.join(install_root, "profiles")) + os.sep
    if not normalized.startswith(prefix):
        return False
    name = normalized[len(prefix):].split(os.sep, 1)[0]
    return bool(name) and name != (os.path.normcase(our_profile) if our_profile else None)


def _argv_home_selection(
    argv: Sequence[str], this_home: str, install_root: Optional[str], our_profile: Optional[str]
) -> Optional[str]:
    """``"ours"``/``"other"``/``None`` from the process's OWN profile/home selection.

    Under one process per host the shared binary path proves nothing about which home a process
    serves; its ``--hermes-home``/``HERMES_HOME=``/``--profile``/``-p`` selection does. Same
    token-exact parsers ``gateway/run.py::_argv_contradicts_home`` uses.
    """
    home_value = _argv_flag_value(argv, _HOME_FLAGS) or _argv_env_home(argv)
    if home_value:
        return "ours" if _norm_path(home_value) == _norm_path(this_home) else "other"
    profile_value = _argv_flag_value(argv, _PROFILE_FLAGS)
    if profile_value:
        if our_profile is not None:
            return "ours" if profile_value == our_profile else "other"
        # Root/custom home: any explicit named profile selects a different home.
        return "ours" if (install_root is not None and profile_value == "default") else "other"
    return None


def _argv_path_tokens(argv: Sequence[str]) -> List[Tuple[int, str]]:
    """``(argv index, normalized absolute path)`` for every path-bearing token."""
    tokens: List[Tuple[int, str]] = []
    for index, token in enumerate(argv):
        if not isinstance(token, str):
            continue
        if token.startswith("/"):
            path_token = token
        elif token.startswith("-") and "=" in token:
            # ``--db=/abs/path``-style options carry a path value; anchor on
            # the text after '=' so normpath does not prepend the option.
            value = token.split("=", 1)[1]
            path_token = value if value.startswith("/") else None
        else:
            path_token = None
        if path_token is not None:
            tokens.append((index, _norm_path(path_token)))
    return tokens


def _argv_scoped_to_other_home(argv: Sequence[str], db_path: Path) -> bool:
    """Return whether argv proves the process belongs to a DIFFERENT instance.

    ``state.db`` lives at the HERMES_HOME root, so an absolute-path token
    containing a ``/.hermes`` segment (or naming a ``state.db``/WAL/SHM under
    some other parent) identifies that token's own Hermes home.  When at least
    one such token exists AND no token references this instance's state.db,
    its sidecars, or its home directory, the process provably works on a
    different generation and must not be counted as an uninspectable holder
    of ours (issue #92401: a second gateway under /home/demo/.hermes deferred
    this instance's stale-FTS rebuild forever despite lsof proving zero open
    handles).  Ambiguous argv without absolute-path tokens returns False and
    keeps the fail-closed suspicion.

    Evidence is ranked, because one host now runs ONE process for every profile:

    1. A token naming our state.db or a sidecar exactly — definitive, ours.
    2. The process's own ``--hermes-home``/``HERMES_HOME=``/``--profile``/``-p``
       selection — that is what decides which home a multiplexer serves.
    3. Path tokens. A token under ``<install root>/profiles/<other>`` is another
       profile's store even though it sits under our root; a token that names only
       the SHARED install root is NEUTRAL (it is the same binary for every profile,
       so it can neither prove nor disprove a hold); ``argv[0]`` locates the INSTALL,
       not the home, so it is not other-home evidence for a store whose home is not
       part of an install layout (a custom ``HERMES_HOME`` is served BY the binary
       under ``~/.hermes`` — dismissing on it admits maintenance under a live writer).
    """
    db_path_str = os.path.abspath(os.fspath(db_path))
    this_home = os.path.dirname(db_path_str)
    install_root, our_profile = _store_install_layout(this_home)
    sidecars = {
        os.path.normcase(candidate)
        for candidate in (db_path_str, db_path_str + "-wal", db_path_str + "-shm")
    }
    this_home_norm = os.path.normcase(this_home)
    root_norm = os.path.normcase(install_root) if install_root else None
    path_tokens = _argv_path_tokens(argv)

    if any(normalized in sidecars for _, normalized in path_tokens):
        return False
    selection = _argv_home_selection(argv, this_home, install_root, our_profile)
    if selection == "ours":
        return False
    # A store whose home is not itself part of an install layout cannot be identified from the
    # install location, so argv[0] alone never dismisses a holder of it.
    argv0_locates_home = install_root is not None
    other_home_seen = selection == "other"
    for index, normalized in path_tokens:
        if _names_other_profile(normalized, install_root, our_profile):
            other_home_seen = True
            continue
        if normalized == this_home_norm or normalized.startswith(this_home_norm + os.sep):
            return False
        if root_norm is not None and (
            normalized == root_norm or normalized.startswith(root_norm + os.sep)
        ):
            continue  # shared install root: neutral, every served profile lives under it
        if index == 0 and not argv0_locates_home:
            continue
        if "/.hermes" in normalized or normalized.endswith("/.hermes"):
            other_home_seen = True
        elif os.path.basename(normalized) in _STATE_DB_NAMES:
            other_home_seen = True
    return other_home_seen


def _windows_restart_manager_holders(db_path: Path) -> List[Tuple[int, str]]:
    """Return foreign processes using state.db or a WAL sidecar via Windows Restart Manager."""
    import ctypes
    from ctypes import wintypes

    error_more_data = 234
    max_app_name, max_svc_name = 255, 63

    class _UniqueProcess(ctypes.Structure):
        _fields_ = [("pid", wintypes.DWORD), ("started", wintypes.FILETIME)]

    class _ProcessInfo(ctypes.Structure):
        _fields_ = [
            ("process", _UniqueProcess),
            ("app_name", wintypes.WCHAR * (max_app_name + 1)),
            ("service_name", wintypes.WCHAR * (max_svc_name + 1)),
            ("app_type", wintypes.DWORD),
            ("app_status", wintypes.ULONG),
            ("ts_session_id", wintypes.DWORD),
            ("restartable", wintypes.BOOL),
        ]

    api = ctypes.WinDLL("rstrtmgr", use_last_error=True)
    start, register, get_list, end = (
        api.RmStartSession, api.RmRegisterResources, api.RmGetList, api.RmEndSession
    )
    start.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD, wintypes.LPWSTR]
    register.argtypes = [
        wintypes.DWORD, wintypes.UINT, ctypes.POINTER(wintypes.LPCWSTR),
        wintypes.UINT, ctypes.c_void_p, wintypes.UINT, ctypes.c_void_p,
    ]
    get_list.argtypes = [
        wintypes.DWORD, ctypes.POINTER(wintypes.UINT), ctypes.POINTER(wintypes.UINT),
        ctypes.POINTER(_ProcessInfo), ctypes.POINTER(wintypes.DWORD),
    ]
    end.argtypes = [wintypes.DWORD]
    for fn in (start, register, get_list, end):
        fn.restype = wintypes.DWORD

    resources = [
        os.path.abspath(path)
        for path in (os.fspath(db_path), os.fspath(db_path) + "-wal", os.fspath(db_path) + "-shm")
        if os.path.exists(path)
    ]
    if not resources:
        return []

    session = wintypes.DWORD()
    key = ctypes.create_unicode_buffer(33)
    rc = start(ctypes.byref(session), 0, key)
    if rc:
        raise OSError(rc, "RmStartSession failed")
    try:
        filenames = (wintypes.LPCWSTR * len(resources))(*resources)
        rc = register(session, len(resources), filenames, 0, None, 0, None)
        if rc:
            raise OSError(rc, "RmRegisterResources failed")

        needed, count, reasons = wintypes.UINT(), wintypes.UINT(), wintypes.DWORD()
        rc = get_list(session, ctypes.byref(needed), ctypes.byref(count), None, ctypes.byref(reasons))
        if rc == 0:
            return []
        if rc != error_more_data:
            raise OSError(rc, "RmGetList failed")

        # The process set can change between the sizing and data calls. Retry the bounded race.
        for _ in range(3):
            apps = (_ProcessInfo * needed.value)()
            count = wintypes.UINT(needed.value)
            rc = get_list(
                session, ctypes.byref(needed), ctypes.byref(count), apps, ctypes.byref(reasons)
            )
            if rc == 0:
                own_pid = os.getpid()
                return [
                    (int(apps[index].process.pid), os.path.abspath(os.fspath(db_path)))
                    for index in range(count.value)
                    if int(apps[index].process.pid) != own_pid
                ]
            if rc != error_more_data:
                raise OSError(rc, "RmGetList failed")
        raise RuntimeError("Restart Manager holder set kept changing")
    finally:
        end(session)


def foreign_state_db_holders(db_path: Path) -> List[Tuple[int, str]]:
    """Return foreign holders of the DB or one of its WAL sidecars.

    A scan failure is represented as an unknown holder. Structural maintenance
    must not assume quiescence when an old, unlinked SQLite generation may
    still be open by another process.
    """
    if _IS_WINDOWS:
        try:
            return _windows_restart_manager_holders(db_path)
        except Exception as exc:
            logger.warning(
                "Could not prove state.db has no Windows holders; deferring structural maintenance: %s",
                exc,
            )
            return [(-1, f"Windows Restart Manager scan failed: {exc}")]

    # realpath, not abspath: psutil/libproc report the kernel-resolved pathname, so a symlinked
    # HERMES_HOME would otherwise make every holder invisible and let maintenance proceed.
    db_path_str = os.path.realpath(os.fspath(db_path))
    watched = {
        canonical_sqlite_path(db_path_str),
        canonical_sqlite_path(db_path_str + "-wal"),
        canonical_sqlite_path(db_path_str + "-shm"),
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
                    if (
                        argv is not None
                        and _looks_like_hermes(argv)
                        and not _argv_scoped_to_other_home(argv, db_path)
                    ):
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
                        if (
                            argv is not None
                            and _looks_like_hermes(argv)
                            and not _argv_scoped_to_other_home(argv, db_path)
                        ):
                            holders.append(
                                (
                                    pid,
                                    f"uninspectable descriptor: {fd_path}: {exc}",
                                )
                            )
                        continue
                    target_is_watched = canonical_sqlite_path(target) in watched
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
                            if (
                                argv is not None
                                and _looks_like_hermes(argv)
                                and not _argv_scoped_to_other_home(argv, db_path)
                            ):
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
                if path and canonical_sqlite_path(os.path.realpath(path)) in watched:
                    holders.append((pid, path))
    except Exception as exc:
        logger.warning(
            "Could not prove state.db has no foreign holders; "
            "deferring structural maintenance: %s",
            exc,
        )
        holders.append((-1, f"open-file scan failed: {exc}"))
    return holders


def in_process_state_db_holders(
    db_path: Path, *, exclude=None
) -> List[Tuple[int, str]]:
    """Return holders of ``db_path`` inside THIS process, other than *exclude*.

    :func:`foreign_state_db_holders` skips ``os.getpid()`` by design, so it answers a
    cross-PROCESS question only. Consumers that read "no holders" as "the store is quiet"
    (auto-VACUUM admission) need this arm too: a VACUUM plus its TRUNCATE checkpoint retires
    the generation a sibling SessionDB in this very process still holds.
    """
    from hermes_state_registry import other_generations_for_path

    return [
        (os.getpid(), description)
        for description in other_generations_for_path(db_path, exclude=exclude)
    ]


def held_store_refusal(db_path: Path, *, command: str, force_hint: Optional[str] = "--force") -> Optional[str]:
    """Operator-facing refusal for structural maintenance (VACUUM, index rebuild, bulk delete) while another
    process holds ``db_path`` or a WAL sidecar; ``None`` when the store is provably quiet.

    Running ``hermes sessions optimize-storage`` underneath a fleet of live gateways put every agent into
    the retired-WAL refusal until all writers were stopped (#110054). Same fail-closed scan doctor and
    repair use: an incomplete scan refuses too, it never reads as an all-clear.
    """
    holders = foreign_state_db_holders(db_path)
    if not holders:
        return None
    from hermes_constants import profile_cli_selector
    from hermes_state_errors import STORAGE_RECOVERY_DOCS_URL

    by_pid: dict[int, Set[str]] = {}
    unknown: List[str] = []
    for pid, target in holders:
        if pid <= 0 or target.startswith("uninspectable"):
            unknown.append(target)
        else:
            by_pid.setdefault(pid, set()).add(Path(target.removesuffix(" (deleted)")).name)
    lines = [f"Refusing `hermes sessions {command}`: another process is using {db_path}."]
    lines += [f"  {describe_holder_pid(pid)}: {', '.join(sorted(by_pid[pid]))}" for pid in sorted(by_pid)]
    if unknown:
        lines.append(f"  cannot prove the database is quiet (holder scan incomplete: {unknown[0][:120]})")
    profile_arg = profile_cli_selector()
    lines += [
        "Rewriting the database under a live writer is how every agent ends up refusing turns with the "
        "retired state.db-wal error. Nothing is lost.",
        f"Stop them first (`hermes {profile_arg}gateway stop`, quit the Desktop app, pause cron), then re-run.",
    ]
    if force_hint:
        lines.append(f"Override with {force_hint} if you accept the risk.")
    lines.append(f"Recovery guide: {STORAGE_RECOVERY_DOCS_URL}")
    return "\n".join(lines)


def live_writer_holds_db(
    db_path: Path,
    *,
    connect_repair_durable: Callable[..., sqlite3.Connection],
) -> bool:
    """Return whether repair lacks proven exclusive ownership of ``db_path``.

    ANY foreign process holding the DB or a sidecar is a live holder (#103339): the lock probe below
    cannot see a DELETE-mode reader (SHARED only) and cannot run at all on a malformed file, and those
    are exactly the states repair/VACUUM/checkpoint get invoked in. The holder scan is the authority and
    fails closed on its own failures (unknown/uninspectable sentinels); the probe only adds a positive
    lock signal on top."""
    if foreign_state_db_holders(db_path):
        return True

    probe = None
    try:
        probe = connect_repair_durable(db_path, timeout=0.0)
        probe.execute("PRAGMA locking_mode=EXCLUSIVE")
        probe.execute("BEGIN IMMEDIATE")
        probe.execute("ROLLBACK")
        return False
    except sqlite3.OperationalError as exc:
        return is_sqlite_lock_error(exc)
    except sqlite3.DatabaseError:
        # Malformed/unreadable with no holder on the scan: nobody else has it open, so repair may run.
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
