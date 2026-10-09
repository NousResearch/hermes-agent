"""Process and descriptor authority for state.db structural maintenance.

This module owns the proof that no foreign process still holds the active or
an unlinked SQLite DB/WAL/SHM generation.  ``hermes_state`` supplies only the
SQLite connection factory needed by the final lock probe.  Profile delete/rename
run the same descriptor census over a whole home (:func:`foreign_tree_holders`).
"""

from __future__ import annotations

import errno
import functools
import logging
import os
import sqlite3
import stat
import sys
from pathlib import Path
from typing import Callable, Collection, Iterator, List, Optional, Sequence, Set, Tuple

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


def _read_proc_argv(pid: int) -> Optional[list[str]]:
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
    """``python``/``pythonw``/``pypy`` (optionally versioned and ``.exe``) under any directory.

    ``pythonw`` is how Windows runs Hermes without a console (gateway launchers, Scheduled Tasks,
    Desktop backends). Both separators split because ``os.path.basename`` keeps ``\\`` on POSIX,
    and Windows command lines are classified on every host.
    """
    name = program.replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".exe")
    return any(
        name.startswith(prefix) and all(char.isdigit() or char == "." for char in name[len(prefix) :])
        for prefix in ("python", "pythonw", "pypy")
    )


def _python_execution_target(argv: Sequence[str]) -> Optional[tuple[str, str, int]]:
    """``(kind, target, index)`` of the module or script interpreter options select, else ``None``.

    ``kind`` is ``"module"`` or ``"script"``. ``index`` is the argv slot holding the target (the
    ``-mX`` cluster itself when the module is attached), so the target's own arguments start at
    ``index + 1``. ``-c`` inline source has no target: its payload is data, not identity.
    """
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            index += 1
            return ("script", argv[index], index) if index < len(argv) else None
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
                        return "module", attached, index
                    index += 1
                    return ("module", argv[index], index) if index < len(argv) else None
                if option in _PYTHON_SHORT_OPTIONS_WITH_OPERANDS:
                    consumed_next = not attached
                    break
                option_index += 1
            index += 2 if consumed_next else 1
            continue
        return "script", arg, index
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
    kind, value, _index = target
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


def _store_install_layout(this_home: str) -> tuple[Optional[str], Optional[str]]:
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


def _argv_path_tokens(argv: Sequence[str]) -> list[tuple[int, str]]:
    """``(argv index, normalized absolute path)`` for every path-bearing token."""
    tokens: list[tuple[int, str]] = []
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
    sidecars = {os.path.normcase(candidate) for candidate in _sqlite_family(db_path_str)}
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
        if "/.hermes" in normalized or normalized.endswith("/.hermes") or os.path.basename(normalized) in _STATE_DB_NAMES:
            other_home_seen = True
    return other_home_seen


_RM_ERROR_MORE_DATA = 234
_RM_SESSION_KEY_LEN = 33  # CCH_RM_SESSION_KEY + 1


@functools.cache
def _rm_ctypes():
    """Restart Manager ctypes metadata, built once on first scan.

    Once, because ``ctypes.POINTER()`` on a freshly defined Structure pins it in
    ``ctypes._pointer_type_cache`` forever (a per-scan rebuild leaked a class set per call);
    lazily, because POSIX processes import this module but never scan with Restart Manager.
    """
    import ctypes
    from ctypes import wintypes

    class _RmUniqueProcess(ctypes.Structure):
        _fields_ = [("pid", wintypes.DWORD), ("started", wintypes.FILETIME)]

    class _RmProcessInfo(ctypes.Structure):
        _fields_ = [
            ("process", _RmUniqueProcess),
            ("app_name", wintypes.WCHAR * 256),  # CCH_RM_MAX_APP_NAME + 1
            ("service_name", wintypes.WCHAR * 64),  # CCH_RM_MAX_SVC_NAME + 1
            ("app_type", wintypes.DWORD),
            ("app_status", wintypes.ULONG),
            ("ts_session_id", wintypes.DWORD),
            ("restartable", wintypes.BOOL),
        ]

    argtypes = {
        "RmStartSession": [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD, wintypes.LPWSTR],
        "RmRegisterResources": [
            wintypes.DWORD, wintypes.UINT, ctypes.POINTER(wintypes.LPCWSTR),
            wintypes.UINT, ctypes.c_void_p, wintypes.UINT, ctypes.c_void_p,
        ],
        "RmGetList": [
            wintypes.DWORD, ctypes.POINTER(wintypes.UINT), ctypes.POINTER(wintypes.UINT),
            ctypes.POINTER(_RmProcessInfo), ctypes.POINTER(wintypes.DWORD),
        ],
        "RmEndSession": [wintypes.DWORD],
    }
    return _RmProcessInfo, argtypes


def _sqlite_family(base: str) -> tuple[str, str, str]:
    """The main database file plus the ``-wal``/``-shm`` sidecars SQLite may hold alongside it."""
    return (base, base + "-wal", base + "-shm")


# Paths per RmRegisterResources call: a profile tree registers thousands of files.
_RM_REGISTER_CHUNK = 1000


def windows_restart_manager_pids(paths: Sequence[str]) -> list[int]:
    """PIDs Windows Restart Manager reports holding any of *paths* open; raises OSError on API failure.

    One ``RmGetList`` answers for every registered file at once, so the cost follows the file
    count rather than the number of processes on the machine.
    """
    import ctypes
    from ctypes import wintypes

    process_info, argtypes = _rm_ctypes()
    # WinDLL per call (not memoised) so the unit test can inject a fake rstrtmgr through ctypes.WinDLL.
    api = ctypes.WinDLL("rstrtmgr", use_last_error=True)
    for name, types in argtypes.items():
        fn = getattr(api, name)
        fn.argtypes = types
        fn.restype = wintypes.DWORD
    start, register, get_list, end = (
        api.RmStartSession, api.RmRegisterResources, api.RmGetList, api.RmEndSession
    )

    session = wintypes.DWORD()
    key = ctypes.create_unicode_buffer(_RM_SESSION_KEY_LEN)
    rc = start(ctypes.byref(session), 0, key)
    if rc:
        raise OSError(rc, "RmStartSession failed")
    try:
        for offset in range(0, len(paths), _RM_REGISTER_CHUNK):
            chunk = paths[offset:offset + _RM_REGISTER_CHUNK]
            filenames = (wintypes.LPCWSTR * len(chunk))(*chunk)
            rc = register(session, len(chunk), filenames, 0, None, 0, None)
            if rc:
                raise OSError(rc, "RmRegisterResources failed")

        # Pass 1 sizes (ERROR_MORE_DATA); the process set can change before the data pass, so retry the
        # bounded race with a re-sized buffer.
        needed, reasons = wintypes.UINT(), wintypes.DWORD()
        for _ in range(4):
            apps = (process_info * needed.value)()
            count = wintypes.UINT(needed.value)
            rc = get_list(
                session, ctypes.byref(needed), ctypes.byref(count),
                apps if needed.value else None, ctypes.byref(reasons),
            )
            if rc == 0:
                return [int(apps[index].process.pid) for index in range(count.value)]
            if rc != _RM_ERROR_MORE_DATA:
                raise OSError(rc, "RmGetList failed")
        raise RuntimeError("Restart Manager holder set kept changing")
    finally:
        end(session)


def _windows_restart_manager_holders(db_path: Path) -> list[tuple[int, str]]:
    """Return foreign processes using state.db or a WAL sidecar via Windows Restart Manager."""
    db_abspath = os.path.abspath(os.fspath(db_path))
    resources = [path for path in _sqlite_family(db_abspath) if os.path.exists(path)]
    if not resources:
        return []
    own_pid = os.getpid()
    return [(pid, db_abspath) for pid in windows_restart_manager_pids(resources) if pid != own_pid]


def _possible_hermes_holder(pid: int, db_path: Path) -> Optional[list[str]]:
    """argv of *pid* when it may serve the home of *db_path* though its descriptors cannot be read.

    Only a Hermes process whose own argv does not place it in another home counts. Every host
    keeps processes whose descriptors only root can read for their whole life, whoever owns them
    (``sshd: <user>``, ``(sd-pam)``, ``gpg-agent``, setuid helpers and Chromium's sandbox are
    non-dumpable), so counting every unreadable process would keep each census failing forever.
    """
    argv = _read_proc_argv(pid)
    if argv is not None and _looks_like_hermes(argv) and not _argv_scoped_to_other_home(argv, db_path):
        return argv
    return None


# ``(target, target_is_watched, descriptor stat)`` -> whether the descriptor is a holder.
_DescriptorMatch = Callable[[str, bool, os.stat_result], bool]


def _proc_descriptor_holder(
    pid: int, fd_path: str, db_path: Path, is_watched: Callable[[str], bool], matches: _DescriptorMatch,
) -> Optional[str]:
    """What one /proc descriptor of *pid* holds when it counts: its target, or why it cannot be read."""
    try:
        target = os.readlink(fd_path)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESRCH) or _possible_hermes_holder(pid, db_path) is None:
            return None
        return f"uninspectable descriptor: {fd_path}: {exc}"
    # Only a path can be watched: ``socket:[...]`` would otherwise resolve against our cwd.
    target_is_watched = target.startswith("/") and is_watched(canonical_sqlite_path(target))
    try:
        fd_stat = os.stat(fd_path)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESRCH):
            return None
        if target_is_watched or _possible_hermes_holder(pid, db_path) is not None:
            return f"uninspectable descriptor: {target}: {exc}"
        return None
    return target if matches(target, target_is_watched, fd_stat) else None


def _proc_descriptor_holders(
    db_path: Path, is_watched: Callable[[str], bool], matches: _DescriptorMatch, pids: Optional[Collection[int]],
) -> Iterator[tuple[int, str]]:
    own_pid = os.getpid()
    if pids is None:
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    for pid in pids:
        if pid == own_pid:
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            argv = _possible_hermes_holder(pid, db_path)
            if argv is not None:
                yield pid, f"uninspectable holder: {' '.join(argv)[:80]}"
            continue
        for fd in fds:
            held = _proc_descriptor_holder(pid, f"{fd_dir}/{fd}", db_path, is_watched, matches)
            if held is not None:
                yield pid, held


def _psutil_descriptor_holders(
    is_watched: Callable[[str], bool], pids: Optional[Collection[int]],
) -> Iterator[tuple[int, str]]:
    if psutil is None:
        raise RuntimeError("open-file scan unavailable")
    for process in psutil.process_iter(["pid", "open_files"]):
        info = process.info
        pid = int(info["pid"])
        if pid == os.getpid() or (pids is not None and pid not in pids):
            continue
        for opened in info.get("open_files") or ():
            path = getattr(opened, "path", "")
            if path and is_watched(canonical_sqlite_path(os.path.realpath(path))):
                yield pid, path


def _descriptor_holders(
    db_path: Path,
    is_watched: Callable[[str], bool],
    matches: _DescriptorMatch,
    pids: Optional[Collection[int]] = None,
) -> Iterator[tuple[int, str]]:
    """``(pid, what)`` per descriptor another process holds that counts; raises when the scan fails.

    The one POSIX census behind state.db maintenance and profile delete/rename; each caller says
    what counts. ``is_watched`` judges a descriptor's canonical path (`` (deleted)`` stripped)
    and ``matches(target, watched, stat)`` decides with the open file's identity. Every process
    /proc lists is read, whoever owns it (only root can read another user's descriptors). A
    process or descriptor that cannot be read counts as :func:`_possible_hermes_holder` says
    (anchored on the home of *db_path*), and a watched descriptor whose identity cannot be read
    always counts. Without /proc (macOS, BSD) psutil reports paths only, so there a watched
    path is the match and a process psutil may not read is skipped.
    """
    if sys.platform.startswith("linux"):
        return _proc_descriptor_holders(db_path, is_watched, matches, pids)
    return _psutil_descriptor_holders(is_watched, pids)


def foreign_state_db_holders(db_path: Path) -> list[tuple[int, str]]:
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
    if psutil is None and not sys.platform.startswith("linux"):
        return [(-1, "open-file scan unavailable")]

    # realpath, not abspath: psutil/libproc report the kernel-resolved pathname, so a symlinked
    # HERMES_HOME would otherwise make every holder invisible and let maintenance proceed.
    db_path_str = os.path.realpath(os.fspath(db_path))
    watched = {canonical_sqlite_path(candidate) for candidate in _sqlite_family(db_path_str)}
    holders: list[tuple[int, str]] = []
    watched_ids: set[tuple[int, int]] = set()
    db_dev: Optional[int] = None
    for candidate in _sqlite_family(db_path_str):
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

    def matches(target: str, target_is_watched: bool, fd_stat: os.stat_result) -> bool:
        # Identity is authoritative even when /proc spells another path; an unlinked generation
        # has no identity left to compare, only its old pathname on the database's device.
        return (fd_stat.st_dev, fd_stat.st_ino) in watched_ids or (
            target_is_watched
            and target.endswith(" (deleted)")
            and db_dev is not None
            and fd_stat.st_dev == db_dev
        )

    try:
        for holder in _descriptor_holders(db_path, watched.__contains__, matches):
            holders.append(holder)
    except Exception as exc:
        logger.warning(
            "Could not prove state.db has no foreign holders; "
            "deferring structural maintenance: %s",
            exc,
        )
        holders.append((-1, f"open-file scan failed: {exc}"))
    return holders


def _tree_identities(root: str) -> set[tuple[int, int]]:
    """``(st_dev, st_ino)`` of every entry deleting *root* frees.

    A symlink, and a non-directory with another link, outlive the tree: a holder of what they
    name is not a holder of the tree.
    """
    identities: set[tuple[int, int]] = set()
    for directory, _dirnames, filenames in os.walk(root):
        for path in (directory, *(os.path.join(directory, name) for name in filenames)):
            try:
                entry = os.lstat(path)
            except OSError:
                continue
            if stat.S_ISDIR(entry.st_mode) or (not stat.S_ISLNK(entry.st_mode) and entry.st_nlink == 1):
                identities.add((entry.st_dev, entry.st_ino))
    return identities


def foreign_tree_holders(root: Path, pids: Optional[Collection[int]] = None) -> list[tuple[int, str]]:
    """``(pid, what)`` per descriptor another process holds in the tree at *root*; raises when the
    scan cannot finish. *pids* narrows the scan. POSIX only: Windows asks Restart Manager.

    Profile delete/rename gate an rmtree/rename of the whole home on this, so a descriptor counts
    by its kernel path under the resolved root (an unlinked file too: /proc keeps its old path
    with `` (deleted)``) or by the identity of an entry the deletion frees. Identity catches a
    holder that opened the file through a bind mount in another mount namespace, which /proc
    spells with that namespace's path: a Docker sandbox run as the host user sees
    ``<home>/sandboxes/<task>`` as ``/workspace``. Unreadable processes follow the state.db rule,
    anchored on this home; that covers another user's gateway in a group-shared (0770) install,
    which no descriptor scan of ours can read.
    """
    resolved = os.path.realpath(os.fspath(root))
    prefix = resolved.rstrip(os.sep) + os.sep
    # Walked on first need: psutil (no /proc) never compares identities.
    identities = functools.cache(lambda: _tree_identities(resolved))

    def is_watched(path: str) -> bool:
        return path == resolved or path.startswith(prefix)

    def matches(_target: str, target_is_watched: bool, fd_stat: os.stat_result) -> bool:
        return target_is_watched or (fd_stat.st_dev, fd_stat.st_ino) in identities()

    # The argv fallback compares homes as the caller spells them, as a holder's argv does.
    anchor = Path(os.path.abspath(os.fspath(root))) / "state.db"
    try:
        return list(_descriptor_holders(anchor, is_watched, matches, pids))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot prove no other process holds files under {root}: open-file scan failed: {exc}"
        ) from exc


def in_process_state_db_holders(
    db_path: Path, *, exclude=None
) -> list[tuple[int, str]]:
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

    by_pid: dict[int, set[str]] = {}
    unknown: list[str] = []
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
