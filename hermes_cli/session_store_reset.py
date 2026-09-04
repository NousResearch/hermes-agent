"""Owner-only reset of the Hermes session store.

This is deliberately a Hermes command, rather than an operational shell
recipe.  It owns only the fixed active ``state.db`` SQLite family and fixed
``sessions`` directory.  It deliberately excludes archives, traces,
checkpoints, snapshots, and backups.  Callers can choose an attempt name but
never paths to move, delete, or initialise.  A failed reset after quarantine
has started is intentionally left quarantined: starting an empty database is
safer than guessing at a rollback of a partially moved SQLite family.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import sqlite3
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from hermes_constants import get_hermes_home


_REPORT_SCHEMA = "hermes-session-store-reset/v1"
_ATTEMPT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SQLITE_PARTS = ("state.db", "state.db-wal", "state.db-shm", "state.db-journal")
_QUARANTINE_DIR = "session-reset-quarantine"
_LOCK_NAME = ".session-store-reset.lock"
_ATTEMPT_PLAN = "plan.json"
_PHASES_DIR = "phases"
_PARTS_DIR = "parts"
_ATTEMPT_PHASES = {
    "prepared",
    "reconciling",
    "quarantining",
    "quarantined",
    "completed",
    "failed",
}
_EMPTY_TABLES = (
    "system_prompts",
    "sessions",
    "messages",
    "session_model_usage",
    "gateway_routing",
    "gateway_heartbeats",
    "compression_locks",
    "session_turn_leases",
    "async_delegations",
    "conversation_generations",
    "delivery_obligations",
)


class SessionStoreResetError(RuntimeError):
    """A reset failure whose public report contains only ``code``."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _safe_report(
    attempt: str,
    status: str,
    code: Optional[str] = None,
    **extra: Any,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": _REPORT_SCHEMA,
        "attempt": attempt,
        "status": status,
    }
    if code is not None:
        report["code"] = code
    report.update(extra)
    return report


def _lstat_at(dir_fd: int, name: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        raise SessionStoreResetError("source_missing") from None
    except OSError:
        raise SessionStoreResetError("source_untrusted") from None


def _assert_owned_regular(dir_fd: int, name: str, uid: int, gid: int) -> os.stat_result:
    info = _lstat_at(dir_fd, name)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != uid
        or info.st_gid != gid
        or info.st_mode & 0o022
    ):
        raise SessionStoreResetError("source_untrusted")
    return info


def _assert_owned_dir(dir_fd: int, name: str, uid: int, gid: int) -> os.stat_result:
    info = _lstat_at(dir_fd, name)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != uid
        or info.st_gid != gid
        or info.st_mode & 0o022
    ):
        raise SessionStoreResetError("source_untrusted")
    return info


def _part_identity(info: os.stat_result) -> dict[str, int]:
    """Non-content identity that an in-filesystem rename must preserve."""

    return {
        "dev": int(info.st_dev),
        "ino": int(info.st_ino),
        "mode": stat.S_IMODE(info.st_mode),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "size": int(info.st_size) if stat.S_ISREG(info.st_mode) else 0,
    }


def _tree_identity(info: os.stat_result) -> dict[str, int]:
    identity = _part_identity(info)
    identity["mtime_ns"] = int(info.st_mtime_ns)
    return identity


def _open_dir_at(dir_fd: int, name: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError:
        raise SessionStoreResetError("source_untrusted") from None


def _walk_owned_tree(dir_fd: int, uid: int, gid: int) -> int:
    """Reject links and special files in a session-artifact tree.

    All traversal remains descriptor-relative.  ``listdir(fd)`` plus
    ``lstat(..., dir_fd=fd)`` prevents a link or an attacker-controlled cwd
    spelling from escaping the Hermes-owned sessions directory.
    """

    entries = 0
    for name in os.listdir(dir_fd):
        info = _lstat_at(dir_fd, name)
        if stat.S_ISDIR(info.st_mode):
            if info.st_uid != uid or info.st_gid != gid or info.st_mode & 0o022:
                raise SessionStoreResetError("source_untrusted")
            child_fd = _open_dir_at(dir_fd, name)
            try:
                entries += 1 + _walk_owned_tree(child_fd, uid, gid)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(info.st_mode):
            if (
                info.st_nlink != 1
                or info.st_uid != uid
                or info.st_gid != gid
                or info.st_mode & 0o022
            ):
                raise SessionStoreResetError("source_untrusted")
            entries += 1
        else:
            raise SessionStoreResetError("source_untrusted")
    return entries


def _session_tree_sha256(dir_fd: int, uid: int, gid: int) -> str:
    """Hash trusted session-tree metadata without disclosing names or content.

    Relative names participate only in the canonical digest material.  The
    returned value is the sole externally stored representation of that tree.
    """

    records: list[dict[str, Any]] = []

    def visit(current_fd: int, prefix: str) -> None:
        for name in sorted(os.listdir(current_fd)):
            info = _lstat_at(current_fd, name)
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(info.st_mode):
                if info.st_uid != uid or info.st_gid != gid or info.st_mode & 0o022:
                    raise SessionStoreResetError("source_untrusted")
                records.append({"kind": "dir", "name": relative, "identity": _tree_identity(info)})
                child_fd = _open_dir_at(current_fd, name)
                try:
                    visit(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(info.st_mode):
                _assert_owned_regular(current_fd, name, uid, gid)
                records.append({"kind": "file", "name": relative, "identity": _tree_identity(info)})
            else:
                raise SessionStoreResetError("source_untrusted")

    visit(dir_fd, "")
    return hashlib.sha256(_json_bytes({"entries": records})).hexdigest()


def _attempt_manifest_sha256(plan: dict[str, Any]) -> str:
    """Return the one non-content digest that authorizes finalization."""

    material = {
        "schema": plan["schema"],
        "attempt": plan["attempt"],
        "parts": [
            {"part": part, "identity": plan["identities"][part]}
            for part in plan["parts"]
        ],
        "session_tree_sha256": plan["session_tree_sha256"],
    }
    return hashlib.sha256(_json_bytes(material)).hexdigest()


def _fsync_dir(fd: int) -> None:
    try:
        os.fsync(fd)
    except OSError:
        raise SessionStoreResetError("durability_failed") from None


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_all(fd: int, payload: bytes) -> None:
    """Write every byte or fail before publishing an immutable record."""

    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short durable-record write")
        view = view[written:]


def _write_json_once(dir_fd: int, name: str, value: dict[str, Any]) -> None:
    """Atomically append a durable, immutable attempt record.

    Attempt state is an event log, rather than a replaceable phase file.  A
    crash cannot make a previously observed boundary disappear, and an
    existing name is never silently accepted as a record for this attempt.
    """

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    payload = _json_bytes(value)
    temporary = f".{name}.tmp"
    try:
        try:
            _lstat_at(dir_fd, name)
        except SessionStoreResetError as exc:
            if exc.code != "source_missing":
                raise SessionStoreResetError("attempt_inconsistent") from None
        else:
            raise SessionStoreResetError("attempt_inconsistent")
        try:
            fd = os.open(temporary, flags, 0o600, dir_fd=dir_fd)
        except FileExistsError:
            # A power loss before replace leaves only an owner-private marker
            # scratch file.  It has no committed meaning and can be safely
            # discarded after its metadata is checked; a committed final
            # record is always written through replace below.
            _assert_owned_regular(dir_fd, temporary, os.geteuid(), os.getegid())
            os.unlink(temporary, dir_fd=dir_fd)
            _fsync_dir(dir_fd)
            fd = os.open(temporary, flags, 0o600, dir_fd=dir_fd)
        try:
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        _fsync_dir(dir_fd)
    except FileExistsError:
        raise SessionStoreResetError("attempt_inconsistent") from None
    except OSError:
        raise SessionStoreResetError("durability_failed") from None


def _mkdir_owned_at(dir_fd: int, name: str, uid: int, gid: int) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=dir_fd)
        _fsync_dir(dir_fd)
    except FileExistsError:
        pass
    except OSError:
        raise SessionStoreResetError("durability_failed") from None
    _assert_owned_dir(dir_fd, name, uid, gid)
    return _open_dir_at(dir_fd, name)


def _canonical_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path.removesuffix(" (deleted)")))


def _linux_foreign_holders(paths: set[str]) -> bool:
    """Return whether a foreign process holds any SQLite-family inode.

    The deployed Linux runtime predates Hermes' later state-holder module, so
    this small, self-contained `/proc` check keeps the reset safe when copied
    into that image.  An unreadable process table is deliberately not treated
    as an all-clear.
    """

    proc = "/proc"
    try:
        pids = os.listdir(proc)
    except OSError:
        raise SessionStoreResetError("holder_check_unavailable") from None
    own_pid = os.getpid()
    for pid_name in pids:
        if not pid_name.isdecimal() or int(pid_name) == own_pid:
            continue
        fd_dir = os.path.join(proc, pid_name, "fd")
        try:
            descriptors = os.listdir(fd_dir)
        except FileNotFoundError:
            continue
        except PermissionError:
            raise SessionStoreResetError("holder_check_unavailable") from None
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(os.path.join(fd_dir, descriptor))
            except FileNotFoundError:
                continue
            except OSError:
                continue
            if _canonical_path(target) in paths:
                return True
    return False


def _foreign_holders_present(state: Path, names: set[str]) -> bool:
    """Use the modern owner helper when available, otherwise inspect Linux."""

    if "state.db" not in names:
        return False

    try:
        from hermes_state_holders import foreign_state_db_holders
    except ImportError:
        foreign_state_db_holders = None
    if foreign_state_db_holders is not None:
        try:
            return bool(foreign_state_db_holders(state))
        except Exception:
            raise SessionStoreResetError("holder_check_unavailable") from None
    if sys.platform.startswith("linux"):
        watched = {
            _canonical_path(str(state.with_name(name)))
            for name in names
            if name in _SQLITE_PARTS
        }
        return _linux_foreign_holders(watched)
    # On non-Linux we still acquire the whole-file advisory locks below.  That
    # is the only portable, parser-free exclusion primitive available on the
    # schema-17 deployment line.
    return False


@contextlib.contextmanager
def _lock_sqlite_family(home_fd: int, names: set[str]) -> Iterator[None]:
    """Hold non-blocking POSIX record locks across all source moves.

    A whole-file ``lockf`` covers SQLite's locking bytes too, so a process
    that acquires a SQLite lock after the `/proc` scan conflicts with this
    critical section.  Unlike opening SQLite, this works for a corrupt source.
    """

    try:
        import fcntl
    except ImportError:
        raise SessionStoreResetError("holder_check_unavailable") from None
    handles: list[int] = []
    try:
        for name in _SQLITE_PARTS:
            if name not in names:
                continue
            try:
                fd = os.open(name, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), dir_fd=home_fd)
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                raise SessionStoreResetError("live_holder") from None
            handles.append(fd)
        yield
    finally:
        for fd in reversed(handles):
            try:
                fcntl.lockf(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


def _revalidate_source_before_rename(
    home_fd: int,
    part: str,
    identity: dict[str, int],
    uid: int,
    gid: int,
    home_device: int,
    expected_session_tree_sha256: Optional[str] = None,
) -> None:
    """Close the plan-to-rename window without ever following a path link."""

    names = set(os.listdir(home_fd))
    if any(name.startswith("state.db-") and name not in _SQLITE_PARTS for name in names):
        raise SessionStoreResetError("unknown_sidecar")
    if part not in names:
        raise SessionStoreResetError("attempt_inconsistent")
    info = (
        _assert_owned_dir(home_fd, part, uid, gid)
        if part == "sessions"
        else _assert_owned_regular(home_fd, part, uid, gid)
    )
    if info.st_dev != home_device or _part_identity(info) != identity:
        raise SessionStoreResetError("attempt_inconsistent")
    if part == "sessions":
        sessions_fd = _open_dir_at(home_fd, part)
        try:
            _walk_owned_tree(sessions_fd, uid, gid)
            if (
                expected_session_tree_sha256 is not None
                and _session_tree_sha256(sessions_fd, uid, gid)
                != expected_session_tree_sha256
            ):
                raise SessionStoreResetError("attempt_inconsistent")
        finally:
            os.close(sessions_fd)


def _assert_active_scope_is_empty(home_fd: int) -> None:
    """The reset owns no live artifact beyond the freshly initialized DB."""

    names = set(os.listdir(home_fd))
    if "sessions" in names or any(name.startswith("state.db-") for name in names):
        raise SessionStoreResetError("verification_failed")


def _write_phase(attempt_fd: int, attempt: str, phase: str, uid: int, gid: int) -> None:
    """Append a durable phase event; phase records are never overwritten."""

    phases_fd = _mkdir_owned_at(attempt_fd, _PHASES_DIR, uid, gid)
    try:
        value = {"schema": _REPORT_SCHEMA, "attempt": attempt, "phase": phase}
        if _record_exists(phases_fd, f"{phase}.json", value, uid, gid):
            return
        _write_json_once(
            phases_fd,
            f"{phase}.json",
            value,
        )
    finally:
        os.close(phases_fd)


def _write_part_record(
    attempt_fd: int,
    attempt: str,
    part: str,
    identity: dict[str, int],
    uid: int,
    gid: int,
) -> None:
    parts_fd = _mkdir_owned_at(attempt_fd, _PARTS_DIR, uid, gid)
    try:
        _write_json_once(
            parts_fd,
            f"{part}.json",
            {
                "schema": _REPORT_SCHEMA,
                "attempt": attempt,
                "part": part,
                "identity": identity,
            },
        )
    finally:
        os.close(parts_fd)


def _record_exists(dir_fd: int, name: str, expected: dict[str, Any], uid: int, gid: int) -> bool:
    try:
        _assert_owned_regular(dir_fd, name, uid, gid)
    except SessionStoreResetError as exc:
        if exc.code == "source_missing":
            return False
        raise SessionStoreResetError("attempt_inconsistent") from None
    try:
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=dir_fd)
        try:
            raw = os.read(fd, 16385)
        finally:
            os.close(fd)
        if len(raw) > 16384 or raw != _json_bytes(expected):
            raise SessionStoreResetError("attempt_inconsistent")
        return True
    except SessionStoreResetError:
        raise
    except (OSError, UnicodeError):
        raise SessionStoreResetError("attempt_inconsistent") from None


def _load_json_record(dir_fd: int, name: str, uid: int, gid: int) -> dict[str, Any]:
    try:
        _assert_owned_regular(dir_fd, name, uid, gid)
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=dir_fd)
        try:
            raw = os.read(fd, 16385)
        finally:
            os.close(fd)
        if len(raw) > 16384:
            raise ValueError
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict) or _json_bytes(decoded) != raw:
            raise ValueError
        return decoded
    except (SessionStoreResetError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
        raise SessionStoreResetError("attempt_inconsistent") from None


def _phase_exists(attempt_fd: int, attempt: str, phase: str, uid: int, gid: int) -> bool:
    try:
        phases_fd = _open_dir_at(attempt_fd, _PHASES_DIR)
    except SessionStoreResetError:
        return False
    try:
        return _record_exists(
            phases_fd,
            f"{phase}.json",
            {"schema": _REPORT_SCHEMA, "attempt": attempt, "phase": phase},
            uid,
            gid,
        )
    finally:
        os.close(phases_fd)


@contextlib.contextmanager
def _reset_lock(home_fd: int, uid: int, gid: int) -> Iterator[None]:
    """Acquire the owner reset lock; inability to prove exclusion is failure."""

    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(_LOCK_NAME, flags, 0o600, dir_fd=home_fd)
    except OSError:
        raise SessionStoreResetError("lock_unavailable") from None
    try:
        info = _assert_owned_regular(home_fd, _LOCK_NAME, uid, gid)
        if info.st_mode & 0o077:
            raise SessionStoreResetError("lock_unavailable")
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, BlockingIOError, OSError):
            raise SessionStoreResetError("lock_unavailable") from None
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)


@dataclass
class _DeferredSignals:
    received: Optional[int] = None
    previous: dict[int, Any] | None = None

    def __enter__(self) -> "_DeferredSignals":
        self.previous = {}

        def defer(signum: int, _frame: Any) -> None:
            if self.received is None:
                self.received = signum

        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.getsignal(signum)
            signal.signal(signum, defer)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        assert self.previous is not None
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)


def _invoke_hook(hook: Optional[Callable[[str], None]], boundary: str) -> None:
    if hook is not None:
        hook(boundary)


def _validate_release_contract(conn: sqlite3.Connection, fts_enabled: bool) -> None:
    """Check the release-owned minimum schema, including FTS wiring.

    This is deliberately object-level rather than a row-count approximation:
    an empty database with a missing ``state_meta`` table or orphaned FTS
    trigger cannot safely be treated as a bootstrapped release store.
    """

    rows = conn.execute(
        "SELECT type, name, COALESCE(sql, '') FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%'"
    ).fetchall()
    objects = {str(name): (str(kind), str(sql).upper()) for kind, name, sql in rows}
    required = {
        "schema_version": ("version",),
        "state_meta": ("key", "value"),
        "sessions": ("id",),
        "messages": ("id", "session_id"),
        "compression_locks": ("session_id",),
    }
    for table, columns in required.items():
        if objects.get(table, (None,))[0] != "table":
            raise SessionStoreResetError("verification_failed")
        actual_columns = tuple(
            str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')
        )
        if not all(column in actual_columns for column in columns):
            raise SessionStoreResetError("verification_failed")

    fts_tables = [
        name
        for name, (kind, sql) in objects.items()
        if name.startswith("messages_fts")
        and kind == "table"
        and "VIRTUAL TABLE" in sql
        and "USING FTS5" in sql
    ]
    for table in fts_tables:
        kind, sql = objects[table]
        if kind != "table" or "VIRTUAL TABLE" not in sql or "USING FTS5" not in sql:
            raise SessionStoreResetError("verification_failed")
    if fts_enabled and "messages_fts" not in fts_tables:
        raise SessionStoreResetError("verification_failed")
    for table in fts_tables:
        prefix = f"{table}_"
        trigger_names = {
            name for name, (kind, _sql) in objects.items() if kind == "trigger" and name.startswith(prefix)
        }
        if not {f"{table}_insert", f"{table}_delete", f"{table}_update"} <= trigger_names:
            raise SessionStoreResetError("verification_failed")


def _validate_empty_store(state_path: Path, uid: int, gid: int) -> dict[str, Any]:
    """Validate the fresh owner-created store, then leave no sidecars behind."""

    from hermes_state import SCHEMA_VERSION, SessionDB

    db = SessionDB(db_path=state_path)
    try:
        conn = db._conn
        if conn is None:
            raise SessionStoreResetError("initialization_failed")
        integrity = [str(row[0]).lower() for row in conn.execute("PRAGMA integrity_check")]
        foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchall()
        journal = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        version_row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if not version_row or int(version_row[0]) != int(SCHEMA_VERSION):
            raise SessionStoreResetError("verification_failed")
        _validate_release_contract(conn, bool(getattr(db, "_fts_enabled", False)))
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        counts: dict[str, int] = {}
        for table in _EMPTY_TABLES:
            if table in tables:
                counts[table] = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        if (
            integrity != ["ok"]
            or foreign_keys
            or journal not in {"wal", "delete"}
            or any(counts.values())
        ):
            raise SessionStoreResetError("verification_failed")
    except SessionStoreResetError:
        raise
    except (sqlite3.Error, ValueError, TypeError):
        raise SessionStoreResetError("verification_failed") from None
    finally:
        db.close()

    state_fd = os.open(state_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        info = _assert_owned_regular(state_fd, state_path.name, uid, gid)
        if info.st_mode & 0o022:
            raise SessionStoreResetError("verification_failed")
        for suffix in ("-wal", "-shm", "-journal"):
            try:
                os.stat(state_path.name + suffix, dir_fd=state_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise SessionStoreResetError("verification_failed")
    finally:
        os.close(state_fd)
    return {"schema_version": int(SCHEMA_VERSION), "journal_mode": journal}


def _validated_plan(
    plan: dict[str, Any], attempt: str
) -> tuple[tuple[str, ...], dict[str, dict[str, int]], str]:
    parts = plan.get("parts")
    identities = plan.get("identities")
    tree_digest = plan.get("session_tree_sha256")
    manifest = plan.get("attempt_manifest_sha256")
    if (
        plan.get("schema") != _REPORT_SCHEMA
        or plan.get("attempt") != attempt
        or not isinstance(parts, list)
        or not isinstance(identities, dict)
        or not isinstance(tree_digest, str)
        or not isinstance(manifest, str)
        or not _SHA256_RE.fullmatch(tree_digest)
        or not _SHA256_RE.fullmatch(manifest)
        or not parts
        or len(parts) != len(set(parts))
        or any(part not in (*_SQLITE_PARTS, "sessions") for part in parts)
        or "state.db" not in parts
        or set(identities) != set(parts)
        or any(
            not isinstance(identities[part], dict)
            or set(identities[part]) != {"dev", "ino", "mode", "uid", "gid", "size"}
            or any(not isinstance(value, int) for value in identities[part].values())
            for part in parts
        )
        or _attempt_manifest_sha256(plan) != manifest
    ):
        raise SessionStoreResetError("attempt_inconsistent")
    return tuple(parts), identities, manifest


def _validate_terminal_attempt(
    attempt_fd: int,
    home_fd: int,
    attempt: str,
    uid: int,
    gid: int,
    home_device: int,
) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    """Revalidate immutable terminal evidence before reporting or purging it."""

    if not _phase_exists(attempt_fd, attempt, "completed", uid, gid):
        raise SessionStoreResetError("attempt_not_completed")
    plan = _load_json_record(attempt_fd, _ATTEMPT_PLAN, uid, gid)
    parts, identities, manifest = _validated_plan(plan, attempt)
    expected_root = {_ATTEMPT_PLAN, _PHASES_DIR, _PARTS_DIR, "sqlite"}
    if "sessions" in parts:
        expected_root.add("sessions")
    if set(os.listdir(attempt_fd)) != expected_root:
        raise SessionStoreResetError("attempt_inconsistent")
    _assert_owned_dir(attempt_fd, "sqlite", uid, gid)
    _assert_owned_dir(attempt_fd, _PARTS_DIR, uid, gid)
    _assert_owned_dir(attempt_fd, _PHASES_DIR, uid, gid)
    phases_fd = _open_dir_at(attempt_fd, _PHASES_DIR)
    try:
        phase_files = set(os.listdir(phases_fd))
        if not phase_files or not phase_files <= {f"{phase}.json" for phase in _ATTEMPT_PHASES}:
            raise SessionStoreResetError("attempt_inconsistent")
        for phase_file in phase_files:
            phase = phase_file.removesuffix(".json")
            if not _record_exists(
                phases_fd,
                phase_file,
                {"schema": _REPORT_SCHEMA, "attempt": attempt, "phase": phase},
                uid,
                gid,
            ):
                raise SessionStoreResetError("attempt_inconsistent")
    finally:
        os.close(phases_fd)
    sqlite_fd = _open_dir_at(attempt_fd, "sqlite")
    try:
        if set(os.listdir(sqlite_fd)) != {part for part in parts if part != "sessions"}:
            raise SessionStoreResetError("attempt_inconsistent")
        parts_fd = _open_dir_at(attempt_fd, _PARTS_DIR)
        try:
            if set(os.listdir(parts_fd)) != {f"{part}.json" for part in parts}:
                raise SessionStoreResetError("attempt_inconsistent")
            for part in parts:
                dst_fd = attempt_fd if part == "sessions" else sqlite_fd
                info = (
                    _assert_owned_dir(dst_fd, part, uid, gid)
                    if part == "sessions"
                    else _assert_owned_regular(dst_fd, part, uid, gid)
                )
                if (
                    info.st_dev != home_device
                    or _part_identity(info) != identities[part]
                    or not _record_exists(
                        parts_fd,
                        f"{part}.json",
                        {
                            "schema": _REPORT_SCHEMA,
                            "attempt": attempt,
                            "part": part,
                            "identity": identities[part],
                        },
                        uid,
                        gid,
                    )
                ):
                    raise SessionStoreResetError("attempt_inconsistent")
                if part == "sessions":
                    sessions_fd = _open_dir_at(attempt_fd, "sessions")
                    try:
                        if _session_tree_sha256(sessions_fd, uid, gid) != plan["session_tree_sha256"]:
                            raise SessionStoreResetError("attempt_inconsistent")
                    finally:
                        os.close(sessions_fd)
            if "sessions" not in parts and plan["session_tree_sha256"] != hashlib.sha256(
                _json_bytes({"entries": []})
            ).hexdigest():
                raise SessionStoreResetError("attempt_inconsistent")
        finally:
            os.close(parts_fd)
    finally:
        os.close(sqlite_fd)
    _assert_active_scope_is_empty(home_fd)
    validation = _validate_empty_store(Path(get_hermes_home()) / "state.db", uid, gid)
    _assert_active_scope_is_empty(home_fd)
    return validation, manifest, parts


def _preflight_new_attempt(
    home_fd: int,
    home_info: os.stat_result,
    names: set[str],
    uid: int,
    gid: int,
) -> dict[str, Any]:
    """Validate every fixed source before creating any quarantine artifact."""

    if "state.db" not in names:
        raise SessionStoreResetError("source_missing")
    if any(name.startswith("state.db-") and name not in _SQLITE_PARTS for name in names):
        raise SessionStoreResetError("unknown_sidecar")
    plan: dict[str, Any] = {
        "schema": _REPORT_SCHEMA,
        "attempt": "",  # assigned only after the caller selects its safe id
        "parts": [name for name in _SQLITE_PARTS if name in names],
        "identities": {},
    }
    for part in plan["parts"]:
        info = _assert_owned_regular(home_fd, part, uid, gid)
        if info.st_dev != home_info.st_dev:
            raise SessionStoreResetError("cross_filesystem")
        plan["identities"][part] = _part_identity(info)
    if "sessions" in names:
        info = _assert_owned_dir(home_fd, "sessions", uid, gid)
        if info.st_dev != home_info.st_dev:
            raise SessionStoreResetError("cross_filesystem")
        sessions_fd = _open_dir_at(home_fd, "sessions")
        try:
            _walk_owned_tree(sessions_fd, uid, gid)
            plan["session_tree_sha256"] = _session_tree_sha256(sessions_fd, uid, gid)
        finally:
            os.close(sessions_fd)
        plan["parts"].append("sessions")
        plan["identities"]["sessions"] = _part_identity(info)
    else:
        plan["session_tree_sha256"] = hashlib.sha256(_json_bytes({"entries": []})).hexdigest()
    if _foreign_holders_present(Path(get_hermes_home()) / "state.db", names):
        raise SessionStoreResetError("live_holder")
    return plan


def reset_session_store(
    attempt: str,
    *,
    failure_hook: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Quarantine the fixed session store and create a verified empty one.

    ``failure_hook`` is intentionally an in-process test seam, not a CLI/API
    setting.  It lets tests cover crash boundaries without granting operators
    a way to alter filesystem targets or reset behavior.
    """

    if not _ATTEMPT_RE.fullmatch(attempt):
        return _safe_report(attempt, "failed", "attempt_invalid")

    home = get_hermes_home()
    uid, gid = os.geteuid(), os.getegid()
    try:
        home_fd = os.open(
            home,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError:
        return _safe_report(attempt, "failed", "home_untrusted")

    moved = False
    attempt_fd: Optional[int] = None
    deferred: Optional[_DeferredSignals] = None
    validation: dict[str, Any] = {}
    sessions_present = False
    session_entries = 0
    terminal_reconciliation = False
    try:
        home_info = os.fstat(home_fd)
        if (
            not stat.S_ISDIR(home_info.st_mode)
            or home_info.st_uid != uid
            or home_info.st_gid != gid
            or home_info.st_mode & 0o022
        ):
            raise SessionStoreResetError("home_untrusted")
        with _DeferredSignals() as deferred, _reset_lock(home_fd, uid, gid):
            names = set(os.listdir(home_fd))
            unknown = [
                name
                for name in names
                if name.startswith("state.db-") and name not in _SQLITE_PARTS
            ]
            if unknown:
                raise SessionStoreResetError("unknown_sidecar")
            # Existence discovery is read-only.  A new attempt must finish
            # all fixed-source preflight before it creates a quarantine root,
            # attempt directory, or durable marker.
            q_fd: Optional[int] = None
            attempt_exists = False
            try:
                _assert_owned_dir(home_fd, _QUARANTINE_DIR, uid, gid)
                q_fd = _open_dir_at(home_fd, _QUARANTINE_DIR)
                q_info = os.fstat(q_fd)
                if q_info.st_dev != home_info.st_dev:
                    raise SessionStoreResetError("cross_filesystem")
                try:
                    _assert_owned_dir(q_fd, attempt, uid, gid)
                    attempt_exists = True
                except SessionStoreResetError as exc:
                    if exc.code != "source_missing":
                        raise SessionStoreResetError("attempt_inconsistent") from None
            except SessionStoreResetError as exc:
                if exc.code != "source_missing":
                    raise
            if not attempt_exists:
                try:
                    new_plan = _preflight_new_attempt(home_fd, home_info, names, uid, gid)
                except Exception:
                    if q_fd is not None:
                        os.close(q_fd)
                    raise
                new_plan["attempt"] = attempt
                new_plan["attempt_manifest_sha256"] = _attempt_manifest_sha256(new_plan)
                if q_fd is None:
                    q_fd = _mkdir_owned_at(home_fd, _QUARANTINE_DIR, uid, gid)
                os.mkdir(attempt, 0o700, dir_fd=q_fd)
                _fsync_dir(q_fd)
            assert q_fd is not None
            try:
                attempt_fd = _open_dir_at(q_fd, attempt)
                try:
                    if attempt_exists:
                        plan = _load_json_record(attempt_fd, _ATTEMPT_PLAN, uid, gid)
                        parts, identities, manifest_digest = _validated_plan(plan, attempt)
                    else:
                        plan = new_plan
                        _write_json_once(attempt_fd, _ATTEMPT_PLAN, plan)
                        _write_phase(attempt_fd, attempt, "prepared", uid, gid)
                        _invoke_hook(failure_hook, "prepared")
                    parts = tuple(plan["parts"])
                    identities = plan["identities"]
                    manifest_digest = plan["attempt_manifest_sha256"]
                    if attempt_exists and _phase_exists(
                        attempt_fd, attempt, "completed", uid, gid
                    ):
                        # A completed attempt is idempotent only after both
                        # its immutable quarantine records and the active
                        # owner-created store still validate.  Do not append a
                        # failure marker merely because a reconciler retried.
                        terminal_reconciliation = True
                        validation, manifest_digest, parts = _validate_terminal_attempt(
                            attempt_fd,
                            home_fd,
                            attempt,
                            uid,
                            gid,
                            home_info.st_dev,
                        )
                        return _safe_report(
                            attempt,
                            "completed",
                            already_completed=True,
                            quarantined=True,
                            session_artifacts_present="sessions" in parts,
                            validation=validation,
                            attempt_manifest_sha256=manifest_digest,
                            preserved_excluded_surfaces=(
                                "moa-traces",
                                "state-snapshots",
                                "backups",
                                "checkpoints",
                                "service-archive",
                            ),
                            deferred_signal=(
                                deferred.received if deferred is not None else None
                            ),
                        )
                    if attempt_exists:
                        _write_phase(attempt_fd, attempt, "reconciling", uid, gid)
                    fresh_state_present = False

                    sqlite_fd = _mkdir_owned_at(attempt_fd, "sqlite", uid, gid)
                    try:
                        parts_fd = _mkdir_owned_at(attempt_fd, _PARTS_DIR, uid, gid)
                        try:
                            recorded = {
                                part
                                for part in parts
                                if _record_exists(
                                    parts_fd,
                                    f"{part}.json",
                                    {
                                        "schema": _REPORT_SCHEMA,
                                        "attempt": attempt,
                                        "part": part,
                                        "identity": identities[part],
                                    },
                                    uid,
                                    gid,
                                )
                            }
                            if set(os.listdir(parts_fd)) != {f"{part}.json" for part in recorded}:
                                raise SessionStoreResetError("attempt_inconsistent")
                        finally:
                            os.close(parts_fd)

                        # A recorded move must have only the destination; an
                        # unrecorded move must have only the source.  Anything
                        # else is a crash in an unknowable boundary.
                        for part in parts:
                            dst_fd = attempt_fd if part == "sessions" else sqlite_fd
                            src_present = part in names
                            try:
                                _lstat_at(dst_fd, part)
                                dst_present = True
                            except SessionStoreResetError as exc:
                                if exc.code != "source_missing":
                                    raise
                                dst_present = False
                            if dst_present and not src_present and part not in recorded:
                                # The only recoverable crash boundary is the
                                # atomic in-filesystem rename before its
                                # durable part event.  The plan's inode-bound
                                # identity proves this is exactly that part.
                                dst_info = (
                                    _assert_owned_dir(dst_fd, part, uid, gid)
                                    if part == "sessions"
                                    else _assert_owned_regular(dst_fd, part, uid, gid)
                                )
                                if _part_identity(dst_info) != identities[part]:
                                    raise SessionStoreResetError("attempt_inconsistent")
                                if part == "sessions":
                                    sessions_fd = _open_dir_at(attempt_fd, "sessions")
                                    try:
                                        if _session_tree_sha256(sessions_fd, uid, gid) != plan[
                                            "session_tree_sha256"
                                        ]:
                                            raise SessionStoreResetError("attempt_inconsistent")
                                    finally:
                                        os.close(sessions_fd)
                                _write_part_record(
                                    attempt_fd, attempt, part, identities[part], uid, gid
                                )
                                recorded.add(part)
                            elif (
                                part == "state.db"
                                and part in recorded
                                and dst_present
                                and src_present
                                and _phase_exists(
                                    attempt_fd, attempt, "quarantined", uid, gid
                                )
                            ):
                                # A crash after the old family was completely
                                # quarantined may leave an owner-created fresh
                                # state.db before the completed event.  Its
                                # identity must differ from the old planned
                                # inode; validation below is still mandatory.
                                fresh_state_present = True
                            elif (part in recorded) != (dst_present and not src_present):
                                raise SessionStoreResetError("attempt_inconsistent")
                            if dst_present:
                                dst_info = (
                                    _assert_owned_dir(dst_fd, part, uid, gid)
                                    if part == "sessions"
                                    else _assert_owned_regular(dst_fd, part, uid, gid)
                                )
                                if dst_info.st_dev != home_info.st_dev:
                                    raise SessionStoreResetError("cross_filesystem")
                                if _part_identity(dst_info) != identities[part]:
                                    raise SessionStoreResetError("attempt_inconsistent")
                            if src_present:
                                info = (
                                    _assert_owned_dir(home_fd, part, uid, gid)
                                    if part == "sessions"
                                    else _assert_owned_regular(home_fd, part, uid, gid)
                                )
                                if info.st_dev != home_info.st_dev:
                                    raise SessionStoreResetError("cross_filesystem")
                                if (
                                    _part_identity(info) != identities[part]
                                    and not (part == "state.db" and fresh_state_present)
                                ):
                                    raise SessionStoreResetError("attempt_inconsistent")
                                if part == "sessions":
                                    part_fd = _open_dir_at(home_fd, part)
                                    try:
                                        session_entries = _walk_owned_tree(part_fd, uid, gid)
                                        if _session_tree_sha256(part_fd, uid, gid) != plan[
                                            "session_tree_sha256"
                                        ]:
                                            raise SessionStoreResetError("attempt_inconsistent")
                                    finally:
                                        os.close(part_fd)
                                    sessions_present = True

                        state = Path(home) / "state.db"
                        if _foreign_holders_present(state, names):
                            raise SessionStoreResetError("live_holder")
                        try:
                            from hermes_cli.sqlite_safe_read import offline_file_access
                        except ImportError:
                            access = contextlib.nullcontext()
                        else:
                            access = (
                                offline_file_access(state, what="reset session store")
                                if "state.db" in names
                                else contextlib.nullcontext()
                            )
                        with access, _lock_sqlite_family(home_fd, names):
                            # The initial plan was made before holder
                            # exclusion.  Re-check every remaining source
                            # after locks are held, then again immediately
                            # before each individual rename below.
                            _invoke_hook(failure_hook, "locks-held")
                            for part in parts:
                                if part not in recorded:
                                    _revalidate_source_before_rename(
                                        home_fd,
                                        part,
                                        identities[part],
                                        uid,
                                        gid,
                                        home_info.st_dev,
                                        plan["session_tree_sha256"]
                                        if part == "sessions"
                                        else None,
                                    )
                            _write_phase(attempt_fd, attempt, "quarantining", uid, gid)
                            for part in parts:
                                if part in recorded:
                                    continue
                                _revalidate_source_before_rename(
                                    home_fd,
                                    part,
                                    identities[part],
                                    uid,
                                    gid,
                                    home_info.st_dev,
                                    plan["session_tree_sha256"]
                                    if part == "sessions"
                                    else None,
                                )
                                dst_fd = attempt_fd if part == "sessions" else sqlite_fd
                                os.rename(part, part, src_dir_fd=home_fd, dst_dir_fd=dst_fd)
                                moved = True
                                _fsync_dir(dst_fd)
                                _fsync_dir(home_fd)
                                _write_part_record(
                                    attempt_fd, attempt, part, identities[part], uid, gid
                                )
                                _invoke_hook(failure_hook, f"moved:{part}")
                    finally:
                        os.close(sqlite_fd)
                    _write_phase(attempt_fd, attempt, "quarantined", uid, gid)
                    _invoke_hook(failure_hook, "quarantined")
                    _assert_active_scope_is_empty(home_fd)
                    validation = _validate_empty_store(Path(home) / "state.db", uid, gid)
                    _assert_active_scope_is_empty(home_fd)
                    _fsync_dir(home_fd)
                    _write_phase(attempt_fd, attempt, "completed", uid, gid)
                    _invoke_hook(failure_hook, "completed")
                finally:
                    pass
            finally:
                os.close(q_fd)

        deferred_signal = deferred.received if deferred is not None else None
        return _safe_report(
            attempt,
            "completed",
            quarantined=True,
            session_artifacts_present=sessions_present,
            session_artifact_entries=session_entries,
            validation=validation,
            attempt_manifest_sha256=manifest_digest,
            preserved_excluded_surfaces=(
                "moa-traces",
                "state-snapshots",
                "backups",
                "checkpoints",
                "service-archive",
            ),
            deferred_signal=deferred_signal,
        )
    except SessionStoreResetError as exc:
        if attempt_fd is not None and not terminal_reconciliation:
            try:
                _write_phase(attempt_fd, attempt, "failed", uid, gid)
            except SessionStoreResetError:
                pass
        return _safe_report(attempt, "failed", exc.code, quarantined=moved)
    except Exception:
        if attempt_fd is not None and not terminal_reconciliation:
            try:
                _write_phase(attempt_fd, attempt, "failed", uid, gid)
            except SessionStoreResetError:
                pass
        return _safe_report(attempt, "failed", "operational_failure", quarantined=moved)
    finally:
        if attempt_fd is not None:
            os.close(attempt_fd)
        os.close(home_fd)


def _remove_owned_tree_at(dir_fd: int, name: str, uid: int, gid: int) -> None:
    """Unlink only a fully validated owner-private attempt tree."""

    info = _lstat_at(dir_fd, name)
    if stat.S_ISREG(info.st_mode):
        _assert_owned_regular(dir_fd, name, uid, gid)
        os.unlink(name, dir_fd=dir_fd)
        return
    _assert_owned_dir(dir_fd, name, uid, gid)
    child_fd = _open_dir_at(dir_fd, name)
    try:
        for child in os.listdir(child_fd):
            _remove_owned_tree_at(child_fd, child, uid, gid)
        _fsync_dir(child_fd)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=dir_fd)


def finalize_session_store_reset(
    attempt: str, expected_manifest_sha256: Optional[str] = None
) -> dict[str, Any]:
    """Purge one *completed and revalidated* quarantine attempt.

    This is intentionally separate from reset: a failed or partial reset is
    forensic evidence and is never eligible for this explicit space-reclaim
    action.
    """

    if not _ATTEMPT_RE.fullmatch(attempt):
        return _safe_report(attempt, "failed", "attempt_invalid")
    if expected_manifest_sha256 is None:
        return _safe_report(attempt, "failed", "manifest_required")
    if not _SHA256_RE.fullmatch(expected_manifest_sha256):
        return _safe_report(attempt, "failed", "manifest_invalid")
    home = get_hermes_home()
    uid, gid = os.geteuid(), os.getegid()
    try:
        home_fd = os.open(
            home,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError:
        return _safe_report(attempt, "failed", "home_untrusted")
    deferred: Optional[_DeferredSignals] = None
    try:
        home_info = os.fstat(home_fd)
        if (
            not stat.S_ISDIR(home_info.st_mode)
            or home_info.st_uid != uid
            or home_info.st_gid != gid
            or home_info.st_mode & 0o022
        ):
            raise SessionStoreResetError("home_untrusted")
        with _DeferredSignals() as deferred, _reset_lock(home_fd, uid, gid):
            q_fd = _open_dir_at(home_fd, _QUARANTINE_DIR)
            try:
                _assert_owned_dir(q_fd, attempt, uid, gid)
                attempt_fd = _open_dir_at(q_fd, attempt)
                try:
                    _validation, manifest_digest, _parts = _validate_terminal_attempt(
                        attempt_fd,
                        home_fd,
                        attempt,
                        uid,
                        gid,
                        home_info.st_dev,
                    )
                    if manifest_digest != expected_manifest_sha256:
                        raise SessionStoreResetError("manifest_mismatch")
                finally:
                    os.close(attempt_fd)
                _remove_owned_tree_at(q_fd, attempt, uid, gid)
                _fsync_dir(q_fd)
            finally:
                os.close(q_fd)
        return _safe_report(
            attempt,
            "completed",
            finalized=True,
            attempt_manifest_sha256=expected_manifest_sha256,
            deferred_signal=deferred.received if deferred is not None else None,
        )
    except SessionStoreResetError as exc:
        return _safe_report(attempt, "failed", exc.code)
    except Exception:
        return _safe_report(attempt, "failed", "operational_failure")
    finally:
        os.close(home_fd)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quarantine and reset fixed active state.db and sessions only"
    )
    parser.add_argument("--attempt", required=True)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm this destructive-to-session-history operation",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Purge only this completed, revalidated quarantine attempt",
    )
    parser.add_argument(
        "--manifest-sha256",
        help="Required exact attempt manifest digest for --finalize",
    )
    parser.add_argument("--json", action="store_true", help="Emit one sanitized JSON result")
    return parser.parse_args(argv)


def report_exit_code(report: dict[str, Any]) -> int:
    """Preserve the first deferred signal after the bounded critical section."""

    if report.get("status") != "completed":
        return 1
    deferred = report.get("deferred_signal")
    if isinstance(deferred, int) and deferred in {
        signal.SIGHUP,
        signal.SIGINT,
        signal.SIGTERM,
    }:
        return 128 + deferred
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if not args.yes:
        report = _safe_report(args.attempt, "failed", "confirmation_required")
    else:
        report = (
            finalize_session_store_reset(args.attempt, args.manifest_sha256)
            if args.finalize
            else reset_session_store(args.attempt)
        )
    if args.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(
            "Session store reset "
            + ("completed." if report["status"] == "completed" else "refused.")
        )
    return report_exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
