"""Cross-profile ownership checks for durable session ids.

A session id is globally unique within a Hermes home even though each profile has its own
SQLite database. The advisory lock serializes the inventory check with the write transaction
performed by callers, so two profiles cannot both observe an unused id and claim it.
"""

from __future__ import annotations

import os
import hashlib
import sqlite3
import stat
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from persistence.core.errors import PersistenceError


class SessionLineageError(PersistenceError):
    """Base class for fail-closed cross-store ownership errors."""

    code = "session_lineage_error"


class SessionLineageInventoryError(SessionLineageError):
    """The store inventory could not be proven complete and readable."""

    code = "session_lineage_unverifiable"


class SessionLineageAmbiguousError(SessionLineageError):
    """Several copies exist but no unambiguous owner can be established."""

    code = "session_lineage_ambiguous"


class SessionIdOwnedElsewhereError(SessionLineageError):
    """A different profile store is the unique owner of the requested session id."""

    code = "session_id_owned_elsewhere"


class FreshSessionCollision(SessionLineageError):
    """A generated fresh identity already belongs to persisted state."""

    code = "fresh_session_collision"


@dataclass(frozen=True)
class SessionStoreRow:
    store_path: Path
    ended_at: object
    end_reason: str | None

    @property
    def is_live(self) -> bool:
        return self.ended_at is None


@dataclass(frozen=True)
class AdoptionOwnership:
    """Store-bound lifecycle evidence loaded from a validated durable transfer."""

    donor_path: Path
    target_path: Path
    states: dict[str, tuple[bool, str | None, bool, str | None]]

    def permits(self, session_id, source_path, target_path, source, local) -> bool:
        expected = self.states.get(session_id)
        if expected is None or not _same_file(self.donor_path, source_path) or not _same_file(self.target_path, target_path):
            return False
        source_state = (source.is_live, source.end_reason)
        return (source_state in {expected[:2], (False, "adopted_by_profile")}
                and (local.is_live, local.end_reason) == expected[2:])


def _explicit_transfer_allowed(rows, source_path, current_path, session_id, adoption):
    if _same_file(source_path, current_path):
        return False
    source = [row for row in rows if _same_file(row.store_path, source_path)]
    local = [row for row in rows if _same_file(row.store_path, current_path)]
    if len(rows) == 1 and len(source) == 1:
        return True
    if len(rows) != 2 or len(source) != 1 or len(local) != 1:
        return False
    if source[0].is_live and (local[0].is_live or local[0].end_reason == "adopted_by_profile"):
        return True
    return isinstance(adoption, AdoptionOwnership) and adoption.permits(
        session_id, source_path, current_path, source[0], local[0])


@dataclass(frozen=True)
class SessionLineageSnapshot:
    current_path: Path
    rows_by_id: dict[str, tuple[SessionStoreRow, ...]]

    def require_transfer_owner(self, session_id, donor_path, adoption) -> None:
        rows = self.rows_by_id.get(session_id, ())
        if not _explicit_transfer_allowed(rows, donor_path, self.current_path, session_id, adoption):
            raise SessionLineageAmbiguousError(
                f"Cannot verify both sides of the declared transfer for session {session_id!r}")

def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


def _validate_lineage_parent(path: Path, info, *, protected_by=None) -> None:
    """Require namespace parents to be controlled by this user or root on POSIX.

    Group-write is acceptable only when a verified private parent makes this path
    unreachable to other users. Other-write is always refused.
    """
    if os.name == "nt":
        return
    if info.st_mode & stat.S_IWOTH:
        raise SessionLineageInventoryError(f"Unsafe permissions on lineage namespace parent: {path}")
    if info.st_mode & stat.S_IWGRP:
        protected = (
            protected_by is not None
            and stat.S_ISDIR(protected_by.st_mode)
            and stat.S_IMODE(protected_by.st_mode) & 0o077 == 0
            and (not hasattr(os, "geteuid") or protected_by.st_uid in (0, os.geteuid()))
        )
        if not protected:
            raise SessionLineageInventoryError(f"Unsafe permissions on lineage namespace parent: {path}")
    if hasattr(os, "geteuid") and info.st_uid not in (0, os.geteuid()):
        raise SessionLineageInventoryError(f"Untrusted owner of lineage namespace parent: {path}")


def _lineage_lock_directory(root: Path, *, create: bool) -> Path:
    """Return the private shared lock directory inside the excluded profiles tree."""
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise SessionLineageInventoryError(f"Cannot inspect Hermes home {root}: {exc}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise SessionLineageInventoryError(f"Unsafe Hermes home for lineage locks: {root}")
    try:
        parent_stat = root.parent.lstat()
    except OSError:
        parent_stat = None
    private_parent = (
        parent_stat is not None and stat.S_ISDIR(parent_stat.st_mode)
        and stat.S_IMODE(parent_stat.st_mode) & 0o077 == 0
        and (not hasattr(os, "geteuid") or parent_stat.st_uid in (0, os.geteuid()))
    )
    _validate_lineage_parent(root, root_stat, protected_by=parent_stat if private_parent else None)

    profiles_root = root / "profiles"
    if create:
        try:
            profiles_root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise SessionLineageInventoryError(
                f"Cannot create profile metadata root {profiles_root}: {exc}") from exc
    try:
        profiles_stat = profiles_root.lstat()
    except OSError as exc:
        raise SessionLineageInventoryError(
            f"Cannot inspect profile metadata root {profiles_root}: {exc}") from exc
    if stat.S_ISLNK(profiles_stat.st_mode) or not stat.S_ISDIR(profiles_stat.st_mode):
        raise SessionLineageInventoryError(f"Unsafe profiles directory: {profiles_root}")
    profile_protector = (
        root_stat if stat.S_IMODE(root_stat.st_mode) & 0o077 == 0
        else parent_stat if private_parent else None
    )
    _validate_lineage_parent(profiles_root, profiles_stat, protected_by=profile_protector)

    lock_root = profiles_root / ".session-lineage-locks"
    if create:
        try:
            lock_root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise SessionLineageInventoryError(
                f"Cannot create session-lineage lock directory {lock_root}: {exc}") from exc
    try:
        lock_stat = lock_root.lstat()
    except OSError as exc:
        raise SessionLineageInventoryError(
            f"Cannot inspect session-lineage lock directory {lock_root}: {exc}") from exc
    if stat.S_ISLNK(lock_stat.st_mode) or not stat.S_ISDIR(lock_stat.st_mode):
        raise SessionLineageInventoryError(f"Unsafe session-lineage lock directory: {lock_root}")
    if os.name != "nt":
        if stat.S_IMODE(lock_stat.st_mode) & 0o077:
            raise SessionLineageInventoryError(
                f"Unsafe permissions on session-lineage lock directory: {lock_root}")
        if hasattr(os, "geteuid") and lock_stat.st_uid != os.geteuid():
            raise SessionLineageInventoryError(
                f"Session-lineage lock directory is not owned by this user: {lock_root}")
    return lock_root


def _prepare_advisory_parent(path: Path, create: bool) -> None:
    if path.parent.name == ".session-lineage-locks" and path.parent.parent.name == "profiles":
        _lineage_lock_directory(path.parent.parent.parent, create=create)
    elif create:
        path.parent.mkdir(parents=True, exist_ok=True)


def _validate_advisory_descriptor(path: Path, fd: int, before):
    opened = os.fstat(fd)
    if not stat.S_ISREG(opened.st_mode):
        raise SessionLineageInventoryError(f"Lineage lock is not a regular file: {path}")
    if before is not None and not os.path.samestat(before, opened):
        raise SessionLineageInventoryError(f"Lineage lock changed while opening: {path}")
    if os.name != "nt":
        if stat.S_IMODE(opened.st_mode) & 0o077:
            raise SessionLineageInventoryError(f"Unsafe permissions on lineage lock file: {path}")
        if hasattr(os, "geteuid") and opened.st_uid != os.geteuid():
            raise SessionLineageInventoryError(f"Lineage lock file is not owned by this user: {path}")
    return opened


@contextmanager
def _advisory_lock_file(path: Path, timeout_s: float, *, create: bool = True) -> Iterator[int]:
    """Acquire an advisory lock; every opened descriptor closes on every exit path."""
    try:
        _prepare_advisory_parent(path, create)
        try:
            before = path.lstat()
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise SessionLineageInventoryError(f"Unsafe lineage lock path: {path}")
        except FileNotFoundError as exc:
            if not create:
                raise SessionLineageInventoryError("Advisory lock disappeared during inventory") from exc
            before = None
        flags = (os.O_CREAT if create else 0) | os.O_RDWR | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            opened = _validate_advisory_descriptor(path, fd, before)
            if os.name == "nt" and opened.st_size == 0:
                os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
            deadline = time.monotonic() + timeout_s
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if time.monotonic() >= deadline:
                            raise SessionLineageInventoryError("Timed out acquiring session-lineage lock") from exc
                        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                try:
                    yield fd
                finally:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError as exc:
                        if time.monotonic() >= deadline:
                            raise SessionLineageInventoryError("Timed out acquiring session-lineage lock") from exc
                        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                try:
                    yield fd
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    except SessionLineageError:
        raise
    except OSError as exc:
        raise SessionLineageInventoryError(f"Cannot acquire session-lineage lock {path}: {exc}") from exc


@dataclass
class _ProcessLockEntry:
    lock: threading.RLock = field(default_factory=threading.RLock)
    users: int = 0


_PROCESS_LOCKS: dict[str, _ProcessLockEntry] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS = threading.local()


def _drop_process_lock_reference(key: str, entry: _ProcessLockEntry) -> None:
    """Discard idle per-session locks without racing waiters that already hold a reference."""
    with _PROCESS_LOCKS_GUARD:
        entry.users -= 1
        if entry.users == 0 and _PROCESS_LOCKS.get(key) is entry:
            _PROCESS_LOCKS.pop(key, None)


@contextmanager
def _lock_file(path: Path, timeout_s: float) -> Iterator[int]:
    """Re-entrant in-process wrapper around the cross-process advisory lock.

    A profile transfer holds each session lock across import and donor retirement; the
    nested import guard must reuse the same descriptor rather than block on its own lock.
    """
    deadline = time.monotonic() + timeout_s
    key = f"{os.getpid()}:{path.resolve(strict=False)}"
    with _PROCESS_LOCKS_GUARD:
        entry = _PROCESS_LOCKS.get(key)
        if entry is None:
            entry = _ProcessLockEntry()
            _PROCESS_LOCKS[key] = entry
        entry.users += 1
    try:
        if not entry.lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise SessionLineageInventoryError("Timed out acquiring session-lineage lock")
        try:
            active = getattr(_THREAD_LOCKS, "active", None)
            if active is None:
                active = {}
                _THREAD_LOCKS.active = active
            held_fd = active.get(key)
            if held_fd is not None:
                yield held_fd
                return
            with _advisory_lock_file(path, max(0.0, deadline - time.monotonic())) as fd:
                active[key] = fd
                try:
                    yield fd
                finally:
                    active.pop(key, None)
        finally:
            entry.lock.release()
    finally:
        _drop_process_lock_reference(key, entry)


def _managed_root_and_current(db_path: str | Path) -> tuple[Path, Path] | None:
    """Return the home root/current store, or None for standalone non-Hermes databases."""
    from persistence.sqlite.conversation._store.sessions import _session_store_install_root

    try:
        root = _session_store_install_root()
        raw = Path(db_path).expanduser().absolute()
        try:
            raw_stat = raw.lstat()
            if stat.S_ISLNK(raw_stat.st_mode):
                raise SessionLineageInventoryError(f"Session database is a symlink: {raw}")
        except FileNotFoundError:
            pass
        try:
            raw.relative_to(root)
        except ValueError:
            resolved = raw.resolve(strict=False)
            try:
                resolved.relative_to(root)
            except ValueError:
                return None
            raw = resolved
        # A store file itself may not be a symlink. Parent symlinks inside the home are
        # rejected during inventory; resolve only after establishing the lexical home.
        try:
            st = raw.lstat()
            if stat.S_ISLNK(st.st_mode):
                raise SessionLineageInventoryError(f"Session database is a symlink: {raw}")
        except FileNotFoundError:
            pass
        return root, raw.resolve(strict=False)
    except SessionLineageError:
        raise
    except Exception as exc:
        raise SessionLineageInventoryError(f"Cannot resolve Hermes state-store root: {exc}") from exc


def _inventory_session_columns(execute, path: Path, deadline: float) -> set[str]:
    """Wait for a sibling's startup DDL without treating an unfinished store as empty."""
    while True:
        # Exhaust each probe so no read cursor holds up the sibling's schema commit.
        table = execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'"
        ).fetchall()
        columns = (
            {row["name"] for row in execute("PRAGMA table_info(sessions)").fetchall()}
            if table else set()
        )
        if {"id", "ended_at"}.issubset(columns):
            return columns
        reason = "has incomplete sessions schema" if table else "lacks sessions table"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SessionLineageInventoryError(f"Session store {reason}: {path}")
        time.sleep(min(0.05, remaining))


def _rows_for_ids(
    path: Path, session_ids: tuple[str, ...], timeout_s: float = 2.0, *,
    deadline: float | None = None,
) -> dict[str, SessionStoreRow]:
    if not session_ids:
        return {}
    if deadline is None:
        deadline = time.monotonic() + max(0.0, timeout_s)
    try:
        uri = path.resolve(strict=True).as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=0.0)
        conn.row_factory = sqlite3.Row

        def execute(sql, parameters=()):
            remaining = max(0.0, deadline - time.monotonic())
            conn.execute(f"PRAGMA busy_timeout = {max(0, int(remaining * 1000))}")
            return conn.execute(sql, parameters)

        try:
            execute("PRAGMA query_only = ON")
            columns = _inventory_session_columns(execute, path, deadline)
            def col(name: str, fallback: str = "NULL") -> str:
                return name if name in columns else fallback
            wanted = ", ".join((
                "id", "ended_at", f"{col('end_reason')} AS end_reason",
            ))
            rows: dict[str, SessionStoreRow] = {}
            for start in range(0, len(session_ids), 500):
                chunk = session_ids[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                result = execute(
                    f"SELECT {wanted} FROM sessions WHERE id IN ({placeholders})", chunk
                ).fetchall()
                for row in result:
                    rows[str(row["id"])] = SessionStoreRow(
                        store_path=path,
                        ended_at=row["ended_at"],
                        end_reason=row["end_reason"],
                    )
            return rows
        finally:
            conn.close()
    except SessionLineageError:
        raise
    except Exception as exc:
        raise SessionLineageInventoryError(f"Cannot read session store {path}: {exc}") from exc


# A bounded stripe table keeps import descriptor use independent of batch size.
# A stripe represents a store-qualified ID. Transfers acquire both stores'
# stripes together, in the same order every scoped writer uses.
_LINEAGE_LOCK_STRIPES = 32


def _lineage_lock_paths(root, stores, ids):
    separator = chr(0)
    stripes = {
        int.from_bytes(hashlib.sha256(f"{path}{separator}{sid}".encode()).digest()[:4], "big") % _LINEAGE_LOCK_STRIPES
        for path in stores for sid in ids
    }
    return [root / "profiles" / ".session-lineage-locks" / f"scoped-stripe-{i:02}.lock" for i in sorted(stripes)]


@contextmanager
def _lock_store_members(root, stores, ids, deadline):
    _lineage_lock_directory(root, create=True)
    lock_paths = _lineage_lock_paths(root, stores, ids)
    held = {key for key in getattr(_THREAD_LOCKS, "active", ()) if key.startswith(f"{os.getpid()}:")}
    requested = {f"{os.getpid()}:{path.resolve(strict=False)}" for path in lock_paths}
    additional = requested - held
    if held and additional and min(additional) < max(held):
        raise SessionLineageInventoryError("Nested lineage lock order would invert; acquire the complete ID set first")
    with ExitStack() as locks:
        for path in lock_paths:
            locks.enter_context(_lock_file(path, max(0.0, deadline - time.monotonic())))
        yield


@contextmanager
def session_lineage_guard(db_path, session_ids, *, transfer_from, adoption=None, timeout_s=10.0):
    """Hold the explicitly selected donor and target through the complete transfer."""
    ids = tuple(dict.fromkeys(session_ids))
    target = _managed_root_and_current(db_path)
    donor = _managed_root_and_current(transfer_from)
    if target is None and donor is None:
        yield None
        return
    if target is None or donor is None or target[0] != donor[0] or _same_file(target[1], donor[1]):
        raise SessionLineageInventoryError("Transfer requires distinct stores in the same managed home")
    root, target_path = target
    donor_path = donor[1]
    deadline = time.monotonic() + timeout_s
    with _lock_store_members(root, (target_path, donor_path), ids, deadline):
        source = _rows_for_ids(donor_path, ids, deadline=deadline)
        destination = _rows_for_ids(target_path, ids, deadline=deadline)
        snapshot = SessionLineageSnapshot(target_path, {
            sid: tuple(rows[sid] for rows in (source, destination) if sid in rows) for sid in ids
        })
        for sid in ids:
            snapshot.require_transfer_owner(sid, donor_path, adoption)
        yield snapshot


DONOR_TRANSFER_PREFIX = "adoption.donor.v1:"


def _require_local_write(conn, ids):
    """Only actual transfer evidence can close a row to ordinary scoped writes."""
    if not ids:
        return
    has_journal = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='adoption_transfers'"
    ).fetchone() is not None
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ",".join("?" for _ in chunk)
        retired = conn.execute(
            f"SELECT id FROM sessions WHERE id IN ({marks}) AND end_reason='adopted_by_profile' LIMIT 1", chunk
        ).fetchone()
        retired_marker = conn.execute(
            f"SELECT key FROM state_meta WHERE key IN ({marks}) LIMIT 1",
            tuple(DONOR_TRANSFER_PREFIX + sid for sid in chunk),
        ).fetchone()
        if retired is not None or retired_marker is not None:
            sid = retired[0] if retired is not None else retired_marker[0][len(DONOR_TRANSFER_PREFIX):]
            raise SessionIdOwnedElsewhereError(f"Session {sid!r} was transferred to another profile store")
        if has_journal and conn.execute(
            f"SELECT 1 FROM adoption_transfers WHERE member_id IN ({marks}) AND status IN ('pending','copied') LIMIT 1",
            chunk,
        ).fetchone():
            raise SessionLineageInventoryError("Donor retirement is unsettled for the requested session")


@contextmanager
def session_lock_budget(holder, ids: tuple[str, ...], patience_s: float | None = None):
    """Hold a store's members until commit, leaving row checks to the writer."""
    from contextlib import nullcontext

    patience = holder._WRITE_PATIENCE_S if patience_s is None else patience_s
    deadline = time.monotonic() + patience
    managed = _managed_root_and_current(holder.db_path)
    locks = _lock_store_members(managed[0], (managed[1],), ids, deadline) if managed and ids else nullcontext()
    with locks:
        yield deadline


@contextmanager
def session_write_budget(holder, ids: str | Iterable[str], patience_s: float | None = None):
    """Serialize operations that check a scoped row before starting their write."""
    ids = (ids,) if isinstance(ids, str) else tuple(dict.fromkeys(sid for sid in ids if sid))
    with session_lock_budget(holder, ids, patience_s) as deadline:
        if ids:
            with holder._read_ctx_with_timeout(max(0.0, deadline - time.monotonic())) as conn:
                _require_local_write(conn, ids)
        yield max(0.0, deadline - time.monotonic())


def guarded_discovered_write(holder, discover, write, *, patience_s: float):
    """Recheck the discovered write set on the actual writer before any mutation."""
    deadline = time.monotonic() + patience_s
    ids: set[str] = set()
    expand = object()
    for _ in range(8):
        with holder._read_ctx_with_timeout(max(0.0, deadline - time.monotonic())) as conn:
            ids.update(discover(conn))
        if time.monotonic() >= deadline:
            break
        needed: set[str] = set()
        with session_lock_budget(holder, tuple(ids), max(0.0, deadline - time.monotonic())):
            def guarded(conn):
                needed.update(discover(conn))
                if needed - ids:
                    return expand
                _require_local_write(conn, tuple(needed))
                return write(conn)

            result = holder._execute_write(guarded, patience_s=max(0.0, deadline - time.monotonic()))
            if result is not expand:
                return result
        ids.update(needed)
    raise SessionLineageInventoryError("Session write set did not settle within its ownership budget")
