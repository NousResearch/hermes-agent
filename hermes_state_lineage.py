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
from typing import Any, Callable, Iterable, Iterator, TypeVar

from hermes_constants import get_default_hermes_root
from hermes_state_keys import profile_from_session_key

_WriteResult = TypeVar("_WriteResult")


class SessionLineageError(RuntimeError):
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


@dataclass(frozen=True)
class SessionStoreRow:
    store_path: Path
    profile: str
    ended_at: object
    end_reason: str | None
    profile_name: str | None
    session_key: str | None
    message_count: int | None

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

    def require_current_owner(self, session_id: str, transfer_from: str | Path | None = None,
                              adoption: AdoptionOwnership | None = None) -> None:
        rows = self.rows_by_id.get(session_id, ())
        if transfer_from is not None and _explicit_transfer_allowed(
                rows, Path(transfer_from), self.current_path, session_id, adoption):
            return
        owner = _select_owner(session_id, rows)
        if owner is None or _same_file(owner.store_path, self.current_path):
            return
        raise SessionIdOwnedElsewhereError(
            f"Session {session_id!r} is owned by profile store {owner.profile!r}; "
            f"refusing to create or resume it in {self.current_path}"
        )


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


def _select_owner(session_id: str, rows: tuple[SessionStoreRow, ...]) -> SessionStoreRow | None:
    if not rows:
        return None
    live = [row for row in rows if row.is_live]
    if len(live) == 1:
        return live[0]
    if len(live) > 1:
        stores = ", ".join(sorted(row.profile for row in live))
        raise SessionLineageAmbiguousError(
            f"Session {session_id!r} has multiple live owners ({stores}); refusing to guess"
        )
    if len(rows) == 1:
        return rows[0]

    # Adopted donor rows are explicitly retired by the lifecycle-aware profile migration path.
    # Prefer the sole non-donor row; all other multi-ended cases require operator adjudication.
    non_donors = [row for row in rows if row.end_reason != "adopted_by_profile"]
    if len(non_donors) == 1:
        return non_donors[0]

    # A namespaced gateway key is stronger evidence than a copied profile_name field.
    key_matches = [
        row for row in rows
        if row.profile is not None and profile_from_session_key(row.session_key) == row.profile
    ]
    if len(key_matches) == 1:
        return key_matches[0]

    profile_matches = [
        row for row in rows
        if row.profile is not None and row.profile_name == row.profile
    ]
    if len(profile_matches) == 1:
        return profile_matches[0]

    stores = ", ".join(sorted(row.profile for row in rows))
    raise SessionLineageAmbiguousError(
        f"Session {session_id!r} has multiple ended copies ({stores}) without a unique owner; "
        "refusing to guess"
    )


@contextmanager
def _advisory_lock_file(path: Path, timeout_s: float) -> Iterator[int]:
    """Acquire an advisory lock; every opened descriptor closes on every exit path."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            before = path.lstat()
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise SessionLineageInventoryError(f"Unsafe lineage lock path: {path}")
        except FileNotFoundError:
            before = None
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise SessionLineageInventoryError(f"Lineage lock is not a regular file: {path}")
            if before is not None and not os.path.samestat(before, opened):
                raise SessionLineageInventoryError(f"Lineage lock changed while opening: {path}")
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
                        time.sleep(0.05)
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
                        time.sleep(0.05)
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
    try:
        root = get_default_hermes_root().resolve(strict=False)
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


def physical_store_candidates(root: Path | None = None, current_path: Path | None = None) -> list[tuple[Path, str]]:
    """Every physical state.db the ownership fence inventories (incl. tombstoned profile dirs).

    Shared inventory for the fence and the repair planner (review finding 3): repair must
    see exactly the stores that can block a write, or its plan omits the blocking store.
    ``current_path`` mode resolves the managed root from a writer's database and keeps the
    mandatory-current-file checks. ``root``-only mode is inventory-only: no current/writer
    database is required, so read-only repair can scan an empty or profile-only home without
    one being created or demanded.
    """
    from hermes_constants import get_default_hermes_root
    if current_path is not None:
        managed = _managed_root_and_current(current_path)
        if managed is None:
            return []
        managed_root, current = managed
        if root is not None and root.expanduser().resolve(strict=False) != managed_root:
            raise SessionLineageInventoryError(
                f"Conflicting physical-store root {root} and current database {current_path}")
        return _store_candidates(managed_root, current)
    managed_root = (root if root is not None else get_default_hermes_root()).expanduser().resolve(strict=False)
    return _store_candidates(managed_root, None)


def _regular_file_or_none(path: Path, label: str) -> os.stat_result | None:
    """lstat a path that must be a plain regular file; None when absent, raise on unsafe."""
    try:
        st = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SessionLineageInventoryError(f"Cannot inspect {label} {path}: {exc}") from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise SessionLineageInventoryError(f"Unsafe {label}: {path}")
    return st


def _store_candidates(root: Path, current_path: Path | None) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    if _regular_file_or_none(root / "state.db", "session store") is not None:
        candidates.append((root / "state.db", "default"))

    profiles_root = root / "profiles"
    try:
        profiles_stat = profiles_root.lstat()
    except FileNotFoundError:
        profiles_stat = None
    except OSError as exc:
        raise SessionLineageInventoryError(f"Cannot inspect profiles directory {profiles_root}: {exc}") from exc
    if profiles_stat is not None:
        if stat.S_ISLNK(profiles_stat.st_mode) or not stat.S_ISDIR(profiles_stat.st_mode):
            raise SessionLineageInventoryError(f"Unsafe profiles directory: {profiles_root}")
        try:
            entries = sorted(profiles_root.iterdir(), key=lambda entry: entry.name)
        except OSError as exc:
            raise SessionLineageInventoryError(f"Cannot enumerate profiles directory {profiles_root}: {exc}") from exc
        for profile_dir in entries:
            try:
                entry_stat = profile_dir.lstat()
            except OSError as exc:
                raise SessionLineageInventoryError(f"Cannot inspect profile entry {profile_dir}: {exc}") from exc
            if stat.S_ISLNK(entry_stat.st_mode):
                raise SessionLineageInventoryError(f"Cannot prove profile inventory: symlinked entry {profile_dir}")
            if not stat.S_ISDIR(entry_stat.st_mode):
                continue
            db_path = profile_dir / "state.db"
            if _regular_file_or_none(db_path, "session store") is not None:
                candidates.append((db_path, profile_dir.name))

    if current_path is not None and not any(_same_file(path, current_path) for path, _ in candidates):
        if _regular_file_or_none(current_path, "current session database") is None:
            raise SessionLineageInventoryError(f"Current session database is missing: {current_path}")
        rel = current_path.relative_to(root)
        candidates.append((current_path, f"custom:{rel.as_posix()}"))
    return candidates


def _rows_for_ids(path: Path, profile: str, session_ids: tuple[str, ...]) -> dict[str, SessionStoreRow]:
    if not session_ids:
        return {}
    try:
        uri = path.resolve(strict=True).as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA query_only = ON")
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'"
            ).fetchone()
            if table is None:
                raise SessionLineageInventoryError(f"Session store lacks sessions table: {path}")
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
            if not {"id", "ended_at"}.issubset(columns):
                raise SessionLineageInventoryError(f"Session store has incomplete sessions schema: {path}")
            def col(name: str, fallback: str = "NULL") -> str:
                return name if name in columns else fallback
            wanted = ", ".join((
                "id", "ended_at", f"{col('end_reason')} AS end_reason",
                f"{col('profile_name')} AS profile_name", f"{col('session_key')} AS session_key",
                f"{col('message_count')} AS message_count",
            ))
            rows: dict[str, SessionStoreRow] = {}
            for start in range(0, len(session_ids), 500):
                chunk = session_ids[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                result = conn.execute(
                    f"SELECT {wanted} FROM sessions WHERE id IN ({placeholders})", chunk
                ).fetchall()
                for row in result:
                    rows[str(row["id"])] = SessionStoreRow(
                        store_path=path,
                        profile=profile,
                        ended_at=row["ended_at"],
                        end_reason=row["end_reason"],
                        profile_name=row["profile_name"],
                        session_key=row["session_key"],
                        message_count=row["message_count"],
                    )
            return rows
        finally:
            conn.close()
    except SessionLineageError:
        raise
    except Exception as exc:
        raise SessionLineageInventoryError(f"Cannot read session store {path}: {exc}") from exc


# Unreleased ownership-fence protocol: all cooperating writers use this fixed mapping.
# Collisions serialize writers; owner selection still uses the complete session IDs.
_LINEAGE_LOCK_STRIPES = 32


def _lineage_lock_paths(root: Path, ids: Iterable[str]) -> list[Path]:
    stripes = {int.from_bytes(hashlib.sha256(sid.encode("utf-8")).digest()[:4], "big")
               % _LINEAGE_LOCK_STRIPES for sid in ids}
    return [root / ".session-lineage-locks" / f"stripe-{index:02}.lock" for index in sorted(stripes)]


@contextmanager
def session_lineage_guard(
    db_path: str | Path,
    session_ids: str | Iterable[str],
    *,
    transfer_from: str | Path | None = None,
    adoption: AdoptionOwnership | None = None,
    timeout_s: float = 10.0,
) -> Iterator[SessionLineageSnapshot | None]:
    """Serialize ownership validation with a caller's DB write transaction.

    Callers must keep this context open until their SQLite write commits or rolls back. The
    context yields ``None`` for standalone DBs outside the active Hermes home.
    ``adoption`` carries store-bound lifecycle evidence for a saved transfer.
    Its additional ended-pair allowance is never granted to a plain ID set.
    """
    ids = (session_ids,) if isinstance(session_ids, str) else tuple(dict.fromkeys(str(s) for s in session_ids if s))
    if not ids:
        yield None
        return
    managed = _managed_root_and_current(db_path)
    if managed is None:
        yield None
        return
    root, current_path = managed
    lock_paths = _lineage_lock_paths(root, ids)
    held = {key for key in getattr(_THREAD_LOCKS, "active", ()) if key.startswith(f"{os.getpid()}:")}
    requested = {f"{os.getpid()}:{path.resolve(strict=False)}" for path in lock_paths}
    additional = requested - held
    if held and additional and min(additional) < max(held):
        raise SessionLineageInventoryError("Nested lineage lock order would invert; acquire the complete ID set first")
    deadline = time.monotonic() + timeout_s
    with ExitStack() as locks:
        for lock_path in lock_paths:
            locks.enter_context(_lock_file(lock_path, max(0.0, deadline - time.monotonic())))
        candidates = _store_candidates(root, current_path)
        rows_by_id: dict[str, list[SessionStoreRow]] = {session_id: [] for session_id in ids}
        for path, profile in candidates:
            rows = _rows_for_ids(path, profile, ids)
            for session_id, row in rows.items():
                rows_by_id[session_id].append(row)
        frozen = {session_id: tuple(rows) for session_id, rows in rows_by_id.items()}
        snapshot = SessionLineageSnapshot(current_path=current_path, rows_by_id=frozen)
        for session_id in ids:
            snapshot.require_current_owner(
                session_id, transfer_from=transfer_from, adoption=adoption)
        yield snapshot


def guarded_session_write(
    db: Any,
    session_ids: str | Iterable[str],
    write: Callable[[sqlite3.Connection], _WriteResult],
    *,
    transfer_from: str | Path | None = None,
) -> _WriteResult:
    """Keep the cross-store ownership lock held through the caller's SQLite commit."""
    with session_lineage_guard(db.db_path, session_ids, transfer_from=transfer_from, timeout_s=db._WRITE_PATIENCE_S):
        return db._execute_write(write)


def guarded_resume_lineage_write(
    db: Any, session_id: str, write: Callable[[sqlite3.Connection, list[str]], _WriteResult],
) -> _WriteResult:
    """Revalidate resolved display segments before writing, retaining their guards through commit."""
    guarded = set(db._resume_lineage_ids(session_id))
    deadline = time.monotonic() + db._WRITE_PATIENCE_S
    for _ in range(8):
        with session_lineage_guard(db.db_path, guarded,
                                   timeout_s=max(0.0, deadline - time.monotonic())):
            lineage = db._resume_lineage_ids(session_id)
            if not set(lineage) <= guarded:
                guarded.update(lineage)
                continue
            return db._execute_write(lambda conn: write(conn, lineage))
    raise SessionLineageInventoryError("Resume lineage changed during ownership discovery")
