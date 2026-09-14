"""Inactive fail-closed primitives for Kanban dispatch ownership and launch.

This module deliberately has no dispatcher, gateway, process-spawn, remote
execution, or live-policy imports.  It defines the lower-level identities and
durable state transitions needed before those call paths can be fenced.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import stat
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TextIO, cast

# Live boards mint task IDs as ``t_`` + ``secrets.token_hex(4)`` (8 hex
# chars); historical/test fixtures use ``token_hex(8)`` (16 chars).  Both
# are canonical live formats, so both lengths are accepted.  Any other
# length, non-hex characters, or a wrong prefix is rejected.
_TASK_ID_RE = re.compile(r"^t_[0-9a-f]{8}(?:[0-9a-f]{8})?$")
_CLAIM_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_LAUNCH_STATES = ("claimed_not_spawned", "spawn_intent", "spawned")


class BoardIdentityError(ValueError):
    """The configured board cannot be bound to a canonical database object."""


class DispatcherOwnerError(RuntimeError):
    """Exclusive dispatcher ownership could not be established or validated."""


class LaunchProtocolError(RuntimeError):
    """A durable launch transition failed closed."""


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _canonical_uuid(value: object) -> str | None:
    if type(value) is not str:
        return None
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return None
    return str(parsed)


def _valid_task_id(value: object) -> bool:
    return type(value) is str and _TASK_ID_RE.fullmatch(value) is not None


def _valid_claim_token(value: object) -> bool:
    return type(value) is str and _CLAIM_TOKEN_RE.fullmatch(value) is not None


@dataclass(frozen=True)
class BoardIdentity:
    """Canonical execution identity for one configured board database."""

    board_uuid: str
    device: int
    inode: int

    def __post_init__(self) -> None:
        canonical = _canonical_uuid(self.board_uuid)
        if canonical != self.board_uuid:
            raise BoardIdentityError("board UUID must be canonical")
        if not _positive_int(self.device) or not _positive_int(self.inode):
            raise BoardIdentityError("board device and inode must be positive integers")


def canonical_board_identity(
    database_path: str | os.PathLike[str], configured_board_uuid: object
) -> BoardIdentity:
    """Bind a configured UUID to the resolved regular database file.

    ``stat`` follows path aliases and symlinks, so aliases of the same object
    produce the same device/inode tuple.  Missing, non-regular, or malformed
    inputs raise instead of falling back to path-string identity.
    """

    board_uuid = _canonical_uuid(configured_board_uuid)
    if board_uuid is None:
        raise BoardIdentityError("invalid configured board UUID")
    try:
        metadata = os.stat(os.fspath(database_path), follow_symlinks=True)
    except (OSError, TypeError, ValueError) as exc:
        raise BoardIdentityError("board database cannot be resolved") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise BoardIdentityError("board database is not a regular file")
    return BoardIdentity(board_uuid, metadata.st_dev, metadata.st_ino)


def _canonical_home(home: str | os.PathLike[str]) -> Path:
    try:
        path = Path(home).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DispatcherOwnerError("dispatcher home cannot be resolved") from exc
    if not path.is_dir():
        raise DispatcherOwnerError("dispatcher home is not a directory")
    return path


def dispatcher_owner_lock_path(home: str | os.PathLike[str]) -> Path:
    """Return the single canonical profile-scoped dispatcher lock path."""

    return _canonical_home(home) / "run" / "kanban-dispatcher-owner.lock"


def _secure_owned_directory(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.geteuid()  # windows-footgun: ok
        and stat.S_IMODE(metadata.st_mode) & 0o022 == 0
    )


def _secure_owned_run_directory(metadata: os.stat_result) -> bool:
    return _secure_owned_directory(metadata) and stat.S_IMODE(metadata.st_mode) & 0o077 == 0


def _secure_owned_lock_file(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.geteuid()  # windows-footgun: ok
        and metadata.st_nlink == 1
        and stat.S_IMODE(metadata.st_mode) & 0o077 == 0
    )


class DispatcherOwnerLease:
    """Process-bound handle for one exclusive dispatcher owner generation."""

    __slots__ = (
        "_acquisition_pid",
        "_active",
        "_guard",
        "_handle",
        "_in_serialized_op",
        "_run_fd",
        "home",
        "lock_path",
        "owner_generation",
    )

    def __init__(
        self,
        *,
        home: Path,
        lock_path: Path,
        owner_generation: int,
        handle: TextIO,
        run_fd: int,
    ) -> None:
        self.home = home
        self.lock_path = lock_path
        self.owner_generation = owner_generation
        self._handle = handle
        self._run_fd = run_fd
        self._acquisition_pid = os.getpid()
        self._guard = threading.RLock()
        self._active = True
        self._in_serialized_op = 0

    def _expected_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "owner_generation": self.owner_generation,
            "pid": self._acquisition_pid,
            "canonical_home": str(self.home),
        }

    def validate(self) -> bool:
        """Verify this process still owns the original stamped lock object."""

        with self._guard:
            if os.getpid() != self._acquisition_pid:
                return False
            if not self._active or self._handle.closed:
                return False
            if not _positive_int(self.owner_generation):
                return False
            try:
                run_held = os.fstat(self._run_fd)
                run_current = os.stat(self.lock_path.parent, follow_symlinks=False)
                if not _secure_owned_run_directory(run_held):
                    return False
                if (run_held.st_dev, run_held.st_ino) != (
                    run_current.st_dev,
                    run_current.st_ino,
                ):
                    return False
                held = os.fstat(self._handle.fileno())
                current = os.stat(
                    self.lock_path.name,
                    dir_fd=self._run_fd,
                    follow_symlinks=False,
                )
                if not _secure_owned_lock_file(held) or not _secure_owned_lock_file(current):
                    return False
                if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
                    return False
                self._handle.seek(0)
                metadata = json.load(self._handle)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return False
            return type(metadata) is dict and metadata == self._expected_metadata()

    @contextlib.contextmanager
    def _serialized(self) -> Iterator[None]:
        """Prevent lease release mid-transition, including reentrant release.

        Cross-thread release blocks on the guard; same-thread reentrant
        release (RLock would admit it) is rejected by _in_serialized_op so
        a serialized kernel operation — through its COMMIT — cannot have its
        lease pulled out from under it on its own thread.
        """

        with self._guard:
            self._in_serialized_op += 1
            try:
                yield
            finally:
                self._in_serialized_op -= 1

    def release(self) -> None:
        """Release ownership exactly once; later validation fails closed."""

        with self._guard:
            if self._in_serialized_op > 0:
                raise DispatcherOwnerError(
                    "lease release is deferred: a serialized kernel "
                    "operation is active on this lease"
                )
            if not self._active:
                return
            self._active = False
            if os.getpid() == self._acquisition_pid:
                try:
                    import fcntl

                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError, ValueError):
                    pass
            self._handle.close()
            os.close(self._run_fd)

    def __enter__(self) -> Self:
        if not self.validate():
            raise DispatcherOwnerError("owner lease is not valid")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def acquire_dispatcher_owner(
    home: str | os.PathLike[str], owner_generation: object
) -> DispatcherOwnerLease:
    """Acquire the profile's non-blocking POSIX dispatcher owner lock.

    Every path component used after validation is held by descriptor. Locking,
    permissions, link identity, and metadata failures are fatal; no unlocked
    or config-only fallback exists.
    """

    if not _positive_int(owner_generation):
        raise DispatcherOwnerError("owner generation must be a positive integer")
    exact_owner_generation = cast(int, owner_generation)
    canonical_home = _canonical_home(home)

    required_flags = ("O_NOFOLLOW", "O_DIRECTORY")
    if any(getattr(os, name, None) is None for name in required_flags):
        raise DispatcherOwnerError("safe descriptor-relative locking is unsupported")
    nofollow = cast(int, os.O_NOFOLLOW)
    directory = cast(int, os.O_DIRECTORY)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)

    try:
        import fcntl
    except ImportError as exc:
        raise DispatcherOwnerError("POSIX dispatcher locking is unavailable") from exc

    home_fd: int | None = None
    run_fd: int | None = None
    descriptor: int | None = None
    handle: TextIO | None = None
    lock_acquired = False
    try:
        home_fd = os.open(
            canonical_home,
            os.O_RDONLY | directory | nofollow | close_on_exec,
        )
        home_held = os.fstat(home_fd)
        home_current = os.stat(canonical_home, follow_symlinks=False)
        if not _secure_owned_directory(home_held):
            raise DispatcherOwnerError("dispatcher home ownership or permissions are unsafe")
        if (home_held.st_dev, home_held.st_ino) != (
            home_current.st_dev,
            home_current.st_ino,
        ):
            raise DispatcherOwnerError("dispatcher home identity changed")

        try:
            os.mkdir("run", mode=0o700, dir_fd=home_fd)
        except FileExistsError:
            pass
        run_fd = os.open(
            "run",
            os.O_RDONLY | directory | nofollow | close_on_exec,
            dir_fd=home_fd,
        )
        run_metadata = os.fstat(run_fd)
        if not _secure_owned_run_directory(run_metadata):
            raise DispatcherOwnerError("dispatcher run directory permissions are unsafe")
        run_current = os.stat("run", dir_fd=home_fd, follow_symlinks=False)
        if (run_metadata.st_dev, run_metadata.st_ino) != (
            run_current.st_dev,
            run_current.st_ino,
        ):
            raise DispatcherOwnerError("dispatcher run directory identity changed")
        os.close(home_fd)
        home_fd = None

        lock_name = "kanban-dispatcher-owner.lock"
        descriptor = os.open(
            lock_name,
            os.O_RDWR | os.O_CREAT | nofollow | close_on_exec,
            0o600,
            dir_fd=run_fd,
        )
        lock_metadata = os.fstat(descriptor)
        if not _secure_owned_lock_file(lock_metadata):
            raise DispatcherOwnerError("dispatcher lock file identity or permissions are unsafe")
        handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        descriptor = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_acquired = True
        except (BlockingIOError, OSError) as exc:
            raise DispatcherOwnerError("dispatcher owner lock is contended") from exc
        if not _secure_owned_lock_file(os.fstat(handle.fileno())):
            raise DispatcherOwnerError("dispatcher lock file changed during acquisition")

        lock_path = canonical_home / "run" / lock_name
        lease = DispatcherOwnerLease(
            home=canonical_home,
            lock_path=lock_path,
            owner_generation=exact_owner_generation,
            handle=handle,
            run_fd=run_fd,
        )
        handle.seek(0)
        handle.truncate()
        json.dump(lease._expected_metadata(), handle, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
        if not lease.validate():
            raise DispatcherOwnerError("dispatcher owner lock validation failed")
        handle = None
        run_fd = None
        return lease
    except DispatcherOwnerError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise DispatcherOwnerError("dispatcher owner lock is unavailable") from exc
    finally:
        if handle is not None:
            if lock_acquired:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except (OSError, ValueError):
                    pass
            handle.close()
        elif descriptor is not None:
            os.close(descriptor)
        if run_fd is not None:
            os.close(run_fd)
        if home_fd is not None:
            os.close(home_fd)


@dataclass(frozen=True)
class LaunchRecord:
    board: BoardIdentity
    task_id: str
    run_generation: int
    dispatcher_owner_generation: int
    policy_generation: int
    route_generation: int
    claim_token: str
    state: str
    pid: int | None
    remote_execution_id: str | None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS kanban_launch_protocol (
    board_uuid TEXT NOT NULL CHECK(
        length(board_uuid) = 36
        AND board_uuid = lower(board_uuid)
        AND board_uuid NOT GLOB '*[^0-9a-f-]*'
        AND substr(board_uuid, 9, 1) = '-'
        AND substr(board_uuid, 14, 1) = '-'
        AND substr(board_uuid, 19, 1) = '-'
        AND substr(board_uuid, 24, 1) = '-'
    ),
    board_device INTEGER NOT NULL CHECK(board_device > 0),
    board_inode INTEGER NOT NULL CHECK(board_inode > 0),
    task_id TEXT NOT NULL CHECK(
        length(task_id) IN (10, 18)
        AND substr(task_id, 1, 2) = 't_'
        AND substr(task_id, 3) NOT GLOB '*[^0-9a-f]*'
    ),
    run_generation INTEGER NOT NULL CHECK(run_generation > 0),
    dispatcher_owner_generation INTEGER NOT NULL CHECK(dispatcher_owner_generation > 0),
    policy_generation INTEGER NOT NULL CHECK(policy_generation > 0),
    route_generation INTEGER NOT NULL CHECK(route_generation > 0),
    claim_token TEXT NOT NULL UNIQUE CHECK(
        length(claim_token) BETWEEN 16 AND 128
        AND claim_token NOT GLOB '*[^A-Za-z0-9_-]*'
    ),
    state TEXT NOT NULL CHECK(state IN ('claimed_not_spawned', 'spawn_intent', 'spawned')),
    pid INTEGER,
    remote_execution_id TEXT CHECK(
        remote_execution_id IS NULL
        OR (
            length(remote_execution_id) BETWEEN 1 AND 256
            AND remote_execution_id = trim(remote_execution_id)
        )
    ),
    PRIMARY KEY (board_uuid, board_device, board_inode, task_id, run_generation),
    CHECK(pid IS NULL OR pid > 0),
    CHECK(NOT (pid IS NOT NULL AND remote_execution_id IS NOT NULL)),
    CHECK(state = 'spawned' OR (pid IS NULL AND remote_execution_id IS NULL)),
    CHECK(state != 'spawned' OR (pid IS NOT NULL) != (remote_execution_id IS NOT NULL))
) STRICT
"""

_POLICY_POINTER_SCHEMA = """
CREATE TABLE IF NOT EXISTS kanban_policy_pointer (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    schema_version INTEGER NOT NULL CHECK(schema_version = 1),
    active_generation INTEGER NOT NULL CHECK(active_generation > 0)
) STRICT
"""


def _normalize_schema_sql(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().rstrip(";"))
    return normalized.replace("CREATE TABLE IF NOT EXISTS ", "CREATE TABLE ", 1)


def _require_exact_schema(
    conn: sqlite3.Connection, table_name: str, expected_sql: str
) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if (
        row is None
        or len(row) != 1
        or type(row[0]) is not str
        or _normalize_schema_sql(row[0]) != _normalize_schema_sql(expected_sql)
    ):
        raise LaunchProtocolError(f"{table_name} schema does not match version 1")
    extra_objects = conn.execute(
        "SELECT type, name FROM sqlite_master "
        "WHERE tbl_name = ? AND type IN ('index', 'trigger') AND sql IS NOT NULL",
        (table_name,),
    ).fetchall()
    if extra_objects:
        raise LaunchProtocolError(f"{table_name} has unrecognized schema objects")


def install_launch_schema(conn: sqlite3.Connection) -> None:
    """Install and verify the exact isolated launch and policy-pointer schemas."""

    try:
        with _immediate_transaction(conn):
            conn.execute(_SCHEMA)
            conn.execute(_POLICY_POINTER_SCHEMA)
            _require_exact_schema(conn, "kanban_launch_protocol", _SCHEMA)
            _require_exact_schema(conn, "kanban_policy_pointer", _POLICY_POINTER_SCHEMA)
    except LaunchProtocolError:
        raise
    except sqlite3.Error as exc:
        raise LaunchProtocolError("launch schema installation failed") from exc


def _require_connection_boundary(conn: sqlite3.Connection) -> None:
    if not isinstance(conn, sqlite3.Connection):
        raise LaunchProtocolError("invalid SQLite connection")
    if conn.in_transaction:
        raise LaunchProtocolError("active transaction is not allowed")


@contextlib.contextmanager
def _nullcontext():
    yield


@contextlib.contextmanager
def _immediate_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    _require_connection_boundary(conn)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise


def _validate_launch_identity(
    board: object,
    task_id: object,
    run_generation: object,
    dispatcher_owner_generation: object,
    policy_generation: object,
    route_generation: object,
    claim_token: object,
) -> tuple[BoardIdentity, str, int, int, int, int, str]:
    if not isinstance(board, BoardIdentity):
        raise LaunchProtocolError("invalid board identity")
    try:
        # Reconstruct to prevent malformed low-level object construction.
        board = BoardIdentity(board.board_uuid, board.device, board.inode)
    except (AttributeError, BoardIdentityError) as exc:
        raise LaunchProtocolError("invalid board identity") from exc
    if not _valid_task_id(task_id):
        raise LaunchProtocolError("invalid task identity")
    values = (run_generation, dispatcher_owner_generation, policy_generation, route_generation)
    if not all(_positive_int(value) for value in values):
        raise LaunchProtocolError("launch generations must be positive integers")
    if not _valid_claim_token(claim_token):
        raise LaunchProtocolError("invalid claim token")
    return (
        board,
        cast(str, task_id),
        cast(int, run_generation),
        cast(int, dispatcher_owner_generation),
        cast(int, policy_generation),
        cast(int, route_generation),
        cast(str, claim_token),
    )


def _identity_values(board: BoardIdentity, task_id: str, run_generation: int) -> tuple[object, ...]:
    return (board.board_uuid, board.device, board.inode, task_id, run_generation)


def _row_to_record(row: sqlite3.Row | tuple[object, ...]) -> LaunchRecord:
    values = tuple(row)
    if len(values) != 12:
        raise LaunchProtocolError("stored launch row has an invalid shape")
    try:
        board = BoardIdentity(values[0], values[1], values[2])  # type: ignore[arg-type]
    except (BoardIdentityError, TypeError) as exc:
        raise LaunchProtocolError("stored launch row has an invalid board identity") from exc
    identity = _validate_launch_identity(
        board,
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
    )
    board, task_id, run_gen, owner_gen, policy_gen, route_gen, claim_token = identity
    state_value, pid_value, remote_value = values[9], values[10], values[11]
    if type(state_value) is not str or state_value not in _LAUNCH_STATES:
        raise LaunchProtocolError("stored launch row has an invalid state")
    if pid_value is not None and not _positive_int(pid_value):
        raise LaunchProtocolError("stored launch row has an invalid PID")
    if remote_value is not None and (
        type(remote_value) is not str
        or not 0 < len(remote_value) <= 256
        or remote_value != remote_value.strip()
    ):
        raise LaunchProtocolError("stored launch row has an invalid remote identity")
    if state_value == "spawned":
        if (pid_value is not None) == (remote_value is not None):
            raise LaunchProtocolError("stored spawned row lacks one exact execution identity")
    elif pid_value is not None or remote_value is not None:
        raise LaunchProtocolError("stored pre-spawn row has an execution identity")
    return LaunchRecord(
        board=board,
        task_id=task_id,
        run_generation=run_gen,
        dispatcher_owner_generation=owner_gen,
        policy_generation=policy_gen,
        route_generation=route_gen,
        claim_token=claim_token,
        state=state_value,
        pid=cast(int | None, pid_value),
        remote_execution_id=cast(str | None, remote_value),
    )


_COLUMNS = """board_uuid, board_device, board_inode, task_id, run_generation,
dispatcher_owner_generation, policy_generation, route_generation, claim_token,
state, pid, remote_execution_id"""


def claim_not_spawned(
    conn: sqlite3.Connection,
    *,
    board: BoardIdentity,
    task_id: str,
    run_generation: int,
    dispatcher_owner_generation: int,
    policy_generation: int,
    route_generation: int,
    claim_token: str,
) -> LaunchRecord:
    """Atomically persist the complete pre-spawn identity."""

    identity = _validate_launch_identity(
        board,
        task_id,
        run_generation,
        dispatcher_owner_generation,
        policy_generation,
        route_generation,
        claim_token,
    )
    board, task_id, run_generation, owner_gen, policy_gen, route_gen, token = identity
    try:
        with _immediate_transaction(conn):
            conn.execute(
                """INSERT INTO kanban_launch_protocol (
                    board_uuid, board_device, board_inode, task_id, run_generation,
                    dispatcher_owner_generation, policy_generation, route_generation,
                    claim_token, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'claimed_not_spawned')""",
                (
                    board.board_uuid,
                    board.device,
                    board.inode,
                    task_id,
                    run_generation,
                    owner_gen,
                    policy_gen,
                    route_gen,
                    token,
                ),
            )
    except sqlite3.IntegrityError as exc:
        # Distinguish a genuinely conflicting/duplicate claim (unique index on
        # claim_token or the (board_uuid, task_id, run_generation) key) from any
        # other integrity violation: mislabeling a CHECK rejection as a duplicate
        # would mislead operators. SQLite's message text is unstable across
        # versions, so inspect post-hoc: a real duplicate leaves the existing
        # claim row in place; a CHECK failure leaves no trace of the insert.
        try:
            cur = conn.execute(
                "SELECT 1 FROM kanban_launch_protocol WHERE task_id = ?",
                (task_id,),
            )
            if cur.fetchone() is not None:
                raise LaunchProtocolError(
                    "duplicate or conflicting launch claim"
                ) from exc
        except sqlite3.Error:
            pass
        raise LaunchProtocolError("launch claim rejected by schema") from exc
    except sqlite3.Error as exc:
        raise LaunchProtocolError("launch claim persistence failed") from exc
    return LaunchRecord(
        board=board,
        task_id=task_id,
        run_generation=run_generation,
        dispatcher_owner_generation=owner_gen,
        policy_generation=policy_gen,
        route_generation=route_gen,
        claim_token=token,
        state="claimed_not_spawned",
        pid=None,
        remote_execution_id=None,
    )


def _cas_record(
    conn: sqlite3.Connection,
    *,
    board: BoardIdentity,
    task_id: str,
    run_generation: int,
    owner_generation: int,
    policy_generation: int,
    route_generation: int,
    claim_token: str,
    source_state: str,
    target_state: str,
    pid: int | None = None,
    remote_execution_id: str | None = None,
) -> None:
    cursor = conn.execute(
        """UPDATE kanban_launch_protocol
           SET state = ?, pid = ?, remote_execution_id = ?
         WHERE board_uuid = ? AND board_device = ? AND board_inode = ?
           AND task_id = ? AND run_generation = ?
           AND dispatcher_owner_generation = ? AND policy_generation = ?
           AND route_generation = ? AND claim_token = ? AND state = ?""",
        (
            target_state,
            pid,
            remote_execution_id,
            *_identity_values(board, task_id, run_generation),
            owner_generation,
            policy_generation,
            route_generation,
            claim_token,
            source_state,
        ),
    )
    if cursor.rowcount != 1:
        raise LaunchProtocolError("claim identity or transition state did not match")


def record_spawn_intent(
    conn: sqlite3.Connection,
    *,
    board: BoardIdentity,
    task_id: str,
    run_generation: int,
    dispatcher_owner_generation: int,
    policy_generation: int,
    route_generation: int,
    claim_token: str,
    owner_lease: DispatcherOwnerLease,
) -> LaunchRecord:
    """Linearize active policy and launch permission before process creation.

    The protected policy pointer is read from the same SQLite database after
    ``BEGIN IMMEDIATE``. A caller may create a process only after this
    transaction commits and returns ``spawn_intent``.
    """

    identity = _validate_launch_identity(
        board,
        task_id,
        run_generation,
        dispatcher_owner_generation,
        policy_generation,
        route_generation,
        claim_token,
    )
    board, task_id, run_generation, owner_gen, policy_gen, route_gen, token = identity
    if not isinstance(owner_lease, DispatcherOwnerLease):
        raise LaunchProtocolError("owner lease is invalid")

    try:
        with owner_lease._serialized():
            if owner_lease.owner_generation != owner_gen or not owner_lease.validate():
                raise LaunchProtocolError("owner lease is not held for this generation")
            with _immediate_transaction(conn):
                if not owner_lease.validate():
                    raise LaunchProtocolError("owner lease was lost before spawn intent")
                policy_row = conn.execute(
                    "SELECT schema_version, active_generation "
                    "FROM kanban_policy_pointer WHERE singleton = 1"
                ).fetchone()
                if (
                    policy_row is None
                    or len(policy_row) != 2
                    or type(policy_row[0]) is not int
                    or policy_row[0] != 1
                    or not _positive_int(policy_row[1])
                ):
                    raise LaunchProtocolError("protected policy pointer is invalid")
                active_policy_generation = policy_row[1]
                if active_policy_generation != policy_gen:
                    raise LaunchProtocolError("active policy generation does not match claim")
                if not owner_lease.validate():
                    raise LaunchProtocolError("owner lease was lost before spawn intent")
                _cas_record(
                    conn,
                    board=board,
                    task_id=task_id,
                    run_generation=run_generation,
                    owner_generation=owner_gen,
                    policy_generation=policy_gen,
                    route_generation=route_gen,
                    claim_token=token,
                    source_state="claimed_not_spawned",
                    target_state="spawn_intent",
                )
                if not owner_lease.validate():
                    raise LaunchProtocolError("owner lease was lost at spawn intent")
    except LaunchProtocolError:
        raise
    except sqlite3.Error as exc:
        raise LaunchProtocolError("spawn-intent persistence failed") from exc

    return LaunchRecord(
        board=board,
        task_id=task_id,
        run_generation=run_generation,
        dispatcher_owner_generation=owner_gen,
        policy_generation=policy_gen,
        route_generation=route_gen,
        claim_token=token,
        state="spawn_intent",
        pid=None,
        remote_execution_id=None,
    )


def record_spawned(
    conn: sqlite3.Connection,
    *,
    board: BoardIdentity,
    task_id: str,
    run_generation: int,
    dispatcher_owner_generation: int,
    policy_generation: int,
    route_generation: int,
    claim_token: str,
    pid: int | None = None,
    remote_execution_id: str | None = None,
    owner_lease: DispatcherOwnerLease | None = None,
) -> LaunchRecord:
    """CAS a durable intent to one concrete local or remote execution.

    When ``owner_lease`` is supplied, the lease is validated and the active
    policy pointer is re-read inside the same ``BEGIN IMMEDIATE`` transaction
    as the CAS: a spawn cannot be recorded under a lost owner lease or a
    revoked policy generation even if the caller checked both just before
    the call. Omitting the lease retains the legacy behavior (used only by
    tests of the pre-spawn transition itself).
    """

    identity = _validate_launch_identity(
        board,
        task_id,
        run_generation,
        dispatcher_owner_generation,
        policy_generation,
        route_generation,
        claim_token,
    )
    board, task_id, run_generation, owner_gen, policy_gen, route_gen, token = identity
    valid_pid = pid is not None and _positive_int(pid)
    valid_remote = (
        type(remote_execution_id) is str
        and 0 < len(remote_execution_id.strip()) <= 256
        and remote_execution_id == remote_execution_id.strip()
    )
    if valid_pid == valid_remote:
        raise LaunchProtocolError("exactly one valid execution identity is required")

    lease = None
    if owner_lease is not None:
        lease = _validate_owner_lease(owner_lease, owner_gen)

    try:
        with lease._serialized() if lease is not None else _nullcontext(), _immediate_transaction(conn):
            if lease is not None:
                if not lease.validate():
                    raise LaunchProtocolError(
                        "owner lease was lost before spawn record"
                    )
                active = _read_active_policy_generation(conn)
                if active != policy_gen:
                    raise LaunchProtocolError(
                        "active policy generation does not match claim"
                    )
            _cas_record(
                conn,
                board=board,
                task_id=task_id,
                run_generation=run_generation,
                owner_generation=owner_gen,
                policy_generation=policy_gen,
                route_generation=route_gen,
                claim_token=token,
                source_state="spawn_intent",
                target_state="spawned",
                pid=pid if valid_pid else None,
                remote_execution_id=remote_execution_id if valid_remote else None,
            )
            # Revalidate while rollback is still possible: a lease lost
            # between the pre-CAS check and the CAS must not commit spawned.
            if lease is not None and not lease.validate():
                raise LaunchProtocolError("owner lease was lost at spawn record")
    except LaunchProtocolError:
        raise
    except sqlite3.Error as exc:
        raise LaunchProtocolError("spawn identity persistence failed") from exc

    return LaunchRecord(
        board=board,
        task_id=task_id,
        run_generation=run_generation,
        dispatcher_owner_generation=owner_gen,
        policy_generation=policy_gen,
        route_generation=route_gen,
        claim_token=token,
        state="spawned",
        pid=cast(int, pid) if valid_pid else None,
        remote_execution_id=cast(str, remote_execution_id) if valid_remote else None,
    )


@dataclass(frozen=True)
class ReleaseOutcome:
    """Evidence of one retired pre-spawn launch claim."""

    board: BoardIdentity
    task_id: str
    run_generation: int
    dispatcher_owner_generation: int
    policy_generation: int
    route_generation: int
    claim_token: str
    previous_state: str


@dataclass(frozen=True)
class RunOutcome:
    """Evidence of one terminal handoff for a spawned run."""

    board: BoardIdentity
    task_id: str
    run_generation: int
    dispatcher_owner_generation: int
    policy_generation: int
    route_generation: int
    claim_token: str
    previous_state: str
    exit_status: int
    detail: str


def _read_active_policy_generation(conn: sqlite3.Connection) -> int:
    policy_row = conn.execute(
        "SELECT schema_version, active_generation "
        "FROM kanban_policy_pointer WHERE singleton = 1"
    ).fetchone()
    if (
        policy_row is None
        or len(policy_row) != 2
        or type(policy_row[0]) is not int
        or policy_row[0] != 1
        or not _positive_int(policy_row[1])
    ):
        raise LaunchProtocolError("protected policy pointer is invalid")
    return cast(int, policy_row[1])


def _validate_owner_lease(
    owner_lease: object, owner_generation: int
) -> DispatcherOwnerLease:
    if not isinstance(owner_lease, DispatcherOwnerLease):
        raise LaunchProtocolError("owner lease is invalid")
    if owner_lease.owner_generation != owner_generation or not owner_lease.validate():
        raise LaunchProtocolError("owner lease is not held for this generation")
    return owner_lease


def release_claim(
    conn: sqlite3.Connection,
    *,
    board: BoardIdentity,
    task_id: str,
    run_generation: int,
    dispatcher_owner_generation: int,
    policy_generation: int,
    route_generation: int,
    claim_token: str,
    owner_lease: DispatcherOwnerLease,
) -> ReleaseOutcome:
    """Atomically retire an unresolved pre-spawn claim (claim rollback).

    Requires a currently-held owner lease and a matching active policy
    generation; only ``claimed_not_spawned`` and ``spawn_intent`` rows may be
    released. Spawned rows require ``record_run_outcome`` instead.
    """

    identity = _validate_launch_identity(
        board,
        task_id,
        run_generation,
        dispatcher_owner_generation,
        policy_generation,
        route_generation,
        claim_token,
    )
    board, task_id, run_generation, owner_gen, policy_gen, route_gen, token = identity
    lease = _validate_owner_lease(owner_lease, owner_gen)

    try:
        with lease._serialized(), _immediate_transaction(conn):
            if not lease.validate():
                raise LaunchProtocolError("owner lease was lost before release")
            active = _read_active_policy_generation(conn)
            if active != policy_gen:
                raise LaunchProtocolError("active policy generation does not match claim")
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM kanban_launch_protocol "
                "WHERE board_uuid = ? AND board_device = ? AND board_inode = ? "
                "AND task_id = ? AND run_generation = ?",
                _identity_values(board, task_id, run_generation),
            ).fetchone()
            if row is None:
                raise LaunchProtocolError("claim identity or transition state did not match")
            record = _row_to_record(row)
            if record.dispatcher_owner_generation != owner_gen:
                raise LaunchProtocolError("claim identity or transition state did not match")
            if record.state == "spawned":
                raise LaunchProtocolError("spawned rows require a terminal run outcome")
            cursor = conn.execute(
                "DELETE FROM kanban_launch_protocol "
                "WHERE board_uuid = ? AND board_device = ? AND board_inode = ? "
                "AND task_id = ? AND run_generation = ? AND claim_token = ? "
                "AND dispatcher_owner_generation = ? AND policy_generation = ? "
                "AND route_generation = ? AND state = ?",
                (
                    *_identity_values(board, task_id, run_generation),
                    token,
                    owner_gen,
                    policy_gen,
                    route_gen,
                    record.state,
                ),
            )
            if cursor.rowcount != 1:
                raise LaunchProtocolError("claim identity or transition state did not match")
            if not lease.validate():
                raise LaunchProtocolError("owner lease was lost at claim release")
    except LaunchProtocolError:
        raise
    except sqlite3.Error as exc:
        raise LaunchProtocolError("claim release persistence failed") from exc

    return ReleaseOutcome(
        board=board,
        task_id=task_id,
        run_generation=run_generation,
        dispatcher_owner_generation=owner_gen,
        policy_generation=policy_gen,
        route_generation=route_gen,
        claim_token=token,
        previous_state=record.state,
    )


def _validate_exit_evidence(exit_status: object, detail: object) -> tuple[int, str]:
    if type(exit_status) is not int:
        raise LaunchProtocolError("exit status must be an exact integer")
    if not -2**31 <= exit_status < 2**31:
        raise LaunchProtocolError("exit status is out of bounds")
    if type(detail) is not str or len(detail) > 256 or detail != detail.strip():
        raise LaunchProtocolError("outcome detail must be bounded trimmed text")
    return exit_status, detail


def record_run_outcome(
    conn: sqlite3.Connection,
    *,
    board: BoardIdentity,
    task_id: str,
    run_generation: int,
    dispatcher_owner_generation: int,
    policy_generation: int,
    route_generation: int,
    claim_token: str,
    owner_lease: DispatcherOwnerLease,
    exit_status: int,
    detail: str = "",
) -> RunOutcome:
    """Atomically retire a spawned row with bounded terminal evidence."""

    identity = _validate_launch_identity(
        board,
        task_id,
        run_generation,
        dispatcher_owner_generation,
        policy_generation,
        route_generation,
        claim_token,
    )
    board, task_id, run_generation, owner_gen, policy_gen, route_gen, token = identity
    exact_exit, exact_detail = _validate_exit_evidence(exit_status, detail)
    lease = _validate_owner_lease(owner_lease, owner_gen)

    try:
        with lease._serialized(), _immediate_transaction(conn):
            if not lease.validate():
                raise LaunchProtocolError("owner lease was lost before run outcome")
            active = _read_active_policy_generation(conn)
            if active != policy_gen:
                raise LaunchProtocolError("active policy generation does not match claim")
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM kanban_launch_protocol "
                "WHERE board_uuid = ? AND board_device = ? AND board_inode = ? "
                "AND task_id = ? AND run_generation = ?",
                _identity_values(board, task_id, run_generation),
            ).fetchone()
            if row is None:
                raise LaunchProtocolError("claim identity or transition state did not match")
            record = _row_to_record(row)
            if record.state != "spawned":
                raise LaunchProtocolError("run outcome requires a spawned row")
            cursor = conn.execute(
                "DELETE FROM kanban_launch_protocol "
                "WHERE board_uuid = ? AND board_device = ? AND board_inode = ? "
                "AND task_id = ? AND run_generation = ? AND claim_token = ? "
                "AND dispatcher_owner_generation = ? AND policy_generation = ? "
                "AND route_generation = ? AND state = 'spawned'",
                (
                    *_identity_values(board, task_id, run_generation),
                    token,
                    owner_gen,
                    policy_gen,
                    route_gen,
                ),
            )
            if cursor.rowcount != 1:
                raise LaunchProtocolError("claim identity or transition state did not match")
            if not lease.validate():
                raise LaunchProtocolError("owner lease was lost at run outcome")
    except LaunchProtocolError:
        raise
    except sqlite3.Error as exc:
        raise LaunchProtocolError("run outcome persistence failed") from exc

    return RunOutcome(
        board=board,
        task_id=task_id,
        run_generation=run_generation,
        dispatcher_owner_generation=owner_gen,
        policy_generation=policy_gen,
        route_generation=route_gen,
        claim_token=token,
        previous_state=record.state,
        exit_status=exact_exit,
        detail=exact_detail,
    )


def get_launch_record(
    conn: sqlite3.Connection,
    board: BoardIdentity,
    task_id: str,
    run_generation: int,
) -> LaunchRecord | None:
    """Read one exact launch row without opening a write transaction."""

    _require_connection_boundary(conn)
    if not isinstance(board, BoardIdentity) or not _valid_task_id(task_id):
        raise LaunchProtocolError("invalid launch lookup identity")
    if not _positive_int(run_generation):
        raise LaunchProtocolError("invalid run generation")
    try:
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM kanban_launch_protocol "
            "WHERE board_uuid = ? AND board_device = ? AND board_inode = ? "
            "AND task_id = ? AND run_generation = ?",
            _identity_values(board, task_id, run_generation),
        ).fetchone()
    except sqlite3.Error as exc:
        raise LaunchProtocolError("launch lookup failed") from exc
    return None if row is None else _row_to_record(row)


def freeze_ack_blockers(conn: sqlite3.Connection) -> list[LaunchRecord]:
    """List unresolved pre-spawn rows in deterministic identity order."""

    _require_connection_boundary(conn)
    try:
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM kanban_launch_protocol "
            "WHERE state IN ('claimed_not_spawned', 'spawn_intent') "
            "ORDER BY board_uuid, board_device, board_inode, task_id, run_generation"
        ).fetchall()
    except sqlite3.Error as exc:
        raise LaunchProtocolError("freeze blocker enumeration failed") from exc
    return [_row_to_record(row) for row in rows]
