"""Durable idempotency reservations for API server runs."""

import hmac
import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from gateway import hosted_room_fence as fence
from gateway.hosted_rooms_common import identifier
from gateway.platforms.api_server_run_scope import room_run_scope_key, validate_room_run_scope
from hermes_cli.sqlite_util import add_column_if_missing


# Keep the extracted store's log records on the API server logger.
logger = logging.getLogger("gateway.platforms.api_server")

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})
_FREEZES = "group_run_freezes"
_COMMANDS = "group_run_stop_commands"
_SCOPES = "group_run_scopes"
_TERMINAL_SQL = "'completed','failed','cancelled','interrupted'"
_KNOWN_SQL = _TERMINAL_SQL + ",'queued','running','waiting_for_approval','stopping'"
_STATUS_SQL = f"""CASE WHEN length(CAST(status_json AS BLOB)) <= 1048576 AND json_valid(status_json)
    THEN CASE WHEN json_extract(status_json,'$.status') IN ({_KNOWN_SQL})
         THEN json_extract(status_json,'$.status') ELSE 'unknown' END ELSE 'unknown' END"""


class GroupRunFreezeError(RuntimeError):
    code = "group_stop_storage_unavailable"
    status = 503
    message = "Durable group Stop storage is unavailable."

    def __init__(self):
        super().__init__(self.message)


class GroupRunFrozen(GroupRunFreezeError):
    code = "group_work_frozen"
    # Not 409: a group's home reads a conflicting admission as possibly accepted work.
    # A frozen scope proves the turn was never admitted here and never will be.
    status = 403
    message = "This participant's group work is permanently frozen."


class GroupStopScopeNotFound(GroupRunFreezeError):
    code = "group_stop_scope_not_found"
    status = 404
    message = "The known participant scope or Stop command was not found."


class GroupStopCommandConflict(GroupRunFreezeError):
    code = "group_stop_command_conflict"
    status = 409
    message = "The Stop command belongs to a different participant scope."


class GroupStopCapacity(GroupRunFreezeError):
    code = "group_stop_capacity"
    status = 507
    message = "Durable group Stop capacity is full."


class GroupStopStorageUnavailable(GroupRunFreezeError):
    pass


def _scope_key(scope: Any) -> str:
    if type(scope) is not str or re.fullmatch(r"[0-9a-f]{64}", scope) is None:
        raise ValueError("invalid internal Runs scope key")
    return scope


def _command_id(value: Any) -> str:
    checked = identifier(value, label="command_id", error=ValueError)
    if checked != value:
        raise ValueError("command_id must be an exact identifier string")
    return checked

_SELECT_BY_KEY = (
    "SELECT fingerprint, run_id, status_json, owner_pid, owner_started, updated_at "
    "FROM run_idempotency WHERE scope=? AND idempotency_key=?")
_EXTEND_RETENTION_BY_KEY = (
    "UPDATE run_idempotency SET retention_until=MAX(retention_until, ?) "
    "WHERE scope=? AND idempotency_key=? AND fingerprint=?")
_EXTEND_RETENTION_BY_RUN = (
    "UPDATE run_idempotency SET retention_until=MAX(retention_until, ?) "
    "WHERE scope=? AND run_id=?")
# Columns added after the first schema shipped; applied when missing.
_MIGRATIONS = {
    "owner_pid": "INTEGER NOT NULL DEFAULT 0",
    "owner_started": "INTEGER NOT NULL DEFAULT 0",
    "retention_until": "REAL NOT NULL DEFAULT 0",
    "acknowledged_at": "REAL"}


def _encode_status(status: Dict[str, Any]) -> str:
    return json.dumps(status, sort_keys=True, separators=(",", ":"))


def _record(run_id, status_json, owner_pid, owner_started, updated_at) -> dict[str, Any]:
    return {
        "run_id": run_id, "status": json.loads(status_json), "owner_pid": int(owner_pid or 0),
        "owner_started": int(owner_started or 0), "updated_at": float(updated_at or 0)}


def _outcome(row, fingerprint):
    """Classify a stored ``(scope, key)`` row against the caller's fingerprint."""
    return ("reused" if hmac.compare_digest(row[0], fingerprint) else "conflict"), _record(*row[1:])


class RunIdempotencyStore:
    """Durable, tenant-scoped reservations for ``POST /v1/runs``: a unique ``(scope, key)`` row
    inserted inside ``BEGIN IMMEDIATE`` so separate workers cannot both admit one request. Only
    fingerprints and public run status are stored — never request bodies or credentials."""

    RETENTION_SECONDS = 24 * 60 * 60
    ACKNOWLEDGED_RETENTION_SECONDS = 24 * 60 * 60
    MAX_GROUP_FREEZES = 512
    MAX_GROUP_STOP_COMMANDS = 4096
    GROUP_STOP_RUN_LIMIT = 128
    GROUP_SCOPE_LIST_LIMIT = 128

    @property
    def durable(self) -> bool:
        """Whether reservations survive this process."""
        return self._db_path is not None

    @property
    def path(self) -> Path | None:
        """Actual main SQLite file captured at open, not a caller's path hint."""
        return Path(self._db_path) if self._db_path else None

    def __init__(self, db_path: str = None):
        if db_path is None:
            try:
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "runs_idempotency.db")
            except Exception:
                db_path = ":memory:"
        self._db_path = None if db_path == ":memory:" else db_path
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        except Exception as exc:
            # Docker may create the container object before `docker run` fails to start it (e.g. exit code
            # 125 when the daemon isn't ready, or a timeout mid-pull). That orphan is left in "Created"
            # state — which the exited-only orphan reaper (reap_orphan_containers, status=exited) never
            # catches, so it leaks permanently. Remove it by its known name before re-raising. See #7439.
            logger.warning(
                "Run idempotency storage is unavailable; falling back to "
                "process memory, so replay will not survive a restart: %s", exc)
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._db_path = None
        main_file = next(row[2] for row in self._conn.execute("PRAGMA database_list") if row[1] == "main")
        self._db_path = str(Path(main_file).resolve()) if main_file else None
        from hermes_state_wal import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="runs_idempotency.db")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS run_idempotency (
                scope TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                run_id TEXT NOT NULL,
                status_json TEXT NOT NULL,
                owner_pid INTEGER NOT NULL DEFAULT 0,
                owner_started INTEGER NOT NULL DEFAULT 0,
                retention_until REAL NOT NULL DEFAULT 0,
                acknowledged_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (scope, idempotency_key)
            )"""
        )
        columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(run_idempotency)")}
        for column, ddl in _MIGRATIONS.items():
            if column not in columns:
                add_column_if_missing(self._conn, "run_idempotency", column, f"{column} {ddl}")
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS run_idempotency_run_id ON run_idempotency(run_id)")
        self._conn.commit()
        self._lock = threading.Lock()
        try:
            with self._immediate_txn():
                self._initialize_group_stop_locked()
                self._conn.commit()
        except sqlite3.Error:
            self._conn.close()
            raise GroupStopStorageUnavailable() from None
        self._tighten_permissions()

    def _tighten_permissions(self) -> None:
        for suffix in ("", "-wal", "-shm") if self._db_path else ():
            candidate = Path(self._db_path + suffix)
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug("Failed to restrict run idempotency store permissions", exc_info=True)

    @contextmanager
    def _immediate_txn(self):
        """Hold the lock inside ``BEGIN IMMEDIATE``; the body commits, errors roll back."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self._conn.rollback()
                raise

    def _initialize_group_stop_locked(self) -> None:
        self._conn.execute(f"""CREATE TABLE IF NOT EXISTS {_FREEZES} (
            scope TEXT PRIMARY KEY, identity_json TEXT NOT NULL, frozen_at REAL NOT NULL,
            retired_terminal INTEGER NOT NULL DEFAULT 0,
            missing_runs INTEGER NOT NULL DEFAULT 0)""")
        self._conn.execute(f"""CREATE TABLE IF NOT EXISTS {_COMMANDS} (
            command_id TEXT PRIMARY KEY, scope TEXT NOT NULL, created_at REAL NOT NULL)""")
        # The exact identity behind each hashed room scope, so the owner can list
        # what this participant holds while the group's home is unreachable.
        self._conn.execute(f"""CREATE TABLE IF NOT EXISTS {_SCOPES} (
            scope TEXT PRIMARY KEY, identity_json TEXT NOT NULL,
            first_admitted_at REAL NOT NULL, last_admitted_at REAL NOT NULL)""")
        # REPLACE can delete a different frozen row without firing DELETE triggers
        # on legacy connections. Fence victims of both unique keys before conflict resolution.
        frozen_victim = f"""SELECT 1 FROM run_idempotency AS victim
            JOIN {_FREEZES} AS frozen ON frozen.scope=victim.scope
            WHERE (victim.run_id=NEW.run_id OR
                (victim.scope=NEW.scope AND victim.idempotency_key=NEW.idempotency_key))"""
        self._conn.execute(f"""CREATE TRIGGER IF NOT EXISTS group_run_frozen_insert_v2
            BEFORE INSERT ON run_idempotency
            WHEN EXISTS (SELECT 1 FROM {_FREEZES} WHERE scope=NEW.scope)
              OR EXISTS ({frozen_victim})
            BEGIN SELECT RAISE(ABORT, 'group run scope frozen'); END""")
        self._conn.execute(f"""CREATE TRIGGER IF NOT EXISTS group_run_frozen_identity_v2
            BEFORE UPDATE ON run_idempotency
            WHEN (EXISTS (SELECT 1 FROM {_FREEZES} WHERE scope IN (OLD.scope,NEW.scope))
              AND (NEW.scope IS NOT OLD.scope OR NEW.idempotency_key IS NOT OLD.idempotency_key
                OR NEW.fingerprint IS NOT OLD.fingerprint OR NEW.run_id IS NOT OLD.run_id
                OR NEW.owner_pid IS NOT OLD.owner_pid OR NEW.owner_started IS NOT OLD.owner_started))
              OR EXISTS ({frozen_victim}
                AND NOT (victim.scope=OLD.scope AND victim.idempotency_key=OLD.idempotency_key))
            BEGIN SELECT RAISE(ABORT, 'group run scope frozen'); END""")
        terminal = f"({_STATUS_SQL.replace('status_json', 'OLD.status_json')}) IN ({_TERMINAL_SQL})"
        # Keep global/legacy pruning usable. A removed nonterminal receipt becomes
        # unresolved evidence, never an empty successful Stop after restart.
        self._conn.execute(f"""CREATE TRIGGER IF NOT EXISTS group_run_frozen_delete_v1
            AFTER DELETE ON run_idempotency
            WHEN EXISTS (SELECT 1 FROM {_FREEZES} WHERE scope=OLD.scope)
            BEGIN UPDATE {_FREEZES}
                SET retired_terminal=retired_terminal + CASE WHEN {terminal} THEN 1 ELSE 0 END,
                    missing_runs=missing_runs + CASE WHEN {terminal} THEN 0 ELSE 1 END
                WHERE scope=OLD.scope; END""")
        # Succession fences share this writer, so a fence and an admission are ordered.
        fence.initialize_fence_schema(self._conn)

    @contextmanager
    def _group_stop_txn(self):
        if not self.durable:
            raise GroupStopStorageUnavailable()
        try:
            with self._immediate_txn():
                yield
                self._conn.commit()
        except sqlite3.Error:
            raise GroupStopStorageUnavailable() from None

    def _frozen_row_locked(self, scope: str) -> bool:
        return self._conn.execute(f"SELECT 1 FROM {_FREEZES} WHERE scope=?", (scope,)).fetchone() is not None

    def _scope_frozen_locked(self, scope: str, identity: dict[str, Any] | None = None) -> bool:
        """Whether a freeze covers this scope: its own, or one of the same participant at an earlier
        or equal epoch of the same group, so a change of the group's host never reopens it."""
        if self._frozen_row_locked(scope):
            return True
        identity = self._scope_identity_locked(scope) if identity is None else identity
        return identity is not None and self._covering_freeze_locked(identity) is not None

    def _covering_freeze_locked(self, identity: dict[str, Any]) -> float | None:
        """When the owner first froze this participant (room, member and local target) at this
        epoch or an earlier one, whichever host the group had then; ``None`` if never."""
        row = self._conn.execute(f"""SELECT MIN(frozen_at) FROM {_FREEZES}
            WHERE json_extract(identity_json,'$.room_id')=? AND json_extract(identity_json,'$.member_id')=?
              AND json_extract(identity_json,'$.target_install_id')=?
              AND json_extract(identity_json,'$.target_profile')=?
              AND json_extract(identity_json,'$.authority_epoch')<=?""", (
            identity["room_id"], identity["member_id"], identity["target_install_id"], identity["target_profile"],
            int(identity["authority_epoch"]))).fetchone()
        return None if row is None or row[0] is None else float(row[0])

    def _scope_identity_locked(self, scope: str) -> dict[str, Any] | None:
        row = self._conn.execute(f"SELECT identity_json FROM {_SCOPES} WHERE scope=?", (scope,)).fetchone()
        try:
            identity = validate_room_run_scope(json.loads(row[0])) if row is not None else None
        except (TypeError, ValueError):
            return None
        return identity if identity is not None and room_run_scope_key(identity) == scope else None

    def _scope_fenced_locked(self, scope: str) -> bool:
        identity = self._scope_identity_locked(scope)
        return identity is not None and identity["authority_epoch"] <= fence.fenced_epoch_locked(
            self._conn, identity["room_id"])

    def is_scope_frozen(self, scope: str) -> bool:
        """Check an internally derived Runs scope, without reconstructing its identity."""
        scope = _scope_key(scope)
        if not self.durable:
            raise GroupStopStorageUnavailable()
        try:
            with self._lock:
                return self._scope_frozen_locked(scope)
        except sqlite3.Error:
            raise GroupStopStorageUnavailable() from None

    @contextmanager
    def group_control_open(self, scope: str, *, freeze: bool = True):
        """Serialize only a short in-memory decision with freeze; no store re-entry or I/O.

        A room epoch fenced for succession refuses this control outright: it passed to the
        promised successor. ``freeze=False`` checks only that fence.
        """
        scope = _scope_key(scope)
        with self._group_stop_txn():
            frozen = freeze and self._scope_frozen_locked(scope)
            if not frozen and self._scope_fenced_locked(scope):
                raise fence.RoomAuthorityFenced()
            yield not frozen

    def freeze_room_scope(self, identity: dict, command_id: str) -> dict[str, Any]:
        """Permanently fence one known participant scope in this durable Runs store."""
        identity = validate_room_run_scope(identity)
        scope, command_id = room_run_scope_key(identity), _command_id(command_id)
        with self._group_stop_txn():
            command = self._conn.execute(
                f"SELECT scope FROM {_COMMANDS} WHERE command_id=?", (command_id,),
            ).fetchone()
            if command is not None and command[0] != scope:
                raise GroupStopCommandConflict()
            frozen = self._frozen_row_locked(scope)
            if command is not None and not frozen:
                raise GroupStopStorageUnavailable()
            if not frozen:
                if self._conn.execute(
                    "SELECT 1 FROM run_idempotency WHERE scope=? LIMIT 1", (scope,),
                ).fetchone() is None:
                    raise GroupStopScopeNotFound()
                if self._conn.execute(f"SELECT COUNT(*) FROM {_FREEZES}").fetchone()[0] >= self.MAX_GROUP_FREEZES:
                    raise GroupStopCapacity()
            if command is None:
                if self._conn.execute(f"SELECT COUNT(*) FROM {_COMMANDS}").fetchone()[0] >= self.MAX_GROUP_STOP_COMMANDS:
                    raise GroupStopCapacity()
                now = time.time()
                if not frozen:
                    self._conn.execute(
                        f"INSERT INTO {_FREEZES}(scope,identity_json,frozen_at) VALUES (?,?,?)",
                        (scope, json.dumps(identity, sort_keys=True, separators=(",", ":")), now),
                    )
                self._conn.execute(
                    f"INSERT INTO {_COMMANDS}(command_id,scope,created_at) VALUES (?,?,?)",
                    (command_id, scope, now),
                )
            return self._room_stop_snapshot_locked(command_id)

    def room_stop_snapshot(self, command_id: str) -> dict[str, Any]:
        """Return live minimal records, not execution-completion or absence proof.

        Counts separate recognized nonterminal, terminal and unknown work. Deleted
        nonterminal records remain unknown/missing and make the summary truncated;
        retained terminal counts survive lawful pruning. Only bounded pending rows
        are returned, without status bodies. Foreign/zero owner IDs need caller review.
        """
        command_id = _command_id(command_id)
        with self._group_stop_txn():
            return self._room_stop_snapshot_locked(command_id)

    def _room_stop_snapshot_locked(self, command_id: str) -> dict[str, Any]:
        row = self._conn.execute(f"""SELECT f.scope,f.identity_json,f.frozen_at,f.retired_terminal,f.missing_runs
            FROM {_COMMANDS} c JOIN {_FREEZES} f ON f.scope=c.scope WHERE c.command_id=?""", (command_id,)).fetchone()
        if row is None:
            raise GroupStopScopeNotFound()
        scope, encoded_identity, frozen_at, retired_terminal, missing = row
        try:
            identity = validate_room_run_scope(json.loads(encoded_identity))
            if room_run_scope_key(identity) != scope:
                raise ValueError("stored scope differs")
        except (TypeError, ValueError):
            raise GroupStopStorageUnavailable() from None
        classified = f"""(SELECT CASE WHEN length(CAST(run_id AS BLOB))<=128 THEN run_id END AS run_id,
            {_STATUS_SQL} AS run_status,
            CASE WHEN typeof(owner_pid)='integer' AND owner_pid>=0 THEN owner_pid ELSE 0 END AS owner_pid,
            CASE WHEN typeof(owner_started)='integer' AND owner_started>=0 THEN owner_started ELSE 0 END AS owner_started,
            created_at
            FROM run_idempotency WHERE scope=?)"""
        total, terminal, unknown = self._conn.execute(f"""SELECT COUNT(*),
            COALESCE(SUM(run_status IN ({_TERMINAL_SQL})),0),COALESCE(SUM(run_status='unknown'),0)
            FROM {classified}""", (scope,)).fetchone()
        pending = self._conn.execute(f"""SELECT run_id,run_status,owner_pid,owner_started FROM {classified}
            WHERE run_status NOT IN ({_TERMINAL_SQL}) ORDER BY created_at,run_id LIMIT ?""",
            (scope, self.GROUP_STOP_RUN_LIMIT)).fetchall()
        nonterminal, runs = total - terminal - unknown, []
        for run_id, status, owner_pid, owner_started in pending:
            try:
                checked_id = identifier(run_id, label="run_id", error=ValueError)
                if checked_id != run_id:
                    raise ValueError("noncanonical run identity")
            except ValueError:
                if status != "unknown":
                    nonterminal, unknown = nonterminal - 1, unknown + 1
                continue
            runs.append({
                "run_id": run_id, "status": status,
                "owner_pid": owner_pid if type(owner_pid) is int and owner_pid >= 0 else 0,
                "owner_started": owner_started if type(owner_started) is int and owner_started >= 0 else 0,
            })
        return {
            "command_id": command_id, "identity": identity, "scope": scope, "frozen_at": frozen_at,
            "runs": runs, "counts": {"total": total + retired_terminal + missing,
                "terminal": terminal + retired_terminal, "nonterminal": nonterminal, "unknown": unknown + missing},
            "truncated": len(runs) < total - terminal + missing, "missing_runs": missing,
        }

    def list_room_scopes(self, *, target_install_id: str, target_profile: str = "default") -> dict[str, Any]:
        """List the room scopes one local target holds here, most recently admitted first.

        A scope is listed while it still has a run record or a freeze, because only
        those can be stopped. Exact identities and counts only, never status bodies.
        Bounded; an unreadable identity is skipped and reported as truncation.
        """
        with self._group_stop_txn():
            rows = self._conn.execute(f"""WITH known AS (
                    SELECT scope,identity_json,first_admitted_at,last_admitted_at FROM {_SCOPES}
                    UNION ALL SELECT scope,identity_json,frozen_at,frozen_at FROM {_FREEZES}
                    WHERE scope NOT IN (SELECT scope FROM {_SCOPES}))
                SELECT k.scope,k.identity_json,k.first_admitted_at,k.last_admitted_at,
                    f.frozen_at,COALESCE(f.retired_terminal,0),COALESCE(f.missing_runs,0)
                FROM known k LEFT JOIN {_FREEZES} f ON f.scope=k.scope
                WHERE json_valid(k.identity_json)
                  AND json_extract(k.identity_json,'$.target_install_id')=?
                  AND json_extract(k.identity_json,'$.target_profile')=?
                  AND (f.scope IS NOT NULL OR EXISTS (SELECT 1 FROM run_idempotency r WHERE r.scope=k.scope))
                ORDER BY k.last_admitted_at DESC,k.scope LIMIT ?""",
                (target_install_id, target_profile, self.GROUP_SCOPE_LIST_LIMIT + 1)).fetchall()
            truncated = len(rows) > self.GROUP_SCOPE_LIST_LIMIT
            participants = []
            for scope, encoded, first_at, last_at, frozen_at, retired, missing in rows[:self.GROUP_SCOPE_LIST_LIMIT]:
                try:
                    identity = validate_room_run_scope(json.loads(encoded))
                    if room_run_scope_key(identity) != scope:
                        raise ValueError("stored scope differs")
                except (TypeError, ValueError):
                    truncated = True
                    continue
                total, terminal, unknown = self._conn.execute(f"""SELECT COUNT(*),
                    COALESCE(SUM(run_status IN ({_TERMINAL_SQL})),0),COALESCE(SUM(run_status='unknown'),0)
                    FROM (SELECT {_STATUS_SQL} AS run_status FROM run_idempotency WHERE scope=?)""",
                    (scope,)).fetchone()
                participants.append({
                    "identity": identity, "first_admitted_at": first_at, "last_admitted_at": last_at,
                    "frozen_at": frozen_at if frozen_at is not None else self._covering_freeze_locked(identity),
                    "counts": {
                        "total": total + retired + missing, "terminal": terminal + retired,
                        "nonterminal": total - terminal - unknown, "unknown": unknown + missing},
                })
            return {"participants": participants, "truncated": truncated}

    def reserve(self, scope: str, key: str, fingerprint: str, run_id: str, status: Dict[str, Any], *,
                owner_pid: int = 0, owner_started: int = 0, retention_until: float = 0,
                identity: dict | None = None):
        """Atomically reserve a key; return ``(outcome, stored_record)``.

        ``identity`` is the exact room scope behind ``scope``. A new run records it
        in the same transaction, so the owner can list the scopes this store holds.
        """
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        encoded = _encode_status(status)
        if identity is not None:
            identity = validate_room_run_scope(identity)
            if room_run_scope_key(identity) != scope:
                raise ValueError("room scope identity does not match its Runs scope")
        with self._immediate_txn():
            self._prune_stale_terminal_locked(now)
            row = self._conn.execute(_SELECT_BY_KEY, (scope, key)).fetchone()
            if row is not None:
                if retention_until:
                    self._conn.execute(_EXTEND_RETENTION_BY_KEY, (retention_until, scope, key, fingerprint))
                self._conn.commit()
                return _outcome(row, fingerprint)
            if self._scope_frozen_locked(scope, identity):
                raise GroupRunFrozen()
            if identity is not None and identity["authority_epoch"] <= fence.fenced_epoch_locked(
                    self._conn, identity["room_id"]):
                raise fence.RoomAuthorityFenced()
            self._conn.execute(
                "INSERT INTO run_idempotency("
                "scope,idempotency_key,fingerprint,run_id,status_json,"
                "owner_pid,owner_started,retention_until,created_at,updated_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?)",
                (scope, key, fingerprint, run_id, encoded, int(owner_pid or 0), int(owner_started or 0),
                 retention_until, now, now))
            if identity is not None:
                self._conn.execute(
                    f"""INSERT INTO {_SCOPES}(scope,identity_json,first_admitted_at,last_admitted_at)
                        VALUES (?,?,?,?) ON CONFLICT(scope) DO UPDATE SET last_admitted_at=excluded.last_admitted_at""",
                    (scope, json.dumps(identity, sort_keys=True, separators=(",", ":")), now, now))
            self._conn.commit()
            return "created", _record(run_id, encoded, owner_pid, owner_started, now) | {"status": status}

    def lookup(self, scope: str, key: str, fingerprint: str, *, retention_until: float = 0):
        """Return ``missing``, ``reused`` or ``conflict`` without reserving."""
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        with self._immediate_txn():
            if retention_until:
                self._conn.execute(_EXTEND_RETENTION_BY_KEY, (retention_until, scope, key, fingerprint))
            self._prune_stale_terminal_locked(now)
            row = self._conn.execute(_SELECT_BY_KEY, (scope, key)).fetchone()
            self._conn.commit()
        return ("missing", None) if row is None else _outcome(row, fingerprint)

    def _prune_stale_terminal_locked(self, now: float) -> None:
        """Prune aged replay records only once their stored run is terminal (caller holds the
        lock + transaction): a long or disconnected room turn may outlive the retention window."""
        stale = self._conn.execute(
            """SELECT scope, idempotency_key, status_json
                 FROM run_idempotency
                WHERE acknowledged_at <= ?
                   OR (retention_until > 0 AND retention_until <= ?)
                   OR (retention_until <= 0 AND updated_at < ?)""",
            (now - self.ACKNOWLEDGED_RETENTION_SECONDS, now, now - self.RETENTION_SECONDS),
        ).fetchall()
        pruned = False
        for stale_scope, stale_key, stale_status in stale:
            try:
                terminal = json.loads(stale_status).get("status") in TERMINAL_STATUSES
            except Exception:
                terminal = False
            if terminal:
                self._conn.execute(
                    "DELETE FROM run_idempotency WHERE scope=? AND idempotency_key=?", (stale_scope, stale_key))
                pruned = True
        if pruned:
            # Without a run record or a freeze, a scope can no longer be stopped or listed.
            self._conn.execute(f"""DELETE FROM {_SCOPES} WHERE scope NOT IN (SELECT scope FROM run_idempotency)
                AND scope NOT IN (SELECT scope FROM {_FREEZES})""")

    def successor_run_scope(self, run_id: str, *, successor: dict) -> str | None:
        """The scope of one existing room run whose Status and Stop passed to ``successor``.

        ``successor`` is the caller's verified room scope. It must name the same room, member and
        local target as the run, and be the live promise here: the promised candidate at the
        promised epoch, above the fence that covers the run's own epoch. Anything else is ``None``.
        """
        successor = validate_room_run_scope(successor)
        if not self.durable:
            return None
        with self._lock:
            row = self._conn.execute("SELECT scope FROM run_idempotency WHERE run_id=?", (run_id,)).fetchone()
            identity = self._scope_identity_locked(row[0]) if row is not None else None
            if identity is None or any(identity[key] != successor[key] for key in (
                    "room_id", "member_id", "target_install_id", "target_profile")):
                return None
            room_id = identity["room_id"]
            if identity["authority_epoch"] > fence.fenced_epoch_locked(self._conn, room_id):
                return None
            controls = fence.successor_controls_locked(
                self._conn, room_id, candidate_install_id=successor["authority_gateway_id"],
                epoch=successor["authority_epoch"])
        return row[0] if controls else None

    def status_for_run(self, scope: str, run_id: str, *, retention_until: float = 0) -> dict[str, Any] | None:
        """Load one durable run status inside its authenticated scope."""
        retention_until = max(0.0, float(retention_until or 0))
        with self._lock:
            if retention_until:
                self._conn.execute(_EXTEND_RETENTION_BY_RUN, (retention_until, scope, run_id))
                self._conn.commit()
            row = self._conn.execute(
                "SELECT status_json, owner_pid, owner_started, updated_at "
                "FROM run_idempotency WHERE scope=? AND run_id=?",
                (scope, run_id)).fetchone()
        if row is None:
            return None
        return {k: v for k, v in _record(None, *row).items() if k != "run_id"}

    def extend_retention(self, scope: str, run_id: str, until: float) -> bool:
        """Persist the latest verified recovery horizon for an active grant."""
        checked_until = max(0.0, float(until or 0))
        if not checked_until:
            return False
        with self._lock:
            changed = self._conn.execute(_EXTEND_RETENTION_BY_RUN, (checked_until, scope, run_id)).rowcount
            self._conn.commit()
        return changed == 1

    def owns_run(self, scope: str, run_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM run_idempotency WHERE scope=? AND run_id=?", (scope, run_id)).fetchone()
        return row is not None

    def update_status(self, run_id: str, status: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE run_idempotency SET status_json=?, updated_at=? WHERE run_id=?",
                (_encode_status(status), time.time(), run_id))
            self._conn.commit()

    def forget(self, scope: str, key: str) -> None:
        """Release a reservation whose run was refused before it existed."""
        with self._lock:
            self._conn.execute("DELETE FROM run_idempotency WHERE scope=? AND idempotency_key=?", (scope, key))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
