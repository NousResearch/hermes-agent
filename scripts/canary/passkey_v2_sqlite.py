#!/usr/bin/env python3
"""Crash-atomic SQLite authority and executor ledgers for passkey v2.

This is the only runtime-eligible passkey v2 state implementation.  The
dedicated owner-gate VM runs it under three distinct no-shell identities:
public web, authority, and privileged executor.  The authority database and
executor database live in separate 0700 directories and are owned by their
respective service identities.

Every state change is an INSERT inside ``BEGIN IMMEDIATE`` with
``synchronous=FULL``.  UPDATE and DELETE triggers make the ledgers logically
append-only.  Consumption and its journal entry commit in one transaction, so
there is no tombstone/journal crash window.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import stat
import threading
import uuid
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import passkey_v2_webauthn as webauthn
from scripts.canary import storage_growth_evidence as growth_evidence
from scripts.canary.passkey_v2_signer import ReceiptSigner


RUNTIME_ELIGIBLE = True
AUTHORITY_DB_SCHEMA = "muncho-passkey-v2-authority-sqlite.v1"
EXECUTOR_DB_SCHEMA = "muncho-passkey-v2-executor-sqlite.v1"
DATABASE_MODE = 0o600
DATABASE_DIRECTORY_MODE = 0o700
SQLITE_BUSY_TIMEOUT_MS = 5_000
SQLITE_BUSY_TIMEOUT_SECONDS = SQLITE_BUSY_TIMEOUT_MS / 1_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_REPEATED_EXECUTION_EVENT_KINDS = frozenset({
    "provider_pending",
    "attempt_failed",
    "reconciliation_observed",
})
_EXECUTION_EVENT_KINDS = frozenset({
    "opened",
    "resize_intent",
    "resize_complete",
    "post_resize_observation_required",
    "post_resize_observation_accepted",
    "stop_intent",
    "stop_complete",
    "post_stop_observation_required",
    "post_stop_observation_accepted",
    "start_intent",
    "start_complete",
    "post_start_observation_required",
    "post_start_observation_accepted",
    "postflight_complete",
    "completed",
}) | _REPEATED_EXECUTION_EVENT_KINDS
_ONLINE_EXECUTION_SEQUENCE = (
    "opened",
    "resize_intent",
    "resize_complete",
    "post_resize_observation_required",
    "post_resize_observation_accepted",
    "postflight_complete",
    "completed",
)
_REBOOT_EXECUTION_SEQUENCE = (
    "opened",
    "resize_intent",
    "resize_complete",
    "post_resize_observation_required",
    "post_resize_observation_accepted",
    "stop_intent",
    "stop_complete",
    "post_stop_observation_required",
    "post_stop_observation_accepted",
    "start_intent",
    "start_complete",
    "post_start_observation_required",
    "post_start_observation_accepted",
    "postflight_complete",
    "completed",
)


def _execution_sequence_is_legal(kinds: tuple[str, ...]) -> bool:
    core = tuple(
        kind for kind in kinds
        if kind not in _REPEATED_EXECUTION_EVENT_KINDS
    )
    if not core or core[0] != "opened":
        return False
    if "completed" in core and kinds[-1] != "completed":
        return False
    return any(core == expected[: len(core)] for expected in (
        _ONLINE_EXECUTION_SEQUENCE,
        _REBOOT_EXECUTION_SEQUENCE,
    ))


def _execution_event_id(
    *,
    transaction_id: str,
    authorization_request_id: str,
    event_kind: str,
    event_payload: Mapping[str, Any],
) -> str:
    return protocol.sha256_json({
        "schema": "muncho-passkey-v2-execution-event-id.v1",
        "transaction_id": transaction_id,
        "authorization_request_id": authorization_request_id,
        "event_kind": event_kind,
        "event_payload": dict(event_payload),
    })


def _deterministic_gce_request_id(transaction_id: str, stage: str) -> str:
    raw = bytearray(hashlib.sha256(protocol.canonical_json_bytes({
        "schema": "muncho-passkey-v2-gce-request-id.v1",
        "transaction_id": transaction_id,
        "stage": stage,
    })).digest()[:16])
    raw[6] = (raw[6] & 0x0F) | 0x50
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


def _deterministic_attempt_id(
    transaction_id: str, stage: str, anchor: str
) -> str:
    return protocol.sha256_json({
        "schema": "muncho-passkey-v2-stage-attempt-id.v1",
        "transaction_id": transaction_id,
        "stage": stage,
        "attempt_anchor_event_head_sha256": anchor,
    })


def _observation_collection_attempt_valid(
    value: Any,
    *,
    expected_context_sha256: str | None = None,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "context_sha256",
        "context_sequence",
        "collection_attempt_id",
        "transaction_id",
        "checkpoint",
        "prior_event_head_sha256",
        "release_sha",
        "plan_sha256",
        "issued_at_unix",
        "expires_at_unix",
        "attempt_sha256",
    }:
        return False
    context = {
        "schema": "muncho-storage-growth-collection-context.v1",
        "transaction_id": value.get("transaction_id"),
        "checkpoint": value.get("checkpoint"),
        "prior_event_head_sha256": value.get("prior_event_head_sha256"),
        "release_sha": value.get("release_sha"),
        "plan_sha256": value.get("plan_sha256"),
    }
    context_sha256 = protocol.sha256_json(context)
    identity = {
        "schema": "muncho-storage-growth-collection-attempt-id.v1",
        "context_sha256": context_sha256,
        "context_sequence": value.get("context_sequence"),
        "issued_at_unix": value.get("issued_at_unix"),
    }
    unsigned = {key: item for key, item in value.items() if key != "attempt_sha256"}
    return bool(
        value.get("schema")
        == "muncho-storage-growth-collection-attempt.v1"
        and _SHA256.fullmatch(str(value.get("transaction_id"))) is not None
        and value.get("checkpoint")
        in {"source", "post_resize", "post_stop", "post_start"}
        and _SHA256.fullmatch(
            str(value.get("prior_event_head_sha256"))
        )
        is not None
        and _REVISION.fullmatch(str(value.get("release_sha"))) is not None
        and _SHA256.fullmatch(str(value.get("plan_sha256"))) is not None
        and type(value.get("context_sequence")) is int
        and value["context_sequence"] >= 1
        and type(value.get("issued_at_unix")) is int
        and value["issued_at_unix"] >= 1
        and value.get("expires_at_unix")
        == value["issued_at_unix"]
        + growth_evidence.OBSERVATION_BUNDLE_TTL_SECONDS
        and value.get("context_sha256") == context_sha256
        and (
            expected_context_sha256 is None
            or context_sha256 == expected_context_sha256
        )
        and value.get("collection_attempt_id")
        == protocol.sha256_json(identity)
        and value.get("attempt_sha256") == protocol.sha256_json(unsigned)
    )


class PasskeyV2SqliteError(RuntimeError):
    """Stable SQLite boundary failure."""


class PasskeyV2SqliteDenied(PasskeyV2SqliteError):
    """The requested state transition or execution is not authorized."""


@dataclass(frozen=True)
class ConsumptionResult:
    disposition: str
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.disposition not in {"authorized_once", "receipt_replay"}:
            raise PasskeyV2SqliteError("passkey_v2_consumption_disposition_invalid")


@dataclass(frozen=True)
class ExecutionClaimResult:
    disposition: str
    intent: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.disposition not in {"claimed_once", "authorization_replay"}:
            raise PasskeyV2SqliteError("passkey_v2_execution_disposition_invalid")


def _validate_db_parent(path: Path, *, uid: int, gid: int) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or path.name in {
        "",
        ".",
        "..",
    }:
        raise PasskeyV2SqliteError("passkey_v2_database_path_invalid")
    try:
        parent = path.parent.lstat()
        resolved = path.parent.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise PasskeyV2SqliteError("passkey_v2_database_parent_invalid") from None
    if (
        resolved != path.parent
        or stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != uid
        or parent.st_gid != gid
        or stat.S_IMODE(parent.st_mode) != DATABASE_DIRECTORY_MODE
    ):
        raise PasskeyV2SqliteError("passkey_v2_database_parent_invalid")


def _validate_database_file(path: Path, *, uid: int, gid: int) -> os.stat_result:
    _validate_db_parent(path, uid=uid, gid=gid)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PasskeyV2SqliteError("passkey_v2_database_file_invalid") from None
    try:
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_uid != uid
            or opened.st_gid != gid
            or stat.S_IMODE(opened.st_mode) != DATABASE_MODE
            or opened.st_nlink != 1
            or opened.st_size < 1
        ):
            raise PasskeyV2SqliteError("passkey_v2_database_file_invalid")
        return opened
    finally:
        os.close(descriptor)


def _create_database_file(path: Path, *, uid: int, gid: int) -> None:
    _validate_db_parent(path, uid=uid, gid=gid)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    parent_fd = os.open(path.parent, directory_flags)
    descriptor: int | None = None
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, DATABASE_MODE, dir_fd=parent_fd)
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, DATABASE_MODE)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(parent_fd)
    except FileExistsError as exc:
        raise PasskeyV2SqliteError("passkey_v2_database_already_exists") from None
    except OSError as exc:
        raise PasskeyV2SqliteError("passkey_v2_database_create_failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _configure(connection: sqlite3.Connection) -> None:
    # Install the bounded busy handler before any PRAGMA that might need to
    # read the database header while another exact transaction is committing.
    connection.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA synchronous=FULL")


def _initialize_journal_mode(connection: sqlite3.Connection) -> None:
    """Set the persistent journal mode once, during exclusive bootstrap."""

    if connection.execute("PRAGMA journal_mode=DELETE").fetchone() != (
        "delete",
    ):
        raise PasskeyV2SqliteError("passkey_v2_database_journal_mode_invalid")


def _append_only_triggers(table: str) -> str:
    return f"""
    CREATE TRIGGER {table}_no_update
    BEFORE UPDATE ON {table}
    BEGIN SELECT RAISE(ABORT, 'append_only'); END;
    CREATE TRIGGER {table}_no_delete
    BEFORE DELETE ON {table}
    BEGIN SELECT RAISE(ABORT, 'append_only'); END;
    """


_AUTHORITY_SCHEMA_SQL = """
CREATE TABLE metadata (
    schema_name TEXT PRIMARY KEY NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at_unix INTEGER NOT NULL,
    sqlite_master_sha256 TEXT NOT NULL
) STRICT;
CREATE TABLE requests (
    request_id TEXT PRIMARY KEY NOT NULL,
    envelope_sha256 TEXT UNIQUE NOT NULL,
    document BLOB NOT NULL,
    created_at_unix INTEGER NOT NULL
) STRICT;
CREATE TABLE challenges (
    request_id TEXT PRIMARY KEY NOT NULL REFERENCES requests(request_id),
    challenge_id TEXT UNIQUE NOT NULL,
    challenge_record_sha256 TEXT UNIQUE NOT NULL,
    document BLOB NOT NULL,
    created_at_unix INTEGER NOT NULL
) STRICT;
CREATE TABLE grants (
    request_id TEXT PRIMARY KEY NOT NULL REFERENCES challenges(request_id),
    grant_id TEXT UNIQUE NOT NULL,
    grant_sha256 TEXT UNIQUE NOT NULL,
    document BLOB NOT NULL,
    granted_at_unix INTEGER NOT NULL
) STRICT;
CREATE TABLE credentials (
    credential_id_sha256 TEXT PRIMARY KEY NOT NULL,
    credential_record_sha256 TEXT UNIQUE NOT NULL,
    document BLOB NOT NULL,
    imported_at_unix INTEGER NOT NULL
) STRICT;
CREATE TABLE credential_uses (
    sequence INTEGER PRIMARY KEY NOT NULL,
    credential_id_sha256 TEXT NOT NULL REFERENCES credentials(credential_id_sha256),
    request_id TEXT UNIQUE NOT NULL REFERENCES challenges(request_id),
    verification_sha256 TEXT UNIQUE NOT NULL,
    sign_count INTEGER NOT NULL,
    document BLOB NOT NULL,
    verified_at_unix INTEGER NOT NULL
) STRICT;
CREATE TABLE consumptions (
    request_id TEXT PRIMARY KEY NOT NULL REFERENCES grants(request_id),
    consume_attempt_id TEXT UNIQUE NOT NULL,
    authorization_receipt_sha256 TEXT UNIQUE NOT NULL,
    receipt BLOB NOT NULL,
    consumed_at_unix INTEGER NOT NULL
) STRICT;
CREATE TABLE authorization_journal (
    sequence INTEGER PRIMARY KEY NOT NULL,
    prior_journal_head_sha256 TEXT NOT NULL,
    authorization_receipt_sha256 TEXT UNIQUE NOT NULL,
    journal_head_sha256 TEXT UNIQUE NOT NULL,
    request_id TEXT UNIQUE NOT NULL REFERENCES consumptions(request_id),
    consume_attempt_id TEXT UNIQUE NOT NULL,
    created_at_unix INTEGER NOT NULL
) STRICT;
""" + "".join(
    _append_only_triggers(table)
    for table in (
        "metadata",
        "requests",
        "challenges",
        "grants",
        "credentials",
        "credential_uses",
        "consumptions",
        "authorization_journal",
    )
)

_EXECUTOR_SCHEMA_SQL = """
CREATE TABLE metadata (
    schema_name TEXT PRIMARY KEY NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at_unix INTEGER NOT NULL,
    sqlite_master_sha256 TEXT NOT NULL
) STRICT;
CREATE TABLE execution_transactions (
    transaction_id TEXT PRIMARY KEY NOT NULL,
    initial_request_id TEXT UNIQUE NOT NULL,
    executor_release_sha TEXT NOT NULL,
    executor_plan_sha256 TEXT NOT NULL,
    source_preflight_sha256 TEXT NOT NULL,
    transaction_sha256 TEXT UNIQUE NOT NULL,
    document BLOB NOT NULL,
    opened_at_unix INTEGER NOT NULL
) STRICT;
CREATE TABLE execution_authorizations (
    request_id TEXT PRIMARY KEY NOT NULL,
    transaction_id TEXT NOT NULL REFERENCES execution_transactions(transaction_id),
    authorization_sequence INTEGER NOT NULL,
    consume_attempt_id TEXT UNIQUE NOT NULL,
    authorization_receipt_sha256 TEXT UNIQUE NOT NULL,
    action_envelope_sha256 TEXT UNIQUE NOT NULL,
    authorization_sha256 TEXT UNIQUE NOT NULL,
    authorization BLOB NOT NULL,
    action_envelope BLOB NOT NULL,
    authorization_receipt BLOB NOT NULL,
    challenge_record BLOB NOT NULL,
    grant_record BLOB NOT NULL,
    authorized_at_unix INTEGER NOT NULL,
    UNIQUE(transaction_id, authorization_sequence)
) STRICT;
CREATE TABLE execution_events (
    sequence INTEGER PRIMARY KEY NOT NULL,
    event_id TEXT UNIQUE NOT NULL,
    transaction_id TEXT NOT NULL REFERENCES execution_transactions(transaction_id),
    authorization_request_id TEXT NOT NULL REFERENCES execution_authorizations(request_id),
    event_kind TEXT NOT NULL,
    prior_event_head_sha256 TEXT NOT NULL,
    event_head_sha256 TEXT UNIQUE NOT NULL,
    event BLOB NOT NULL,
    created_at_unix INTEGER NOT NULL
) STRICT;
CREATE TABLE observation_collection_attempts (
    sequence INTEGER PRIMARY KEY NOT NULL,
    context_sha256 TEXT NOT NULL,
    context_sequence INTEGER NOT NULL,
    collection_attempt_id TEXT UNIQUE NOT NULL,
    transaction_id TEXT NOT NULL,
    checkpoint TEXT NOT NULL,
    prior_event_head_sha256 TEXT NOT NULL,
    release_sha TEXT NOT NULL,
    plan_sha256 TEXT NOT NULL,
    issued_at_unix INTEGER NOT NULL,
    expires_at_unix INTEGER NOT NULL,
    document BLOB NOT NULL,
    UNIQUE(context_sha256, context_sequence)
) STRICT;
""" + "".join(
    _append_only_triggers(table)
    for table in (
        "metadata",
        "execution_transactions",
        "execution_authorizations",
        "execution_events",
        "observation_collection_attempts",
    )
)


def _sqlite_master_sha256(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name,tbl_name"
    ).fetchall()
    projection = [
        {
            "type": str(row[0]),
            "name": str(row[1]),
            "table": str(row[2]),
            "sql": None if row[3] is None else str(row[3]),
        }
        for row in rows
    ]
    return protocol.sha256_json(projection)


@lru_cache(maxsize=2)
def _expected_schema_sha256(schema_name: str) -> str:
    if schema_name == AUTHORITY_DB_SCHEMA:
        schema_sql = _AUTHORITY_SCHEMA_SQL
    elif schema_name == EXECUTOR_DB_SCHEMA:
        schema_sql = _EXECUTOR_SCHEMA_SQL
    else:
        raise PasskeyV2SqliteError("passkey_v2_database_schema_unknown")
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        _configure(connection)
        connection.executescript("BEGIN IMMEDIATE;\n" + schema_sql)
        digest = _sqlite_master_sha256(connection)
        connection.rollback()
        return digest
    finally:
        connection.close()


def _bootstrap_database(
    path: Path,
    *,
    uid: int,
    gid: int,
    schema_name: str,
    schema_sql: str,
    now_unix: int,
    require_root: bool,
) -> None:
    if require_root and os.geteuid() != 0:  # windows-footgun: ok — Debian root boundary
        raise PasskeyV2SqliteError("passkey_v2_database_bootstrap_requires_root")
    if not isinstance(now_unix, int) or isinstance(now_unix, bool) or now_unix < 1:
        raise PasskeyV2SqliteError("passkey_v2_database_time_invalid")
    _create_database_file(path, uid=uid, gid=gid)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            path,
            timeout=SQLITE_BUSY_TIMEOUT_SECONDS,
            isolation_level=None,
        )
        _configure(connection)
        _initialize_journal_mode(connection)
        if not schema_name.replace("-", "").replace(".", "").isalnum():
            raise PasskeyV2SqliteError("passkey_v2_database_schema_name_invalid")
        connection.executescript("BEGIN IMMEDIATE;\n" + schema_sql)
        schema_sha256 = _sqlite_master_sha256(connection)
        connection.execute(
            "INSERT INTO metadata(schema_name,schema_version,created_at_unix,"
            "sqlite_master_sha256) VALUES(?,?,?,?)",
            (schema_name, 1, now_unix, schema_sha256),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(FULL)")
    except BaseException as exc:
        if connection is not None:
            connection.rollback()
        if isinstance(exc, (OSError, sqlite3.Error)):
            raise PasskeyV2SqliteError(
                "passkey_v2_database_bootstrap_failed"
            ) from None
        raise
    finally:
        if connection is not None:
            connection.close()
    installed = path.lstat()
    if (installed.st_uid, installed.st_gid) != (uid, gid):
        os.chown(path, uid, gid)
    if stat.S_IMODE(installed.st_mode) != DATABASE_MODE:
        os.chmod(path, DATABASE_MODE)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    _validate_database_file(path, uid=uid, gid=gid)


def bootstrap_authority_database(
    path: Path,
    *,
    authority_uid: int,
    authority_gid: int,
    now_unix: int,
    require_root: bool = True,
) -> None:
    _bootstrap_database(
        path,
        uid=authority_uid,
        gid=authority_gid,
        schema_name=AUTHORITY_DB_SCHEMA,
        schema_sql=_AUTHORITY_SCHEMA_SQL,
        now_unix=now_unix,
        require_root=require_root,
    )


def bootstrap_executor_database(
    path: Path,
    *,
    executor_uid: int,
    executor_gid: int,
    now_unix: int,
    require_root: bool = True,
) -> None:
    _bootstrap_database(
        path,
        uid=executor_uid,
        gid=executor_gid,
        schema_name=EXECUTOR_DB_SCHEMA,
        schema_sql=_EXECUTOR_SCHEMA_SQL,
        now_unix=now_unix,
        require_root=require_root,
    )


class _Database:
    def __init__(self, path: Path, *, uid: int, gid: int, schema_name: str) -> None:
        self.path = path
        self.uid = uid
        self.gid = gid
        self.schema_name = schema_name
        self._transaction_lock = threading.RLock()
        self.preflight()

    def _connect(self) -> sqlite3.Connection:
        before = _validate_database_file(self.path, uid=self.uid, gid=self.gid)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=SQLITE_BUSY_TIMEOUT_SECONDS,
                isolation_level=None,
                check_same_thread=False,
            )
            _configure(connection)
            if connection.execute("PRAGMA journal_mode").fetchone() != (
                "delete",
            ):
                raise PasskeyV2SqliteError(
                    "passkey_v2_database_journal_mode_invalid"
                )
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            raise PasskeyV2SqliteError("passkey_v2_database_open_failed") from None
        except PasskeyV2SqliteError:
            if connection is not None:
                connection.close()
            raise
        assert connection is not None
        after = self.path.lstat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            connection.close()
            raise PasskeyV2SqliteError("passkey_v2_database_changed")
        expected_schema = _expected_schema_sha256(self.schema_name)
        actual_schema = _sqlite_master_sha256(connection)
        metadata = connection.execute(
            "SELECT schema_name,schema_version,sqlite_master_sha256 FROM metadata"
        ).fetchall()
        if metadata != [(self.schema_name, 1, expected_schema)] or actual_schema != expected_schema:
            connection.close()
            raise PasskeyV2SqliteError("passkey_v2_database_schema_drift")
        return connection

    def preflight(self) -> Mapping[str, Any]:
        state = _validate_database_file(self.path, uid=self.uid, gid=self.gid)
        connection = self._connect() if hasattr(self, "path") else None
        assert connection is not None
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            synchronous = connection.execute("PRAGMA synchronous").fetchone()
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
            metadata = connection.execute(
                "SELECT schema_name,schema_version,sqlite_master_sha256 FROM metadata"
            ).fetchall()
            foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
            foreign_key_violations = connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
            if (
                integrity != ("ok",)
                or synchronous != (2,)
                or journal_mode != ("delete",)
                or foreign_keys != (1,)
                or foreign_key_violations
                or metadata
                != [(self.schema_name, 1, _expected_schema_sha256(self.schema_name))]
            ):
                raise PasskeyV2SqliteError("passkey_v2_database_preflight_failed")
        except sqlite3.Error as exc:
            raise PasskeyV2SqliteError("passkey_v2_database_preflight_failed") from None
        finally:
            connection.close()
        unsigned = {
            "schema": "muncho-passkey-v2-sqlite-preflight.v1",
            "ok": True,
            "database_schema": self.schema_name,
            "sqlite_master_sha256": _expected_schema_sha256(
                self.schema_name
            ),
            "database_device": state.st_dev,
            "database_inode": state.st_ino,
            "database_uid": state.st_uid,
            "database_gid": state.st_gid,
            "database_mode_octal": "0600",
            "journal_mode": "delete",
            "synchronous": "FULL",
            "foreign_keys": True,
            "logical_append_only": True,
        }
        return {**unsigned, "preflight_sha256": protocol.sha256_json(unsigned)}

    @staticmethod
    def _begin_immediate(connection: sqlite3.Connection) -> None:
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            error_code = getattr(exc, "sqlite_errorcode", None)
            primary_code = (
                error_code & 0xFF if type(error_code) is int else None
            )
            if primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise PasskeyV2SqliteDenied("passkey_v2_concurrent_attempt") from None
            raise PasskeyV2SqliteError("passkey_v2_transaction_failed") from None


class PasskeyV2AuthorityDatabase(_Database):
    def __init__(self, path: Path, *, authority_uid: int, authority_gid: int) -> None:
        super().__init__(
            path,
            uid=authority_uid,
            gid=authority_gid,
            schema_name=AUTHORITY_DB_SCHEMA,
        )

    def create_request(self, envelope: Mapping[str, Any]) -> Mapping[str, Any]:
        action = protocol.validate_action_envelope(envelope)
        protocol.require_production_webauthn_identity(action)
        document = protocol.canonical_json_bytes(action)
        connection = self._connect()
        try:
            self._begin_immediate(connection)
            connection.execute(
                "INSERT INTO requests(request_id,envelope_sha256,document,created_at_unix) "
                "VALUES(?,?,?,?)",
                (
                    action["request_id"],
                    action["envelope_sha256"],
                    document,
                    action["issued_at_unix"],
                ),
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PasskeyV2SqliteDenied("passkey_v2_request_not_rearmable") from None
        finally:
            connection.close()
        return action

    def read_request_state(self, request_id: str) -> Mapping[str, Any]:
        """Read one exact request bundle without creating or rotating state."""

        protocol.validate_request_id(request_id)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT r.document,c.document,g.document FROM requests r "
                "LEFT JOIN challenges c ON c.request_id=r.request_id "
                "LEFT JOIN grants g ON g.request_id=r.request_id "
                "WHERE r.request_id=?",
                (request_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise PasskeyV2SqliteDenied("passkey_v2_request_unknown")
        documents: list[Mapping[str, Any] | None] = []
        for raw in row:
            if raw is None:
                documents.append(None)
                continue
            value = protocol.decode_canonical_json(bytes(raw))
            if not isinstance(value, Mapping):
                raise PasskeyV2SqliteError("passkey_v2_request_state_invalid")
            documents.append(dict(value))
        action, challenge, grant = documents
        assert action is not None
        checked_action = protocol.validate_action_envelope(action)
        if challenge is not None:
            protocol.validate_challenge_record(
                challenge,
                envelope=checked_action,
            )
        if grant is not None:
            if challenge is None:
                raise PasskeyV2SqliteError("passkey_v2_request_state_invalid")
            protocol.validate_passkey_grant(
                grant,
                envelope=checked_action,
                challenge=challenge,
            )
        return {
            "action_envelope": checked_action,
            "challenge_record": challenge,
            "grant_record": grant,
        }

    def read_active_credentials(self) -> tuple[Mapping[str, Any], ...]:
        """Return validated public credential records for allowCredentials."""

        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT document FROM credentials ORDER BY credential_id_sha256"
            ).fetchall()
        finally:
            connection.close()
        result: list[Mapping[str, Any]] = []
        for row in rows:
            value = protocol.decode_canonical_json(bytes(row[0]))
            result.append(webauthn.validate_migrated_credential(value))
        return tuple(result)

    def create_challenge(
        self,
        challenge: Mapping[str, Any],
        *,
        envelope: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        action = protocol.validate_action_envelope(envelope)
        checked = protocol.validate_challenge_record(challenge, envelope=action)
        connection = self._connect()
        try:
            self._begin_immediate(connection)
            request = connection.execute(
                "SELECT document FROM requests WHERE request_id=?",
                (action["request_id"],),
            ).fetchone()
            if request is None or bytes(request[0]) != protocol.canonical_json_bytes(action):
                raise PasskeyV2SqliteDenied("passkey_v2_request_binding_mismatch")
            connection.execute(
                "INSERT INTO challenges(request_id,challenge_id,"
                "challenge_record_sha256,document,created_at_unix) VALUES(?,?,?,?,?)",
                (
                    action["request_id"],
                    checked["challenge_id"],
                    checked["challenge_record_sha256"],
                    protocol.canonical_json_bytes(checked),
                    checked["created_at_unix"],
                ),
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PasskeyV2SqliteDenied("passkey_v2_challenge_not_rearmable") from None
        finally:
            connection.close()
        return checked

    def record_passkey_grant(
        self,
        grant: Mapping[str, Any],
        *,
        envelope: Mapping[str, Any],
        challenge: Mapping[str, Any],
        now_unix: int,
    ) -> Mapping[str, Any]:
        raise PasskeyV2SqliteDenied("passkey_v2_untrusted_grant_forbidden")

    def _record_verified_grant(
        self,
        grant: Mapping[str, Any],
        *,
        envelope: Mapping[str, Any],
        challenge: Mapping[str, Any],
        verification: Mapping[str, Any],
        now_unix: int,
        connection: sqlite3.Connection,
    ) -> Mapping[str, Any]:
        action = protocol.validate_action_envelope(envelope)
        checked_challenge = protocol.validate_challenge_record(
            challenge, envelope=action
        )
        checked = protocol.validate_passkey_grant(
            grant, envelope=action, challenge=checked_challenge
        )
        if (
            verification.get("request_id") != action["request_id"]
            or verification.get("action_envelope_sha256")
            != action["envelope_sha256"]
            or verification.get("credential_id_sha256")
            != checked["credential_id_sha256"]
            or verification.get("credential_sign_count")
            != checked["credential_sign_count"]
            or verification.get("user_verified") is not True
        ):
            raise PasskeyV2SqliteDenied("passkey_v2_verified_grant_invalid")
        protocol.require_dangerous_approval_method(checked["method"])
        if not checked["granted_at_unix"] <= now_unix < checked["expires_at_unix"]:
            raise PasskeyV2SqliteDenied("passkey_v2_grant_expired")
        try:
            row = connection.execute(
                "SELECT r.document,c.document FROM requests r JOIN challenges c "
                "ON c.request_id=r.request_id WHERE r.request_id=?",
                (action["request_id"],),
            ).fetchone()
            if (
                row is None
                or bytes(row[0]) != protocol.canonical_json_bytes(action)
                or bytes(row[1]) != protocol.canonical_json_bytes(checked_challenge)
            ):
                raise PasskeyV2SqliteDenied("passkey_v2_grant_binding_mismatch")
            connection.execute(
                "INSERT INTO grants(request_id,grant_id,grant_sha256,document,"
                "granted_at_unix) VALUES(?,?,?,?,?)",
                (
                    action["request_id"],
                    checked["grant_id"],
                    checked["grant_sha256"],
                    protocol.canonical_json_bytes(checked),
                    checked["granted_at_unix"],
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise PasskeyV2SqliteDenied("passkey_v2_grant_not_reapprovable") from None
        return checked

    def import_migrated_credential(
        self,
        credential: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        checked = webauthn.validate_migrated_credential(credential)
        connection = self._connect()
        try:
            self._begin_immediate(connection)
            connection.execute(
                "INSERT INTO credentials(credential_id_sha256,"
                "credential_record_sha256,document,imported_at_unix) VALUES(?,?,?,?)",
                (
                    checked["credential_id_sha256"],
                    checked["credential_record_sha256"],
                    protocol.canonical_json_bytes(checked),
                    checked["imported_at_unix"],
                ),
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PasskeyV2SqliteDenied(
                "passkey_v2_credential_not_reimportable"
            ) from None
        finally:
            connection.close()
        return checked

    def verify_assertion_and_record_grant(
        self,
        *,
        assertion: Mapping[str, Any],
        envelope: Mapping[str, Any],
        challenge: Mapping[str, Any],
        grant_id: str,
        now_unix: int,
    ) -> Mapping[str, Any]:
        action = protocol.validate_action_envelope(envelope)
        checked_challenge = protocol.validate_challenge_record(
            challenge, envelope=action
        )
        credential_id_sha256 = webauthn.assertion_credential_id_sha256(assertion)
        connection = self._connect()
        try:
            self._begin_immediate(connection)
            row = connection.execute(
                "SELECT document FROM credentials WHERE credential_id_sha256=?",
                (credential_id_sha256,),
            ).fetchone()
            if row is None:
                raise PasskeyV2SqliteDenied("passkey_v2_credential_unknown")
            credential = protocol.decode_canonical_json(bytes(row[0]))
            if not isinstance(credential, Mapping):
                raise PasskeyV2SqliteError("passkey_v2_credential_state_invalid")
            last_use = connection.execute(
                "SELECT sign_count FROM credential_uses WHERE credential_id_sha256=? "
                "ORDER BY sequence DESC LIMIT 1",
                (credential_id_sha256,),
            ).fetchone()
            prior_sign_count = (
                int(credential["initial_sign_count"])
                if last_use is None
                else int(last_use[0])
            )
            verification = webauthn.verify_assertion(
                assertion,
                credential=credential,
                challenge=checked_challenge,
                envelope=action,
                prior_sign_count=prior_sign_count,
            )
            grant = protocol.build_passkey_grant(
                envelope=action,
                challenge=checked_challenge,
                grant_id=grant_id,
                approver_discord_user_id=verification[
                    "approver_discord_user_id"
                ],
                credential_id_sha256=verification["credential_id_sha256"],
                credential_record_sha256=verification[
                    "credential_record_sha256"
                ],
                credential_migration_receipt_sha256=credential[
                    "migration_receipt_sha256"
                ],
                assertion_verification_sha256=verification[
                    "verification_sha256"
                ],
                credential_sign_count=verification["credential_sign_count"],
                credential_backed_up=verification["credential_backed_up"],
                granted_at_unix=now_unix,
            )
            next_sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 FROM credential_uses"
            ).fetchone()
            assert next_sequence is not None
            connection.execute(
                "INSERT INTO credential_uses(sequence,credential_id_sha256,"
                "request_id,verification_sha256,sign_count,document,verified_at_unix) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    int(next_sequence[0]),
                    credential_id_sha256,
                    action["request_id"],
                    verification["verification_sha256"],
                    verification["credential_sign_count"],
                    protocol.canonical_json_bytes(verification),
                    now_unix,
                ),
            )
            self._record_verified_grant(
                grant,
                envelope=action,
                challenge=checked_challenge,
                verification=verification,
                now_unix=now_unix,
                connection=connection,
            )
            connection.commit()
            return grant
        except (
            protocol.PasskeyV2ProtocolError,
            webauthn.PasskeyV2WebAuthnError,
            sqlite3.IntegrityError,
        ) as exc:
            connection.rollback()
            raise PasskeyV2SqliteDenied(
                "passkey_v2_assertion_verification_failed"
            ) from None
        except PasskeyV2SqliteError:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _journal_state(connection: sqlite3.Connection) -> tuple[int, str]:
        counts = connection.execute(
            "SELECT (SELECT COUNT(*) FROM consumptions),"
            "(SELECT COUNT(*) FROM authorization_journal)"
        ).fetchone()
        if counts is None or counts[0] != counts[1]:
            raise PasskeyV2SqliteError("passkey_v2_tombstone_journal_mismatch")
        rows = connection.execute(
            "SELECT sequence,prior_journal_head_sha256,"
            "authorization_receipt_sha256,journal_head_sha256,request_id,"
            "consume_attempt_id,created_at_unix FROM authorization_journal "
            "ORDER BY sequence"
        ).fetchall()
        prior = protocol.GENESIS_JOURNAL_HEAD_SHA256
        expected_sequence = 1
        for row in rows:
            unsigned = {
                "schema": "muncho-passkey-v2-authorization-journal-entry.v1",
                "sequence": int(row[0]),
                "prior_journal_head_sha256": str(row[1]),
                "authorization_receipt_sha256": str(row[2]),
                "request_id": str(row[4]),
                "consume_attempt_id": str(row[5]),
                "created_at_unix": int(row[6]),
            }
            linked = connection.execute(
                "SELECT 1 FROM consumptions WHERE request_id=? AND "
                "consume_attempt_id=? AND authorization_receipt_sha256=?",
                (row[4], row[5], row[2]),
            ).fetchone()
            if (
                int(row[0]) != expected_sequence
                or str(row[1]) != prior
                or str(row[3]) != protocol.sha256_json(unsigned)
                or linked != (1,)
            ):
                raise PasskeyV2SqliteError("passkey_v2_journal_chain_invalid")
            prior = str(row[3])
            expected_sequence += 1
        return expected_sequence - 1, prior

    def consume_or_replay(
        self,
        *,
        envelope: Mapping[str, Any],
        runtime_binding: Mapping[str, Any],
        consume_attempt_id: str,
        signer: ReceiptSigner,
        now_unix: int,
    ) -> ConsumptionResult:
        action = protocol.validate_action_envelope(envelope)
        runtime = protocol.validate_runtime_binding(runtime_binding)
        if not isinstance(consume_attempt_id, str) or len(consume_attempt_id) != 64:
            raise PasskeyV2SqliteError("passkey_v2_consume_attempt_invalid")
        try:
            int(consume_attempt_id, 16)
        except ValueError as exc:
            raise PasskeyV2SqliteError("passkey_v2_consume_attempt_invalid") from None
        connection = self._connect()
        try:
            self._begin_immediate(connection)
            existing = connection.execute(
                "SELECT consume_attempt_id,receipt FROM consumptions WHERE request_id=?",
                (action["request_id"],),
            ).fetchone()
            if existing is not None:
                if existing[0] != consume_attempt_id:
                    raise PasskeyV2SqliteDenied(
                        "passkey_v2_consumed_by_different_attempt"
                    )
                stored = protocol.decode_canonical_json(bytes(existing[1]))
                if not isinstance(stored, Mapping):
                    raise PasskeyV2SqliteError("passkey_v2_stored_receipt_invalid")
                state = connection.execute(
                    "SELECT r.document,c.document,g.document FROM requests r "
                    "JOIN challenges c ON c.request_id=r.request_id "
                    "JOIN grants g ON g.request_id=r.request_id WHERE r.request_id=?",
                    (action["request_id"],),
                ).fetchone()
                if (
                    state is None
                    or bytes(state[0]) != protocol.canonical_json_bytes(action)
                ):
                    raise PasskeyV2SqliteDenied(
                        "passkey_v2_replay_action_binding_mismatch"
                    )
                challenge = protocol.decode_canonical_json(bytes(state[1]))
                grant = protocol.decode_canonical_json(bytes(state[2]))
                if not isinstance(challenge, Mapping) or not isinstance(grant, Mapping):
                    raise PasskeyV2SqliteError("passkey_v2_replay_state_invalid")
                checked_receipt = protocol.validate_authorization_receipt(
                    stored,
                    envelope=action,
                    grant=grant,
                    challenge=challenge,
                    receipt_public_key=signer.public_key,
                )
                if (
                    protocol.canonical_json_bytes(
                        checked_receipt["runtime_binding"]
                    )
                    != protocol.canonical_json_bytes(runtime)
                    or now_unix >= checked_receipt[
                        "execution_window_expires_at_unix"
                    ]
                ):
                    raise PasskeyV2SqliteDenied(
                        "passkey_v2_replay_runtime_or_window_mismatch"
                    )
                self._journal_state(connection)
                connection.rollback()
                return ConsumptionResult("receipt_replay", checked_receipt)
            row = connection.execute(
                "SELECT r.document,c.document,g.document FROM requests r "
                "JOIN challenges c ON c.request_id=r.request_id "
                "JOIN grants g ON g.request_id=r.request_id WHERE r.request_id=?",
                (action["request_id"],),
            ).fetchone()
            if row is None or bytes(row[0]) != protocol.canonical_json_bytes(action):
                raise PasskeyV2SqliteDenied("passkey_v2_consumption_binding_mismatch")
            challenge = protocol.decode_canonical_json(bytes(row[1]))
            grant = protocol.decode_canonical_json(bytes(row[2]))
            if not isinstance(challenge, Mapping) or not isinstance(grant, Mapping):
                raise PasskeyV2SqliteError("passkey_v2_authority_state_invalid")
            checked_challenge = protocol.validate_challenge_record(
                challenge, envelope=action
            )
            checked_grant = protocol.validate_passkey_grant(
                grant, envelope=action, challenge=checked_challenge
            )
            if not checked_grant["granted_at_unix"] <= now_unix < checked_grant[
                "expires_at_unix"
            ]:
                raise PasskeyV2SqliteDenied("passkey_v2_grant_expired")
            sequence, prior_head = self._journal_state(connection)
            unsigned = protocol.build_authorization_receipt_unsigned(
                envelope=action,
                grant=checked_grant,
                challenge=checked_challenge,
                runtime_binding=runtime,
                consume_attempt_id=consume_attempt_id,
                consumed_at_unix=now_unix,
                prior_journal_head_sha256=prior_head,
                receipt_public_key_id=signer.key_id,
            )
            receipt = signer.sign(unsigned)
            protocol.validate_authorization_receipt(
                receipt,
                envelope=action,
                grant=checked_grant,
                challenge=checked_challenge,
                receipt_public_key=signer.public_key,
            )
            journal_unsigned = {
                "schema": "muncho-passkey-v2-authorization-journal-entry.v1",
                "sequence": sequence + 1,
                "prior_journal_head_sha256": prior_head,
                "authorization_receipt_sha256": receipt["receipt_sha256"],
                "request_id": action["request_id"],
                "consume_attempt_id": consume_attempt_id,
                "created_at_unix": now_unix,
            }
            journal_head = protocol.sha256_json(journal_unsigned)
            connection.execute(
                "INSERT INTO consumptions(request_id,consume_attempt_id,"
                "authorization_receipt_sha256,receipt,consumed_at_unix) "
                "VALUES(?,?,?,?,?)",
                (
                    action["request_id"],
                    consume_attempt_id,
                    receipt["receipt_sha256"],
                    protocol.canonical_json_bytes(receipt),
                    now_unix,
                ),
            )
            connection.execute(
                "INSERT INTO authorization_journal(sequence,"
                "prior_journal_head_sha256,authorization_receipt_sha256,"
                "journal_head_sha256,request_id,consume_attempt_id,created_at_unix) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    sequence + 1,
                    prior_head,
                    receipt["receipt_sha256"],
                    journal_head,
                    action["request_id"],
                    consume_attempt_id,
                    now_unix,
                ),
            )
            if self._journal_state(connection)[0] != sequence + 1:
                raise PasskeyV2SqliteError("passkey_v2_atomic_commit_invalid")
            connection.commit()
            return ConsumptionResult("authorized_once", receipt)
        except PasskeyV2SqliteError:
            connection.rollback()
            raise
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PasskeyV2SqliteDenied("passkey_v2_consumption_collision") from None
        except sqlite3.Error as exc:
            connection.rollback()
            raise PasskeyV2SqliteError("passkey_v2_consumption_failed") from None
        finally:
            connection.close()

    def assert_bijection(self) -> Mapping[str, Any]:
        connection = self._connect()
        try:
            sequence, head = self._journal_state(connection)
            orphan = connection.execute(
                "SELECT COUNT(*) FROM consumptions c LEFT JOIN authorization_journal j "
                "ON j.authorization_receipt_sha256=c.authorization_receipt_sha256 "
                "WHERE j.authorization_receipt_sha256 IS NULL"
            ).fetchone()
            reverse = connection.execute(
                "SELECT COUNT(*) FROM authorization_journal j LEFT JOIN consumptions c "
                "ON c.authorization_receipt_sha256=j.authorization_receipt_sha256 "
                "WHERE c.authorization_receipt_sha256 IS NULL"
            ).fetchone()
            if orphan != (0,) or reverse != (0,):
                raise PasskeyV2SqliteError("passkey_v2_tombstone_journal_mismatch")
        finally:
            connection.close()
        return {
            "ok": True,
            "authorization_count": sequence,
            "journal_head_sha256": head,
            "bijection": True,
        }

    def preflight(self) -> Mapping[str, Any]:
        base = dict(super().preflight())
        if not hasattr(self, "path"):
            return base
        connection = self._connect()
        try:
            sequence, head = self._journal_state(connection)
        finally:
            connection.close()
        unsigned = {
            **{key: item for key, item in base.items() if key != "preflight_sha256"},
            "atomic_authorization_sequence": sequence,
            "atomic_authorization_head_sha256": head,
            "tombstone_journal_bijection": True,
        }
        return {**unsigned, "preflight_sha256": protocol.sha256_json(unsigned)}


class PasskeyV2ExecutorDatabase(_Database):
    """Privileged executor-owned append-only mutation intent/event ledger."""

    def __init__(
        self,
        path: Path,
        *,
        executor_uid: int,
        executor_gid: int,
        pinned_authority_receipt_public_key: Ed25519PublicKey,
        pinned_authority_receipt_key_id: str,
    ) -> None:
        if not isinstance(pinned_authority_receipt_public_key, Ed25519PublicKey):
            raise PasskeyV2SqliteError("passkey_v2_pinned_authority_key_invalid")
        actual_key_id = protocol.sha256_bytes(
            pinned_authority_receipt_public_key.public_bytes_raw()
        )
        if actual_key_id != pinned_authority_receipt_key_id:
            raise PasskeyV2SqliteError("passkey_v2_pinned_authority_key_invalid")
        self._pinned_authority_receipt_public_key = (
            pinned_authority_receipt_public_key
        )
        self._pinned_authority_receipt_key_id = pinned_authority_receipt_key_id
        super().__init__(
            path,
            uid=executor_uid,
            gid=executor_gid,
            schema_name=EXECUTOR_DB_SCHEMA,
        )

    def issue_observation_collection_attempt(
        self,
        *,
        transaction_id: str,
        checkpoint: str,
        prior_event_head_sha256: str,
        release_sha: str,
        plan_sha256: str,
        now_unix: int,
        ttl_seconds: int,
    ) -> Mapping[str, Any]:
        """Return one durable attempt, rotating only after exact expiry."""

        if (
            _SHA256.fullmatch(transaction_id or "") is None
            or checkpoint not in {
                "source", "post_resize", "post_stop", "post_start"
            }
            or _SHA256.fullmatch(prior_event_head_sha256 or "") is None
            or _REVISION.fullmatch(release_sha or "") is None
            or _SHA256.fullmatch(plan_sha256 or "") is None
            or type(now_unix) is not int
            or now_unix < 1
            or type(ttl_seconds) is not int
            or ttl_seconds
            != growth_evidence.OBSERVATION_BUNDLE_TTL_SECONDS
        ):
            raise PasskeyV2SqliteError(
                "passkey_v2_observation_collection_context_invalid"
            )
        context = {
            "schema": "muncho-storage-growth-collection-context.v1",
            "transaction_id": transaction_id,
            "checkpoint": checkpoint,
            "prior_event_head_sha256": prior_event_head_sha256,
            "release_sha": release_sha,
            "plan_sha256": plan_sha256,
        }
        context_sha256 = protocol.sha256_json(context)
        connection = self._connect()
        try:
            self._begin_immediate(connection)
            row = connection.execute(
                "SELECT document FROM observation_collection_attempts "
                "WHERE context_sha256=? ORDER BY context_sequence DESC LIMIT 1",
                (context_sha256,),
            ).fetchone()
            context_sequence = 1
            if row is not None:
                existing = protocol.decode_canonical_json(bytes(row[0]))
                if not _observation_collection_attempt_valid(
                    existing,
                    expected_context_sha256=context_sha256,
                ):
                    raise PasskeyV2SqliteError(
                        "passkey_v2_observation_collection_ledger_invalid"
                    )
                context_sequence = int(existing.get("context_sequence", 0)) + 1
                if now_unix < int(existing.get("expires_at_unix", 0)):
                    connection.rollback()
                    return dict(existing)
            identity = {
                "schema": "muncho-storage-growth-collection-attempt-id.v1",
                "context_sha256": context_sha256,
                "context_sequence": context_sequence,
                "issued_at_unix": now_unix,
            }
            attempt_id = protocol.sha256_json(identity)
            unsigned = {
                "schema": "muncho-storage-growth-collection-attempt.v1",
                "context_sha256": context_sha256,
                "context_sequence": context_sequence,
                "collection_attempt_id": attempt_id,
                **{key: value for key, value in context.items() if key != "schema"},
                "issued_at_unix": now_unix,
                "expires_at_unix": now_unix + ttl_seconds,
            }
            attempt = {
                **unsigned,
                "attempt_sha256": protocol.sha256_json(unsigned),
            }
            sequence_row = connection.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 "
                "FROM observation_collection_attempts"
            ).fetchone()
            assert sequence_row is not None
            connection.execute(
                "INSERT INTO observation_collection_attempts("
                "sequence,context_sha256,context_sequence,collection_attempt_id,"
                "transaction_id,checkpoint,prior_event_head_sha256,release_sha,"
                "plan_sha256,issued_at_unix,expires_at_unix,document) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    int(sequence_row[0]),
                    context_sha256,
                    context_sequence,
                    attempt_id,
                    transaction_id,
                    checkpoint,
                    prior_event_head_sha256,
                    release_sha,
                    plan_sha256,
                    now_unix,
                    now_unix + ttl_seconds,
                    protocol.canonical_json_bytes(attempt),
                ),
            )
            connection.commit()
            return attempt
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PasskeyV2SqliteDenied(
                "passkey_v2_observation_collection_collision"
            ) from None
        finally:
            connection.close()

    def claim_execution(
        self,
        *,
        receipt: Mapping[str, Any],
        envelope: Mapping[str, Any],
        grant: Mapping[str, Any],
        challenge: Mapping[str, Any],
        now_unix: int,
    ) -> ExecutionClaimResult:
        action = protocol.validate_action_envelope(envelope)
        checked_receipt = protocol.validate_authorization_receipt(
            receipt,
            envelope=action,
            grant=grant,
            challenge=challenge,
            receipt_public_key=self._pinned_authority_receipt_public_key,
        )
        if (
            checked_receipt["receipt_public_key_id"]
            != self._pinned_authority_receipt_key_id
        ):
            raise PasskeyV2SqliteDenied("passkey_v2_authority_key_not_pinned")
        if not checked_receipt["consumed_at_unix"] <= now_unix < checked_receipt[
            "execution_window_expires_at_unix"
        ]:
            raise PasskeyV2SqliteDenied("passkey_v2_execution_window_expired")
        authorization_unsigned = {
            "schema": "muncho-passkey-v2-execution-authorization.v1",
            "request_id": action["request_id"],
            "consume_attempt_id": checked_receipt["consume_attempt_id"],
            "authorization_receipt_sha256": checked_receipt["receipt_sha256"],
            "action_envelope_sha256": action["envelope_sha256"],
            "action_payload_sha256": action["action_payload_sha256"],
            "executor_release_sha": action["executor_release_sha"],
            "executor_plan_sha256": action["executor_plan_sha256"],
            "transaction_id": action["transaction_id"],
            "stage": action["stage"],
            "authorized_at_unix": now_unix,
        }
        authorization = {
            **authorization_unsigned,
            "authorization_sha256": protocol.sha256_json(
                authorization_unsigned
            ),
        }
        connection = self._connect()
        try:
            self._begin_immediate(connection)
            existing = connection.execute(
                "SELECT consume_attempt_id,authorization,action_envelope,"
                "authorization_receipt,challenge_record,grant_record "
                "FROM execution_authorizations "
                "WHERE request_id=?",
                (action["request_id"],),
            ).fetchone()
            if existing is not None:
                connection.rollback()
                if existing[0] != checked_receipt["consume_attempt_id"]:
                    raise PasskeyV2SqliteDenied(
                        "passkey_v2_execution_claimed_by_different_attempt"
                    )
                stored = protocol.decode_canonical_json(bytes(existing[1]))
                if (
                    not isinstance(stored, Mapping)
                    or bytes(existing[2]) != protocol.canonical_json_bytes(action)
                    or bytes(existing[3])
                    != protocol.canonical_json_bytes(checked_receipt)
                    or bytes(existing[4])
                    != protocol.canonical_json_bytes(challenge)
                    or bytes(existing[5]) != protocol.canonical_json_bytes(grant)
                ):
                    raise PasskeyV2SqliteError(
                        "passkey_v2_execution_authorization_invalid"
                    )
                return ExecutionClaimResult(
                    "authorization_replay", dict(stored)
                )

            transaction_row = connection.execute(
                "SELECT document FROM execution_transactions "
                "WHERE transaction_id=?",
                (action["transaction_id"],),
            ).fetchone()
            latest_event = connection.execute(
                "SELECT event_head_sha256 FROM execution_events "
                "WHERE transaction_id=? ORDER BY sequence DESC LIMIT 1",
                (action["transaction_id"],),
            ).fetchone()
            prior_event_head = (
                protocol.GENESIS_JOURNAL_HEAD_SHA256
                if latest_event is None
                else str(latest_event[0])
            )
            if action["prior_event_head_sha256"] != prior_event_head:
                raise PasskeyV2SqliteDenied(
                    "passkey_v2_authorization_event_head_stale"
                )
            if transaction_row is None:
                if action["stage"] != "intent":
                    raise PasskeyV2SqliteDenied(
                        "passkey_v2_initial_authorization_stage_invalid"
                    )
                if action["prior_authoritative_receipt_sha256"] != (
                    protocol.GENESIS_JOURNAL_HEAD_SHA256
                ):
                    raise PasskeyV2SqliteDenied(
                        "passkey_v2_initial_authorization_binding_invalid"
                    )
                transaction_unsigned = {
                    "schema": "muncho-passkey-v2-execution-transaction.v1",
                    "transaction_id": action["transaction_id"],
                    "initial_request_id": action["request_id"],
                    "executor_release_sha": action["executor_release_sha"],
                    "executor_plan_sha256": action["executor_plan_sha256"],
                    "source_preflight_sha256": action[
                        "source_preflight_sha256"
                    ],
                    "opened_at_unix": now_unix,
                }
                transaction = {
                    **transaction_unsigned,
                    "transaction_sha256": protocol.sha256_json(
                        transaction_unsigned
                    ),
                }
                connection.execute(
                    "INSERT INTO execution_transactions(transaction_id,"
                    "initial_request_id,executor_release_sha,"
                    "executor_plan_sha256,source_preflight_sha256,"
                    "transaction_sha256,document,opened_at_unix) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        action["transaction_id"],
                        action["request_id"],
                        action["executor_release_sha"],
                        action["executor_plan_sha256"],
                        action["source_preflight_sha256"],
                        transaction["transaction_sha256"],
                        protocol.canonical_json_bytes(transaction),
                        now_unix,
                    ),
                )
                authorization_sequence = 1
            else:
                transaction = protocol.decode_canonical_json(
                    bytes(transaction_row[0])
                )
                if (
                    not isinstance(transaction, Mapping)
                    or action["stage"] == "intent"
                    or action["executor_release_sha"]
                    != transaction.get("executor_release_sha")
                    or action["executor_plan_sha256"]
                    != transaction.get("executor_plan_sha256")
                ):
                    raise PasskeyV2SqliteDenied(
                        "passkey_v2_resume_authorization_binding_invalid"
                    )
                terminal = connection.execute(
                    "SELECT 1 FROM execution_events WHERE transaction_id=? "
                    "AND event_kind='completed'",
                    (action["transaction_id"],),
                ).fetchone()
                if terminal is not None:
                    raise PasskeyV2SqliteDenied(
                        "passkey_v2_terminal_transaction_reauthorization_forbidden"
                    )
                sequence_row = connection.execute(
                    "SELECT COUNT(*) FROM execution_authorizations "
                    "WHERE transaction_id=?",
                    (action["transaction_id"],),
                ).fetchone()
                authorization_sequence = int(sequence_row[0]) + 1
                prior_authorization_row = connection.execute(
                    "SELECT authorization_receipt_sha256 FROM "
                    "execution_authorizations WHERE transaction_id=? "
                    "ORDER BY authorization_sequence DESC LIMIT 1",
                    (action["transaction_id"],),
                ).fetchone()
                if (
                    prior_authorization_row is None
                    or action["prior_authoritative_receipt_sha256"]
                    != str(prior_authorization_row[0])
                ):
                    raise PasskeyV2SqliteDenied(
                        "passkey_v2_resume_authorization_receipt_stale"
                    )
            authorization = {
                **authorization,
                "authorization_sequence": authorization_sequence,
            }
            # Sequence is part of the signed ledger identity even though the
            # authority receipt was issued before this executor-owned number
            # was allocated.
            authorization = {
                **{
                    key: item
                    for key, item in authorization.items()
                    if key != "authorization_sha256"
                },
                "authorization_sha256": protocol.sha256_json(
                    {
                        key: item
                        for key, item in authorization.items()
                        if key != "authorization_sha256"
                    }
                ),
            }
            connection.execute(
                "INSERT INTO execution_authorizations(request_id,transaction_id,"
                "authorization_sequence,consume_attempt_id,"
                "authorization_receipt_sha256,action_envelope_sha256,"
                "authorization_sha256,authorization,action_envelope,"
                "authorization_receipt,challenge_record,grant_record,"
                "authorized_at_unix) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    action["request_id"],
                    action["transaction_id"],
                    authorization_sequence,
                    checked_receipt["consume_attempt_id"],
                    checked_receipt["receipt_sha256"],
                    action["envelope_sha256"],
                    authorization["authorization_sha256"],
                    protocol.canonical_json_bytes(authorization),
                    protocol.canonical_json_bytes(action),
                    protocol.canonical_json_bytes(checked_receipt),
                    protocol.canonical_json_bytes(challenge),
                    protocol.canonical_json_bytes(grant),
                    now_unix,
                ),
            )
            if authorization_sequence == 1:
                prior = connection.execute(
                    "SELECT sequence,event_head_sha256 FROM execution_events "
                    "ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                sequence = 1 if prior is None else int(prior[0]) + 1
                global_prior = (
                    protocol.GENESIS_JOURNAL_HEAD_SHA256
                    if prior is None
                    else str(prior[1])
                )
                opened_unsigned = {
                    "schema": "muncho-passkey-v2-execution-event.v1",
                    "sequence": sequence,
                    "event_id": _execution_event_id(
                        transaction_id=action["transaction_id"],
                        authorization_request_id=action["request_id"],
                        event_kind="opened",
                        event_payload={
                            "transaction_sha256": transaction[
                                "transaction_sha256"
                            ],
                            "source_preflight_sha256": action[
                                "source_preflight_sha256"
                            ],
                        },
                    ),
                    "transaction_id": action["transaction_id"],
                    "authorization_request_id": action["request_id"],
                    "authorization_sha256": authorization[
                        "authorization_sha256"
                    ],
                    "event_kind": "opened",
                    "event_payload": {
                        "transaction_sha256": transaction[
                            "transaction_sha256"
                        ],
                        "source_preflight_sha256": action[
                            "source_preflight_sha256"
                        ],
                    },
                    "prior_event_head_sha256": global_prior,
                    "created_at_unix": now_unix,
                }
                opened = {
                    **opened_unsigned,
                    "event_head_sha256": protocol.sha256_json(
                        opened_unsigned
                    ),
                }
                connection.execute(
                    "INSERT INTO execution_events(sequence,event_id,transaction_id,"
                    "authorization_request_id,event_kind,"
                    "prior_event_head_sha256,event_head_sha256,event,"
                    "created_at_unix) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        sequence,
                        opened["event_id"],
                        action["transaction_id"],
                        action["request_id"],
                        "opened",
                        global_prior,
                        opened["event_head_sha256"],
                        protocol.canonical_json_bytes(opened),
                        now_unix,
                    ),
                )
            connection.commit()
            return ExecutionClaimResult("claimed_once", authorization)
        except PasskeyV2SqliteError:
            connection.rollback()
            raise
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PasskeyV2SqliteDenied("passkey_v2_execution_claim_collision") from None
        finally:
            connection.close()

    def append_execution_event(
        self,
        *,
        request_id: str,
        event_kind: str,
        event_payload: Mapping[str, Any],
        now_unix: int,
    ) -> Mapping[str, Any]:
        protocol.validate_request_id(request_id)
        if event_kind not in _EXECUTION_EVENT_KINDS:
            raise PasskeyV2SqliteError("passkey_v2_execution_event_kind_invalid")
        connection = self._connect()
        try:
            self._begin_immediate(connection)
            authorization = connection.execute(
                "SELECT transaction_id,authorization_sha256 FROM "
                "execution_authorizations WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if authorization is None:
                raise PasskeyV2SqliteDenied(
                    "passkey_v2_execution_authorization_missing"
                )
            transaction_id = str(authorization[0])
            transaction_row = connection.execute(
                "SELECT document FROM execution_transactions "
                "WHERE transaction_id=?",
                (transaction_id,),
            ).fetchone()
            transaction = (
                None
                if transaction_row is None
                else protocol.decode_canonical_json(bytes(transaction_row[0]))
            )
            if (
                not isinstance(transaction, Mapping)
                or not self._execution_event_payload_valid(
                    event_kind,
                    event_payload,
                    transaction=transaction,
                )
            ):
                raise PasskeyV2SqliteError(
                    "passkey_v2_execution_event_payload_invalid"
                )
            event_id = _execution_event_id(
                transaction_id=transaction_id,
                authorization_request_id=request_id,
                event_kind=event_kind,
                event_payload=event_payload,
            )
            existing = connection.execute(
                "SELECT event FROM execution_events "
                + (
                    "WHERE event_id=?"
                    if event_kind in _REPEATED_EXECUTION_EVENT_KINDS
                    else "WHERE transaction_id=? AND event_kind=?"
                ),
                (
                    (event_id,)
                    if event_kind in _REPEATED_EXECUTION_EVENT_KINDS
                    else (transaction_id, event_kind)
                ),
            ).fetchone()
            if existing is not None:
                connection.rollback()
                stored = protocol.decode_canonical_json(bytes(existing[0]))
                if (
                    not isinstance(stored, Mapping)
                    or stored.get("event_payload") != dict(event_payload)
                ):
                    raise PasskeyV2SqliteError("passkey_v2_execution_event_invalid")
                return dict(stored)
            prior_for_request = connection.execute(
                "SELECT event_kind,event FROM execution_events "
                "WHERE transaction_id=? "
                "ORDER BY sequence",
                (transaction_id,),
            ).fetchall()
            prior_events: list[Mapping[str, Any]] = []
            for row in prior_for_request:
                prior_event = protocol.decode_canonical_json(bytes(row[1]))
                if not isinstance(prior_event, Mapping):
                    raise PasskeyV2SqliteError(
                        "passkey_v2_execution_ledger_invalid"
                    )
                prior_events.append(prior_event)
            kinds = tuple(str(row[0]) for row in prior_for_request)
            if not _execution_sequence_is_legal((*kinds, event_kind)):
                raise PasskeyV2SqliteDenied(
                    "passkey_v2_execution_event_transition_invalid"
                )
            if event_kind in _REPEATED_EXECUTION_EVENT_KINDS:
                stage = str(event_payload["stage"])
                intent_kind = f"{stage}_intent"
                complete_kind = f"{stage}_complete"
                intents = [
                    item for item in prior_events
                    if item["event_kind"] == intent_kind
                ]
                same_attempt = [
                    item for item in prior_events
                    if item["event_kind"] in _REPEATED_EXECUTION_EVENT_KINDS
                    and item["event_payload"].get("attempt_id")
                    == event_payload["attempt_id"]
                ]
                anchor = event_payload[
                    "attempt_anchor_event_head_sha256"
                ]
                if (
                    len(intents) != 1
                    or complete_kind in kinds
                    or event_payload["attempt_id"]
                    != _deterministic_attempt_id(
                        transaction_id, stage, anchor
                    )
                    or (
                        not same_attempt
                        and (
                            not prior_events
                            or anchor
                            != prior_events[-1]["event_head_sha256"]
                        )
                    )
                    or (
                        same_attempt
                        and any(
                            item["event_payload"].get(
                                "attempt_anchor_event_head_sha256"
                            ) != anchor
                            for item in same_attempt
                        )
                    )
                    or (
                        event_kind != "reconciliation_observed"
                        and event_payload["gce_request_id"]
                        != intents[0]["event_payload"]["gce_request_id"]
                    )
                    or any(
                        item["event_kind"] == event_kind
                        for item in same_attempt
                    )
                ):
                    raise PasskeyV2SqliteDenied(
                        "passkey_v2_execution_attempt_binding_invalid"
                    )
            elif event_kind.endswith("_observation_accepted"):
                stage = event_kind.removeprefix("post_").removesuffix(
                    "_observation_accepted"
                )
                bundle = event_payload["observation_bundle"]
                if (
                    kinds.count(f"post_{stage}_observation_required") != 1
                    or bundle["transaction_id"] != transaction_id
                    or bundle["checkpoint"] != f"post_{stage}"
                    or bundle["observation_nonce_sha256"]
                    != event_payload["observation_nonce_sha256"]
                ):
                    raise PasskeyV2SqliteDenied(
                        "passkey_v2_execution_observation_binding_invalid"
                    )
            prior = connection.execute(
                "SELECT sequence,event_head_sha256 FROM execution_events "
                "ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            sequence = 1 if prior is None else int(prior[0]) + 1
            prior_head = (
                protocol.GENESIS_JOURNAL_HEAD_SHA256
                if prior is None
                else str(prior[1])
            )
            unsigned = {
                "schema": "muncho-passkey-v2-execution-event.v1",
                "sequence": sequence,
                "event_id": event_id,
                "transaction_id": transaction_id,
                "authorization_request_id": request_id,
                "authorization_sha256": str(authorization[1]),
                "event_kind": event_kind,
                "event_payload": dict(event_payload),
                "prior_event_head_sha256": prior_head,
                "created_at_unix": now_unix,
            }
            event = {**unsigned, "event_head_sha256": protocol.sha256_json(unsigned)}
            connection.execute(
                "INSERT INTO execution_events(sequence,event_id,transaction_id,"
                "authorization_request_id,event_kind,prior_event_head_sha256,"
                "event_head_sha256,event,created_at_unix) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    sequence,
                    event_id,
                    transaction_id,
                    request_id,
                    event_kind,
                    prior_head,
                    event["event_head_sha256"],
                    protocol.canonical_json_bytes(event),
                    now_unix,
                ),
            )
            connection.commit()
            return event
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PasskeyV2SqliteDenied("passkey_v2_execution_event_replay") from None
        finally:
            connection.close()

    @staticmethod
    def _execution_event_payload_valid(
        event_kind: str,
        payload: Mapping[str, Any],
        *,
        transaction: Mapping[str, Any],
    ) -> bool:
        """Validate the fixed mechanical executor event vocabulary."""

        def sha(name: str) -> bool:
            return _SHA256.fullmatch(str(payload.get(name))) is not None

        def attested_bundle_valid() -> bool:
            try:
                bundle = growth_evidence.validate_attested_observation_structure(
                    payload.get("observation_bundle")
                )
            except growth_evidence.StorageGrowthEvidenceError:
                return False
            return bundle["bundle_sha256"] == payload.get(
                "observation_bundle_sha256"
            )

        if event_kind == "opened":
            return (
                set(payload)
                == {"transaction_sha256", "source_preflight_sha256"}
                and payload.get("transaction_sha256")
                == transaction["transaction_sha256"]
                and payload.get("source_preflight_sha256")
                == transaction["source_preflight_sha256"]
            )
        if event_kind.endswith("_intent"):
            stage = event_kind.removesuffix("_intent")
            return (
                set(payload)
                == {
                    "stage",
                    "requested_operation",
                    "observation_bundle_sha256",
                    "live_before_sha256",
                    "gce_request_id",
                    "activation_seal_sha256",
                    "firewall_readiness_receipt_sha256",
                }
                and payload.get("stage") == stage
                and isinstance(payload.get("requested_operation"), str)
                and 8 <= len(payload["requested_operation"]) <= 128
                and payload["requested_operation"].isascii()
                and sha("observation_bundle_sha256")
                and sha("live_before_sha256")
                and _UUID.fullmatch(str(payload.get("gce_request_id")))
                is not None
                and payload.get("gce_request_id")
                == _deterministic_gce_request_id(
                    transaction["transaction_id"], stage
                )
                and sha("activation_seal_sha256")
                and sha("firewall_readiness_receipt_sha256")
            )
        if event_kind in {"resize_complete", "stop_complete", "start_complete"}:
            stage = event_kind.removesuffix("_complete")
            return (
                set(payload)
                == {"stage", "disposition", "live_after_sha256"}
                and payload.get("stage") == stage
                and payload.get("disposition") in {"accepted", "reconciled"}
                and sha("live_after_sha256")
            )
        if event_kind.endswith("_observation_required"):
            stage = event_kind.removeprefix("post_").removesuffix(
                "_observation_required"
            )
            return (
                set(payload)
                == {
                    "after_stage",
                    "checkpoint",
                    "required_states",
                    "live_resource_sha256",
                }
                and payload.get("after_stage") == stage
                and payload.get("checkpoint") == f"post_{stage}"
                and isinstance(payload.get("required_states"), list)
                and 1 <= len(payload["required_states"]) <= 2
                and all(
                    isinstance(item, str) and item
                    for item in payload["required_states"]
                )
                and len(set(payload["required_states"]))
                == len(payload["required_states"])
                and sha("live_resource_sha256")
            )
        if event_kind.endswith("_observation_accepted"):
            stage = event_kind.removeprefix("post_").removesuffix(
                "_observation_accepted"
            )
            return (
                set(payload)
                == {
                    "after_stage",
                    "checkpoint",
                    "state",
                    "observation_bundle",
                    "observation_bundle_sha256",
                    "observation_nonce_sha256",
                }
                and payload.get("after_stage") == stage
                and payload.get("checkpoint") == f"post_{stage}"
                and isinstance(payload.get("state"), str)
                and payload["state"]
                and isinstance(payload.get("observation_bundle"), Mapping)
                and attested_bundle_valid()
                and sha("observation_bundle_sha256")
                and sha("observation_nonce_sha256")
            )
        if event_kind == "postflight_complete":
            return (
                set(payload)
                == {
                    "state",
                    "observation_bundle_sha256",
                    "live_resource_sha256",
                }
                and payload.get("state") == "target_ready"
                and sha("observation_bundle_sha256")
                and sha("live_resource_sha256")
            )
        if event_kind == "completed":
            receipt = payload.get("terminal_receipt")
            return (
                set(payload) == {"terminal_receipt"}
                and isinstance(receipt, Mapping)
                and receipt.get("schema")
                == "muncho-passkey-v2-storage-terminal-receipt.v1"
                and receipt.get("terminal") is True
                and receipt.get("transaction_id")
                == transaction["transaction_id"]
                and receipt.get("release_sha")
                == transaction["executor_release_sha"]
                and receipt.get("plan_sha256")
                == transaction["executor_plan_sha256"]
                and receipt.get("receipt_sha256")
                == protocol.sha256_json({
                    key: item for key, item in receipt.items()
                    if key != "receipt_sha256"
                })
            )
        if event_kind in _REPEATED_EXECUTION_EVENT_KINDS:
            common = (
                isinstance(payload.get("stage"), str)
                and payload["stage"] in {"resize", "stop", "start"}
                and _SHA256.fullmatch(str(payload.get("attempt_id")))
                is not None
                and _SHA256.fullmatch(
                    str(payload.get("attempt_anchor_event_head_sha256"))
                ) is not None
                and sha("live_resource_sha256")
            )
            if event_kind == "provider_pending":
                return (
                    common
                    and set(payload)
                    == {
                        "stage", "attempt_id",
                        "attempt_anchor_event_head_sha256", "gce_request_id",
                        "provider_status", "live_resource_sha256",
                    }
                    and _UUID.fullmatch(str(payload.get("gce_request_id")))
                    is not None
                    and isinstance(payload.get("provider_status"), str)
                    and 1 <= len(payload["provider_status"]) <= 64
                    and payload["provider_status"].isascii()
                )
            if event_kind == "attempt_failed":
                return (
                    common
                    and set(payload)
                    == {
                        "stage", "attempt_id",
                        "attempt_anchor_event_head_sha256", "gce_request_id", "failure",
                        "live_resource_sha256",
                    }
                    and _UUID.fullmatch(str(payload.get("gce_request_id")))
                    is not None
                    and isinstance(payload.get("failure"), str)
                    and 1 <= len(payload["failure"]) <= 128
                    and payload["failure"].isascii()
                )
            return (
                common
                and set(payload)
                == {
                    "stage", "attempt_id",
                    "attempt_anchor_event_head_sha256", "disposition",
                    "live_resource_sha256",
                }
                and payload.get("disposition")
                in {"pending", "already_applied", "ready"}
            )
        return False

    def _validate_execution_ledger(
        self,
        connection: sqlite3.Connection,
    ) -> Mapping[str, Any]:
        """Validate the full append-only executor truth, not only its schema."""

        transaction_rows = connection.execute(
            "SELECT transaction_id,initial_request_id,executor_release_sha,"
            "executor_plan_sha256,source_preflight_sha256,transaction_sha256,"
            "document,opened_at_unix FROM execution_transactions "
            "ORDER BY transaction_id"
        ).fetchall()
        transactions: dict[str, Mapping[str, Any]] = {}
        for row in transaction_rows:
            document = protocol.decode_canonical_json(bytes(row[6]))
            if not isinstance(document, Mapping):
                raise PasskeyV2SqliteError(
                    "passkey_v2_execution_transaction_invalid"
                )
            unsigned = {
                key: item for key, item in document.items()
                if key != "transaction_sha256"
            }
            transaction_id = str(row[0])
            if (
                set(document) != {
                    "schema", "transaction_id", "initial_request_id",
                    "executor_release_sha", "executor_plan_sha256",
                    "source_preflight_sha256", "opened_at_unix",
                    "transaction_sha256",
                }
                or document.get("schema")
                != "muncho-passkey-v2-execution-transaction.v1"
                or document.get("transaction_id") != transaction_id
                or document.get("initial_request_id") != str(row[1])
                or document.get("executor_release_sha") != str(row[2])
                or document.get("executor_plan_sha256") != str(row[3])
                or document.get("source_preflight_sha256") != str(row[4])
                or document.get("transaction_sha256") != str(row[5])
                or protocol.sha256_json(unsigned) != str(row[5])
                or document.get("opened_at_unix") != int(row[7])
            ):
                raise PasskeyV2SqliteError(
                    "passkey_v2_execution_transaction_invalid"
                )
            transactions[transaction_id] = dict(document)

        authorization_rows = connection.execute(
            "SELECT request_id,transaction_id,authorization_sequence,"
            "consume_attempt_id,authorization_receipt_sha256,"
            "action_envelope_sha256,authorization_sha256,authorization,"
            "action_envelope,authorization_receipt,challenge_record,"
            "grant_record,authorized_at_unix FROM execution_authorizations "
            "ORDER BY transaction_id,authorization_sequence"
        ).fetchall()
        authorizations: dict[str, Mapping[str, Any]] = {}
        per_transaction_authorizations: dict[str, list[Mapping[str, Any]]] = {
            transaction_id: [] for transaction_id in transactions
        }
        for row in authorization_rows:
            request_id = str(row[0])
            transaction_id = str(row[1])
            values = [
                protocol.decode_canonical_json(bytes(row[index]))
                for index in range(7, 12)
            ]
            authorization, envelope, receipt, challenge, grant = values
            if (
                transaction_id not in transactions
                or not all(isinstance(item, Mapping) for item in values)
            ):
                raise PasskeyV2SqliteError(
                    "passkey_v2_execution_authorization_invalid"
                )
            action = protocol.validate_action_envelope(envelope)
            checked_challenge = protocol.validate_challenge_record(
                challenge, envelope=action
            )
            checked_grant = protocol.validate_passkey_grant(
                grant, envelope=action, challenge=checked_challenge
            )
            checked_receipt = protocol.validate_authorization_receipt(
                receipt,
                envelope=action,
                grant=checked_grant,
                challenge=checked_challenge,
                receipt_public_key=self._pinned_authority_receipt_public_key,
            )
            unsigned = {
                key: item for key, item in authorization.items()
                if key != "authorization_sha256"
            }
            transaction = transactions[transaction_id]
            expected_sequence = len(
                per_transaction_authorizations[transaction_id]
            ) + 1
            if (
                action["request_id"] != request_id
                or action["transaction_id"] != transaction_id
                or action["executor_release_sha"]
                != transaction["executor_release_sha"]
                or action["executor_plan_sha256"]
                != transaction["executor_plan_sha256"]
                or (
                    expected_sequence == 1
                    and action["source_preflight_sha256"]
                    != transaction["source_preflight_sha256"]
                )
                or int(row[2]) != expected_sequence
                or authorization.get("authorization_sequence")
                != expected_sequence
                or authorization.get("request_id") != request_id
                or authorization.get("transaction_id") != transaction_id
                or authorization.get("consume_attempt_id") != str(row[3])
                or authorization.get("authorization_receipt_sha256")
                != str(row[4])
                or authorization.get("action_envelope_sha256") != str(row[5])
                or authorization.get("authorization_sha256") != str(row[6])
                or protocol.sha256_json(unsigned) != str(row[6])
                or action["envelope_sha256"] != str(row[5])
                or receipt.get("receipt_sha256") != str(row[4])
                or checked_receipt["receipt_public_key_id"]
                != self._pinned_authority_receipt_key_id
                or authorization.get("authorized_at_unix") != int(row[12])
                or (expected_sequence == 1 and action["stage"] != "intent")
                or (expected_sequence > 1 and action["stage"] == "intent")
                or (
                    expected_sequence == 1
                    and transaction["initial_request_id"] != request_id
                )
            ):
                raise PasskeyV2SqliteError(
                    "passkey_v2_execution_authorization_invalid"
                )
            record = {
                "authorization": dict(authorization),
                "action_envelope": dict(action),
                "authorization_receipt": dict(receipt),
                "challenge_record": dict(challenge),
                "grant_record": dict(grant),
            }
            authorizations[request_id] = record
            per_transaction_authorizations[transaction_id].append(record)

        for records in per_transaction_authorizations.values():
            for index, record in enumerate(records):
                expected_prior_receipt = (
                    protocol.GENESIS_JOURNAL_HEAD_SHA256
                    if index == 0
                    else records[index - 1]["authorization_receipt"][
                        "receipt_sha256"
                    ]
                )
                if record["action_envelope"][
                    "prior_authoritative_receipt_sha256"
                ] != expected_prior_receipt:
                    raise PasskeyV2SqliteError(
                        "passkey_v2_execution_authorization_chain_invalid"
                    )

        event_rows = connection.execute(
            "SELECT sequence,event_id,transaction_id,authorization_request_id,"
            "event_kind,prior_event_head_sha256,event_head_sha256,event,"
            "created_at_unix FROM execution_events ORDER BY sequence"
        ).fetchall()
        prior_head = protocol.GENESIS_JOURNAL_HEAD_SHA256
        per_transaction_events: dict[str, list[Mapping[str, Any]]] = {
            transaction_id: [] for transaction_id in transactions
        }
        for expected_sequence, row in enumerate(event_rows, start=1):
            sequence = int(row[0])
            event_id = str(row[1])
            transaction_id = str(row[2])
            request_id = str(row[3])
            event_kind = str(row[4])
            event = protocol.decode_canonical_json(bytes(row[7]))
            if (
                not isinstance(event, Mapping)
                or transaction_id not in transactions
                or request_id not in authorizations
                or authorizations[request_id]["action_envelope"][
                    "transaction_id"
                ] != transaction_id
            ):
                raise PasskeyV2SqliteError(
                    "passkey_v2_execution_ledger_invalid"
                )
            payload = event.get("event_payload")
            authorization = authorizations[request_id]["authorization"]
            prior_transaction_events = per_transaction_events[transaction_id]
            cross_event_valid = True
            if (
                isinstance(payload, Mapping)
                and event_kind in _REPEATED_EXECUTION_EVENT_KINDS
            ):
                stage = str(payload.get("stage"))
                intent_kind = f"{stage}_intent"
                complete_kind = f"{stage}_complete"
                intents = [
                    prior_event for prior_event in prior_transaction_events
                    if prior_event["event_kind"] == intent_kind
                ]
                same_attempt = [
                    prior_event for prior_event in prior_transaction_events
                    if prior_event["event_kind"]
                    in _REPEATED_EXECUTION_EVENT_KINDS
                    and prior_event["event_payload"].get("attempt_id")
                    == payload.get("attempt_id")
                ]
                anchor = str(
                    payload.get("attempt_anchor_event_head_sha256")
                )
                expected_attempt_id = _deterministic_attempt_id(
                    transaction_id, stage, anchor
                )
                first_for_attempt = not same_attempt
                cross_event_valid = (
                    len(intents) == 1
                    and not any(
                        prior_event["event_kind"] == complete_kind
                        for prior_event in prior_transaction_events
                    )
                    and payload.get("attempt_id") == expected_attempt_id
                    and (
                        (
                            first_for_attempt
                            and bool(prior_transaction_events)
                            and anchor
                            == prior_transaction_events[-1][
                                "event_head_sha256"
                            ]
                        )
                        or (
                            not first_for_attempt
                            and all(
                                prior_event["event_payload"].get(
                                    "attempt_anchor_event_head_sha256"
                                ) == anchor
                                for prior_event in same_attempt
                            )
                        )
                    )
                    and (
                        event_kind == "reconciliation_observed"
                        or payload.get("gce_request_id")
                        == intents[0]["event_payload"]["gce_request_id"]
                    )
                    and not any(
                        prior_event["event_kind"] == event_kind
                        for prior_event in same_attempt
                    )
                    and not (
                        event_kind == "provider_pending" and same_attempt
                    )
                )
            elif (
                isinstance(payload, Mapping)
                and event_kind.endswith("_observation_accepted")
            ):
                stage = event_kind.removeprefix("post_").removesuffix(
                    "_observation_accepted"
                )
                required_kind = f"post_{stage}_observation_required"
                required = [
                    prior_event for prior_event in prior_transaction_events
                    if prior_event["event_kind"] == required_kind
                ]
                bundle = payload.get("observation_bundle")
                cross_event_valid = (
                    len(required) == 1
                    and isinstance(bundle, Mapping)
                    and bundle.get("transaction_id") == transaction_id
                    and bundle.get("checkpoint") == f"post_{stage}"
                    and bundle.get("observation_nonce_sha256")
                    == payload.get("observation_nonce_sha256")
                    and bundle.get("bundle_sha256")
                    == payload.get("observation_bundle_sha256")
                )
            unsigned = {
                key: item for key, item in event.items()
                if key != "event_head_sha256"
            }
            if (
                sequence != expected_sequence
                or event_id != event.get("event_id")
                or event_id != _execution_event_id(
                    transaction_id=transaction_id,
                    authorization_request_id=request_id,
                    event_kind=event_kind,
                    event_payload=payload if isinstance(payload, Mapping) else {},
                )
                or str(row[5]) != prior_head
                or str(row[6]) != protocol.sha256_json(unsigned)
                or event.get("event_head_sha256") != str(row[6])
                or event.get("sequence") != sequence
                or event.get("transaction_id") != transaction_id
                or event.get("authorization_request_id") != request_id
                or event.get("authorization_sha256")
                != authorization["authorization_sha256"]
                or event.get("event_kind") != event_kind
                or event.get("prior_event_head_sha256") != prior_head
                or event.get("created_at_unix") != int(row[8])
                or event_kind not in _EXECUTION_EVENT_KINDS
                or not isinstance(payload, Mapping)
                or not cross_event_valid
                or not self._execution_event_payload_valid(
                    event_kind,
                    payload,
                    transaction=transactions[transaction_id],
                )
            ):
                raise PasskeyV2SqliteError(
                    "passkey_v2_execution_event_payload_invalid"
                )
            per_transaction_events[transaction_id].append(dict(event))
            prior_head = str(row[6])

        states: dict[str, str] = {}
        for transaction_id, events in per_transaction_events.items():
            kinds = tuple(str(event["event_kind"]) for event in events)
            milestone_kinds = tuple(
                kind for kind in kinds
                if kind not in _REPEATED_EXECUTION_EVENT_KINDS
            )
            if len(milestone_kinds) != len(set(milestone_kinds)):
                raise PasskeyV2SqliteError(
                    "passkey_v2_execution_ledger_invalid"
                )
            if not _execution_sequence_is_legal(kinds):
                raise PasskeyV2SqliteError(
                    "passkey_v2_execution_ledger_invalid"
                )
            states[transaction_id] = (
                "terminal" if milestone_kinds[-1] == "completed"
                else "recovery_required"
            )
        collection_rows = connection.execute(
            "SELECT sequence,context_sha256,context_sequence,"
            "collection_attempt_id,transaction_id,checkpoint,"
            "prior_event_head_sha256,release_sha,plan_sha256,issued_at_unix,"
            "expires_at_unix,document FROM observation_collection_attempts "
            "ORDER BY sequence"
        ).fetchall()
        contexts: dict[str, list[Mapping[str, Any]]] = {}
        for expected_sequence, row in enumerate(collection_rows, start=1):
            attempt = protocol.decode_canonical_json(bytes(row[11]))
            context_sha256 = str(row[1])
            prior_context = contexts.setdefault(context_sha256, [])
            if (
                int(row[0]) != expected_sequence
                or not _observation_collection_attempt_valid(
                    attempt,
                    expected_context_sha256=context_sha256,
                )
                or attempt.get("context_sequence") != len(prior_context) + 1
                or attempt.get("context_sequence") != int(row[2])
                or attempt.get("collection_attempt_id") != str(row[3])
                or attempt.get("transaction_id") != str(row[4])
                or attempt.get("checkpoint") != str(row[5])
                or attempt.get("prior_event_head_sha256") != str(row[6])
                or attempt.get("release_sha") != str(row[7])
                or attempt.get("plan_sha256") != str(row[8])
                or attempt.get("issued_at_unix") != int(row[9])
                or attempt.get("expires_at_unix") != int(row[10])
                or (
                    prior_context
                    and int(attempt["issued_at_unix"])
                    < int(prior_context[-1]["expires_at_unix"])
                )
            ):
                raise PasskeyV2SqliteError(
                    "passkey_v2_observation_collection_ledger_invalid"
                )
            prior_context.append(dict(attempt))
        return {
            "transaction_count": len(transactions),
            "authorization_count": len(authorizations),
            "event_count": len(event_rows),
            "event_head_sha256": prior_head,
            "transaction_states": states,
            "transactions": transactions,
            "authorizations": authorizations,
            "transaction_authorizations": per_transaction_authorizations,
            "events": per_transaction_events,
            "observation_collection_attempt_count": len(collection_rows),
        }

    def read_execution_state(self, request_id: str) -> Mapping[str, Any]:
        """Return one fully chain-validated recovery bundle."""

        protocol.validate_request_id(request_id)
        connection = self._connect()
        try:
            ledger = self._validate_execution_ledger(connection)
        finally:
            connection.close()
        if request_id not in ledger["authorizations"]:
            raise PasskeyV2SqliteDenied(
                "passkey_v2_execution_authorization_missing"
            )
        authorization = ledger["authorizations"][request_id]
        transaction_id = authorization["action_envelope"]["transaction_id"]
        return {
            "transaction": ledger["transactions"][transaction_id],
            **authorization,
            "authorizations": tuple(
                ledger["transaction_authorizations"][transaction_id]
            ),
            "events": tuple(ledger["events"][transaction_id]),
            "state": ledger["transaction_states"][transaction_id],
        }

    def read_transaction_state(self, transaction_id: str) -> Mapping[str, Any]:
        """Read canonical transaction state without needing a live grant."""

        if not isinstance(transaction_id, str) or _SHA256.fullmatch(
            transaction_id
        ) is None:
            raise PasskeyV2SqliteError("passkey_v2_transaction_id_invalid")
        connection = self._connect()
        try:
            ledger = self._validate_execution_ledger(connection)
        finally:
            connection.close()
        if transaction_id not in ledger["transactions"]:
            raise PasskeyV2SqliteDenied(
                "passkey_v2_execution_transaction_missing"
            )
        authorizations = tuple(
            ledger["transaction_authorizations"][transaction_id]
        )
        return {
            "transaction": ledger["transactions"][transaction_id],
            "authorizations": authorizations,
            "authorization": authorizations[-1],
            "events": tuple(ledger["events"][transaction_id]),
            "state": ledger["transaction_states"][transaction_id],
        }

    def read_terminal_receipt(self, transaction_id: str) -> Mapping[str, Any] | None:
        """Return stored terminal bytes after local ledger validation only."""

        state = self.read_transaction_state(transaction_id)
        if state["state"] != "terminal":
            return None
        completed = tuple(
            event for event in state["events"]
            if event["event_kind"] == "completed"
        )
        if len(completed) != 1:
            raise PasskeyV2SqliteError(
                "passkey_v2_terminal_receipt_invalid"
            )
        return dict(completed[0]["event_payload"]["terminal_receipt"])

    def preflight(self) -> Mapping[str, Any]:
        base = dict(super().preflight())
        if not hasattr(self, "path"):
            return base
        connection = self._connect()
        try:
            ledger = self._validate_execution_ledger(connection)
        finally:
            connection.close()
        states = ledger["transaction_states"]
        unsigned = {
            **{key: item for key, item in base.items() if key != "preflight_sha256"},
            "execution_transaction_count": ledger["transaction_count"],
            "execution_authorization_count": ledger["authorization_count"],
            "execution_event_count": ledger["event_count"],
            "execution_event_head_sha256": ledger["event_head_sha256"],
            "observation_collection_attempt_count": ledger[
                "observation_collection_attempt_count"
            ],
            "execution_terminal_count": sum(
                state == "terminal" for state in states.values()
            ),
            "execution_recovery_required_count": sum(
                state != "terminal" for state in states.values()
            ),
            "execution_transaction_authorization_bijection": True,
            "execution_global_sequence_contiguous": True,
            "execution_legal_transitions": True,
        }
        return {**unsigned, "preflight_sha256": protocol.sha256_json(unsigned)}
