"""Atomic SQLite state, budgets, replay fences, and receipt chaining."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterator, Mapping

from .errors import (
    BudgetExceeded,
    LeaseReplay,
    ReceiptChainError,
    ReplayDetected,
    StorageError,
)
from .model import CapabilityLease, canonical_json, new_id, parse_utc, utc_now, utc_text

_SCHEMA_VERSION = 1
_ZERO_HASH = "0" * 64
_MICRO = Decimal("1000000")


def _usd_to_micro(value: float | Decimal | None) -> int:
    if value is None:
        return 0
    amount = Decimal(str(value))
    if amount < 0:
        raise ValueError("cost cannot be negative")
    return int((amount * _MICRO).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _micro_to_usd(value: int) -> float:
    return float(Decimal(value) / _MICRO)


def _window_start(now: datetime, kind: str) -> str:
    current = now.astimezone(timezone.utc)
    if kind == "hour":
        current = current.replace(minute=0, second=0, microsecond=0)
    elif kind == "day":
        current = current.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"unsupported budget window: {kind}")
    return utc_text(current)


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    hourly_tokens: int | None = None
    daily_tokens: int | None = None
    hourly_cost_usd: float | None = None
    daily_cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("hourly_tokens", "daily_tokens"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("hourly_cost_usd", "daily_cost_usd"):
            value = getattr(self, name)
            if value is not None and Decimal(str(value)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: str
    scope_digest: str
    reserved_tokens: int
    reserved_cost_usd: float
    state: str
    created_at: str


@dataclass(frozen=True, slots=True)
class ReceiptRecord:
    sequence: int
    receipt_id: str
    event_id: str
    kind: str
    payload: Mapping[str, Any]
    previous_hash: str
    receipt_hash: str
    created_at: str


class HermesTagLedger:
    """One profile-local durable authority.

    A fresh connection is used per operation so independent gateway tasks and
    processes serialize through SQLite rather than through process-local locks.
    """

    def __init__(self, path: str | Path) -> None:
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise StorageError("Hermes Tag ledger path must not be a symlink")
        self.path = requested.resolve()
        self._initialize_lock = threading.Lock()
        self._initialized = False

    def _secure_posix_paths(self) -> None:
        if os.name == "nt":
            return
        for path in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            if not path.exists():
                continue
            try:
                path.chmod(0o600)
            except OSError as exc:
                raise StorageError(
                    f"cannot secure Hermes Tag state file {path.name}: {exc}"
                ) from exc

    def _connect(self) -> sqlite3.Connection:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                try:
                    self.path.parent.chmod(0o700)
                except OSError as exc:
                    raise StorageError(
                        f"cannot secure Hermes Tag state directory: {exc}"
                    ) from exc
            connection = sqlite3.connect(
                self.path,
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"cannot open Hermes Tag ledger: {exc}") from exc
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = FULL")
        self._secure_posix_paths()
        return connection

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._initialize_lock:
            if self._initialized:
                return
            connection = self._connect()
            try:
                # Journal mode is durable database state. Re-negotiating it on
                # every connection can wait behind an unrelated active writer
                # and turns ordinary reads/receipts into lock-amplification.
                # Establish WAL once per process initialization instead.
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("BEGIN IMMEDIATE")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS hermes_tag_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS hermes_tag_principals (
                        principal_id TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        roles_json TEXT NOT NULL,
                        guest INTEGER NOT NULL CHECK (guest IN (0, 1)),
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS hermes_tag_aliases (
                        alias_key TEXT PRIMARY KEY,
                        platform TEXT NOT NULL,
                        profile TEXT NOT NULL,
                        scope_id TEXT NOT NULL,
                        external_id TEXT NOT NULL,
                        display_name TEXT,
                        principal_id TEXT NOT NULL
                            REFERENCES hermes_tag_principals(principal_id),
                        bound_at TEXT NOT NULL,
                        revoked_at TEXT
                    );

                    CREATE TABLE IF NOT EXISTS hermes_tag_continuities (
                        continuity_id TEXT PRIMARY KEY,
                        principal_id TEXT NOT NULL
                            REFERENCES hermes_tag_principals(principal_id),
                        mode TEXT NOT NULL,
                        project_id TEXT,
                        version INTEGER NOT NULL CHECK (version >= 1),
                        objective TEXT NOT NULL DEFAULT '',
                        state_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS hermes_tag_surface_bindings (
                        surface_key TEXT PRIMARY KEY,
                        continuity_id TEXT NOT NULL
                            REFERENCES hermes_tag_continuities(continuity_id),
                        principal_id TEXT NOT NULL
                            REFERENCES hermes_tag_principals(principal_id),
                        platform TEXT NOT NULL,
                        profile TEXT NOT NULL,
                        scope_id TEXT NOT NULL,
                        chat_id TEXT NOT NULL,
                        thread_id TEXT,
                        bound_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS hermes_tag_checkpoints (
                        checkpoint_id TEXT PRIMARY KEY,
                        continuity_id TEXT NOT NULL
                            REFERENCES hermes_tag_continuities(continuity_id),
                        version INTEGER NOT NULL CHECK (version >= 1),
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(continuity_id, version)
                    );

                    CREATE TABLE IF NOT EXISTS hermes_tag_facts (
                        fact_id TEXT PRIMARY KEY,
                        subject TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        value_json TEXT NOT NULL,
                        scope_json TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        source_revision TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        authority INTEGER NOT NULL,
                        sensitivity INTEGER NOT NULL,
                        valid_from TEXT NOT NULL,
                        valid_until TEXT,
                        supersedes TEXT,
                        tags_json TEXT NOT NULL,
                        content_hash TEXT NOT NULL UNIQUE,
                        active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS hermes_tag_facts_lookup
                        ON hermes_tag_facts(subject, predicate, active, authority);

                    CREATE TABLE IF NOT EXISTS hermes_tag_approvals (
                        approval_id TEXT PRIMARY KEY,
                        principal_id TEXT NOT NULL
                            REFERENCES hermes_tag_principals(principal_id),
                        approver_id TEXT NOT NULL
                            REFERENCES hermes_tag_principals(principal_id),
                        intent_digest TEXT NOT NULL,
                        scope_digest TEXT NOT NULL,
                        issued_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        used_at TEXT
                    );
                    CREATE INDEX IF NOT EXISTS hermes_tag_approvals_lookup
                        ON hermes_tag_approvals(principal_id, intent_digest, scope_digest);

                    CREATE TABLE IF NOT EXISTS hermes_tag_budget_usage (
                        scope_digest TEXT NOT NULL,
                        window_kind TEXT NOT NULL CHECK (window_kind IN ('hour', 'day')),
                        window_start TEXT NOT NULL,
                        tokens INTEGER NOT NULL DEFAULT 0 CHECK (tokens >= 0),
                        cost_micro_usd INTEGER NOT NULL DEFAULT 0 CHECK (cost_micro_usd >= 0),
                        PRIMARY KEY(scope_digest, window_kind, window_start)
                    );

                    CREATE TABLE IF NOT EXISTS hermes_tag_budget_reservations (
                        reservation_id TEXT PRIMARY KEY,
                        scope_digest TEXT NOT NULL,
                        reserved_tokens INTEGER NOT NULL CHECK (reserved_tokens >= 0),
                        reserved_cost_micro_usd INTEGER NOT NULL CHECK (reserved_cost_micro_usd >= 0),
                        actual_tokens INTEGER,
                        actual_cost_micro_usd INTEGER,
                        state TEXT NOT NULL CHECK (state IN ('reserved', 'settled', 'released')),
                        created_at TEXT NOT NULL,
                        settled_at TEXT
                    );

                    CREATE TABLE IF NOT EXISTS hermes_tag_lease_uses (
                        lease_id TEXT PRIMARY KEY,
                        principal_id TEXT NOT NULL
                            REFERENCES hermes_tag_principals(principal_id),
                        decision_id TEXT NOT NULL,
                        capability TEXT NOT NULL,
                        intent_digest TEXT NOT NULL,
                        scope_digest TEXT NOT NULL,
                        budget_reservation_id TEXT
                            REFERENCES hermes_tag_budget_reservations(reservation_id),
                        approval_id TEXT
                            REFERENCES hermes_tag_approvals(approval_id),
                        state TEXT NOT NULL CHECK (state IN ('reserved', 'complete')),
                        reserved_at TEXT NOT NULL,
                        completed_at TEXT,
                        success INTEGER CHECK (success IN (0, 1)),
                        receipt_hash TEXT
                    );

                    CREATE TABLE IF NOT EXISTS hermes_tag_receipts (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        receipt_id TEXT NOT NULL UNIQUE,
                        event_id TEXT NOT NULL UNIQUE,
                        kind TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        previous_hash TEXT NOT NULL,
                        receipt_hash TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS hermes_tag_replay_fingerprints (
                        fingerprint TEXT PRIMARY KEY,
                        event_id TEXT NOT NULL,
                        continuity_id TEXT NOT NULL,
                        seen_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS hermes_tag_turn_events (
                        event_id TEXT PRIMARY KEY,
                        state TEXT NOT NULL CHECK (state IN ('pending', 'complete')),
                        admission_id TEXT,
                        principal_id TEXT,
                        surface_key TEXT,
                        continuity_id TEXT,
                        scope_digest TEXT,
                        created_at TEXT NOT NULL,
                        completed_at TEXT
                    );
                    """
                )
                existing = connection.execute(
                    "SELECT value FROM hermes_tag_meta WHERE key='schema_version'"
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO hermes_tag_meta(key, value) VALUES('schema_version', ?)",
                        (str(_SCHEMA_VERSION),),
                    )
                elif int(existing["value"]) != _SCHEMA_VERSION:
                    raise StorageError(
                        f"unsupported Hermes Tag schema version {existing['value']}"
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
            self._initialized = True

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._secure_posix_paths()

    def connection(self) -> sqlite3.Connection:
        """Open an initialized read connection; callers must close it."""
        self.initialize()
        return self._connect()

    def append_receipt(
        self,
        *,
        event_id: str,
        kind: str,
        payload: Mapping[str, Any],
        created_at: str | None = None,
    ) -> ReceiptRecord:
        """Append idempotently to the hash chain.

        Reusing an event id with different content is a hard conflict rather
        than a second receipt or a silent overwrite.
        """
        if (
            not isinstance(event_id, str)
            or not event_id.strip()
            or len(event_id) > 512
        ):
            raise ValueError("event_id must be a bounded non-empty string")
        if not isinstance(kind, str) or not kind.strip() or len(kind) > 128:
            raise ValueError("receipt kind must be a bounded non-empty string")
        if not isinstance(payload, Mapping):
            raise TypeError("receipt payload must be a mapping")
        payload_json = canonical_json(payload)
        if len(payload_json.encode("utf-8")) > 1_000_000:
            raise ValueError("receipt payload exceeds one megabyte")
        timestamp = created_at or utc_text()
        parse_utc(timestamp)
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM hermes_tag_receipts WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if existing is not None:
                if existing["kind"] != kind or existing["payload_json"] != payload_json:
                    raise ReceiptChainError(
                        "event id already exists with different receipt content"
                    )
                return self._receipt_from_row(existing)

            previous = connection.execute(
                "SELECT receipt_hash FROM hermes_tag_receipts ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_hash = previous["receipt_hash"] if previous else _ZERO_HASH
            receipt_id = new_id("receipt")
            body = {
                "receipt_id": receipt_id,
                "event_id": event_id,
                "kind": kind,
                "payload": json.loads(payload_json),
                "previous_hash": previous_hash,
                "created_at": timestamp,
            }
            receipt_hash = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
            connection.execute(
                """
                INSERT INTO hermes_tag_receipts(
                    receipt_id, event_id, kind, payload_json, previous_hash,
                    receipt_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    event_id,
                    kind,
                    payload_json,
                    previous_hash,
                    receipt_hash,
                    timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM hermes_tag_receipts WHERE receipt_id=?",
                (receipt_id,),
            ).fetchone()
            assert row is not None
            return self._receipt_from_row(row)

    @staticmethod
    def _receipt_from_row(row: sqlite3.Row) -> ReceiptRecord:
        return ReceiptRecord(
            sequence=int(row["sequence"]),
            receipt_id=row["receipt_id"],
            event_id=row["event_id"],
            kind=row["kind"],
            payload=json.loads(row["payload_json"]),
            previous_hash=row["previous_hash"],
            receipt_hash=row["receipt_hash"],
            created_at=row["created_at"],
        )

    def verify_receipt_chain(self) -> tuple[int, str]:
        connection = self.connection()
        try:
            rows = connection.execute(
                "SELECT * FROM hermes_tag_receipts ORDER BY sequence"
            ).fetchall()
        finally:
            connection.close()
        previous_hash = _ZERO_HASH
        expected_sequence = 1
        for row in rows:
            if int(row["sequence"]) != expected_sequence:
                raise ReceiptChainError("receipt sequence has a gap")
            if row["previous_hash"] != previous_hash:
                raise ReceiptChainError("receipt previous hash does not match")
            body = {
                "receipt_id": row["receipt_id"],
                "event_id": row["event_id"],
                "kind": row["kind"],
                "payload": json.loads(row["payload_json"]),
                "previous_hash": row["previous_hash"],
                "created_at": row["created_at"],
            }
            calculated = hashlib.sha256(
                canonical_json(body).encode("utf-8")
            ).hexdigest()
            if calculated != row["receipt_hash"]:
                raise ReceiptChainError(
                    f"receipt hash mismatch at sequence {expected_sequence}"
                )
            previous_hash = calculated
            expected_sequence += 1
        return len(rows), previous_hash

    def register_replay_fingerprint(
        self,
        *,
        fingerprint: str,
        event_id: str,
        continuity_id: str,
        seen_at: str | None = None,
    ) -> None:
        if len(fingerprint) != 64:
            raise ValueError("fingerprint must be SHA-256")
        timestamp = seen_at or utc_text()
        with self.transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO hermes_tag_replay_fingerprints(
                        fingerprint, event_id, continuity_id, seen_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (fingerprint, event_id, continuity_id, timestamp),
                )
            except sqlite3.IntegrityError as exc:
                raise ReplayDetected("continuity event fingerprint was already admitted") from exc

    def reserve_turn_event(
        self,
        event_id: str,
        *,
        now: datetime | None = None,
        stale_after_seconds: int = 300,
    ) -> None:
        """Reserve one provider event before any turn admission side effects."""
        if not event_id or len(event_id) > 512:
            raise ValueError("event_id must be a bounded non-empty string")
        current = now or utc_now()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM hermes_tag_turn_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if existing is not None:
                if existing["state"] == "pending":
                    age = current - parse_utc(existing["created_at"])
                    if age > timedelta(seconds=stale_after_seconds):
                        connection.execute(
                            "DELETE FROM hermes_tag_turn_events WHERE event_id=?",
                            (event_id,),
                        )
                    else:
                        raise ReplayDetected("turn event is already pending")
                else:
                    raise ReplayDetected("turn event was already admitted")
            connection.execute(
                """
                INSERT INTO hermes_tag_turn_events(event_id, state, created_at)
                VALUES (?, 'pending', ?)
                """,
                (event_id, utc_text(current)),
            )

    def complete_turn_event(
        self,
        event_id: str,
        *,
        admission_id: str,
        principal_id: str,
        surface_key: str,
        continuity_id: str,
        scope_digest: str,
        now: datetime | None = None,
    ) -> None:
        with self.transaction() as connection:
            changed = connection.execute(
                """
                UPDATE hermes_tag_turn_events
                SET state='complete', admission_id=?, principal_id=?,
                    surface_key=?, continuity_id=?, scope_digest=?, completed_at=?
                WHERE event_id=? AND state='pending'
                """,
                (
                    admission_id,
                    principal_id,
                    surface_key,
                    continuity_id,
                    scope_digest,
                    utc_text(now or utc_now()),
                    event_id,
                ),
            ).rowcount
            if changed != 1:
                raise ReplayDetected("turn event reservation is no longer active")

    def release_turn_event(self, event_id: str) -> None:
        """Release only a still-pending event after admission failure."""
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM hermes_tag_turn_events WHERE event_id=? AND state='pending'",
                (event_id,),
            )

    @staticmethod
    def _lease_use_matches(row: sqlite3.Row, lease: CapabilityLease) -> bool:
        return all(
            row[name] == getattr(lease, name)
            for name in (
                "lease_id",
                "principal_id",
                "decision_id",
                "capability",
                "intent_digest",
                "scope_digest",
                "budget_reservation_id",
                "approval_id",
            )
        )

    def reserve_lease_use(
        self,
        lease: CapabilityLease,
        *,
        now: datetime | None = None,
    ) -> None:
        """Atomically reserve one signed lease for exactly one effect attempt."""
        timestamp = utc_text(now or utc_now())
        with self.transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO hermes_tag_lease_uses(
                        lease_id, principal_id, decision_id, capability,
                        intent_digest, scope_digest, budget_reservation_id,
                        approval_id, state, reserved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?)
                    """,
                    (
                        lease.lease_id,
                        lease.principal_id,
                        lease.decision_id,
                        lease.capability,
                        lease.intent_digest,
                        lease.scope_digest,
                        lease.budget_reservation_id,
                        lease.approval_id,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                existing = connection.execute(
                    "SELECT * FROM hermes_tag_lease_uses WHERE lease_id=?",
                    (lease.lease_id,),
                ).fetchone()
                if existing is not None:
                    if not self._lease_use_matches(existing, lease):
                        raise LeaseReplay(
                            "capability lease use conflicts with signed authority"
                        ) from exc
                    raise LeaseReplay(
                        "one-shot capability lease was already presented"
                    ) from exc
                raise StorageError(
                    "capability lease use could not be persisted"
                ) from exc

    def require_reserved_lease_use(self, lease: CapabilityLease) -> None:
        """Require a prior pre-effect reservation for this exact signed lease."""
        connection = self.connection()
        try:
            row = connection.execute(
                "SELECT * FROM hermes_tag_lease_uses WHERE lease_id=?",
                (lease.lease_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise LeaseReplay(
                "capability lease must be verified before effect completion"
            )
        if not self._lease_use_matches(row, lease):
            raise LeaseReplay("capability lease use does not match signed authority")
        if row["state"] != "reserved":
            raise LeaseReplay("one-shot capability lease was already completed")

    def complete_lease_use(
        self,
        lease: CapabilityLease,
        *,
        success: bool,
        receipt_hash: str,
        now: datetime | None = None,
    ) -> None:
        """Finalize a reserved lease after its completion receipt is durable."""
        if len(receipt_hash) != 64:
            raise ValueError("receipt_hash must be SHA-256")
        timestamp = utc_text(now or utc_now())
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM hermes_tag_lease_uses WHERE lease_id=?",
                (lease.lease_id,),
            ).fetchone()
            if row is None:
                raise LeaseReplay(
                    "capability lease must be verified before effect completion"
                )
            if not self._lease_use_matches(row, lease):
                raise LeaseReplay(
                    "capability lease use does not match signed authority"
                )
            if row["state"] != "reserved":
                raise LeaseReplay("one-shot capability lease was already completed")
            changed = connection.execute(
                """
                UPDATE hermes_tag_lease_uses
                SET state='complete', completed_at=?, success=?, receipt_hash=?
                WHERE lease_id=? AND state='reserved'
                """,
                (timestamp, int(success), receipt_hash, lease.lease_id),
            ).rowcount
            if changed != 1:
                raise LeaseReplay("capability lease reservation was lost")

    def _reserve_budget_in_connection(
        self,
        connection: sqlite3.Connection,
        *,
        scope_digest: str,
        tokens: int,
        cost_usd: float | Decimal | None,
        limits: BudgetLimits,
        now: datetime,
    ) -> BudgetReservation:
        """Reserve budget inside an existing immediate transaction."""
        if tokens < 0:
            raise ValueError("tokens cannot be negative")
        cost_micro = _usd_to_micro(cost_usd)
        if cost_usd is None and (
            limits.hourly_cost_usd is not None
            or limits.daily_cost_usd is not None
        ):
            raise BudgetExceeded("priced budget requires a known worst-case cost")
        reservation_id = new_id("budget")
        for kind in ("hour", "day"):
            token_limit = (
                limits.hourly_tokens if kind == "hour" else limits.daily_tokens
            )
            cost_limit = (
                limits.hourly_cost_usd
                if kind == "hour"
                else limits.daily_cost_usd
            )
            if token_limit is None and cost_limit is None:
                continue
            start = _window_start(now, kind)
            row = connection.execute(
                """
                SELECT tokens, cost_micro_usd
                FROM hermes_tag_budget_usage
                WHERE scope_digest=? AND window_kind=? AND window_start=?
                """,
                (scope_digest, kind, start),
            ).fetchone()
            used_tokens = int(row["tokens"]) if row else 0
            used_cost = int(row["cost_micro_usd"]) if row else 0
            if token_limit is not None and used_tokens + tokens > token_limit:
                raise BudgetExceeded(f"{kind} token budget exhausted")
            if cost_limit is not None and used_cost + cost_micro > _usd_to_micro(
                cost_limit
            ):
                raise BudgetExceeded(f"{kind} cost budget exhausted")
            connection.execute(
                """
                INSERT INTO hermes_tag_budget_usage(
                    scope_digest, window_kind, window_start, tokens,
                    cost_micro_usd
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scope_digest, window_kind, window_start)
                DO UPDATE SET
                    tokens=tokens + excluded.tokens,
                    cost_micro_usd=cost_micro_usd + excluded.cost_micro_usd
                """,
                (scope_digest, kind, start, tokens, cost_micro),
            )
        created_at = utc_text(now)
        connection.execute(
            """
            INSERT INTO hermes_tag_budget_reservations(
                reservation_id, scope_digest, reserved_tokens,
                reserved_cost_micro_usd, state, created_at
            ) VALUES (?, ?, ?, ?, 'reserved', ?)
            """,
            (reservation_id, scope_digest, tokens, cost_micro, created_at),
        )
        return BudgetReservation(
            reservation_id=reservation_id,
            scope_digest=scope_digest,
            reserved_tokens=tokens,
            reserved_cost_usd=_micro_to_usd(cost_micro),
            state="reserved",
            created_at=created_at,
        )

    def reserve_budget(
        self,
        *,
        scope_digest: str,
        tokens: int,
        cost_usd: float | Decimal | None,
        limits: BudgetLimits,
        now: datetime | None = None,
    ) -> BudgetReservation:
        """Atomically reserve both hourly and daily capacity."""
        current = now or utc_now()
        with self.transaction() as connection:
            return self._reserve_budget_in_connection(
                connection,
                scope_digest=scope_digest,
                tokens=tokens,
                cost_usd=cost_usd,
                limits=limits,
                now=current,
            )

    def settle_budget(
        self,
        reservation_id: str,
        *,
        actual_tokens: int,
        actual_cost_usd: float | Decimal,
        now: datetime | None = None,
    ) -> BudgetReservation:
        """Settle a reservation and refund unused capacity atomically."""
        if actual_tokens < 0:
            raise ValueError("actual_tokens cannot be negative")
        actual_micro = _usd_to_micro(actual_cost_usd)
        current = now or utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM hermes_tag_budget_reservations WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise StorageError("unknown budget reservation")
            if row["state"] != "reserved":
                return BudgetReservation(
                    reservation_id=row["reservation_id"],
                    scope_digest=row["scope_digest"],
                    reserved_tokens=int(row["reserved_tokens"]),
                    reserved_cost_usd=_micro_to_usd(int(row["reserved_cost_micro_usd"])),
                    state=row["state"],
                    created_at=row["created_at"],
                )
            reserved_tokens = int(row["reserved_tokens"])
            reserved_micro = int(row["reserved_cost_micro_usd"])
            if actual_tokens > reserved_tokens or actual_micro > reserved_micro:
                raise BudgetExceeded("actual usage exceeds reserved authority")
            token_refund = reserved_tokens - actual_tokens
            cost_refund = reserved_micro - actual_micro
            created = parse_utc(row["created_at"])
            for kind in ("hour", "day"):
                start = _window_start(created, kind)
                usage = connection.execute(
                    """
                    SELECT tokens, cost_micro_usd
                    FROM hermes_tag_budget_usage
                    WHERE scope_digest=? AND window_kind=? AND window_start=?
                    """,
                    (row["scope_digest"], kind, start),
                ).fetchone()
                if usage is None:
                    continue
                connection.execute(
                    """
                    UPDATE hermes_tag_budget_usage
                    SET tokens=?, cost_micro_usd=?
                    WHERE scope_digest=? AND window_kind=? AND window_start=?
                    """,
                    (
                        max(0, int(usage["tokens"]) - token_refund),
                        max(0, int(usage["cost_micro_usd"]) - cost_refund),
                        row["scope_digest"],
                        kind,
                        start,
                    ),
                )
            connection.execute(
                """
                UPDATE hermes_tag_budget_reservations
                SET actual_tokens=?, actual_cost_micro_usd=?, state='settled',
                    settled_at=?
                WHERE reservation_id=?
                """,
                (actual_tokens, actual_micro, utc_text(current), reservation_id),
            )
        return BudgetReservation(
            reservation_id=reservation_id,
            scope_digest=row["scope_digest"],
            reserved_tokens=reserved_tokens,
            reserved_cost_usd=_micro_to_usd(reserved_micro),
            state="settled",
            created_at=row["created_at"],
        )

    def release_budget(
        self,
        reservation_id: str,
        *,
        now: datetime | None = None,
    ) -> BudgetReservation:
        return self._release_budget(reservation_id, now=now or utc_now())

    def _release_budget(
        self,
        reservation_id: str,
        *,
        now: datetime,
    ) -> BudgetReservation:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM hermes_tag_budget_reservations WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise StorageError("unknown budget reservation")
            if row["state"] == "reserved":
                created = parse_utc(row["created_at"])
                for kind in ("hour", "day"):
                    start = _window_start(created, kind)
                    usage = connection.execute(
                        """
                        SELECT tokens, cost_micro_usd
                        FROM hermes_tag_budget_usage
                        WHERE scope_digest=? AND window_kind=? AND window_start=?
                        """,
                        (row["scope_digest"], kind, start),
                    ).fetchone()
                    if usage is None:
                        continue
                    connection.execute(
                        """
                        UPDATE hermes_tag_budget_usage
                        SET tokens=?, cost_micro_usd=?
                        WHERE scope_digest=? AND window_kind=? AND window_start=?
                        """,
                        (
                            max(0, int(usage["tokens"]) - int(row["reserved_tokens"])),
                            max(
                                0,
                                int(usage["cost_micro_usd"])
                                - int(row["reserved_cost_micro_usd"]),
                            ),
                            row["scope_digest"],
                            kind,
                            start,
                        ),
                    )
                connection.execute(
                    """
                    UPDATE hermes_tag_budget_reservations
                    SET state='released', settled_at=?
                    WHERE reservation_id=?
                    """,
                    (utc_text(now), reservation_id),
                )
                state = "released"
            else:
                state = row["state"]
        return BudgetReservation(
            reservation_id=row["reservation_id"],
            scope_digest=row["scope_digest"],
            reserved_tokens=int(row["reserved_tokens"]),
            reserved_cost_usd=_micro_to_usd(int(row["reserved_cost_micro_usd"])),
            state=state,
            created_at=row["created_at"],
        )

    def budget_usage(
        self,
        scope_digest: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, dict[str, float | int]]:
        current = now or utc_now()
        connection = self.connection()
        result: dict[str, dict[str, float | int]] = {}
        try:
            for kind in ("hour", "day"):
                row = connection.execute(
                    """
                    SELECT tokens, cost_micro_usd
                    FROM hermes_tag_budget_usage
                    WHERE scope_digest=? AND window_kind=? AND window_start=?
                    """,
                    (scope_digest, kind, _window_start(current, kind)),
                ).fetchone()
                result[kind] = {
                    "tokens": int(row["tokens"]) if row else 0,
                    "cost_usd": _micro_to_usd(int(row["cost_micro_usd"]))
                    if row
                    else 0.0,
                }
        finally:
            connection.close()
        return result
