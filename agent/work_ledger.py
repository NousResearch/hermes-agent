"""Work ledger — the shared claim / lease / settle core of Kanban tasks, hosted-room driver tasks and
async delegations. DRAFT: not wired into any subsystem; this is the reviewable shape.

One SQLite file (path passed in; nothing resolves ``HERMES_HOME`` here) holds ``work_units`` (one
row per unit, CHECK-constrained status) and ``work_events`` (append-only, contiguous per-unit
``seq``). Every state change appends exactly one event in the same ``BEGIN IMMEDIATE`` transaction.
Time is always an injected ``now`` (default ``time.time()``) so sweeps and tests are deterministic.

State machine::

    admit → queued ─claim→ claimed ─settle→ settled | failed | cancelled
              ▲  ▲          │  │  └─renew (same generation, later expiry)
              │  └─release/defer (defer = queued with a not-before in expires_at)
              │             └─reclaim_expired, holder dead:
              │                 retryable kind → queued (attempts+1), or failed once attempts ≥ max
              └─resolve──── indeterminate ◄── non-retryable kind (default: never silently re-run)

Semantics taken from the strongest existing implementation of each primitive:
admission idempotency and settlement replay from ``gateway/hosted_room_driver.py`` (``admit_task``,
``_settlement``: identical → original, divergent → typed conflict, replay checked BEFORE the lease
fence so a late identical settle still succeeds); fencing on a monotonic ``lease_generation``
(rooms' driver lease) rather than an opaque holder string; failure accounting against a per-kind
cap from Kanban (``kanban_db_dispatch._record_task_failure``); the holder identity is Kanban's
restart-stable ``hermes_cli.process_identity.process_fingerprint`` ("<boot epoch>|<start time>").

Migration map
-------------

=================  ===========================================  ===================================  ==================================
Ledger             Kanban (``hermes_cli/kanban_db*.py``)        Rooms (``gateway/hosted_room_driver``) Async delegation (``tools/async_delegation``)
=================  ===========================================  ===================================  ==================================
unit_id            ``tasks.id``                                 ``(room_id, task_id)`` joined         ``delegation_id``
kind               board / ``workflow_template_id``             ``"room-turn"``                       ``"delegation"``
status             todo/ready→queued, running→claimed,          queued, running→claimed,              running→claimed, completed→settled,
                   done→settled, blocked/gave_up→failed         indeterminate, settled/failed/        error/timeout→failed,
                                                                cancelled (stopping/deferred stay     unknown→indeterminate
                                                                private)
payload_digest     (none; ``idempotency_key`` lookup only)      ``payload_digest``                    (none; ``INSERT OR REPLACE`` — drops)
holder_id          ``claim_lock`` ("host:pid")                  ``gateway_id`` + ``process_generation`` ``owner_pid``
holder_fingerprint ``worker_started_at``                        (logical identity only)               ``owner_started_at`` (start time only)
lease_generation   ``current_run_id`` (de-facto epoch)          driver ``lease_generation``           (none; ``delivery_claim`` uuid)
expires_at         ``claim_expires``                            driver lease ``expires_at``           (none for work)
attempts / max     ``consecutive_failures`` / ``max_retries``   (none; explicit requeue)              (none for work)
settlement_id      (none; CAS on status, no replay value)       ``settlement_id``                     (none; ``_finalize`` no-op)
result_json        ``tasks.result``                             ``result_json``                       ``result_json``
work_events        ``task_events``                              (``hosted_room_events`` is room-level) (none)
admit              ``create_task``                              ``admit_task``                        ``_persist_dispatch``
claim              ``claim_task`` / ``_claim_and_open_run``     ``acquire_lease`` + ``start_task``    (implicit at spawn)
renew              ``heartbeat_claim``                          ``renew_lease``                       (none)
settle             ``complete_task`` / ``block_task``           ``settle_task``                       ``_finalize`` / ``_persist_completion``
release / defer    ``release_stale_claims`` release branch      ``requeue_*`` / ``defer_indeterminate`` (none)
reclaim_expired    ``release_stale_claims`` +                   ``recover_room``                      ``recover_abandoned_delegations``
                   ``detect_crashed_workers``
=================  ===========================================  ===================================  ==================================

Kept private by each subsystem: Kanban — task_runs history, parents/links/comments, review lane,
exit-code taxonomy (rate-limit / infra failures that do not count), worker spawn/terminate,
notify cursors. Rooms — room authority epoch, FIFO-per-room ordering, execution/cancel generations
and two-phase stop, remote-run receipts, publication planning. Async delegation — in-process stall
monitor, routing origin, the delivery sub-machine (claim/complete/release/drop to the parent turn),
partial child results.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, get_args

from agent.deadline import clamp_timeout
from gateway.hosted_rooms_common import compact_json, fenced_update
from hermes_cli.process_identity import UNVERIFIED_WORKER_FINGERPRINT, process_fingerprint
from hermes_cli.sqlite_util import open_db, transaction

Status = Literal["queued", "claimed", "settled", "failed", "cancelled", "indeterminate"]
Outcome = Literal["settled", "failed", "cancelled"]
STATUSES: tuple[str, ...] = get_args(Status)
OUTCOMES: tuple[str, ...] = get_args(Outcome)
_RESOLVE_TARGETS = ("queued", "failed", "cancelled")
# Liveness oracle for reclaim: ``alive(holder_id, holder_fingerprint)``. Called OUTSIDE any write
# transaction (it may probe the process table), then the reclaim CAS re-checks the generation.
Liveness = Callable[[str, str], bool]

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS work_units (
    unit_id            TEXT PRIMARY KEY,
    kind               TEXT NOT NULL,
    status             TEXT NOT NULL CHECK (status IN ({", ".join(repr(s) for s in STATUSES)})),
    payload_json       TEXT NOT NULL,
    payload_digest     TEXT NOT NULL,
    holder_id          TEXT,
    holder_fingerprint TEXT,
    lease_generation   INTEGER NOT NULL DEFAULT 0,
    expires_at         REAL,
    attempts           INTEGER NOT NULL DEFAULT 0,
    max_attempts       INTEGER NOT NULL CHECK (max_attempts >= 1),
    settlement_id      TEXT,
    result_json        TEXT,
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_work_units_claimable ON work_units(status, kind, created_at);
CREATE TABLE IF NOT EXISTS work_events (
    event_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id   TEXT NOT NULL REFERENCES work_units(unit_id),
    seq       INTEGER NOT NULL,
    kind      TEXT NOT NULL,
    at        REAL NOT NULL,
    data_json TEXT NOT NULL,
    UNIQUE (unit_id, seq)
);
CREATE TRIGGER IF NOT EXISTS work_events_append_only BEFORE UPDATE ON work_events
BEGIN SELECT RAISE(ABORT, 'work_events is append-only'); END;
"""


class WorkLedgerError(Exception):
    """Base class for every ledger refusal."""


class LedgerValidationError(WorkLedgerError, ValueError):
    """An argument (ttl, outcome, policy, target status) is invalid."""


class UnknownUnitError(WorkLedgerError, LookupError):
    """No unit with that id."""


class AdmissionConflictError(WorkLedgerError):
    """``unit_id`` is already admitted with a different kind or payload digest."""


class SettlementConflictError(WorkLedgerError):
    """The unit already carries a different terminal settlement."""


class StaleLeaseError(WorkLedgerError):
    """The lease generation (or holder) no longer owns the unit, or the lease already expired."""


class InvalidTransitionError(WorkLedgerError):
    """The unit is not in a state the operation can leave from."""


@dataclass(frozen=True)
class KindPolicy:
    """Per-kind recovery policy. ``retryable=False`` (default) sends dead-holder work to
    ``indeterminate``; ``True`` requeues it until ``attempts`` reaches ``max_attempts``."""
    retryable: bool = False
    max_attempts: int = 1


@dataclass(frozen=True)
class Lease:
    unit_id: str
    holder_id: str
    generation: int
    expires_at: float


@dataclass(frozen=True)
class Settlement:
    unit_id: str
    settlement_id: str
    outcome: str
    result: Any
    idempotent: bool


@dataclass(frozen=True)
class WorkEvent:
    event_id: int
    unit_id: str
    seq: int
    kind: str
    at: float
    data: dict[str, Any]


@dataclass(frozen=True)
class Reclaimed:
    unit_id: str
    status: str
    attempts: int


def payload_digest_of(payload: Any) -> str:
    """sha256 of the canonical (sorted-key, compact) JSON form — the rooms digest shape."""
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()


def _now(now: float | None) -> float:
    return time.time() if now is None else float(now)


def _ttl(ttl: float) -> float:
    value = clamp_timeout(ttl)
    if value is None:
        raise LedgerValidationError("ttl must be a positive number")
    return value


def _append_event(conn: sqlite3.Connection, unit_id: str, kind: str, at: float, data: dict[str, Any]) -> None:
    (seq,) = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM work_events WHERE unit_id=?", (unit_id,)).fetchone()
    conn.execute("INSERT INTO work_events (unit_id, seq, kind, at, data_json) VALUES (?, ?, ?, ?, ?)",
                 (unit_id, seq, kind, at, compact_json(data)))


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


# CAS fence for every lease-holder write: the generation is the authority, holder_id a cross-check.
_FENCE = "unit_id=? AND status='claimed' AND lease_generation=? AND holder_id=?"


class WorkLedger:
    """One ledger file. Each operation opens, runs one IMMEDIATE transaction, and closes."""

    def __init__(self, db_path: Path | str, policies: Mapping[str, KindPolicy] | None = None) -> None:
        self.db_path = Path(db_path)
        self._policies = dict(policies or {})
        for kind, policy in self._policies.items():
            if policy.max_attempts < 1:
                raise LedgerValidationError(f"max_attempts for kind {kind!r} must be >= 1")

    def policy(self, kind: str) -> KindPolicy:
        return self._policies.get(kind, KindPolicy())

    def _txn(self, *, immediate: bool = True):
        conn = open_db(self.db_path, db_label="work-ledger", busy_timeout_ms=10_000, foreign_keys=True,
                       initialize=_initialize_schema)
        return transaction(conn, immediate=immediate)

    @staticmethod
    def _load(conn: sqlite3.Connection, unit_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM work_units WHERE unit_id=?", (unit_id,)).fetchone()
        if row is None:
            raise UnknownUnitError(unit_id)
        return row

    def get(self, unit_id: str) -> dict[str, Any]:
        with self._txn(immediate=False) as conn:
            return dict(self._load(conn, unit_id))

    def admit(self, unit_id: str, kind: str, payload: Any, payload_digest: str | None = None, *,
              now: float | None = None) -> tuple[dict[str, Any], bool]:
        """Queue a unit. Returns ``(row, created)``; an identical re-admission returns the existing row
        with ``created=False``, a different kind or digest raises :class:`AdmissionConflictError`."""
        at = _now(now)
        payload_json = compact_json(payload)
        digest = payload_digest or payload_digest_of(payload)
        with self._txn() as conn:
            existing = conn.execute("SELECT * FROM work_units WHERE unit_id=?", (unit_id,)).fetchone()
            if existing is not None:
                if (existing["kind"], existing["payload_digest"]) != (kind, digest):
                    raise AdmissionConflictError(f"{unit_id} is already admitted with different work")
                return dict(existing), False
            conn.execute("""INSERT INTO work_units (unit_id, kind, status, payload_json, payload_digest,
                            max_attempts, created_at, updated_at) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)""",
                         (unit_id, kind, payload_json, digest, self.policy(kind).max_attempts, at, at))
            _append_event(conn, unit_id, "admitted", at, {"to": "queued", "payload_digest": digest})
            return dict(self._load(conn, unit_id)), True

    def claim(self, holder_id: str, ttl: float, *, unit_id: str | None = None, kind: str | None = None,
              holder_fingerprint: str | None = None, now: float | None = None) -> Lease | None:
        """CAS the oldest claimable queued unit (optionally one id / one kind) to ``claimed`` under a new
        ``lease_generation``. ``None`` when nothing is claimable or the CAS lost. The fingerprint defaults
        to this process's ``process_fingerprint`` (``UNVERIFIED_WORKER_FINGERPRINT`` when unreadable)."""
        at, window = _now(now), _ttl(ttl)
        fingerprint = holder_fingerprint or process_fingerprint(os.getpid()) or UNVERIFIED_WORKER_FINGERPRINT
        where: list[str] = ["status='queued'", "(expires_at IS NULL OR expires_at <= ?)"]
        params: list[Any] = [at]
        for column, value in (("unit_id", unit_id), ("kind", kind)):
            if value is not None:
                where.append(f"{column}=?")
                params.append(value)
        with self._txn() as conn:
            row = conn.execute(f"SELECT unit_id, lease_generation FROM work_units WHERE {' AND '.join(where)} "
                               "ORDER BY created_at, unit_id LIMIT 1", params).fetchone()
            if row is None:
                return None
            generation, expires_at = int(row["lease_generation"]) + 1, at + window
            if conn.execute("""UPDATE work_units SET status='claimed', holder_id=?, holder_fingerprint=?,
                               lease_generation=?, expires_at=?, updated_at=?
                               WHERE unit_id=? AND status='queued' AND lease_generation=?""",
                            (holder_id, fingerprint, generation, expires_at, at, row["unit_id"],
                             row["lease_generation"])).rowcount != 1:
                return None
            _append_event(conn, row["unit_id"], "claimed", at, {"from": "queued", "to": "claimed", "holder_id": holder_id,
                                                                "generation": generation, "expires_at": expires_at})
            return Lease(row["unit_id"], holder_id, generation, expires_at)

    def renew(self, lease: Lease, ttl: float, *, now: float | None = None) -> Lease:
        """Extend an unexpired lease; same generation. Fails closed once expired (rooms ``renew_lease``)."""
        at, window = _now(now), _ttl(ttl)
        with self._txn() as conn:
            fenced_update(conn, f"UPDATE work_units SET expires_at=?, updated_at=? WHERE {_FENCE} "
                          "AND expires_at > ?", (at + window, at, lease.unit_id, lease.generation,
                                                 lease.holder_id, at),
                          StaleLeaseError(f"lease g{lease.generation} on {lease.unit_id} is stale or expired"))
            _append_event(conn, lease.unit_id, "renewed", at, {"generation": lease.generation, "expires_at": at + window})
        return Lease(lease.unit_id, lease.holder_id, lease.generation, at + window)

    def settle(self, lease: Lease, settlement_id: str, outcome: str, result: Any = None, *,
               now: float | None = None) -> Settlement:
        """Terminal write, replay-idempotent: the same ``(settlement_id, outcome, result)`` returns the
        original even after the lease moved on; anything else on a settled unit is a conflict. A fresh
        settlement requires the lease generation to still own the unit (expiry alone does not void it
        until ``reclaim_expired`` moves the unit — the generation is the fence)."""
        if outcome not in OUTCOMES:
            raise LedgerValidationError(f"outcome must be one of {OUTCOMES}")
        at, result_json = _now(now), compact_json(result)
        with self._txn() as conn:
            row = self._load(conn, lease.unit_id)
            if row["settlement_id"] is not None:
                if (row["settlement_id"], row["status"], row["result_json"]) != (settlement_id, outcome, result_json):
                    raise SettlementConflictError(f"{lease.unit_id} already has a different settlement")
                return Settlement(lease.unit_id, settlement_id, outcome, json.loads(result_json), idempotent=True)
            fenced_update(conn, f"""UPDATE work_units SET status=?, settlement_id=?, result_json=?, expires_at=NULL,
                                    updated_at=? WHERE {_FENCE}""",
                          (outcome, settlement_id, result_json, at, lease.unit_id, lease.generation, lease.holder_id),
                          StaleLeaseError(f"lease g{lease.generation} no longer owns {lease.unit_id}"))
            _append_event(conn, lease.unit_id, outcome, at, {"from": "claimed", "to": outcome,
                                                             "settlement_id": settlement_id, "generation": lease.generation})
        return Settlement(lease.unit_id, settlement_id, outcome, result, idempotent=False)

    def release(self, lease: Lease, *, not_before: float | None = None, now: float | None = None) -> None:
        """Give the unit back to ``queued`` without charging an attempt. ``not_before`` makes it a defer:
        claim skips the unit until then (Kanban cooldown / ``scheduled``, AD ``defer``)."""
        at = _now(now)
        with self._txn() as conn:
            fenced_update(conn, f"""UPDATE work_units SET status='queued', holder_id=NULL, holder_fingerprint=NULL,
                                    expires_at=?, updated_at=? WHERE {_FENCE}""",
                          (not_before, at, lease.unit_id, lease.generation, lease.holder_id),
                          StaleLeaseError(f"lease g{lease.generation} no longer owns {lease.unit_id}"))
            _append_event(conn, lease.unit_id, "deferred" if not_before is not None else "released", at,
                          {"from": "claimed", "to": "queued", "generation": lease.generation, "not_before": not_before})

    def defer(self, lease: Lease, until: float, *, now: float | None = None) -> None:
        self.release(lease, not_before=float(until), now=now)

    def reclaim_expired(self, liveness: Liveness, *, now: float | None = None) -> list[Reclaimed]:
        """Recover expired leases whose holder is dead per ``liveness``. Live holders are left alone
        (their next ``renew`` fails closed; the caller decides whether to terminate them). Dead holder:
        the attempt is charged; a retryable kind requeues until ``max_attempts`` then fails, a
        non-retryable kind goes to ``indeterminate`` — uncertain work is never silently re-run."""
        at = _now(now)
        with self._txn(immediate=False) as conn:
            expired = conn.execute("""SELECT unit_id, kind, holder_id, holder_fingerprint, lease_generation,
                                      attempts, max_attempts FROM work_units WHERE status='claimed' AND expires_at <= ?
                                      ORDER BY expires_at, unit_id""", (at,)).fetchall()
        dead = [row for row in expired if not liveness(row["holder_id"], row["holder_fingerprint"])]
        reclaimed: list[Reclaimed] = []
        for row in dead:
            attempts = int(row["attempts"]) + 1
            if not self.policy(row["kind"]).retryable:
                target = "indeterminate"
            else:
                target = "failed" if attempts >= int(row["max_attempts"]) else "queued"
            with self._txn() as conn:
                if conn.execute("""UPDATE work_units SET status=?, attempts=?, holder_id=NULL, holder_fingerprint=NULL,
                                   expires_at=NULL, updated_at=? WHERE unit_id=? AND status='claimed'
                                   AND lease_generation=? AND expires_at <= ?""",
                                (target, attempts, at, row["unit_id"], row["lease_generation"], at)).rowcount != 1:
                    continue  # renewed / settled / reclaimed by a peer since the scan
                _append_event(conn, row["unit_id"], "reclaimed", at, {
                    "from": "claimed", "to": target, "attempts": attempts, "holder_id": row["holder_id"],
                    "generation": int(row["lease_generation"])})
            reclaimed.append(Reclaimed(row["unit_id"], target, attempts))
        return reclaimed

    def resolve_indeterminate(self, unit_id: str, to: str, *, reason: str = "", now: float | None = None) -> None:
        """Explicit operator/receipt resolution of ``indeterminate`` work (rooms ``requeue_indeterminate_task``
        accepts at-least-once; ``failed``/``cancelled`` close it)."""
        if to not in _RESOLVE_TARGETS:
            raise LedgerValidationError(f"indeterminate work resolves to one of {_RESOLVE_TARGETS}")
        at = _now(now)
        with self._txn() as conn:
            fenced_update(conn, "UPDATE work_units SET status=?, updated_at=? WHERE unit_id=? AND status='indeterminate'",
                          (to, at, unit_id), InvalidTransitionError(f"{unit_id} is not indeterminate"))
            _append_event(conn, unit_id, "resolved", at, {"from": "indeterminate", "to": to, "reason": reason})

    def events(self, unit_id: str | None = None, *, after_event_id: int = 0, limit: int = 1000) -> list[WorkEvent]:
        """Events in commit order: one unit's history, or a global tail from ``after_event_id``."""
        sql = "SELECT * FROM work_events WHERE event_id > ?"
        params: list[Any] = [after_event_id]
        if unit_id is not None:
            sql += " AND unit_id=?"
            params.append(unit_id)
        with self._txn(immediate=False) as conn:
            rows = conn.execute(sql + " ORDER BY event_id LIMIT ?", (*params, limit)).fetchall()
        return [WorkEvent(r["event_id"], r["unit_id"], r["seq"], r["kind"], r["at"], json.loads(r["data_json"]))
                for r in rows]
