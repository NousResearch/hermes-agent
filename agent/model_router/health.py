"""Bounded provider/model health state for Hermes model routing.

The store shares the router SQLite database, is profile-safe through the
caller's db path, and serializes state transitions per database in-process.
Only retryable provider/infrastructure failures affect circuits; request 4xx,
auth, billing, safety, and local payload/context errors never poison health.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HEALTH_CLOSED = "closed"
HEALTH_OPEN = "open"
HEALTH_HALF_OPEN = "half_open"

OUTCOME_SUCCESS = "success"
OUTCOME_RETRYABLE_INFRASTRUCTURE = "retryable_infrastructure"
OUTCOME_NON_RETRYABLE = "non_retryable"
OUTCOME_IGNORED = "ignored"

DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_RESET_TIMEOUT_SECONDS = 30.0
DEFAULT_HALF_OPEN_SUCCESSES = 2
DEFAULT_MAX_ENTRIES = 256

_RETRYABLE_REASONS = frozenset(
    {
        "timeout",
        "rate_limit",
        "upstream_rate_limit",
        "overloaded",
        "server_error",
        "unknown",
    }
)
_NON_RETRYABLE_REASONS = frozenset(
    {
        "auth",
        "auth_permanent",
        "billing",
        "model_not_found",
        "provider_policy_blocked",
        "content_policy_blocked",
        "format_error",
        "ssl_cert_verification",
    }
)
_IGNORED_REASONS = frozenset(
    {
        "context_overflow",
        "payload_too_large",
        "image_too_large",
        "image_corrupt",
        "invalid_encrypted_content",
        "multimodal_tool_content_unsupported",
        "reasoning_mandatory",
        "thinking_signature",
        "long_context_tier",
        "oauth_long_context_beta_forbidden",
        "llama_cpp_grammar_pattern",
    }
)
_TRANSIENT_ERROR_TYPES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "ConnectionResetError",
        "NetworkError",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
        "WriteError",
        "WriteTimeout",
    }
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS router_health (
    endpoint_key TEXT PRIMARY KEY,
    provider TEXT NOT NULL DEFAULT '',
    model_id TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'closed',
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    consecutive_successes INTEGER NOT NULL DEFAULT 0,
    last_failure_at REAL,
    last_state_change_at REAL NOT NULL,
    probe_in_flight INTEGER NOT NULL DEFAULT 0,
    last_touched_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_router_health_touched
ON router_health(last_touched_at);
"""

_DB_LOCKS: dict[str, threading.RLock] = {}
_DB_LOCKS_GUARD = threading.Lock()


def _db_lock(path: str) -> threading.RLock:
    with _DB_LOCKS_GUARD:
        lock = _DB_LOCKS.get(path)
        if lock is None:
            lock = threading.RLock()
            _DB_LOCKS[path] = lock
        return lock


def endpoint_key(provider: str, model_id: str) -> str:
    """Stable provider/model key; no URL or credential material is persisted."""
    return f"{str(provider or '').strip().lower()}\x1f{str(model_id or '').strip()}"


@dataclass(frozen=True)
class HealthConfig:
    enabled: bool = True
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    reset_timeout_seconds: float = DEFAULT_RESET_TIMEOUT_SECONDS
    half_open_successes: int = DEFAULT_HALF_OPEN_SUCCESSES
    max_entries: int = DEFAULT_MAX_ENTRIES


@dataclass(frozen=True)
class HealthSnapshot:
    provider: str
    model_id: str
    state: str = HEALTH_CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_at: Optional[float] = None
    last_state_change_at: float = 0.0
    probe_in_flight: bool = False


@dataclass(frozen=True)
class HealthOutcome:
    category: str
    retryable_infrastructure: bool


def classify_health_outcome(
    *,
    success: bool = False,
    status_code: Optional[int] = None,
    retryable: Optional[bool] = None,
    reason: Optional[str] = None,
    error_type: Optional[str] = None,
) -> HealthOutcome:
    """Classify an API outcome without retaining an error body or prompt.

    Structured Hermes classifier fields win. HTTP 429/5xx and known transport
    exceptions are infrastructure failures. Other 4xx and safety/auth reasons
    are explicitly non-retryable; context/payload recovery is ignored.
    """
    if success:
        return HealthOutcome(OUTCOME_SUCCESS, False)

    normalized_reason = str(reason or "").strip().lower()
    if normalized_reason in _IGNORED_REASONS:
        return HealthOutcome(OUTCOME_IGNORED, False)
    if normalized_reason in _NON_RETRYABLE_REASONS:
        return HealthOutcome(OUTCOME_NON_RETRYABLE, False)
    if normalized_reason in _RETRYABLE_REASONS:
        return HealthOutcome(OUTCOME_RETRYABLE_INFRASTRUCTURE, True)

    try:
        status = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status = None
    if status == 429 or (status is not None and status >= 500):
        return HealthOutcome(OUTCOME_RETRYABLE_INFRASTRUCTURE, True)
    if status is not None and 400 <= status < 500:
        return HealthOutcome(OUTCOME_NON_RETRYABLE, False)
    if str(error_type or "").strip() in _TRANSIENT_ERROR_TYPES:
        return HealthOutcome(OUTCOME_RETRYABLE_INFRASTRUCTURE, True)
    if retryable is True:
        return HealthOutcome(OUTCOME_RETRYABLE_INFRASTRUCTURE, True)
    if retryable is False:
        return HealthOutcome(OUTCOME_NON_RETRYABLE, False)
    return HealthOutcome(OUTCOME_IGNORED, False)


def bind_agent_health(agent, store: Optional["RouterHealthStore"]) -> None:
    """Attach a privacy-safe health outcome adapter to a Hermes AIAgent."""
    if agent is None:
        return
    if store is None or not store.available:
        agent._router_health_callback = None
        agent._router_health_store = None
        return
    agent._router_health_store = store

    def record(**outcome) -> None:
        store.record_outcome(
            str(outcome.get("provider") or ""),
            str(outcome.get("model") or ""),
            success=bool(outcome.get("success", False)),
            status_code=outcome.get("status_code"),
            retryable=outcome.get("retryable"),
            reason=outcome.get("reason"),
            error_type=outcome.get("error_type"),
        )

    agent._router_health_callback = record


class RouterHealthStore:
    """SQLite-backed deterministic CLOSED/OPEN/HALF_OPEN circuit state."""

    def __init__(
        self,
        db_path,
        config: HealthConfig = HealthConfig(),
        *,
        clock=None,
        read_only: bool = False,
    ):
        self._db_path = str(Path(db_path))
        self._config = config
        self._clock = clock or time.time
        self._read_only = bool(read_only)
        self._lock = _db_lock(self._db_path)
        self._available = bool(config.enabled)
        if not self._available:
            return
        if self._read_only:
            self._available = Path(self._db_path).is_file()
            return
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self._connect() as conn:
                conn.executescript(_SCHEMA)
            os.chmod(self._db_path, 0o600)
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _connect(self) -> sqlite3.Connection:
        if self._read_only:
            uri = Path(self._db_path).resolve().as_uri() + "?mode=ro"
            return sqlite3.connect(uri, uri=True, timeout=5)
        conn = sqlite3.connect(self._db_path, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _read(self, conn: sqlite3.Connection, provider: str, model_id: str):
        return conn.execute(
            "SELECT provider, model_id, state, consecutive_failures,"
            " consecutive_successes, last_failure_at, last_state_change_at,"
            " probe_in_flight FROM router_health WHERE endpoint_key = ?",
            (endpoint_key(provider, model_id),),
        ).fetchone()

    @staticmethod
    def _snapshot(row, provider: str, model_id: str) -> HealthSnapshot:
        if row is None:
            return HealthSnapshot(provider=str(provider or ""), model_id=model_id)
        return HealthSnapshot(
            provider=row[0],
            model_id=row[1],
            state=row[2],
            consecutive_failures=int(row[3]),
            consecutive_successes=int(row[4]),
            last_failure_at=float(row[5]) if row[5] is not None else None,
            last_state_change_at=float(row[6]),
            probe_in_flight=bool(row[7]),
        )

    def snapshot(self, provider: str, model_id: str) -> HealthSnapshot:
        if not self._available:
            return HealthSnapshot(str(provider or ""), model_id)
        try:
            with self._lock, self._connect() as conn:
                return self._snapshot(self._read(conn, provider, model_id), provider, model_id)
        except Exception:
            return HealthSnapshot(str(provider or ""), model_id)

    def is_available(self, provider: str, model_id: str) -> bool:
        """Non-mutating eligibility check used while scoring candidates."""
        snap = self.snapshot(provider, model_id)
        if snap.state == HEALTH_CLOSED:
            return True
        if snap.state == HEALTH_HALF_OPEN:
            return (
                not snap.probe_in_flight
                or (self._clock() - snap.last_state_change_at)
                >= self._config.reset_timeout_seconds
            )
        return (self._clock() - snap.last_state_change_at) >= self._config.reset_timeout_seconds

    def claim_dispatch(self, provider: str, model_id: str) -> bool:
        """Atomically claim a dispatch, allowing only one HALF_OPEN probe."""
        if not self._available or self._read_only:
            return self.is_available(provider, model_id)
        now = float(self._clock())
        try:
            with self._lock, self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = self._read(conn, provider, model_id)
                if row is None or row[2] == HEALTH_CLOSED:
                    return True
                if row[2] == HEALTH_OPEN:
                    if now - float(row[6]) < self._config.reset_timeout_seconds:
                        return False
                    conn.execute(
                        "UPDATE router_health SET state=?, consecutive_successes=0,"
                        " probe_in_flight=1, last_state_change_at=?, last_touched_at=?"
                        " WHERE endpoint_key=?",
                        (HEALTH_HALF_OPEN, now, now, endpoint_key(provider, model_id)),
                    )
                    return True
                if (
                    bool(row[7])
                    and now - float(row[6]) < self._config.reset_timeout_seconds
                ):
                    return False
                # Reclaim a stale probe lease after the reset timeout so a
                # crashed/cancelled dispatch cannot wedge a durable circuit.
                conn.execute(
                    "UPDATE router_health SET probe_in_flight=1,"
                    " last_state_change_at=?, last_touched_at=? WHERE endpoint_key=?",
                    (now, now, endpoint_key(provider, model_id)),
                )
                return True
        except Exception:
            return True  # health is advisory; never block all inference on I/O failure

    def record_outcome(
        self,
        provider: str,
        model_id: str,
        *,
        success: bool = False,
        status_code: Optional[int] = None,
        retryable: Optional[bool] = None,
        reason: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> HealthOutcome:
        outcome = classify_health_outcome(
            success=success,
            status_code=status_code,
            retryable=retryable,
            reason=reason,
            error_type=error_type,
        )
        if not self._available or self._read_only or not model_id:
            return outcome
        if outcome.category == OUTCOME_SUCCESS:
            self.record_success(provider, model_id)
        elif outcome.retryable_infrastructure:
            self.record_failure(provider, model_id)
        return outcome

    def record_failure(self, provider: str, model_id: str) -> None:
        if self._read_only:
            return
        now = float(self._clock())
        key = endpoint_key(provider, model_id)
        try:
            with self._lock, self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = self._read(conn, provider, model_id)
                failures = (int(row[3]) if row else 0) + 1
                old_state = row[2] if row else HEALTH_CLOSED
                new_state = old_state
                changed_at = float(row[6]) if row else now
                if old_state == HEALTH_HALF_OPEN or failures >= self._config.failure_threshold:
                    new_state = HEALTH_OPEN
                    changed_at = now
                conn.execute(
                    "INSERT INTO router_health (endpoint_key, provider, model_id, state,"
                    " consecutive_failures, consecutive_successes, last_failure_at,"
                    " last_state_change_at, probe_in_flight, last_touched_at)"
                    " VALUES (?, ?, ?, ?, ?, 0, ?, ?, 0, ?)"
                    " ON CONFLICT(endpoint_key) DO UPDATE SET state=excluded.state,"
                    " consecutive_failures=excluded.consecutive_failures,"
                    " consecutive_successes=0, last_failure_at=excluded.last_failure_at,"
                    " last_state_change_at=excluded.last_state_change_at,"
                    " probe_in_flight=0, last_touched_at=excluded.last_touched_at",
                    (key, str(provider or ""), model_id, new_state, failures, now, changed_at, now),
                )
                self._prune(conn)
        except Exception:
            return

    def record_success(self, provider: str, model_id: str) -> None:
        if self._read_only:
            return
        now = float(self._clock())
        key = endpoint_key(provider, model_id)
        try:
            with self._lock, self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = self._read(conn, provider, model_id)
                if row is None:
                    return
                state = row[2]
                successes = int(row[4])
                if state == HEALTH_HALF_OPEN:
                    successes += 1
                    if successes >= self._config.half_open_successes:
                        conn.execute("DELETE FROM router_health WHERE endpoint_key=?", (key,))
                    else:
                        conn.execute(
                            "UPDATE router_health SET consecutive_failures=0,"
                            " consecutive_successes=?, probe_in_flight=0, last_touched_at=?"
                            " WHERE endpoint_key=?",
                            (successes, now, key),
                        )
                elif state == HEALTH_OPEN:
                    # A request may run through the fail-open current-model safety net.
                    conn.execute("DELETE FROM router_health WHERE endpoint_key=?", (key,))
                else:
                    conn.execute("DELETE FROM router_health WHERE endpoint_key=?", (key,))
        except Exception:
            return

    def _prune(self, conn: sqlite3.Connection) -> None:
        limit = max(1, int(self._config.max_entries))
        conn.execute(
            "DELETE FROM router_health WHERE endpoint_key IN ("
            " SELECT endpoint_key FROM router_health ORDER BY last_touched_at DESC"
            " LIMIT -1 OFFSET ?)",
            (limit,),
        )

    def list_snapshots(self, *, limit: int = 100) -> list[dict]:
        if not self._available:
            return []
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    "SELECT provider, model_id, state, consecutive_failures,"
                    " consecutive_successes, last_failure_at, last_state_change_at,"
                    " probe_in_flight FROM router_health"
                    " ORDER BY last_touched_at DESC LIMIT ?",
                    (max(1, min(int(limit), self._config.max_entries)),),
                ).fetchall()
        except Exception:
            return []
        return [
            {
                "provider": row[0],
                "model": row[1],
                "state": row[2],
                "consecutive_failures": int(row[3]),
                "consecutive_successes": int(row[4]),
                "last_failure_at": row[5],
                "last_state_change_at": row[6],
                "probe_in_flight": bool(row[7]),
            }
            for row in rows
        ]
