"""Typed, route-neutral metadata receipts for delegated model calls.

The receipt boundary deliberately excludes prompts, outputs, provider envelopes,
and credentials.  It records only identities, routing state, bounded status, and
token counts needed by local reporting.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home


UTC = timezone.utc
RUN_KINDS = frozenset({"production", "canary", "synthetic", "evaluation"})
ROUTES = frozenset({"gemini", "frontier"})
TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "timeout", "cancelled", "malformed", "denied", "oversized", "interrupted"}
)
DISPOSITIONS = frozenset(
    {"accepted_as_is", "materially_revised", "rejected", "fallback_replacement", "unknown"}
)
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "total_tokens",
)
MAX_SAFE_INTEGER = 9_007_199_254_740_991


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime | None = None) -> str:
    return _as_utc(value).isoformat()


def _safe_identifier(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty) or len(value) > 256:
        raise ValueError(f"{field} must be a bounded string")
    allowed = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/")
    if any(char not in allowed for char in value):
        raise ValueError(f"{field} contains unsupported characters")
    return value


def _normalized_usage(
    usage: Mapping[str, Any] | None, usage_status: str
) -> dict[str, int | None]:
    if usage_status == "unavailable":
        if usage not in (None, {}):
            raise ValueError("usage must be absent when usage_status is unavailable")
        return {field: None for field in TOKEN_FIELDS}
    if usage_status != "complete" or not isinstance(usage, Mapping):
        raise ValueError("usage_status must be complete or unavailable")
    counts = {field: usage.get(field) for field in TOKEN_FIELDS}
    if not all(type(value) is int and 0 <= value <= MAX_SAFE_INTEGER for value in counts.values()):
        raise ValueError("usage counts must be non-negative safe integers")
    if counts["total_tokens"] != sum(counts[field] for field in TOKEN_FIELDS[:-1]):
        raise ValueError("usage total_tokens must equal the component sum")
    return counts


def route_usage_from_mapping(usage: Mapping[str, Any] | None) -> dict[str, int] | None:
    """Normalize known provider aliases into the closed route-receipt grammar."""
    if not isinstance(usage, Mapping):
        return None
    aliases = {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_read_tokens": usage.get(
            "cache_read_tokens",
            usage.get("cache_read_input_tokens", usage.get("cached_input_tokens", 0)),
        ),
        "cache_write_tokens": usage.get(
            "cache_write_tokens", usage.get("cache_creation_input_tokens", 0)
        ),
        "reasoning_tokens": usage.get(
            "reasoning_tokens", usage.get("thoughts_tokens", 0)
        ),
    }
    if not all(type(value) is int and 0 <= value <= MAX_SAFE_INTEGER for value in aliases.values()):
        return None
    normalized = {field: int(value) for field, value in aliases.items()}
    normalized["total_tokens"] = sum(normalized.values())
    reported_total = usage.get("total_tokens")
    if reported_total is not None and reported_total != normalized["total_tokens"]:
        return None
    return normalized


class RouteReceiptStore:
    """Private SQLite store for route-neutral call and parent-disposition receipts."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else get_hermes_home() / "routing" / "gemini-routing.sqlite3"
        self._initialize()

    def _initialize(self) -> None:
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise RuntimeError("route receipt path must not be a symlink")
        if self.path.exists() and not self.path.is_file():
            raise RuntimeError("route receipt database path must be a regular file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS route_attempts (
                    stable_call_id TEXT PRIMARY KEY,
                    parent_session_id TEXT NOT NULL,
                    parent_turn_id TEXT NOT NULL,
                    child_session_id TEXT NOT NULL,
                    task_index INTEGER NOT NULL,
                    routing_day TEXT NOT NULL,
                    run_kind TEXT NOT NULL,
                    route_requested TEXT NOT NULL,
                    route_decision TEXT NOT NULL,
                    route_reason TEXT NOT NULL,
                    data_classification TEXT NOT NULL,
                    output_contract TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at_utc TEXT NOT NULL,
                    completed_at_utc TEXT,
                    terminal_route TEXT,
                    terminal_provider TEXT,
                    terminal_model TEXT,
                    usage_status TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cache_read_tokens INTEGER,
                    cache_write_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    total_tokens INTEGER,
                    fallback_used INTEGER NOT NULL DEFAULT 0,
                    fallback_from_call_id TEXT,
                    fallback_call_id TEXT,
                    error_code TEXT,
                    created_at_utc TEXT NOT NULL,
                    CHECK (run_kind IN ('production','canary','synthetic','evaluation')),
                    CHECK (route_decision IN ('gemini','frontier')),
                    CHECK (usage_status IN ('complete','unavailable')),
                    CHECK (fallback_used IN (0,1)),
                    FOREIGN KEY (fallback_from_call_id) REFERENCES route_attempts(stable_call_id),
                    FOREIGN KEY (fallback_call_id) REFERENCES route_attempts(stable_call_id)
                );
                CREATE INDEX IF NOT EXISTS idx_route_attempts_population
                    ON route_attempts(run_kind, routing_day, route_decision, stable_call_id);
                CREATE TABLE IF NOT EXISTS parent_disposition_receipts (
                    disposition_receipt_id TEXT PRIMARY KEY,
                    stable_call_id TEXT NOT NULL,
                    parent_session_id TEXT NOT NULL,
                    parent_turn_id TEXT NOT NULL,
                    disposition TEXT NOT NULL,
                    proof_kind TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL,
                    CHECK (disposition IN (
                        'accepted_as_is','materially_revised','rejected','fallback_replacement','unknown'
                    )),
                    FOREIGN KEY (stable_call_id) REFERENCES route_attempts(stable_call_id)
                );
                CREATE INDEX IF NOT EXISTS idx_parent_disposition_call
                    ON parent_disposition_receipts(stable_call_id, recorded_at_utc, disposition_receipt_id);
                """
            )
        self._enforce_modes()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _enforce_modes(self) -> None:
        if self.path.exists():
            self.path.chmod(0o600)
            if stat.S_IMODE(self.path.stat().st_mode) != 0o600:
                raise RuntimeError("could not enforce private route receipt permissions")

    def prepare_attempt(
        self,
        *,
        parent_session_id: str,
        parent_turn_id: str,
        child_session_id: str,
        task_index: int,
        run_kind: str,
        route_requested: str,
        route_decision: str,
        route_reason: str,
        data_classification: str,
        output_contract: str,
        provider: str,
        model: str,
        stable_call_id: str | None = None,
        fallback_from_call_id: str | None = None,
        started_at: datetime | None = None,
    ) -> str:
        if run_kind not in RUN_KINDS:
            raise ValueError("run_kind must be production, canary, synthetic, or evaluation")
        if route_decision not in ROUTES:
            raise ValueError("route_decision must be gemini or frontier")
        call_id = _safe_identifier(stable_call_id or f"rr_{secrets.token_hex(16)}", "stable_call_id")
        parent_session_id = _safe_identifier(parent_session_id, "parent_session_id")
        parent_turn_id = _safe_identifier(parent_turn_id, "parent_turn_id", allow_empty=True)
        child_session_id = _safe_identifier(child_session_id, "child_session_id")
        provider = _safe_identifier(provider, "provider")
        model = _safe_identifier(model, "model")
        if fallback_from_call_id is not None:
            fallback_from_call_id = _safe_identifier(fallback_from_call_id, "fallback_from_call_id")
        started = _as_utc(started_at)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO route_attempts (
                       stable_call_id, parent_session_id, parent_turn_id, child_session_id,
                       task_index, routing_day, run_kind, route_requested, route_decision,
                       route_reason, data_classification, output_contract, provider, model,
                       status, started_at_utc, usage_status, fallback_from_call_id, created_at_utc
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?,
                             'unavailable', ?, ?)""",
                (
                    call_id,
                    parent_session_id,
                    parent_turn_id,
                    child_session_id,
                    int(task_index),
                    started.date().isoformat(),
                    run_kind,
                    str(route_requested),
                    route_decision,
                    str(route_reason),
                    str(data_classification),
                    str(output_contract),
                    provider,
                    model,
                    started.isoformat(),
                    fallback_from_call_id,
                    _iso(),
                ),
            )
        self._enforce_modes()
        return call_id

    def complete_attempt(
        self,
        stable_call_id: str,
        *,
        status: str,
        terminal_route: str,
        terminal_provider: str,
        terminal_model: str,
        usage: Mapping[str, Any] | None,
        usage_status: str,
        fallback_used: bool = False,
        fallback_call_id: str | None = None,
        error_code: str | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError("status must be terminal")
        if terminal_route not in ROUTES:
            raise ValueError("terminal_route must be gemini or frontier")
        counts = _normalized_usage(usage, usage_status)
        if fallback_call_id is not None:
            fallback_call_id = _safe_identifier(fallback_call_id, "fallback_call_id")
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE route_attempts SET
                       status=?, completed_at_utc=?, terminal_route=?, terminal_provider=?,
                       terminal_model=?, usage_status=?, input_tokens=?, output_tokens=?,
                       cache_read_tokens=?, cache_write_tokens=?, reasoning_tokens=?, total_tokens=?,
                       fallback_used=?, fallback_call_id=?, error_code=?
                   WHERE stable_call_id=? AND status='prepared'""",
                (
                    status,
                    _iso(completed_at),
                    terminal_route,
                    _safe_identifier(terminal_provider, "terminal_provider"),
                    _safe_identifier(terminal_model, "terminal_model"),
                    usage_status,
                    *(counts[field] for field in TOKEN_FIELDS),
                    int(bool(fallback_used)),
                    fallback_call_id,
                    error_code,
                    _safe_identifier(stable_call_id, "stable_call_id"),
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("route attempt is missing or already terminal")
        self._enforce_modes()

    def get_attempt(self, stable_call_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM route_attempts WHERE stable_call_id=?", (stable_call_id,)
            ).fetchone()
        if row is None:
            raise KeyError(stable_call_id)
        return dict(row)

    def link_fallback(self, stable_call_id: str, fallback_call_id: str) -> None:
        """Bind a terminal primary call to its separately receipted fallback call."""
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE route_attempts SET fallback_used=1, fallback_call_id=?
                   WHERE stable_call_id=? AND status!='prepared'
                     AND fallback_call_id IS NULL
                     AND EXISTS (
                         SELECT 1 FROM route_attempts child
                         WHERE child.stable_call_id=?
                           AND child.fallback_from_call_id=route_attempts.stable_call_id
                     )""",
                (
                    _safe_identifier(fallback_call_id, "fallback_call_id"),
                    _safe_identifier(stable_call_id, "stable_call_id"),
                    fallback_call_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("fallback receipt link is missing or already frozen")
        self._enforce_modes()

    def record_parent_disposition(
        self,
        *,
        stable_call_id: str,
        parent_session_id: str,
        parent_turn_id: str,
        disposition: str,
        proof_kind: str,
        recorded_at: datetime | None = None,
        disposition_receipt_id: str | None = None,
    ) -> str:
        if disposition not in DISPOSITIONS:
            raise ValueError("disposition is not recognized")
        receipt_id = _safe_identifier(
            disposition_receipt_id or f"prd_{secrets.token_hex(16)}", "disposition_receipt_id"
        )
        stable_call_id = _safe_identifier(stable_call_id, "stable_call_id")
        parent_session_id = _safe_identifier(parent_session_id, "parent_session_id")
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO parent_disposition_receipts (
                       disposition_receipt_id, stable_call_id, parent_session_id,
                       parent_turn_id, disposition, proof_kind, recorded_at_utc
                   )
                   SELECT ?, stable_call_id, parent_session_id, ?, ?, ?, ?
                   FROM route_attempts
                   WHERE stable_call_id=? AND parent_session_id=?""",
                (
                    receipt_id,
                    _safe_identifier(parent_turn_id, "parent_turn_id", allow_empty=True),
                    disposition,
                    _safe_identifier(proof_kind, "proof_kind"),
                    _iso(recorded_at),
                    stable_call_id,
                    parent_session_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("parent session does not match the route attempt")
        self._enforce_modes()
        return receipt_id

    def get_parent_disposition(self, disposition_receipt_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT disposition_receipt_id, stable_call_id, parent_session_id,
                          parent_turn_id, disposition, proof_kind, recorded_at_utc
                   FROM parent_disposition_receipts WHERE disposition_receipt_id=?""",
                (disposition_receipt_id,),
            ).fetchone()
        if row is None:
            raise KeyError(disposition_receipt_id)
        return dict(row)


class RouteReceiptChild:
    """AIAgent-compatible frontier wrapper that records one metadata-only call."""

    _LOCAL_ATTRS = {
        "child",
        "store",
        "parent_session_id",
        "parent_turn_id",
        "task_index",
        "run_kind",
        "route_requested",
        "route_reason",
        "data_classification",
        "output_contract",
        "fallback_from_call_id",
        "route_receipt_id",
        "_route_metadata",
        "_delegate_release_ownership",
    }

    def __init__(
        self,
        *,
        child: Any,
        store: RouteReceiptStore,
        parent_session_id: str,
        parent_turn_id: str,
        task_index: int,
        run_kind: str,
        route_requested: str,
        route_reason: str,
        data_classification: str,
        output_contract: str,
        fallback_from_call_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "child", child)
        object.__setattr__(self, "store", store)
        object.__setattr__(self, "parent_session_id", parent_session_id)
        object.__setattr__(self, "parent_turn_id", parent_turn_id)
        object.__setattr__(self, "task_index", int(task_index))
        object.__setattr__(self, "run_kind", run_kind)
        object.__setattr__(self, "route_requested", route_requested)
        object.__setattr__(self, "route_reason", route_reason)
        object.__setattr__(self, "data_classification", data_classification)
        object.__setattr__(self, "output_contract", output_contract)
        object.__setattr__(self, "fallback_from_call_id", fallback_from_call_id)
        object.__setattr__(self, "route_receipt_id", "")
        object.__setattr__(self, "_route_metadata", {})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.child, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._LOCAL_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(self.child, name, value)

    def bind_fallback_from(self, stable_call_id: str) -> None:
        if self.route_receipt_id:
            raise ValueError("fallback identity is already frozen")
        object.__setattr__(self, "fallback_from_call_id", stable_call_id)

    def prepare_receipt(self) -> str:
        if self.route_receipt_id:
            return self.route_receipt_id
        receipt_id = self.store.prepare_attempt(
            parent_session_id=self.parent_session_id,
            parent_turn_id=self.parent_turn_id,
            child_session_id=str(getattr(self.child, "session_id", "") or "unknown-child"),
            task_index=self.task_index,
            run_kind=self.run_kind,
            route_requested=self.route_requested,
            route_decision="frontier",
            route_reason=self.route_reason,
            data_classification=self.data_classification,
            output_contract=self.output_contract,
            provider=str(getattr(self.child, "provider", "") or "unknown-provider"),
            model=str(getattr(self.child, "model", "") or "unknown-model"),
            fallback_from_call_id=self.fallback_from_call_id,
        )
        object.__setattr__(self, "route_receipt_id", receipt_id)
        object.__setattr__(
            self,
            "_route_metadata",
            {
                # ``frontier`` is the route-neutral receipt vocabulary.  The
                # parent delegate_task result retains its established public
                # ``sol`` label for compatibility.
                "route": "sol",
                "route_reason": self.route_reason,
                "worker_route": "sol",
                "worker_provider": str(getattr(self.child, "provider", "") or ""),
                "worker_model_requested": str(getattr(self.child, "model", "") or ""),
                "route_receipt_id": receipt_id,
                "run_kind": self.run_kind,
                "fallback_used": self.fallback_from_call_id is not None,
            },
        )
        return receipt_id

    def _usage(self) -> dict[str, int]:
        values = {
            "input_tokens": int(getattr(self.child, "session_prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(self.child, "session_completion_tokens", 0) or 0),
            "cache_read_tokens": int(getattr(self.child, "session_cache_read_tokens", 0) or 0),
            "cache_write_tokens": int(getattr(self.child, "session_cache_write_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(self.child, "session_reasoning_tokens", 0) or 0),
        }
        values["total_tokens"] = sum(values.values())
        return values

    def run_conversation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        receipt_id = self.prepare_receipt()
        try:
            result = self.child.run_conversation(*args, **kwargs)
        except BaseException as exc:
            self.store.complete_attempt(
                receipt_id,
                status="failed",
                terminal_route="frontier",
                terminal_provider=str(getattr(self.child, "provider", "") or "unknown-provider"),
                terminal_model=str(getattr(self.child, "model", "") or "unknown-model"),
                usage=None,
                usage_status="unavailable",
                error_code=f"child_{type(exc).__name__}",
            )
            raise
        if not isinstance(result, dict):
            result = {"completed": False, "final_response": "", "error": "invalid child result"}
        if result.get("interrupted") is True:
            status = "interrupted"
        elif result.get("completed") is True and isinstance(result.get("final_response"), str) and result["final_response"]:
            status = "completed"
        else:
            status = "failed"
        self.store.complete_attempt(
            receipt_id,
            status=status,
            terminal_route="frontier",
            terminal_provider=str(result.get("worker_provider") or getattr(self.child, "provider", "") or "unknown-provider"),
            terminal_model=str(result.get("worker_model_requested") or getattr(self.child, "model", "") or "unknown-model"),
            usage=self._usage(),
            usage_status="complete",
            fallback_used=False,
            error_code=(None if status == "completed" else "frontier_child_failed"),
        )
        result = dict(result)
        result.update(self._route_metadata)
        return result

    def close(self) -> None:
        close = getattr(self.child, "close", None)
        if callable(close):
            close()
