"""Fixed PostgreSQL operations for context reads and durable approvals."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import json
import logging
import re
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..approval import ApprovalRecord, ApprovalScope


logger = logging.getLogger(__name__)


_CREATE_APPROVAL_SQL = """
SELECT research_protocol.store_approval(
    $1, $2, $3, $4, $5, $6, $7, $8, $9
)
"""

_CLAIM_APPROVAL_SQL = """
SELECT approval_id, scope_sha256, plan_sha256, consumed_count, max_executions
FROM research_protocol.claim_approval($1, $2, $3)
"""

_SET_LOCAL_STATEMENT_TIMEOUT_SQL = "SELECT set_config('statement_timeout', $1, true)"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class UnknownContextQueryError(ValueError):
    """Raised before database access when a query identifier is not registered."""


class ContextReadLimitError(RuntimeError):
    """Raised when a bounded context result exceeds a configured cap."""


class ContextReadTimeoutError(TimeoutError):
    """Raised when the wall-clock bound is exceeded."""


class ApprovalStoreTimeoutError(TimeoutError):
    """Raised when a bounded approval store operation exceeds its deadline."""


@dataclass(frozen=True)
class _ReadQuery:
    sql: str
    bind: Callable[[Mapping[str, Any], int], tuple[Any, ...]]


def _bind_approval_status(
    parameters: Mapping[str, Any], _max_rows: int
) -> tuple[Any, ...]:
    if set(parameters) != {"approval_id"}:
        raise ValueError("approval_status.v1 requires only approval_id")
    approval_id = parameters["approval_id"]
    if not isinstance(approval_id, str) or not approval_id or len(approval_id) > 256:
        raise ValueError("invalid approval_id")
    return (approval_id,)


def _bind_approvals_for_run(
    parameters: Mapping[str, Any], max_rows: int
) -> tuple[Any, ...]:
    if not {"run_id"}.issubset(parameters) or not set(parameters).issubset({
        "run_id",
        "limit",
    }):
        raise ValueError("approvals_for_run.v1 requires run_id and optional limit")
    run_id = parameters["run_id"]
    if (
        not isinstance(run_id, str)
        or len(run_id) > 128
        or _IDENTIFIER_RE.fullmatch(run_id) is None
    ):
        raise ValueError("invalid run_id")
    requested = parameters.get("limit", max_rows)
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise ValueError("invalid limit")
    if requested < 1 or requested > min(max_rows, 100):
        raise ValueError("limit exceeds configured row cap")
    return run_id, requested + 1


_READ_QUERIES: Mapping[str, _ReadQuery] = MappingProxyType({
    "approval_status.v1": _ReadQuery(
        sql="""
SELECT approval_id, scope_sha256, plan_sha256, verdict, surface,
       created_at, expires_at, max_executions, consumed_count,
       last_consumed_at
FROM research_protocol.approvals
WHERE approval_id = $1
""",
        bind=_bind_approval_status,
    ),
    "approvals_for_run.v1": _ReadQuery(
        sql="""
SELECT approval_id, scope_sha256, plan_sha256, verdict, surface,
       created_at, expires_at, max_executions, consumed_count,
       last_consumed_at
FROM research_protocol.approvals
WHERE scope_json ->> 'run_id' = $1
ORDER BY created_at DESC, approval_id DESC
LIMIT $2
""",
        bind=_bind_approvals_for_run,
    ),
})
CONTEXT_QUERY_IDS = tuple(_READ_QUERIES)


class PostgresContextReader:
    """Execute only registered SELECTs under independent resource bounds."""

    def __init__(
        self,
        pool: Any,
        *,
        max_rows: int = 100,
        max_bytes: int = 256_000,
        timeout_seconds: float = 5.0,
    ):
        if pool is None:
            raise ValueError("PostgreSQL pool is required")
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
            raise ValueError("max_rows must be a positive integer")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 1
        ):
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self._pool = pool
        self._max_rows = max_rows
        self._max_bytes = max_bytes
        self._timeout_seconds = float(timeout_seconds)

    async def read(
        self,
        query_id: str,
        parameters: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        query = _READ_QUERIES.get(query_id)
        if query is None:
            raise UnknownContextQueryError("unknown context query identifier")
        if not isinstance(parameters, Mapping):
            raise ValueError("context query parameters must be an object")
        bind_values = query.bind(parameters, self._max_rows)
        row_cap = self._max_rows
        if query_id == "approvals_for_run.v1":
            row_cap = int(parameters.get("limit", self._max_rows))
        started_at = time.perf_counter()
        try:
            rows = await asyncio.wait_for(
                self._read(query, bind_values, row_cap),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise ContextReadTimeoutError("context read timed out") from None
        logger.info(
            "research_protocol_context_read query_id=%s row_count=%d latency_ms=%.3f",
            query_id,
            len(rows),
            (time.perf_counter() - started_at) * 1000,
        )
        return rows

    async def _read(
        self,
        query: _ReadQuery,
        bind_values: tuple[Any, ...],
        row_cap: int,
    ) -> list[dict[str, Any]]:
        timeout_ms = max(1, int(self._timeout_seconds * 1000))
        async with self._pool.acquire() as connection:
            async with connection.transaction(readonly=True):
                await connection.execute(
                    _SET_LOCAL_STATEMENT_TIMEOUT_SQL,
                    f"{timeout_ms}ms",
                )
                records = await connection.fetch(query.sql, *bind_values)

        if len(records) > row_cap:
            raise ContextReadLimitError("context row cap exceeded")
        rows = [_normalize_row(record) for record in records]
        serialized = json.dumps(
            rows,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(serialized) > self._max_bytes:
            raise ContextReadLimitError("context byte cap exceeded")
        return rows


class PostgresApprovalStore:
    """Persist and consume exact workflow approvals through fixed SQL only."""

    def __init__(self, pool: Any, *, timeout_seconds: float = 5.0):
        if pool is None:
            raise ValueError("PostgreSQL pool is required")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self._pool = pool
        self._timeout_seconds = float(timeout_seconds)
        self._statement_timeout_ms = max(1, int(self._timeout_seconds * 1000))

    async def create_approval(self, record: ApprovalRecord) -> None:
        """Insert one approval record without accepting caller-provided SQL."""
        if not isinstance(record, ApprovalRecord):
            raise TypeError("record must be a validated ApprovalRecord")
        if record.consumed_count != 0:
            raise ValueError("only fresh decision records can be stored")
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._pool.acquire() as connection:
                    async with connection.transaction():
                        await connection.execute(
                            _SET_LOCAL_STATEMENT_TIMEOUT_SQL,
                            f"{self._statement_timeout_ms}ms",
                        )
                        await connection.execute(
                            _CREATE_APPROVAL_SQL,
                            record.approval_id,
                            record.scope_sha256,
                            record.plan_sha256,
                            record.scope_json,
                            record.verdict.value,
                            record.surface,
                            record.created_at,
                            record.expires_at,
                            record.max_executions,
                        )
        except TimeoutError as exc:
            raise ApprovalStoreTimeoutError(
                "approval store operation timed out"
            ) from exc

    async def claim_approval(
        self,
        *,
        approval_id: str,
        scope: ApprovalScope,
        plan_sha256: str,
    ) -> bool:
        """Atomically claim one execution bound to the exact scope and plan."""
        if not isinstance(approval_id, str) or not approval_id:
            return False
        if not isinstance(scope, ApprovalScope):
            return False
        if (
            not isinstance(plan_sha256, str)
            or _SHA256_RE.fullmatch(plan_sha256) is None
        ):
            return False
        try:
            scope_sha256 = scope.scope_sha256()
        except (AttributeError, TypeError, ValueError):
            return False
        if _SHA256_RE.fullmatch(scope_sha256) is None:
            return False

        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._pool.acquire() as connection:
                    async with connection.transaction():
                        await connection.execute(
                            _SET_LOCAL_STATEMENT_TIMEOUT_SQL,
                            f"{self._statement_timeout_ms}ms",
                        )
                        row = await connection.fetchrow(
                            _CLAIM_APPROVAL_SQL,
                            approval_id,
                            scope_sha256,
                            plan_sha256,
                        )
        except TimeoutError as exc:
            raise ApprovalStoreTimeoutError(
                "approval store operation timed out"
            ) from exc
        return row is not None


def _normalize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalize_row(record: Any) -> dict[str, Any]:
    try:
        values = dict(record)
    except (TypeError, ValueError) as exc:
        raise ContextReadLimitError("context row is not a mapping") from exc
    return {str(key): _normalize_value(value) for key, value in values.items()}


__all__ = [
    "ApprovalStoreTimeoutError",
    "CONTEXT_QUERY_IDS",
    "ContextReadLimitError",
    "ContextReadTimeoutError",
    "PostgresApprovalStore",
    "PostgresContextReader",
    "UnknownContextQueryError",
]
