"""Kanban task usage ledger (HERMES-OBS-001).

Persistent per-board/task/run/profile/provider/model token and cost
accounting for Kanban orchestration. Records provider-authoritative usage,
runtime-reported tokens, estimated costs, and auxiliary model consumption.

Privacy constraints:
- Never store prompts, credentials, OAuth tokens, or raw payloads
- Never estimate tokens from text length
- Secret detection on provider/model fields
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Secret patterns to reject
_SECRET_PATTERNS = [
    re.compile(r"^sk-[a-zA-Z0-9]{20,}$"),  # OpenAI API keys
    re.compile(r"^sk-proj-[a-zA-Z0-9-]{20,}$"),  # OpenAI project keys
    re.compile(r"^xai-[a-zA-Z0-9-]{20,}$"),  # xAI keys
    re.compile(r"^Bearer\s+", re.IGNORECASE),  # Bearer tokens
    re.compile(r"^Basic\s+[a-zA-Z0-9+/=]+$"),  # Basic auth
    re.compile(r"^token:", re.IGNORECASE),  # Generic token prefix
    re.compile(r"^eyJ[a-zA-Z0-9_-]+$"),  # JWT (base64url encoded)
]

VALID_TOKEN_SOURCES = {
    "provider_authoritative",
    "runtime_reported",
    "estimated",
    "incomplete",
    "unknown",
}

# Stable usage-event identity for aggregation (must match run_usage PK).
# COUNT(DISTINCT api_call_index) alone undercounts when the same local index
# appears on different boards/tasks/runs/call_kinds.
_USAGE_EVENT_KEY_SQL = (
    "board || '|' || task_id || '|' || run_id || '|' || call_kind || '|' || "
    "CAST(api_call_index AS TEXT)"
)


def _check_for_secrets(field: str, value: str) -> None:
    """Raise ValueError if value matches secret patterns."""
    for pattern in _SECRET_PATTERNS:
        if pattern.search(value):
            raise ValueError(f"{field} contains potential secret: {pattern.pattern}")


def record_run_usage(
    conn: sqlite3.Connection,
    *,
    board: str,
    task_id: str,
    run_id: int,
    call_kind: str,
    api_call_index: int,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    token_source: str,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
    elapsed_ms: int = 0,
    aux_input_tokens: Optional[int] = None,
    aux_output_tokens: Optional[int] = None,
    aux_cache_read_tokens: Optional[int] = None,
    aux_cache_write_tokens: Optional[int] = None,
    parent_task_id: Optional[str] = None,
    profile: Optional[str] = None,
    cost_usd: Optional[float] = None,
    cost_status: Optional[str] = None,
    checker_result: Optional[str] = None,
    repair_cycle: int = 0,
    accepted_result_tokens: Optional[int] = None,
) -> int:
    """Record usage for a single API call in a Kanban task run.

    Args:
        conn: Database connection
        board: Board identifier (e.g., "default")
        task_id: Task identifier (e.g., "t_abc123")
        run_id: Run identifier for this task execution
        call_kind: Call type (e.g., "primary", "auxiliary")
        api_call_index: Index of this API call within the run (0-based)
        provider: Provider name (e.g., "openrouter", "anthropic")
        model: Model name (e.g., "gpt-4", "claude-3-opus")
        input_tokens: Input tokens consumed
        output_tokens: Output tokens generated
        token_source: One of: provider_authoritative, runtime_reported,
                      estimated, incomplete, unknown
        cache_read_tokens: Tokens read from cache
        cache_write_tokens: Tokens written to cache
        reasoning_tokens: Reasoning tokens (if applicable)
        elapsed_ms: Wall-clock time in milliseconds
        aux_input_tokens: Auxiliary model input tokens (nullable)
        aux_output_tokens: Auxiliary model output tokens (nullable)
        aux_cache_read_tokens: Auxiliary model cache read tokens (nullable)
        aux_cache_write_tokens: Auxiliary model cache write tokens (nullable)
        parent_task_id: Parent task ID (if this is a subtask)
        profile: Profile name that executed the task
        cost_usd: Cost in USD (if available)
        cost_status: Cost status (e.g., "actual", "estimated")
        checker_result: Checker verification result
        repair_cycle: Number of repair attempts (0 = first attempt)

    Returns:
        Row ID of inserted/updated record

    Raises:
        ValueError: If token_source is missing/invalid or provider/model contain secrets
    """
    # Validate token_source
    if token_source is None or token_source == "":
        raise ValueError("token_source required")
    if token_source not in VALID_TOKEN_SOURCES:
        raise ValueError(
            f"Invalid token_source '{token_source}'. "
            f"Must be one of: {', '.join(sorted(VALID_TOKEN_SOURCES))}"
        )

    # Reject secrets
    _check_for_secrets("provider", provider)
    _check_for_secrets("model", model)

    # Idempotent upsert
    conn.execute(
        """
        INSERT INTO run_usage (
            board, task_id, run_id, call_kind, api_call_index,
            provider, model,
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            reasoning_tokens, elapsed_ms,
            aux_input_tokens, aux_output_tokens,
            aux_cache_read_tokens, aux_cache_write_tokens,
            parent_task_id, profile, token_source,
            cost_usd, cost_status, checker_result, repair_cycle,
            accepted_result_tokens, api_calls
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(board, task_id, run_id, call_kind, api_call_index) DO UPDATE SET
            provider=excluded.provider,
            model=excluded.model,
            input_tokens=excluded.input_tokens,
            output_tokens=excluded.output_tokens,
            cache_read_tokens=excluded.cache_read_tokens,
            cache_write_tokens=excluded.cache_write_tokens,
            reasoning_tokens=excluded.reasoning_tokens,
            elapsed_ms=excluded.elapsed_ms,
            aux_input_tokens=excluded.aux_input_tokens,
            aux_output_tokens=excluded.aux_output_tokens,
            aux_cache_read_tokens=excluded.aux_cache_read_tokens,
            aux_cache_write_tokens=excluded.aux_cache_write_tokens,
            parent_task_id=COALESCE(run_usage.parent_task_id, excluded.parent_task_id),
            profile=excluded.profile,
            token_source=excluded.token_source,
            cost_usd=excluded.cost_usd,
            cost_status=excluded.cost_status,
            checker_result=excluded.checker_result,
            repair_cycle=excluded.repair_cycle,
            accepted_result_tokens=excluded.accepted_result_tokens,
            api_calls=excluded.api_calls
        """,
        (
            board, task_id, run_id, call_kind, api_call_index,
            provider, model,
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            reasoning_tokens, elapsed_ms,
            aux_input_tokens, aux_output_tokens,
            aux_cache_read_tokens, aux_cache_write_tokens,
            parent_task_id, profile, token_source,
            cost_usd, cost_status, checker_result, repair_cycle,
            accepted_result_tokens, 0,
        )
    )
    # Accumulate every parent association for this event (multi-parent safe).
    if parent_task_id:
        record_parent(
            conn,
            board=board,
            task_id=task_id,
            run_id=run_id,
            call_kind=call_kind,
            api_call_index=api_call_index,
            parent_task_id=parent_task_id,
        )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def record_parent(
    conn: sqlite3.Connection,
    *,
    board: str,
    task_id: str,
    run_id: int,
    call_kind: str,
    api_call_index: int,
    parent_task_id: str,
) -> None:
    """Record a parent-child relationship for a usage event.

    Multi-parent support: each parent is stored as a separate row in
    run_usage_parents, avoiding UPSERT overwrite. Duplicate recordings
    are idempotent (INSERT OR IGNORE).

    Args:
        conn: Database connection
        board: Board identifier
        task_id: Child task identifier
        run_id: Run identifier
        call_kind: Call type (primary/auxiliary)
        api_call_index: API call index
        parent_task_id: Parent task identifier
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO run_usage_parents (
            board, task_id, run_id, call_kind, api_call_index, parent_task_id
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (board, task_id, run_id, call_kind, api_call_index, parent_task_id),
    )


def list_parents(
    conn: sqlite3.Connection,
    *,
    board: str,
    task_id: str,
    run_id: int,
    call_kind: str,
    api_call_index: int,
) -> list[str]:
    """Return every parent task id associated with a usage event.

    Parents are ordered lexicographically for stable assertions and display.
    """
    rows = conn.execute(
        """
        SELECT parent_task_id
        FROM run_usage_parents
        WHERE board = ?
          AND task_id = ?
          AND run_id = ?
          AND call_kind = ?
          AND api_call_index = ?
        ORDER BY parent_task_id
        """,
        (board, task_id, run_id, call_kind, api_call_index),
    ).fetchall()
    return [row[0] for row in rows]


def record_from_canonical_usage(
    conn: sqlite3.Connection,
    *,
    board: str,
    task_id: str,
    run_id: int,
    call_kind: str,
    api_call_index: int,
    provider: str,
    model: str,
    canonical_usage: dict[str, int],
    token_source: str,
    elapsed_ms: int = 0,
    aux_input_tokens: Optional[int] = None,
    aux_output_tokens: Optional[int] = None,
    aux_cache_read_tokens: Optional[int] = None,
    aux_cache_write_tokens: Optional[int] = None,
    parent_task_id: Optional[str] = None,
    profile: Optional[str] = None,
    cost_usd: Optional[float] = None,
    cost_status: Optional[str] = None,
    checker_result: Optional[str] = None,
    repair_cycle: int = 0,
    accepted_result_tokens: Optional[int] = None,
) -> int:
    """Record usage from a CanonicalUsage-like dict at a call boundary.

    This is the primary hook for runtime instrumentation: conversation
    loop, Codex runtime, and observable auxiliary call paths pass their
    usage through this function.

    Args:
        conn: Database connection
        board: Board identifier
        task_id: Task identifier
        run_id: Run identifier
        call_kind: Call type (primary/auxiliary)
        api_call_index: Index of this API call
        provider: Provider name
        model: Model name
        canonical_usage: Dict with input_tokens, output_tokens,
            cache_read_tokens, cache_write_tokens, reasoning_tokens
        token_source: Source classification
        elapsed_ms: Wall-clock time in milliseconds
        aux_input_tokens: Auxiliary model input tokens (nullable)
        aux_output_tokens: Auxiliary model output tokens (nullable)
        aux_cache_read_tokens: Auxiliary model cache read tokens (nullable)
        aux_cache_write_tokens: Auxiliary model cache write tokens (nullable)
        parent_task_id: Parent task ID
        profile: Profile that executed the task
        cost_usd: Cost in USD
        cost_status: Cost status
        checker_result: Checker result
        repair_cycle: Repair cycle number
        accepted_result_tokens: Accepted result output tokens

    Returns:
        Row ID of inserted/updated record
    """
    return record_run_usage(
        conn,
        board=board,
        task_id=task_id,
        run_id=run_id,
        call_kind=call_kind,
        api_call_index=api_call_index,
        provider=provider,
        model=model,
        input_tokens=canonical_usage.get("input_tokens", 0),
        output_tokens=canonical_usage.get("output_tokens", 0),
        cache_read_tokens=canonical_usage.get("cache_read_tokens", 0),
        cache_write_tokens=canonical_usage.get("cache_write_tokens", 0),
        reasoning_tokens=canonical_usage.get("reasoning_tokens", 0),
        token_source=token_source,
        elapsed_ms=elapsed_ms,
        aux_input_tokens=aux_input_tokens,
        aux_output_tokens=aux_output_tokens,
        aux_cache_read_tokens=aux_cache_read_tokens,
        aux_cache_write_tokens=aux_cache_write_tokens,
        parent_task_id=parent_task_id,
        profile=profile,
        cost_usd=cost_usd,
        cost_status=cost_status,
        checker_result=checker_result,
        repair_cycle=repair_cycle,
        accepted_result_tokens=accepted_result_tokens,
    )


def safe_record_from_canonical_usage(
    conn: sqlite3.Connection,
    *,
    board: str,
    task_id: str,
    run_id: int,
    call_kind: str,
    api_call_index: int,
    provider: str,
    model: str,
    canonical_usage: dict[str, int],
    token_source: str,
    elapsed_ms: int = 0,
    aux_input_tokens: Optional[int] = None,
    aux_output_tokens: Optional[int] = None,
    aux_cache_read_tokens: Optional[int] = None,
    aux_cache_write_tokens: Optional[int] = None,
    parent_task_id: Optional[str] = None,
    profile: Optional[str] = None,
    cost_usd: Optional[float] = None,
    cost_status: Optional[str] = None,
    checker_result: Optional[str] = None,
    repair_cycle: int = 0,
    accepted_result_tokens: Optional[int] = None,
) -> Optional[int]:
    """Fail-safe wrapper around record_from_canonical_usage.

    Ledger failure must never break model execution. This function
    catches all exceptions and returns None on failure, so callers
    can continue without disruption.

    Returns:
        Row ID on success, None on failure.
    """
    try:
        return record_from_canonical_usage(
            conn,
            board=board,
            task_id=task_id,
            run_id=run_id,
            call_kind=call_kind,
            api_call_index=api_call_index,
            provider=provider,
            model=model,
            canonical_usage=canonical_usage,
            token_source=token_source,
            elapsed_ms=elapsed_ms,
            aux_input_tokens=aux_input_tokens,
            aux_output_tokens=aux_output_tokens,
            aux_cache_read_tokens=aux_cache_read_tokens,
            aux_cache_write_tokens=aux_cache_write_tokens,
            parent_task_id=parent_task_id,
            profile=profile,
            cost_usd=cost_usd,
            cost_status=cost_status,
            checker_result=checker_result,
            repair_cycle=repair_cycle,
            accepted_result_tokens=accepted_result_tokens,
        )
    except Exception:
        # Fail-safe: never break model execution, but leave a diagnosable trail.
        logger.debug(
            "Kanban usage ledger persist failed (board=%s task=%s run_id=%s "
            "call_kind=%s api_call_index=%s provider=%s model=%s):",
            board,
            task_id,
            run_id,
            call_kind,
            api_call_index,
            provider,
            model,
            exc_info=True,
        )
        return None


def aggregate_usage(
    conn: sqlite3.Connection,
    *,
    board: Optional[str] = None,
    task_id: Optional[str] = None,
    run_id: Optional[int] = None,
    profile: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    call_kind: Optional[str] = None,
    parent_task_id: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregate usage statistics with optional filters.

    Aggregation is always over distinct usage events keyed by
    ``(board, task_id, run_id, call_kind, api_call_index)``. Parent
    associations live in ``run_usage_parents``; filtering by parent uses
    an EXISTS subquery so multi-parent events are never multiplied.

    Args:
        conn: Database connection
        board: Filter by board (optional)
        task_id: Filter by task (optional)
        run_id: Filter by run (optional)
        profile: Filter by profile (optional)
        provider: Filter by provider (optional)
        model: Filter by model (optional)
        call_kind: Filter by call kind primary/auxiliary (optional)
        parent_task_id: Filter events associated with this parent (optional).
            Matches denormalized ``run_usage.parent_task_id`` or any row in
            ``run_usage_parents`` without JOIN-multiplying token totals.

    Returns:
        Dict with aggregated totals:
        {
            "total_input_tokens": int,
            "total_output_tokens": int,
            "total_cache_read_tokens": int,
            "total_cache_write_tokens": int,
            "total_reasoning_tokens": int,
            "total_aux_input_tokens": int,
            "total_aux_output_tokens": int,
            "total_aux_cache_read_tokens": int,
            "total_aux_cache_write_tokens": int,
            "total_cost_usd": float or None,
            "record_count": int,
            "total_api_calls": int,
            "total_accepted_result_tokens": int,
        }
    """
    where_clauses = []
    params: list[Any] = []

    if board is not None:
        where_clauses.append("board = ?")
        params.append(board)
    if task_id is not None:
        where_clauses.append("task_id = ?")
        params.append(task_id)
    if run_id is not None:
        where_clauses.append("run_id = ?")
        params.append(run_id)
    if profile is not None:
        where_clauses.append("profile = ?")
        params.append(profile)
    if provider is not None:
        where_clauses.append("provider = ?")
        params.append(provider)
    if model is not None:
        where_clauses.append("model = ?")
        params.append(model)
    if call_kind is not None:
        where_clauses.append("call_kind = ?")
        params.append(call_kind)
    if parent_task_id is not None:
        # EXISTS (not JOIN) so multi-parent events count once. Also honor the
        # denormalized parent_task_id column for rows that predate the join table.
        where_clauses.append(
            """(
                parent_task_id = ?
                OR EXISTS (
                    SELECT 1 FROM run_usage_parents p
                    WHERE p.board = run_usage.board
                      AND p.task_id = run_usage.task_id
                      AND p.run_id = run_usage.run_id
                      AND p.call_kind = run_usage.call_kind
                      AND p.api_call_index = run_usage.api_call_index
                      AND p.parent_task_id = ?
                )
            )"""
        )
        params.append(parent_task_id)
        params.append(parent_task_id)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
        SELECT
            SUM(input_tokens) as total_input,
            SUM(output_tokens) as total_output,
            SUM(cache_read_tokens) as total_cache_read,
            SUM(cache_write_tokens) as total_cache_write,
            SUM(reasoning_tokens) as total_reasoning,
            COALESCE(SUM(aux_input_tokens), 0) as total_aux_input,
            COALESCE(SUM(aux_output_tokens), 0) as total_aux_output,
            COALESCE(SUM(aux_cache_read_tokens), 0) as total_aux_cache_read,
            COALESCE(SUM(aux_cache_write_tokens), 0) as total_aux_cache_write,
            SUM(cost_usd) as total_cost,
            COUNT(*) as record_count,
            COUNT(DISTINCT {_USAGE_EVENT_KEY_SQL}) as total_api_calls,
            COALESCE(SUM(accepted_result_tokens), 0) as total_accepted_result_tokens
        FROM run_usage
        WHERE {where_sql}
    """

    row = conn.execute(query, params).fetchone()
    return {
        "total_input_tokens": row[0] or 0,
        "total_output_tokens": row[1] or 0,
        "total_cache_read_tokens": row[2] or 0,
        "total_cache_write_tokens": row[3] or 0,
        "total_reasoning_tokens": row[4] or 0,
        "total_aux_input_tokens": row[5] or 0,
        "total_aux_output_tokens": row[6] or 0,
        "total_aux_cache_read_tokens": row[7] or 0,
        "total_aux_cache_write_tokens": row[8] or 0,
        "total_cost_usd": row[9],
        "record_count": row[10] or 0,
        "total_api_calls": row[11] or 0,
        "total_accepted_result_tokens": row[12] or 0,
    }


def query_usage(
    conn: sqlite3.Connection,
    *,
    board: Optional[str] = None,
    task_id: Optional[str] = None,
    run_id: Optional[int] = None,
    profile: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    call_kind: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Query individual usage records with optional filters.

    Args:
        conn: Database connection
        board: Filter by board (optional)
        task_id: Filter by task (optional)
        run_id: Filter by run (optional)
        profile: Filter by profile (optional)
        provider: Filter by provider (optional)
        model: Filter by model (optional)

    Returns:
        List of dicts, one per API call:
        [
            {
                "board": str,
                "task_id": str,
                "run_id": int,
                "api_call_index": int,
                "provider": str,
                "model": str,
                "input_tokens": int,
                "output_tokens": int,
                ...
            },
            ...
        ]
    """
    where_clauses = []
    params = []

    if board is not None:
        where_clauses.append("board = ?")
        params.append(board)
    if task_id is not None:
        where_clauses.append("task_id = ?")
        params.append(task_id)
    if run_id is not None:
        where_clauses.append("run_id = ?")
        params.append(run_id)
    if profile is not None:
        where_clauses.append("profile = ?")
        params.append(profile)
    if provider is not None:
        where_clauses.append("provider = ?")
        params.append(provider)
    if model is not None:
        where_clauses.append("model = ?")
        params.append(model)
    if call_kind is not None:
        where_clauses.append("call_kind = ?")
        params.append(call_kind)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
            SELECT *
            FROM run_usage
            WHERE {where_sql}
            ORDER BY board, task_id, run_id, call_kind, api_call_index
        """

    cursor = conn.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _normalize_project_task_ids(task_ids: Any) -> tuple[str, ...]:
    """Normalize a supplied project task set without consulting Kanban state."""
    if isinstance(task_ids, str):
        task_ids = (task_ids,)
    try:
        normalized = {task_id for task_id in task_ids if isinstance(task_id, str) and task_id}
    except TypeError as exc:
        raise TypeError("task_ids must be an iterable of non-empty strings") from exc
    if not normalized:
        raise ValueError("task_ids must contain at least one task id")
    return tuple(sorted(normalized))


def query_project_usage(
    conn: sqlite3.Connection,
    *,
    board: str,
    task_ids: Any = None,
    project_task_ids: Any = None,
) -> list[dict[str, Any]]:
    """Return each distinct usage event for the exact supplied project set.

    The query deliberately does not join ``run_usage_parents``.  Parent rows
    are reachability metadata and never additional usage events.  A Python
    identity guard also protects callers using a legacy table without the
    composite primary key.
    """
    if task_ids is not None and project_task_ids is not None:
        raise ValueError("provide only one of task_ids or project_task_ids")
    normalized = _normalize_project_task_ids(
        task_ids if task_ids is not None else project_task_ids
    )
    placeholders = ", ".join("?" for _ in normalized)
    cursor = conn.execute(
        f"""
        SELECT *
        FROM run_usage
        WHERE board = ? AND task_id IN ({placeholders})
        ORDER BY board, task_id, run_id, call_kind, api_call_index
        """,
        (board, *normalized),
    )
    columns = [desc[0] for desc in cursor.description]
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in cursor.fetchall():
        item = dict(zip(columns, row))
        identity = (
            item.get("board"), item.get("task_id"), item.get("run_id"),
            item.get("call_kind"), item.get("api_call_index"),
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(item)
    return result


def aggregate_project_usage(
    conn: sqlite3.Connection,
    *,
    board: str,
    task_ids: Any = None,
    project_task_ids: Any = None,
    task_roles: Optional[dict[str, str]] = None,
    role_by_task_id: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Aggregate exact project usage, grouped by role and model dimensions.

    ``task_roles`` is supplied by the immutable project snapshot.  No task,
    parent, event, or conversation table is read here.  Missing dimensions
    are rendered as ``"unknown"`` and missing usage remains explicitly
    unknown rather than being estimated.
    """
    rows = query_project_usage(
        conn, board=board, task_ids=task_ids, project_task_ids=project_task_ids
    )
    if task_roles is not None and role_by_task_id is not None:
        raise ValueError("provide only one of task_roles or role_by_task_id")
    roles = task_roles or role_by_task_id or {}
    total_fields = (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
        "reasoning_tokens", "aux_input_tokens", "aux_output_tokens",
        "aux_cache_read_tokens", "aux_cache_write_tokens", "accepted_result_tokens",
    )
    if not rows:
        return {
            "usage_status": "unknown",
            "unknown": ["no usage events for supplied project task set"],
            "record_count": 0,
            "total_api_calls": 0,
            **{f"total_{field}": None for field in total_fields},
            "total_cost_usd": None,
            "groups": [],
        }

    totals = {field: 0 for field in total_fields}
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    total_cost = 0.0
    unknown_fields: set[str] = set()
    for row in rows:
        role = roles.get(row.get("task_id"), "unknown") or "unknown"
        profile = row.get("profile") or "unknown"
        provider = row.get("provider") or "unknown"
        model = row.get("model") or "unknown"
        call_kind = row.get("call_kind") or "unknown"
        key = (str(role), str(profile), str(provider), str(model), str(call_kind))
        token_source = row.get("token_source")
        if token_source in {"unknown", "incomplete", None, ""}:
            unknown_fields.update(f"total_{field}" for field in total_fields)
        group = groups.setdefault(
            key,
            {
                "role": key[0], "profile": key[1], "provider": key[2],
                "model": key[3], "call_kind": key[4], "record_count": 0,
                "api_calls": 0, **{field: 0 for field in total_fields},
            },
        )
        group["record_count"] += 1
        group["api_calls"] += 1
        for field in total_fields:
            value = row.get(field)
            if value is not None:
                numeric = int(value)
                totals[field] += numeric
                group[field] += numeric
        cost = row.get("cost_usd")
        if cost is None:
            costs_complete = False
            unknown_fields.add("total_cost_usd")
        else:
            total_cost += float(cost)

    ordered_groups = [groups[key] for key in sorted(groups)]
    result = {
        "usage_status": "known" if not unknown_fields else "partial",
        "record_count": len(rows),
        "total_api_calls": len(rows),
        **{f"total_{field}": value for field, value in totals.items()},
        "total_cost_usd": total_cost if "total_cost_usd" not in unknown_fields else None,
        "groups": ordered_groups,
    }
    if unknown_fields:
        result["unknown"] = sorted(unknown_fields)
        for field in unknown_fields:
            if field in result:
                result[field] = None
    return result


aggregate_usage_for_project = aggregate_project_usage
query_usage_for_project = query_project_usage


# Per-run auxiliary call counter: keyed by (board, task_id, run_id),
# auto-incremented so each auxiliary API call gets a distinct index.
_aux_call_indices: dict[tuple[str, str, int], int] = {}


def _record_kanban_usage_at_boundary(
    conn: sqlite3.Connection,
    *,
    call_kind: str,
    provider: str,
    model: str,
    canonical_usage: dict[str, int],
    token_source: str,
    elapsed_ms: int = 0,
    api_call_index: Optional[int] = None,
    aux_input_tokens: Optional[int] = None,
    aux_output_tokens: Optional[int] = None,
    aux_cache_read_tokens: Optional[int] = None,
    aux_cache_write_tokens: Optional[int] = None,
    parent_task_id: Optional[str] = None,
    profile: Optional[str] = None,
    cost_usd: Optional[float] = None,
    cost_status: Optional[str] = None,
    checker_result: Optional[str] = None,
    repair_cycle: int = 0,
    accepted_result_tokens: Optional[int] = None,
) -> Optional[int]:
    """Record Kanban usage at a production API call boundary.

    This is the primary runtime hook for conversation loop, Codex, and
    auxiliary call paths. It reads Kanban context from env vars and
    delegates to :func:`safe_record_from_canonical_usage`.

    Primary callers MUST pass a stable ``api_call_index`` (typically
    ``session_api_calls - 1``) so each observable model call becomes its
    own ledger event. Auxiliary callers may omit it to receive a
    per-run auto-incremented index.

    Returns:
        Row ID on success, None on failure / when not in a Kanban worker.
    """
    _kanban_task = os.environ.get("HERMES_KANBAN_TASK")
    if not _kanban_task:
        return None

    _board = os.environ.get("HERMES_KANBAN_BOARD", "default")
    _run_id_raw = os.environ.get("HERMES_KANBAN_RUN_ID")
    try:
        _run_id = int(_run_id_raw) if _run_id_raw else 0
    except (TypeError, ValueError):
        _run_id = 0
    # Dispatcher sets HERMES_PROFILE; keep HERMES_KANBAN_PROFILE as a
    # soft alias for older / test callers.
    _profile = (
        profile
        or os.environ.get("HERMES_PROFILE")
        or os.environ.get("HERMES_KANBAN_PROFILE")
    )

    resolved_index = api_call_index
    if resolved_index is None:
        if call_kind == "auxiliary":
            # Per-(board, task, run) identity. Prefer process-local counter
            # for fast consecutive calls; seed from the DB max so retries /
            # fresh processes do not overwrite index 0.
            key = (_board, _kanban_task, _run_id)
            if key not in _aux_call_indices:
                next_idx = 0
                try:
                    row = conn.execute(
                        "SELECT COALESCE(MAX(api_call_index), -1) FROM run_usage "
                        "WHERE board = ? AND task_id = ? AND run_id = ? "
                        "AND call_kind = 'auxiliary'",
                        (_board, _kanban_task, _run_id),
                    ).fetchone()
                    if row is not None and row[0] is not None:
                        next_idx = int(row[0]) + 1
                except Exception:
                    # Table missing / conn closed: start at 0; safe_record
                    # will swallow write failures without breaking callers.
                    next_idx = 0
                _aux_call_indices[key] = next_idx
            idx = _aux_call_indices[key]
            _aux_call_indices[key] = idx + 1
            resolved_index = idx
        else:
            # Primary without an explicit index collapses every call onto
            # row 0; prefer caller-supplied stable indices.
            resolved_index = 0

    return safe_record_from_canonical_usage(
        conn,
        board=_board,
        task_id=_kanban_task,
        run_id=_run_id,
        call_kind=call_kind,
        api_call_index=int(resolved_index),
        provider=provider,
        model=model,
        canonical_usage=canonical_usage,
        token_source=token_source,
        elapsed_ms=elapsed_ms,
        aux_input_tokens=aux_input_tokens,
        aux_output_tokens=aux_output_tokens,
        aux_cache_read_tokens=aux_cache_read_tokens,
        aux_cache_write_tokens=aux_cache_write_tokens,
        parent_task_id=parent_task_id,
        profile=_profile,
        cost_usd=cost_usd,
        cost_status=cost_status,
        checker_result=checker_result,
        repair_cycle=repair_cycle,
        accepted_result_tokens=accepted_result_tokens,
    )
