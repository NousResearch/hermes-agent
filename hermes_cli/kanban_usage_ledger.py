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

import re
import sqlite3
from typing import Any, Optional

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
            cost_usd, cost_status, checker_result, repair_cycle
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            parent_task_id=excluded.parent_task_id,
            profile=excluded.profile,
            token_source=excluded.token_source,
            cost_usd=excluded.cost_usd,
            cost_status=excluded.cost_status,
            checker_result=excluded.checker_result,
            repair_cycle=excluded.repair_cycle
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
        )
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def aggregate_usage(
    conn: sqlite3.Connection,
    *,
    board: Optional[str] = None,
    task_id: Optional[str] = None,
    run_id: Optional[int] = None,
    profile: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregate usage statistics with optional filters.

    Args:
        conn: Database connection
        board: Filter by board (optional)
        task_id: Filter by task (optional)
        run_id: Filter by run (optional)
        profile: Filter by profile (optional)
        provider: Filter by provider (optional)
        model: Filter by model (optional)

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
        }
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
            COUNT(*) as record_count
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

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
        SELECT *
        FROM run_usage
        WHERE {where_sql}
        ORDER BY board, task_id, run_id, api_call_index
    """

    cursor = conn.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]
