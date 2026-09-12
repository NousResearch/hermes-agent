"""Per-attempt execution receipts for Hermes Kanban workers.

Lifecycle state remains owned by ``task_runs``. This module adds observable,
provider-neutral facts without widening that table or changing dispatch.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, cast


_USAGE_KEYS = (
    "api_calls",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "estimated_cost_usd",
    "actual_cost_usd",
)
_RUNNER_METADATA_KEYS = frozenset(("workspace_id", "pane_id", "agent_id", "runner_id"))
_MAX_RUNNER_METADATA_VALUE_CHARS = 256


def _json_dict(raw: Optional[str]) -> Optional[dict[str, Any]]:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _usage_json(usage: Optional[Mapping[str, Any]]) -> Optional[str]:
    if usage is None:
        return None
    bounded: dict[str, int | float] = {}
    for key in _USAGE_KEYS:
        value = usage.get(key)
        if value is None:
            continue
        if key.endswith("cost_usd"):
            bounded[key] = max(0.0, float(value))
        else:
            bounded[key] = max(0, int(value))
    return json.dumps(bounded, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class RunReceipt:
    run_id: int
    task_id: str
    runner: Optional[str]
    worker_session_id: Optional[str]
    session_lineage_root: Optional[str]
    profile: Optional[str]
    requested_provider: Optional[str]
    requested_model: Optional[str]
    requested_reasoning: Optional[str]
    effective_provider: Optional[str]
    effective_model: Optional[str]
    effective_reasoning: Optional[str]
    system_prompt_hash: Optional[str]
    toolset_hash: Optional[str]
    skills_hash: Optional[str]
    context_schema_version: Optional[int]
    context_fingerprint: Optional[str]
    context_chars: Optional[int]
    usage_start: Optional[dict[str, Any]]
    usage_end: Optional[dict[str, Any]]
    api_call_delta: Optional[int]
    input_token_delta: Optional[int]
    output_token_delta: Optional[int]
    cache_read_token_delta: Optional[int]
    cache_write_token_delta: Optional[int]
    reasoning_token_delta: Optional[int]
    estimated_cost_delta: Optional[float]
    actual_cost_delta: Optional[float]
    worktree_start_fingerprint: Optional[str]
    worktree_end_fingerprint: Optional[str]
    runner_metadata: Optional[dict[str, Any]]
    receipt_completeness: str
    receipt_source: str
    fresh_or_resumed: Optional[str]
    created_at: int
    updated_at: int
    finalized_at: Optional[int]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "RunReceipt":
        values = cast(dict[str, Any], dict(row))
        values["run_id"] = int(values["run_id"])
        for key in ("context_schema_version", "context_chars", "api_call_delta",
                    "input_token_delta", "output_token_delta", "cache_read_token_delta",
                    "cache_write_token_delta", "reasoning_token_delta", "finalized_at"):
            values[key] = int(values[key]) if values[key] is not None else None
        values["created_at"] = int(values["created_at"])
        values["updated_at"] = int(values["updated_at"])
        values["usage_start"] = _json_dict(values["usage_start"])
        values["usage_end"] = _json_dict(values["usage_end"])
        values["runner_metadata"] = _json_dict(values["runner_metadata"])
        return cls(**values)


def get_run_receipt(conn: sqlite3.Connection, run_id: int) -> Optional[RunReceipt]:
    row = conn.execute(
        "SELECT * FROM task_run_receipts WHERE run_id = ?", (int(run_id),)
    ).fetchone()
    return RunReceipt.from_row(row) if row else None


def get_run_receipts(
    conn: sqlite3.Connection, run_ids: list[int],
) -> dict[int, RunReceipt]:
    if not run_ids:
        return {}
    placeholders = ",".join("?" for _ in run_ids)
    rows = conn.execute(
        f"SELECT * FROM task_run_receipts WHERE run_id IN ({placeholders})",
        tuple(int(run_id) for run_id in run_ids),
    ).fetchall()
    receipts = [RunReceipt.from_row(row) for row in rows]
    return {receipt.run_id: receipt for receipt in receipts}


def _active_run_matches(
    conn: sqlite3.Connection, task_id: str, run_id: int,
) -> bool:
    row = conn.execute(
        """SELECT 1
             FROM tasks AS t
             JOIN task_runs AS r ON r.id = t.current_run_id
            WHERE t.id = ? AND t.current_run_id = ?
              AND r.task_id = ? AND r.ended_at IS NULL""",
        (task_id, int(run_id), task_id),
    ).fetchone()
    return row is not None


def record_run_session(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: int,
    *,
    worker_session_id: str,
    session_lineage_root: Optional[str] = None,
    profile: Optional[str] = None,
    requested_provider: Optional[str] = None,
    requested_model: Optional[str] = None,
    requested_reasoning: Optional[str] = None,
    effective_provider: Optional[str] = None,
    effective_model: Optional[str] = None,
    effective_reasoning: Optional[str] = None,
    fresh_or_resumed: Optional[str] = None,
    usage_start: Optional[Mapping[str, Any]] = None,
    receipt_source: str = "worker",
) -> bool:
    """Start or enrich the active run's receipt; reject stale run identities."""
    if not worker_session_id or not _active_run_matches(conn, task_id, run_id):
        return False
    now = int(time.time())
    conn.execute(
        """INSERT INTO task_run_receipts (
               run_id, task_id, worker_session_id, session_lineage_root, profile,
               requested_provider, requested_model, requested_reasoning,
               effective_provider, effective_model, effective_reasoning,
               usage_start, receipt_completeness, receipt_source,
               fresh_or_resumed, created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'started', ?, ?, ?, ?)
           ON CONFLICT(run_id) DO UPDATE SET
               worker_session_id = excluded.worker_session_id,
               session_lineage_root = excluded.session_lineage_root,
               profile = COALESCE(excluded.profile, task_run_receipts.profile),
               requested_provider = excluded.requested_provider,
               requested_model = excluded.requested_model,
               requested_reasoning = excluded.requested_reasoning,
               effective_provider = excluded.effective_provider,
               effective_model = excluded.effective_model,
               effective_reasoning = excluded.effective_reasoning,
               usage_start = COALESCE(task_run_receipts.usage_start, excluded.usage_start),
               receipt_source = excluded.receipt_source,
               fresh_or_resumed = excluded.fresh_or_resumed,
               updated_at = excluded.updated_at""",
        (
            int(run_id), task_id, worker_session_id, session_lineage_root, profile,
            requested_provider, requested_model, requested_reasoning,
            effective_provider, effective_model, effective_reasoning,
            _usage_json(usage_start), receipt_source, fresh_or_resumed, now, now,
        ),
    )
    return True


def record_run_context_receipt(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: int,
    *,
    context_schema_version: int,
    context_fingerprint: str,
    context_chars: int,
    system_prompt_hash: Optional[str] = None,
    toolset_hash: Optional[str] = None,
    skills_hash: Optional[str] = None,
) -> bool:
    """Record bounded context identity for the active run."""
    if not _active_run_matches(conn, task_id, run_id):
        return False
    now = int(time.time())
    conn.execute(
        """INSERT INTO task_run_receipts (
               run_id, task_id, context_schema_version, context_fingerprint,
               context_chars, system_prompt_hash, toolset_hash, skills_hash,
               created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(run_id) DO UPDATE SET
               context_schema_version = excluded.context_schema_version,
               context_fingerprint = excluded.context_fingerprint,
               context_chars = excluded.context_chars,
               system_prompt_hash = excluded.system_prompt_hash,
               toolset_hash = excluded.toolset_hash,
               skills_hash = excluded.skills_hash,
               updated_at = excluded.updated_at""",
        (
            int(run_id), task_id, max(1, int(context_schema_version)),
            str(context_fingerprint)[:128], max(0, int(context_chars)),
            (str(system_prompt_hash)[:128] if system_prompt_hash else None),
            (str(toolset_hash)[:128] if toolset_hash else None),
            (str(skills_hash)[:128] if skills_hash else None), now, now,
        ),
    )
    return True


def _safe_runner_metadata(metadata: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not metadata:
        return None
    safe = {
        key: str(metadata[key])[:_MAX_RUNNER_METADATA_VALUE_CHARS]
        for key in sorted(_RUNNER_METADATA_KEYS & metadata.keys())
        if metadata[key] is not None
    }
    return json.dumps(safe, sort_keys=True, separators=(",", ":")) if safe else None


def record_run_runner_handle(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: int,
    *,
    runner: str,
    runner_metadata: Optional[Mapping[str, Any]] = None,
    worktree_start_fingerprint: Optional[str] = None,
    receipt_source: str = "runner",
) -> bool:
    """Record a safe supervisor handle for the active run."""
    runner_name = (runner or "").strip().lower()
    if runner_name not in {"hermes", "herdr"}:
        raise ValueError("runner must be 'hermes' or 'herdr'")
    if not _active_run_matches(conn, task_id, run_id):
        return False
    now = int(time.time())
    conn.execute(
        """INSERT INTO task_run_receipts (
               run_id, task_id, runner, runner_metadata,
               worktree_start_fingerprint, receipt_source, created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(run_id) DO UPDATE SET
               runner = excluded.runner,
               runner_metadata = excluded.runner_metadata,
               worktree_start_fingerprint = excluded.worktree_start_fingerprint,
               receipt_source = excluded.receipt_source,
               updated_at = excluded.updated_at""",
        (
            int(run_id), task_id, runner_name, _safe_runner_metadata(runner_metadata),
            (str(worktree_start_fingerprint)[:128] if worktree_start_fingerprint else None),
            receipt_source, now, now,
        ),
    )
    return True


def _counter_delta(
    start: Mapping[str, Any], end: Mapping[str, Any], key: str, *, cost: bool = False,
) -> Optional[int | float]:
    if key not in start or key not in end:
        return None
    value = float(end[key]) - float(start[key]) if cost else int(end[key]) - int(start[key])
    if value < 0:
        return None
    return float(value) if cost else int(value)


def finalize_run_usage(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: int,
    *,
    worker_session_id: str,
    usage_end: Mapping[str, Any],
    worktree_end_fingerprint: Optional[str] = None,
    receipt_source: str = "worker",
) -> bool:
    """Finalize one exact run/session receipt using per-run usage deltas."""
    row = conn.execute(
        """SELECT rr.worker_session_id, rr.usage_start
             FROM task_run_receipts AS rr
             JOIN task_runs AS r ON r.id = rr.run_id
            WHERE rr.run_id = ? AND rr.task_id = ? AND r.task_id = ?""",
        (int(run_id), task_id, task_id),
    ).fetchone()
    if row is None or row["worker_session_id"] != worker_session_id:
        return False
    start = _json_dict(row["usage_start"])
    if start is None:
        return False
    normalized_end_raw = _usage_json(usage_end)
    normalized_end = _json_dict(normalized_end_raw)
    if normalized_end is None:
        return False
    deltas = {
        "api_call_delta": _counter_delta(start, normalized_end, "api_calls"),
        "input_token_delta": _counter_delta(start, normalized_end, "input_tokens"),
        "output_token_delta": _counter_delta(start, normalized_end, "output_tokens"),
        "cache_read_token_delta": _counter_delta(start, normalized_end, "cache_read_tokens"),
        "cache_write_token_delta": _counter_delta(start, normalized_end, "cache_write_tokens"),
        "reasoning_token_delta": _counter_delta(start, normalized_end, "reasoning_tokens"),
        "estimated_cost_delta": _counter_delta(start, normalized_end, "estimated_cost_usd", cost=True),
        "actual_cost_delta": _counter_delta(start, normalized_end, "actual_cost_usd", cost=True),
    }
    completeness = "complete" if all(value is not None for value in deltas.values()) else "partial"
    now = int(time.time())
    cursor = conn.execute(
        """UPDATE task_run_receipts
              SET usage_end = ?, api_call_delta = ?, input_token_delta = ?,
                  output_token_delta = ?, cache_read_token_delta = ?,
                  cache_write_token_delta = ?, reasoning_token_delta = ?,
                  estimated_cost_delta = ?, actual_cost_delta = ?,
                  worktree_end_fingerprint = ?, receipt_completeness = ?,
                  receipt_source = ?, updated_at = ?, finalized_at = ?
           WHERE run_id = ? AND task_id = ? AND worker_session_id = ?
             AND receipt_completeness != 'complete'""",
        (
            normalized_end_raw, deltas["api_call_delta"], deltas["input_token_delta"],
            deltas["output_token_delta"], deltas["cache_read_token_delta"],
            deltas["cache_write_token_delta"], deltas["reasoning_token_delta"],
            deltas["estimated_cost_delta"], deltas["actual_cost_delta"],
            (str(worktree_end_fingerprint)[:128] if worktree_end_fingerprint else None),
            completeness, receipt_source, now, now, int(run_id), task_id, worker_session_id,
        ),
    )
    return cursor.rowcount == 1


def mark_run_receipt_partial(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: int,
    *,
    receipt_source: str = "lifecycle",
) -> bool:
    """Ensure a terminal run has an honest partial receipt after crashes or early exits."""
    row = conn.execute(
        "SELECT profile, started_at FROM task_runs WHERE id = ? AND task_id = ?",
        (int(run_id), task_id),
    ).fetchone()
    if row is None:
        return False
    now = int(time.time())
    conn.execute(
        """INSERT INTO task_run_receipts (
               run_id, task_id, profile, receipt_completeness, receipt_source,
               created_at, updated_at, finalized_at
           ) VALUES (?, ?, ?, 'partial', ?, ?, ?, ?)
           ON CONFLICT(run_id) DO UPDATE SET
               profile = COALESCE(task_run_receipts.profile, excluded.profile),
               receipt_completeness = CASE
                   WHEN task_run_receipts.receipt_completeness = 'complete'
                   THEN 'complete' ELSE 'partial' END,
               receipt_source = CASE
                   WHEN task_run_receipts.receipt_completeness = 'complete'
                   THEN task_run_receipts.receipt_source ELSE excluded.receipt_source END,
               updated_at = excluded.updated_at,
               finalized_at = COALESCE(task_run_receipts.finalized_at, excluded.finalized_at)""",
        (
            int(run_id), task_id, row["profile"], receipt_source,
            int(row["started_at"]), now, now,
        ),
    )
    return True


def run_receipt_dict(receipt: RunReceipt) -> dict[str, Any]:
    """Return the JSON-safe additive representation used by run views."""
    return asdict(receipt)


def aggregate_run_receipts(
    conn: sqlite3.Connection, *, task_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Aggregate usage by workflow, route, runner, and outcome dimensions."""
    where = "WHERE r.task_id = ?" if task_id is not None else ""
    params = (task_id,) if task_id is not None else ()
    rows = conn.execute(
        f"""SELECT r.task_id, r.step_key, COALESCE(rr.profile, r.profile) AS profile,
                   rr.effective_provider, rr.effective_model, rr.runner, r.outcome,
                   rr.fresh_or_resumed, COUNT(*) AS run_count,
                   SUM(CASE WHEN rr.receipt_completeness = 'complete' THEN 1 ELSE 0 END)
                       AS complete_receipt_count,
                   SUM(rr.api_call_delta) AS api_calls,
                   SUM(rr.input_token_delta) AS input_tokens,
                   SUM(rr.output_token_delta) AS output_tokens,
                   SUM(rr.cache_read_token_delta) AS cache_read_tokens,
                   SUM(rr.cache_write_token_delta) AS cache_write_tokens,
                   SUM(rr.reasoning_token_delta) AS reasoning_tokens,
                   SUM(rr.estimated_cost_delta) AS estimated_cost_usd,
                   SUM(rr.actual_cost_delta) AS actual_cost_usd
              FROM task_runs AS r
              LEFT JOIN task_run_receipts AS rr ON rr.run_id = r.id
              {where}
             GROUP BY r.task_id, r.step_key, COALESCE(rr.profile, r.profile),
                      rr.effective_provider, rr.effective_model, rr.runner, r.outcome,
                      rr.fresh_or_resumed
             ORDER BY r.task_id, r.step_key, profile, rr.effective_provider,
                      rr.effective_model, rr.runner, r.outcome, rr.fresh_or_resumed""",
        params,
    ).fetchall()
    integer_fields = (
        "run_count", "complete_receipt_count", "api_calls", "input_tokens",
        "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
    )
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in integer_fields:
            item[key] = int(item[key]) if item[key] is not None else None
        item["estimated_cost_usd"] = (
            float(item["estimated_cost_usd"])
            if item["estimated_cost_usd"] is not None else None
        )
        item["actual_cost_usd"] = (
            float(item["actual_cost_usd"])
            if item["actual_cost_usd"] is not None else None
        )
        result.append(item)
    return result