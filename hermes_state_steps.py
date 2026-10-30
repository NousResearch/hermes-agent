"""Step ledger (``turn_steps``) writes for SessionDB.

One row per provider call. Rows are written in the *same* transaction as the usage delta
they describe (``hermes_state_usage``), so every token/cost column reconciles exactly with
``session_model_usage``. ``step_seq`` is a dense, per-session ordinal assigned here for
main-loop LLM steps; auxiliary and aggregated rows leave it NULL and never tick the clock.

Design: website/docs/developer-guide/scheduled-tasks.md (P0, step ledger).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Mapping, Optional

logger = logging.getLogger("hermes_state")

_ROLES = frozenset(("assistant", "tool", "aux"))
_STEP_KINDS = frozenset(("llm", "tool", "aux"))
_USAGE_STATUSES = frozenset(("ok", "missing", "aggregated"))
_STEP_STATUSES = frozenset(("ok", "error", "timeout", "interrupted"))

_TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
)
_ROUTE_FIELDS = ("model", "billing_provider", "billing_base_url", "billing_mode")
_COST_FIELDS = ("estimated_cost_usd", "actual_cost_usd", "cost_status", "cost_source", "pricing_version")

# Fields a producer may set. Token/cost/route columns are taken from the usage payload when
# one is present (the ledger must mirror it), so they are accepted here only for StepOnly rows.
STEP_FIELDS = frozenset((
    "turn_id", "turn_seq", "step_index", "parent_step_id", "role", "step_kind", "agent_name",
    "profile_name", "model", "provider", "billing_provider", "billing_base_url", "billing_mode",
    "task", "usage_status", "api_calls", "tool_call_count", "tool_names", "total_tokens",
    *(_TOKEN_FIELDS), *(_COST_FIELDS),
    "started_at", "ended_at", "duration_ms", "status", "finish_reason", "request_id",
    "iteration_uid", "error",
))

_STR_STEP_FIELDS = frozenset((
    "turn_id", "role", "step_kind", "agent_name", "profile_name", "model", "provider",
    "billing_provider", "billing_base_url", "billing_mode", "task", "usage_status",
    "cost_status", "cost_source", "pricing_version", "status", "finish_reason", "request_id",
    "iteration_uid", "error",
))
_INT_STEP_FIELDS = frozenset((
    "turn_seq", "step_index", "parent_step_id", "api_calls", "tool_call_count", "total_tokens",
    "duration_ms", *(_TOKEN_FIELDS),
))
_REAL_STEP_FIELDS = frozenset(("estimated_cost_usd", "actual_cost_usd", "started_at", "ended_at"))

_INSERT_STEP_SQL = """INSERT INTO turn_steps (
    session_id, turn_id, step_seq, turn_seq, step_index, parent_step_id, role, step_kind,
    agent_name, profile_name, model, provider, billing_provider, billing_base_url, billing_mode,
    task, usage_status, api_calls, tool_call_count, tool_names, input_tokens, output_tokens,
    cache_read_tokens, cache_write_tokens, reasoning_tokens, total_tokens, estimated_cost_usd,
    actual_cost_usd, cost_status, cost_source, pricing_version, started_at, ended_at,
    duration_ms, status, finish_reason, request_id, iteration_uid, error
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""


def aux_step(task: str, api_calls: int = 1, *, agent_name: str = "", **extra: Any) -> dict[str, Any]:
    """Step payload for an auxiliary call: one row per ``record_auxiliary_usage`` write.

    ``api_calls > 1`` marks an aggregate (background review / named agent run) that summarizes
    N provider calls whose per-call detail does not exist upstream.
    """
    calls = max(1, int(api_calls or 1))
    step: dict[str, Any] = {
        "role": "aux",
        "step_kind": "aux",
        "task": task or "",
        "api_calls": calls,
        "usage_status": "aggregated" if calls > 1 else "ok",
        "agent_name": agent_name or "",
        "status": "ok",
    }
    step.update({k: v for k, v in extra.items() if v is not None})
    return step


def normalize_step_payload(raw: Any) -> Optional[dict[str, Any]]:
    """Validate a step payload crossing a trust boundary (the managed-worker RPC).

    Returns ``None`` for ``None``; raises ``ValueError`` on an unknown field or bad type.
    Kept strict because the worker journals payloads as JSON and must reject garbage before
    it reaches a receipt transaction.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("step must be an object")
    unknown = set(raw) - STEP_FIELDS
    if unknown:
        raise ValueError(f"unknown step field(s): {sorted(unknown)}")
    step: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            step[key] = None
        elif key == "tool_names":
            if isinstance(value, str):
                if len(value) > 8192:
                    raise ValueError("tool_names too long")
                step[key] = value
            elif isinstance(value, (list, tuple)) and all(isinstance(v, str) and len(v) <= 128 for v in value):
                step[key] = list(value)[:64]
            else:
                raise ValueError("tool_names must be a string or list of short strings")
        elif key in _STR_STEP_FIELDS:
            if not isinstance(value, str) or len(value) > 4096:
                raise ValueError(f"invalid step string field: {key}")
            step[key] = value
        elif key in _INT_STEP_FIELDS:
            if type(value) is not int or not 0 <= value <= 2**53:
                raise ValueError(f"invalid step integer field: {key}")
            step[key] = value
        elif key in _REAL_STEP_FIELDS:
            if type(value) not in (int, float) or not 0 <= float(value) <= 1e12:
                raise ValueError(f"invalid step number field: {key}")
            step[key] = float(value)
        else:  # pragma: no cover - STEP_FIELDS and the type sets cover every key
            raise ValueError(f"invalid step field: {key}")
    return step


def _tool_names_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    return json.dumps(list(value)[:64], ensure_ascii=False) if value else None


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


class SessionStepsMixin:
    """``turn_steps`` ledger + the durable turn/step clocks it advances."""

    def _insert_turn_step(
        self, conn, session_id: str, step: Mapping[str, Any], usage: Optional[Mapping[str, Any]] = None,
    ) -> Optional[int]:
        """Insert one step row inside the caller's write transaction.

        Returns the assigned ``step_seq`` (LLM rows) or ``None`` (aux / aggregated rows).
        Never raises into a turn: persistence failures are logged and counted by the caller's
        best-effort guards.
        """
        row = conn.execute(
            "SELECT turn_seq, step_seq, profile_name, model, billing_provider, billing_base_url,"
            " billing_mode FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None  # FK target absent: accounting loss, never a turn failure
        session = dict(row)
        usage = dict(usage or {})
        task = str(usage.get("task") or step.get("task") or "")
        role = str(step.get("role") or "assistant")
        kind = str(step.get("step_kind") or "llm")
        if role not in _ROLES or kind not in _STEP_KINDS:
            raise ValueError(f"invalid step role/kind: {role}/{kind}")

        # Route identity must match _record_model_usage exactly, or invariant 1 (tokens/cost
        # reconcile with session_model_usage) cannot hold. Aux rows never inherit the session
        # route for missing fields; main-loop rows do.
        route_fallback = {} if task else session

        def route(key: str, default: str) -> str:
            return str(usage.get(key) or route_fallback.get(key) or default)

        counts = {f: _positive_int(usage.get(f)) for f in _TOKEN_FIELDS}
        total_tokens = _positive_int(step.get("total_tokens"))
        if not total_tokens:
            total_tokens = counts["input_tokens"] + counts["output_tokens"]
        has_usage = any(counts.values()) or bool(
            usage.get("api_call_count") or usage.get("estimated_cost_usd") or usage.get("actual_cost_usd")
        )
        usage_status = str(step.get("usage_status") or ("ok" if has_usage else "missing"))
        if usage_status not in _USAGE_STATUSES:
            raise ValueError(f"invalid usage_status: {usage_status}")

        api_calls = _positive_int(usage.get("api_call_count")) or _positive_int(step.get("api_calls"), 1) or 1
        status = str(step.get("status") or "ok")
        if status not in _STEP_STATUSES:
            status = "ok"

        now = time.time()
        started_at = float(step.get("started_at") or now)
        ended_at = max(float(step.get("ended_at") or now), started_at)
        duration_ms = _positive_int(step.get("duration_ms"))
        if not duration_ms:
            duration_ms = int((ended_at - started_at) * 1000)

        turn_id = step.get("turn_id") or None
        turn_seq = _positive_int(step.get("turn_seq")) or None
        step_seq: Optional[int] = None
        if kind == "llm" and turn_id:
            step_seq = _positive_int(session.get("step_seq")) + 1
            bump_turn = max(_positive_int(session.get("turn_seq")), turn_seq or 0)
            conn.execute(
                "UPDATE sessions SET step_seq = MAX(COALESCE(step_seq, 0), ?),"
                " turn_seq = MAX(COALESCE(turn_seq, 0), ?) WHERE id = ?",
                (step_seq, bump_turn, session_id),
            )
            turn_seq = turn_seq or bump_turn or None

        params = (
            session_id, turn_id, step_seq, turn_seq, step.get("step_index"),
            step.get("parent_step_id"), role, kind,
            str(step.get("agent_name") or ""), str(step.get("profile_name") or session.get("profile_name") or ""),
            route("model", "unknown"), route("billing_provider", ""), route("billing_provider", ""),
            route("billing_base_url", ""), route("billing_mode", ""), task, usage_status, api_calls,
            _positive_int(step.get("tool_call_count")), _tool_names_value(step.get("tool_names")),
            counts["input_tokens"], counts["output_tokens"], counts["cache_read_tokens"],
            counts["cache_write_tokens"], counts["reasoning_tokens"], total_tokens,
            float(usage.get("estimated_cost_usd") or 0.0), float(usage.get("actual_cost_usd") or 0.0),
            usage.get("cost_status"), usage.get("cost_source"), usage.get("pricing_version"),
            started_at, ended_at, duration_ms, status, step.get("finish_reason"),
            step.get("request_id"), step.get("iteration_uid"), step.get("error"),
        )
        conn.execute(_INSERT_STEP_SQL, params)
        return step_seq

    def record_turn_step(self, session_id: str, step: Mapping[str, Any]) -> Optional[int]:
        """Write a step row with no usage delta (a usage-less provider response).

        Returns the assigned ``step_seq`` or ``None``. Never raises into a turn.
        """
        try:
            assigned: list[Optional[int]] = []

            def _do(conn):
                assigned.append(self._insert_turn_step(conn, session_id, step))

            self._execute_write(_do)
            return assigned[0] if assigned else None
        except Exception as exc:  # accounting loss is logged, never raised into a turn
            logger.warning("step ledger: insert failed (session=%s): %s", session_id, exc)
            return None

    def turn_steps_for_session(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Newest-first step rows, for reconciliation reads and the scheduler CLI."""
        rows = self._read_all(
            "SELECT * FROM turn_steps WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, max(0, int(limit))),
        )
        return [dict(r) for r in rows]

    def prune_turn_steps(
        self, session_id: str, *, max_rows: int = 2000, retention_days: int = 30,
    ) -> int:
        """Trim a session's ledger: keep the newest ``max_rows`` plus every non-ok row.

        Never rewrites aggregates; returns the number of rows deleted. Best-effort callers
        (``core.steps.prune``) treat failures as logged accounting loss.
        """
        max_rows = max(0, int(max_rows))
        cutoff = time.time() - int(retention_days) * 86400 if retention_days and int(retention_days) > 0 else None

        def _do(conn) -> int:
            deleted = conn.execute(
                "DELETE FROM turn_steps WHERE session_id = ? AND status = 'ok' AND id NOT IN ("
                " SELECT id FROM turn_steps WHERE session_id = ? ORDER BY id DESC LIMIT ?)",
                (session_id, session_id, max_rows),
            ).rowcount
            if cutoff is not None:
                deleted += conn.execute(
                    "DELETE FROM turn_steps WHERE session_id = ? AND status = 'ok' AND started_at < ?",
                    (session_id, cutoff),
                ).rowcount
            return int(deleted or 0)

        return self._execute_write(_do) or 0

    def turn_step_cache_stats(self, session_id: str, *, last_n: int = 10) -> dict[str, float]:
        """Cache hit ratio over the newest ``last_n`` LLM steps (0.0 when there is no usage)."""
        rows = self._read_all(
            "SELECT cache_read_tokens, input_tokens FROM turn_steps"
            " WHERE session_id = ? AND step_kind = 'llm' AND usage_status = 'ok'"
            " AND (cache_read_tokens > 0 OR input_tokens > 0) ORDER BY step_seq DESC LIMIT ?",
            (session_id, max(1, int(last_n))),
        )
        cached = sum(int(r["cache_read_tokens"] or 0) for r in rows)
        uncached = sum(int(r["input_tokens"] or 0) for r in rows)
        total = cached + uncached
        return {"steps": float(len(rows)), "cache_read_tokens": float(cached),
                "input_tokens": float(uncached), "hit_ratio": (cached / total) if total else 0.0}
