"""User-visible beta status helpers for the opt-in DAG context engine."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUE_VALUES


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _config_dag_enabled(config: Optional[Dict[str, Any]]) -> bool:
    context_cfg = (config or {}).get("context") or {}
    if not isinstance(context_cfg, dict):
        return False
    return str(context_cfg.get("engine") or "compressor").strip().lower() == "dag"


def _dag_cfg(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    context_cfg = (config or {}).get("context") or {}
    if not isinstance(context_cfg, dict):
        return {}
    dag_cfg = context_cfg.get("dag") or {}
    return dag_cfg if isinstance(dag_cfg, dict) else {}


def _flag(config: Optional[Dict[str, Any]], key: str, env_name: str) -> bool:
    dag_cfg = _dag_cfg(config)
    if key in dag_cfg:
        return _truthy(dag_cfg.get(key))
    return _truthy(os.getenv(env_name, ""))


def _engine_status(engine: Any) -> Dict[str, Any]:
    if engine is None or str(getattr(engine, "name", "")).lower() != "dag":
        return {}
    try:
        status = engine.get_status()
        return status if isinstance(status, dict) else {}
    except Exception:
        return {"engine": "dag", "status_error": True}


def _count_by_status(session_db: Any, session_id: str, table: str) -> Dict[str, int]:
    if session_db is None or not session_id:
        return {}
    try:
        with session_db._lock:
            rows = session_db._conn.execute(
                f"SELECT status, COUNT(*) AS count FROM {table} WHERE session_id = ? GROUP BY status",
                (session_id,),
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}
    except Exception:
        return {}


def _sidecar_count(session_db: Any, session_id: str) -> int:
    if session_db is None or not session_id:
        return 0
    try:
        with session_db._lock:
            row = session_db._conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM context_message_parts p
                JOIN messages m ON m.id = p.message_id
                WHERE m.session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return int(row["count"] if row is not None else 0)
    except Exception:
        return 0


def _latest_projection(session_db: Any, session_id: str, engine_version: str) -> Dict[str, Any]:
    if session_db is None or not session_id:
        return {}
    try:
        with session_db._lock:
            row = session_db._conn.execute(
                """
                SELECT status, token_estimate, fresh_tail_start_message_id, latest_raw_message_id, updated_at
                FROM context_projection
                WHERE session_id = ? AND engine_version = ?
                ORDER BY CASE status WHEN 'active' THEN 0 WHEN 'stale' THEN 1 ELSE 2 END, updated_at DESC
                LIMIT 1
                """,
                (session_id, engine_version),
            ).fetchone()
        return dict(row) if row is not None else {}
    except Exception:
        return {}


def _format_counts(counts: Dict[str, int], *, empty: str = "none") -> str:
    if not counts:
        return empty
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def dag_context_status_lines(
    *,
    config: Optional[Dict[str, Any]] = None,
    session_id: str = "",
    session_db: Any = None,
    engine: Any = None,
    include_disabled: bool = False,
) -> List[str]:
    """Return concise beta/safety status lines for CLI and gateway surfaces.

    Legacy/default users get no DAG noise unless ``include_disabled`` is set.
    """

    engine_status = _engine_status(engine)
    config_enabled = _config_dag_enabled(config)
    dag_enabled = config_enabled or bool(engine_status)
    if not dag_enabled:
        if not include_disabled:
            return []
        return ["DAG context: disabled (default legacy compressor; safe/off by default)."]

    dag_cfg = _dag_cfg(config)
    gateway_enabled = bool(engine_status.get("gateway_enabled", False)) or _flag(
        config, "gateway_enabled", "HERMES_DAG_CONTEXT_GATEWAY_ENABLED"
    )
    mutation_queue_enabled = bool(engine_status.get("mutation_queue_enabled", False)) or _flag(
        config, "mutation_queue_enabled", "HERMES_DAG_CONTEXT_MUTATION_QUEUE_ENABLED"
    )
    engine_version = str(engine_status.get("engine_version") or "dag-v1")
    projection = _latest_projection(session_db, session_id, engine_version)

    projection_status = projection.get("status") or ("active" if engine_status.get("projection_token_estimate") is not None else "none")
    projection_tokens = projection.get("token_estimate", engine_status.get("projection_token_estimate"))
    fresh_tail = projection.get("fresh_tail_start_message_id", engine_status.get("fresh_tail_start_message_id"))
    latest_raw = projection.get("latest_raw_message_id")

    checkpoint = engine_status.get("last_checkpoint") or None
    if checkpoint is None and session_db is not None and session_id:
        try:
            from agent.context_dag_store import ContextDAGStore

            cp = ContextDAGStore(session_db).read_checkpoint(session_id)
            checkpoint = cp.__dict__ if cp is not None else None
        except Exception:
            checkpoint = None

    mutation_counts = _count_by_status(session_db, session_id, "context_mutation_log")
    sidecar_parts = _sidecar_count(session_db, session_id)
    reconciliation = engine_status.get("reconciliation") or {}
    warnings = engine_status.get("reconciliation_warnings") or []
    fallback = engine_status.get("fallback_reason")

    lines = [
        "DAG context: projection-only/no transcript rewrite ENABLED (BETA, explicit opt-in).",
        "DAG safety: raw transcript is not rewritten; raw messages remain canonical; rollback: set context.engine: compressor.",
        f"DAG flags: gateway_enabled={'on' if gateway_enabled else 'off'}; mutation_queue_enabled={'on' if mutation_queue_enabled else 'off'}; defaults are safe/off unless explicitly set.",
        f"DAG projection: status={projection_status}; token_estimate={projection_tokens if projection_tokens is not None else 'unknown'}; fresh_tail_start_message_id={fresh_tail if fresh_tail is not None else 'unknown'}; latest_raw_message_id={latest_raw if latest_raw is not None else 'unknown'}.",
    ]

    if checkpoint:
        raw_checkpoint_metadata = checkpoint.get("metadata")
        checkpoint_metadata = raw_checkpoint_metadata if isinstance(raw_checkpoint_metadata, dict) else {}
        transcript_count = checkpoint_metadata.get("transcript_message_count")
        inserted_count = checkpoint_metadata.get("inserted")
        warning_count = len(checkpoint_metadata.get("warnings") or [])
        checkpoint_detail = ""
        if transcript_count is not None:
            checkpoint_detail = (
                f" ({transcript_count} transcript message(s), "
                f"{inserted_count if inserted_count is not None else 0} mirrored, "
                f"{warning_count} warning(s))"
            )
        lines.append(
            "DAG reconciliation: checkpointed"
            f"{checkpoint_detail}; DAG checkpoint/reconciliation: "
            f"last_ingested_message_id={checkpoint.get('last_ingested_message_id')}; "
            f"last_projection_message_id={checkpoint.get('last_projection_message_id')}; "
            f"last_anchor_message_id={checkpoint.get('last_anchor_message_id')}."
        )
    elif reconciliation:
        lines.append(f"DAG checkpoint/reconciliation: queued/in-memory {reconciliation}.")
    else:
        lines.append("DAG checkpoint/reconciliation: no checkpoint yet; projection may be cold/stale until first reconcile/compaction.")

    lines.append(f"DAG mutation queue: enabled={'yes' if mutation_queue_enabled else 'no'}; counts={_format_counts(mutation_counts)}.")
    lines.append(f"DAG sidecar: stored_parts={sidecar_parts}; large tool outputs use preview+ref/hash when sidecar is active.")

    if projection_status == "stale" or warnings or fallback:
        detail = []
        if projection_status == "stale":
            detail.append("projection_status=stale")
        if warnings:
            detail.append(f"reconciliation_warnings={len(warnings)}")
        if fallback:
            detail.append(f"fallback={fallback}")
        lines.append("DAG warning: " + "; ".join(detail) + ".")

    return lines
