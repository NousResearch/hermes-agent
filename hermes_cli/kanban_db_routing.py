"""Routing audit persistence for existing Kanban run and event storage."""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Optional

from hermes_cli.kanban_db_connect import write_txn

_ROUTING_EVENT_KINDS = frozenset({"routing_selected", "routing_fallback", "routing_primary_restored"})


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def merged_run_metadata(
    conn, *, task_id: str, run_id: int, metadata: Optional[Mapping[str, Any]]
) -> dict[str, Any]:
    """Merge terminal metadata without allowing it to erase an existing route decision."""
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE id = ? AND task_id = ?",
        (int(run_id), task_id),
    ).fetchone()
    existing = _json_dict(row["metadata"] if row else None)
    routing_decision = existing.get("routing_decision")
    merged = {**existing, **dict(metadata or {})}
    if routing_decision is not None:
        merged["routing_decision"] = routing_decision
    return merged


def _event_payload(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": decision.get("decision_id"),
        "profile": decision.get("selected_profile"),
        "provider": decision.get("selected_provider"),
        "model": decision.get("selected_model"),
        "fallback_used": decision.get("fallback_used"),
        "fallback_reason": decision.get("fallback_reason"),
    }


def _history_extends(existing: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    if existing.get("decision_id") != candidate.get("decision_id"):
        return False
    mutable_fields = {
        "selected_provider",
        "selected_model",
        "fallback_used",
        "fallback_from_provider",
        "fallback_from_model",
        "fallback_reason",
        "routing_history",
    }
    existing_identity = {key: value for key, value in existing.items() if key not in mutable_fields}
    candidate_identity = {key: value for key, value in candidate.items() if key not in mutable_fields}
    if existing_identity != candidate_identity:
        return False
    existing_history = existing.get("routing_history")
    candidate_history = candidate.get("routing_history")
    if not isinstance(existing_history, list) or not isinstance(candidate_history, list):
        return False
    return candidate_history[:len(existing_history)] == existing_history


def build_dispatch_routing_context(task: Any, *, board: str, lane: str) -> dict[str, Any]:
    """Build the policy facts the dispatcher actually knows before provider resolution."""
    selected_profile = str(getattr(task, "assignee", "") or "").strip() or None
    run_id = getattr(task, "current_run_id", None)
    if lane in {"implementation", "ready"}:
        reviewer_profile = {
            "rozmilo-codex": "rozmilo-claude",
            "rozmilo-claude": "rozmilo-codex",
        }.get(selected_profile)
        return {
            "task_id": getattr(task, "id", None),
            "board": board,
            "run_id": run_id if isinstance(run_id, int) else None,
            "task_type": "implementation",
            "capability": "implement",
            "risk": None,
            "code_change": None,
            "independent_review": None,
            "preferred_profile": selected_profile,
            "reviewer_profile": reviewer_profile,
            "selected_profile": selected_profile,
            "human_gate_required": None,
            "independence_valid": None,
            "policy_digest": None,
            "selected_by": "dispatcher",
        }
    return {
        "task_id": getattr(task, "id", None),
        "board": board,
        "run_id": run_id if isinstance(run_id, int) else None,
        "task_type": "review" if lane == "review" else None,
        "capability": "review" if lane == "review" else None,
        "risk": None,
        "code_change": None,
        "independent_review": None,
        "preferred_profile": selected_profile,
        "reviewer_profile": None,
        "selected_profile": selected_profile,
        "human_gate_required": None,
        "independence_valid": None,
        "policy_digest": None,
        "selected_by": "dispatcher",
    }


def load_run_routing_decision(
    conn, *, task_id: str, run_id: int
) -> Optional[dict[str, Any]]:
    """Read the current structured decision for one run without inferring legacy data."""
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE id = ? AND task_id = ?",
        (int(run_id), task_id),
    ).fetchone()
    decision = _json_dict(row["metadata"] if row else None).get("routing_decision")
    return dict(decision) if isinstance(decision, dict) else None


def persist_run_routing_decision(
    conn,
    *,
    task_id: str,
    run_id: int,
    decision: Mapping[str, Any],
    event_kind: str,
) -> bool:
    """Atomically update one active run's route and append its compact audit event."""
    if event_kind not in _ROUTING_EVENT_KINDS:
        raise ValueError(f"unsupported routing event kind: {event_kind}")
    if decision.get("routing_contract_version") != "phase-c-v1":
        raise ValueError("routing decision must use phase-c-v1")
    if decision.get("task_id") != task_id or decision.get("run_id") != int(run_id):
        return False
    with write_txn(conn):
        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ? AND task_id = ? AND ended_at IS NULL",
            (int(run_id), task_id),
        ).fetchone()
        if row is None:
            return False
        metadata = _json_dict(row["metadata"])
        existing = metadata.get("routing_decision")
        if isinstance(existing, Mapping):
            if existing == decision:
                return True
            if not _history_extends(existing, decision):
                return False
        metadata["routing_decision"] = dict(decision)
        updated = conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ? AND task_id = ? AND ended_at IS NULL",
            (json.dumps(metadata, ensure_ascii=False, sort_keys=True), int(run_id), task_id),
        )
        if updated.rowcount != 1:
            return False
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                task_id,
                int(run_id),
                event_kind,
                json.dumps(_event_payload(decision), ensure_ascii=False, sort_keys=True),
                int(time.time()),
            ),
        )
    return True
