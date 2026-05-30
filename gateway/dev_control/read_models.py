"""Oryn-facing Dev control-plane read-model helpers."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional


BOARD_LANES = ("queued", "running", "needs_input", "failed", "completed")
logger = logging.getLogger(__name__)


def build_agent_board_rows(
    *,
    store: Any,
    params: Any,
    limit: int = 250,
    ao_snapshot_cache: Any = None,
) -> list[Dict[str, Any]]:
    """Build Agent Board rows from normalized subagent events plus live AO state."""

    events = store.list_events(
        session_id=params.get("session_id") or None,
        run_id=params.get("run_id") or None,
        subagent_id=params.get("subagent_id") or None,
        ao_session_id=params.get("ao_session_id") or None,
        limit=2000,
    )
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    counts_by_id: Dict[str, int] = {}
    latest_actions_by_id: Dict[str, Dict[str, Any]] = {}
    latest_lifecycle_at_by_id: Dict[str, float] = {}
    for event in events:
        row_id = str(event.get("subagent_id") or event.get("ao_session_id") or "")
        if not row_id:
            continue
        counts_by_id[row_id] = counts_by_id.get(row_id, 0) + 1
        previous_row = rows_by_id.get(row_id)
        next_row = _subagent_board_item_from_event(event, counts_by_id[row_id])
        if event.get("event") == "subagent.action" and previous_row:
            for key in ("goal", "summary", "created_at"):
                if not next_row.get(key):
                    next_row[key] = previous_row.get(key)
        rows_by_id[row_id] = next_row
        if event.get("event") == "subagent.action":
            latest_actions_by_id[row_id] = event
        else:
            latest_lifecycle_at_by_id[row_id] = _subagent_event_order_value(event)

    project_id = params.get("project_id") or None
    try:
        from tools.ao_bridge import AOBridge, AOSession

        def _load_ao_snapshot() -> Dict[str, Any]:
            bridge = AOBridge()
            live_sessions = bridge.list(project_id=project_id)
            live_health_by_id = {}
            batch_health = getattr(bridge, "runtime_health_many", None)
            if callable(batch_health):
                try:
                    live_health_by_id = batch_health(live_sessions)
                except Exception as exc:
                    logger.debug("AO batch runtime health unavailable for subagent board: %s", exc)
            if not live_health_by_id:
                for live_session in live_sessions:
                    live_health_by_id[live_session.id] = bridge.runtime_health(live_session)
            return {
                "sessions": [_ao_session_snapshot_payload(session) for session in live_sessions],
                "health_by_id": live_health_by_id,
            }

        bridge = None
        if ao_snapshot_cache is not None:
            snapshot = ao_snapshot_cache.get_or_load(project_id=project_id, load=_load_ao_snapshot)
            sessions = [AOSession.from_payload(item) for item in snapshot.sessions]
            health_by_id = snapshot.health_by_id
        else:
            bridge = AOBridge()
            sessions = bridge.list(project_id=project_id)
            health_by_id = {}
            batch_health = getattr(bridge, "runtime_health_many", None)
            if callable(batch_health):
                try:
                    health_by_id = batch_health(sessions)
                except Exception as exc:
                    logger.debug("AO batch runtime health unavailable for subagent board: %s", exc)
        for session in sessions:
            row_id = f"ao:{session.id}"
            existing = rows_by_id.get(row_id) or {}
            runtime_health = health_by_id.get(session.id)
            if runtime_health is None:
                if bridge is None:
                    bridge = AOBridge()
                runtime_health = bridge.runtime_health(session)
            rows_by_id[row_id] = _merge_ao_session_into_board_item(
                existing,
                session,
                store,
                runtime_health=runtime_health,
            )
    except Exception as exc:
        logger.debug("AO sessions unavailable for subagent board: %s", exc)

    for row_id, row in list(rows_by_id.items()):
        action_event = latest_actions_by_id.get(row_id)
        if action_event and _subagent_action_belongs_to_current_lifecycle(
            action_event,
            latest_lifecycle_at_by_id.get(row_id),
        ):
            _apply_recent_action_fields(row, action_event)
        _apply_summary_quality_fields(row)

    filtered = [
        item for item in rows_by_id.values()
        if _subagent_board_item_matches(item, params)
    ]
    filtered.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
    return filtered[:limit]


def build_agent_board_response(
    items: Iterable[Dict[str, Any]],
    *,
    updated_at: Optional[float] = None,
) -> Dict[str, Any]:
    data = list(items)
    lanes = {
        lane: sum(1 for item in data if item.get("lane") == lane)
        for lane in BOARD_LANES
    }
    groups_by_key: Dict[str, Dict[str, Any]] = {}
    for item in data:
        group_key = str(item.get("group_key") or "runtime:unknown")
        group = groups_by_key.setdefault(group_key, {
            "key": group_key,
            "label": item.get("group_label") or "Unknown",
            "kind": item.get("group_kind") or "runtime",
            "count": 0,
            "attention_count": 0,
        })
        group["count"] += 1
        if item.get("lane") in {"needs_input", "failed"}:
            group["attention_count"] += 1
    return {
        "object": "list",
        "data": data,
        "total": len(data),
        "lanes": lanes,
        "groups": list(groups_by_key.values()),
        "attention_count": sum(1 for item in data if item.get("lane") in {"needs_input", "failed"}),
        "updated_at": updated_at or time.time(),
    }


def _ao_session_snapshot_payload(session: Any) -> Dict[str, Any]:
    if is_dataclass(session):
        payload = asdict(session)
    else:
        payload = {
            "id": getattr(session, "id", None),
            "project_id": getattr(session, "project_id", None),
            "status": getattr(session, "status", None),
            "activity": getattr(session, "activity", None),
            "branch": getattr(session, "branch", None),
            "issue_id": getattr(session, "issue_id", None),
            "workspace_path": getattr(session, "workspace_path", None),
            "tmux_name": getattr(session, "tmux_name", None),
            "agent": getattr(session, "agent", None),
            "model": getattr(session, "model", None),
            "reasoning_effort": getattr(session, "reasoning_effort", None),
            "pr": getattr(session, "pr", None),
            "summary": getattr(session, "summary", None),
            "open_command": getattr(session, "open_command", None),
        }
    payload["id"] = str(payload.get("id") or "")
    return payload


def build_dev_plans_response(plans: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    data = list(plans)
    return {"object": "list", "data": data, "total": len(data)}


def build_worker_detail_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    return dict(item)


def _subagent_runtime(payload: Dict[str, Any]) -> str:
    runtime = str(payload.get("runtime") or "").strip().lower()
    if runtime:
        return runtime
    return "ao" if payload.get("ao_session_id") else "hermes"


def _subagent_lane(status: Optional[str]) -> str:
    raw = str(status or "").strip().lower()
    if raw in {"queued", "pending", "created", "scheduled"}:
        return "queued"
    if raw in {"needs_input", "input_required", "waiting_for_input", "blocked", "paused", "approval_required"}:
        return "needs_input"
    if raw in {"failed", "fail", "error", "errored", "killed", "terminated", "timed_out", "timeout", "cancelled", "canceled"}:
        return "failed"
    if raw in {"completed", "complete", "done", "success", "succeeded", "merged"}:
        return "completed"
    if raw in {"spawned", "running", "thinking", "progress", "working", "active", ""}:
        return "running"
    return "running"


def _subagent_lane_reason(status: Optional[str], lane: str, runtime: str) -> str:
    raw = str(status or "").strip().lower()
    if lane == "queued":
        return "Worker is queued and has not started active work yet."
    if lane == "running":
        return "Worker is active and reporting progress."
    if lane == "needs_input":
        return "Worker is waiting for input or approval."
    if lane == "failed":
        if raw in {"killed", "terminated", "cancelled", "canceled"}:
            return "Worker was stopped before completing."
        if raw in {"timed_out", "timeout"}:
            return "Worker timed out before completing."
        return "Worker ended with a failed status."
    if lane == "completed":
        return "Worker reached a terminal completed state."
    return f"{runtime.title()} worker status is {raw or 'unknown'}."


def _subagent_attention_level(lane: str) -> str:
    if lane == "failed":
        return "high"
    if lane == "needs_input":
        return "medium"
    return "none"


def _subagent_group_fields(payload: Dict[str, Any], runtime: str) -> Dict[str, str]:
    project_id = str(payload.get("ao_project_id") or "").strip()
    if project_id:
        return {
            "group_key": f"project:{project_id}",
            "group_label": project_id,
            "group_kind": "project",
        }
    session_id = str(payload.get("session_id") or "").strip()
    if session_id:
        short_session = session_id[:8]
        return {
            "group_key": f"session:{session_id}",
            "group_label": f"Session {short_session}",
            "group_kind": "session",
        }
    label = "AO" if runtime == "ao" else "Hermes Native"
    return {
        "group_key": f"runtime:{runtime}",
        "group_label": label,
        "group_kind": "runtime",
    }


def _subagent_numeric(payload: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _subagent_token_total(payload: Dict[str, Any]) -> Optional[int]:
    direct = payload.get("token_total") or payload.get("total_tokens")
    if direct is not None:
        try:
            return int(direct)
        except (TypeError, ValueError):
            pass
    context_usage = payload.get("context_usage")
    if isinstance(context_usage, dict):
        session = context_usage.get("session")
        if isinstance(session, dict):
            session_total = session.get("total_tokens")
            if session_total is not None:
                try:
                    return int(session_total)
                except (TypeError, ValueError):
                    pass
    total = 0
    seen = False
    for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            total += int(value)
            seen = True
        except (TypeError, ValueError):
            pass
    return total if seen else None


def _subagent_current_activity(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("message", "text", "preview", "activity", "summary", "tool_name", "tool"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _apply_recent_action_fields(item: Dict[str, Any], event: Dict[str, Any]) -> None:
    item["recent_action"] = event.get("action")
    item["recent_action_status"] = event.get("action_status") or event.get("status")
    item["recent_action_message"] = event.get("message") or event.get("preview")
    item["recent_action_at"] = event.get("created_at") or event.get("timestamp")


def _subagent_event_order_value(event: Dict[str, Any]) -> float:
    for key in ("created_at", "timestamp", "event_id"):
        value = event.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _subagent_action_belongs_to_current_lifecycle(
    action_event: Dict[str, Any],
    latest_lifecycle_at: Optional[float],
) -> bool:
    if latest_lifecycle_at is None:
        return True
    return _subagent_event_order_value(action_event) >= latest_lifecycle_at


def _apply_summary_quality_fields(item: Dict[str, Any]) -> None:
    summary = str(item.get("summary") or "").strip()
    status = str(item.get("status") or "").lower()
    goal = str(item.get("goal") or "")
    current = str(item.get("current_activity") or "")
    text = summary or current
    warning: Optional[str] = None

    if status in {"completed", "complete", "done", "success", "succeeded"}:
        if not summary:
            warning = "Worker completed without a final summary."
        elif len(summary) < 24:
            warning = "Worker summary is very short."

    expected_prefixes = sorted(set(re.findall(r"\b[A-Z][A-Z0-9_]{3,}_DONE\b", f"{goal}\n{text}")))
    for prefix in expected_prefixes:
        if prefix not in summary:
            warning = f"Worker summary is missing expected completion marker {prefix}."
            break

    weak_patterns = (
        "did not produce a clear",
        "was still searching",
        "no definitive answer",
        "cannot confirm",
        "verification gap",
        "partial exploration",
        "focused on project activation",
    )
    lower_summary = summary.lower()
    if summary and any(pattern in lower_summary for pattern in weak_patterns):
        warning = "Worker summary looks incomplete or inconclusive."

    tool_log_lines = sum(
        1 for line in summary.splitlines()
        if re.match(r"^\s*(ran|read|searched|grep|rg|sed|cat|terminal|mcp_|activated)\b", line.lower())
    )
    if summary and tool_log_lines >= max(2, len([line for line in summary.splitlines() if line.strip()]) // 2):
        warning = "Worker summary mostly describes tool activity instead of a conclusion."

    item["summary_quality"] = "warning" if warning else "ok"
    item["summary_warning"] = warning


def _subagent_board_item_from_event(event: Dict[str, Any], event_count: int) -> Dict[str, Any]:
    status = event.get("status")
    runtime = _subagent_runtime(event)
    row_id = str(event.get("subagent_id") or event.get("ao_session_id"))
    created_at = event.get("created_at")
    lane = _subagent_lane(status)
    group_fields = _subagent_group_fields(event, runtime)
    has_prompt_meta = False
    return {
        "id": row_id,
        "subagent_id": event.get("subagent_id"),
        "parent_id": event.get("parent_id"),
        "session_id": event.get("session_id"),
        "run_id": event.get("run_id"),
        "runtime": runtime,
        "runtime_session_id": event.get("runtime_session_id") or event.get("ao_session_id"),
        "runtime_project_id": event.get("runtime_project_id") or event.get("ao_project_id"),
        "runtime_selection": event.get("runtime_selection"),
        "selected_runtime": event.get("selected_runtime") or runtime,
        "runtime_selection_reason": event.get("runtime_selection_reason"),
        "runtime_fallback_reason": event.get("runtime_fallback_reason"),
        "runtime_policy_evidence": event.get("runtime_policy_evidence") or ((event.get("runtime_selection") or {}).get("runtime_policy_evidence") if isinstance(event.get("runtime_selection"), dict) else None),
        "runtime_policy_status": event.get("runtime_policy_status") or ((event.get("runtime_selection") or {}).get("runtime_policy_status") if isinstance(event.get("runtime_selection"), dict) else None),
        "runtime_policy_reason": event.get("runtime_policy_reason") or ((event.get("runtime_selection") or {}).get("runtime_policy_reason") if isinstance(event.get("runtime_selection"), dict) else None),
        "status": status,
        "lane": lane,
        "lane_reason": _subagent_lane_reason(status, lane, runtime),
        "attention_level": _subagent_attention_level(lane),
        "goal": event.get("goal"),
        "summary": event.get("summary"),
        "current_activity": _subagent_current_activity(event),
        "created_at": created_at,
        "updated_at": created_at,
        "last_activity_at": created_at,
        "event_count": event_count,
        "ao_session_id": event.get("ao_session_id"),
        "ao_project_id": event.get("ao_project_id"),
        "workspace_path": event.get("workspace_path"),
        "branch": event.get("branch"),
        "issue_id": event.get("issue_id"),
        "tmux_name": event.get("tmux_name"),
        "open_url": event.get("open_url"),
        "open_command": event.get("open_command"),
        "agent": event.get("agent"),
        "model": event.get("model"),
        "reasoning_effort": event.get("reasoning_effort"),
        "launch_profile_id": event.get("launch_profile_id"),
        "launch_plan_id": event.get("launch_plan_id"),
        "launch_task_id": event.get("launch_task_id"),
        "permissions": event.get("permissions"),
        "acceptance_criteria": event.get("acceptance_criteria") or [],
        "duration_seconds": _subagent_numeric(event, "duration_seconds"),
        "token_total": _subagent_token_total(event),
        "context_usage": event.get("context_usage"),
        "context_usage_categories": (
            event.get("context_usage_categories")
            or ((event.get("context_usage") or {}).get("categories") if isinstance(event.get("context_usage"), dict) else None)
            or []
        ),
        "cost_usd": _subagent_numeric(event, "cost_usd"),
        "files_read": event.get("files_read") or [],
        "files_written": event.get("files_written") or [],
        "files_changed": event.get("files_changed") or event.get("files_written") or [],
        "commands_run": event.get("commands_run") or [],
        "verification_status": event.get("verification_status"),
        "verification_evidence": event.get("verification_evidence") or [],
        "unresolved_gaps": event.get("unresolved_gaps") or [],
        "findings": event.get("findings") or [],
        "structured_summary": event.get("structured_summary"),
        "worker_confidence": event.get("worker_confidence"),
        "final_marker": event.get("final_marker"),
        "output_contract_version": event.get("output_contract_version"),
        "output_contract_status": event.get("output_contract_status"),
        "output_contract_warning": event.get("output_contract_warning"),
        "output_contract_score": event.get("output_contract_score"),
        "output_tail": event.get("output_tail") or [],
        "recent_action": event.get("action") if event.get("event") == "subagent.action" else None,
        "recent_action_status": event.get("action_status") if event.get("event") == "subagent.action" else None,
        "recent_action_message": event.get("message") if event.get("event") == "subagent.action" else None,
        "recent_action_at": event.get("created_at") or event.get("timestamp") if event.get("event") == "subagent.action" else None,
        "summary_quality": "ok",
        "summary_warning": None,
        "runtime_health": None,
        "runtime_warning": None,
        "diagnostic_status": None,
        "diagnostic_message": None,
        "recovery_recommendation": None,
        "transcript_available": False,
        "transcript_tail": None,
        "transcript_captured_at": None,
        "has_prompt_metadata": has_prompt_meta,
        "action_unavailable_reason": "AO controls are only available for AO-backed workers." if runtime != "ao" else None,
        **group_fields,
        "can_open": runtime == "ao" and bool(event.get("ao_session_id")),
        "can_stop": False,
        "can_follow_up": False,
        "can_retry": False,
        "can_reassign": False,
    }


def _merge_ao_session_into_board_item(
    item: Dict[str, Any],
    session: Any,
    store: Any,
    *,
    runtime_health: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = dict(item)
    row_id = f"ao:{session.id}"
    status = row.get("status") or session.display_status
    runtime_health = runtime_health or {"runtime_health": "ok", "runtime_warning": None}
    is_stale = runtime_health.get("runtime_health") == "stale"
    if is_stale:
        status = "terminated"
    lane = _subagent_lane(status)
    if lane == "running":
        status = session.display_status
        lane = _subagent_lane(status)
        if is_stale:
            status = "terminated"
            lane = "failed"
    prompt_meta = store.get_ao_prompt(session.id)
    group_fields = _subagent_group_fields({
        **row,
        "ao_project_id": session.project_id or row.get("ao_project_id"),
    }, "ao")
    action_unavailable_reason = None
    if lane == "running":
        action_unavailable_reason = "Retry and reassign are available after the worker reaches a terminal state."
    elif not prompt_meta:
        action_unavailable_reason = "Original AO prompt metadata is unavailable for retry or reassign."
    prompt_agent = (prompt_meta or {}).get("agent")
    prompt_model = (prompt_meta or {}).get("model")
    prompt_reasoning_effort = (prompt_meta or {}).get("reasoning_effort")
    prompt_launch_profile_id = (prompt_meta or {}).get("launch_profile_id")
    prompt_launch_plan_id = (prompt_meta or {}).get("launch_plan_id")
    prompt_launch_task_id = (prompt_meta or {}).get("launch_task_id")
    row.update({
        "id": row.get("id") or row_id,
        "subagent_id": row.get("subagent_id") or row_id,
        "runtime": "ao",
        "runtime_session_id": session.id,
        "runtime_project_id": session.project_id or row.get("runtime_project_id") or row.get("ao_project_id"),
        "status": status,
        "lane": lane,
        "lane_reason": runtime_health.get("runtime_warning") or _subagent_lane_reason(status, lane, "ao"),
        "attention_level": _subagent_attention_level(lane),
        "goal": row.get("goal") or (prompt_meta or {}).get("goal") or session.activity or f"AO session {session.id}",
        "summary": row.get("summary") or session.summary,
        "current_activity": session.activity or row.get("current_activity"),
        "updated_at": row.get("updated_at") or time.time(),
        "last_activity_at": row.get("last_activity_at") or row.get("updated_at") or time.time(),
        "event_count": int(row.get("event_count") or 0),
        "ao_session_id": session.id,
        "ao_project_id": session.project_id or row.get("ao_project_id"),
        "workspace_path": session.workspace_path or row.get("workspace_path"),
        "branch": session.branch or row.get("branch"),
        "issue_id": session.issue_id or row.get("issue_id"),
        "tmux_name": session.tmux_name or row.get("tmux_name"),
        "open_command": session.open_command or row.get("open_command"),
        "agent": row.get("agent") or prompt_agent or session.agent,
        "model": row.get("model") or prompt_model or session.model,
        "reasoning_effort": row.get("reasoning_effort") or prompt_reasoning_effort or session.reasoning_effort,
        "launch_profile_id": row.get("launch_profile_id") or prompt_launch_profile_id,
        "launch_plan_id": row.get("launch_plan_id") or prompt_launch_plan_id,
        "launch_task_id": row.get("launch_task_id") or prompt_launch_task_id,
        "permissions": row.get("permissions") or (prompt_meta or {}).get("permissions"),
        "acceptance_criteria": row.get("acceptance_criteria") or (prompt_meta or {}).get("acceptance_criteria") or [],
        "duration_seconds": row.get("duration_seconds"),
        "token_total": row.get("token_total"),
        "context_usage": row.get("context_usage"),
        "context_usage_categories": row.get("context_usage_categories") or [],
        "cost_usd": row.get("cost_usd"),
        "files_read": row.get("files_read") or [],
        "files_written": row.get("files_written") or [],
        "files_changed": row.get("files_changed") or row.get("files_written") or [],
        "commands_run": row.get("commands_run") or [],
        "verification_status": row.get("verification_status"),
        "verification_evidence": row.get("verification_evidence") or [],
        "unresolved_gaps": row.get("unresolved_gaps") or [],
        "findings": row.get("findings") or [],
        "structured_summary": row.get("structured_summary"),
        "worker_confidence": row.get("worker_confidence"),
        "final_marker": row.get("final_marker"),
        "output_contract_version": row.get("output_contract_version"),
        "output_contract_status": row.get("output_contract_status"),
        "output_contract_warning": row.get("output_contract_warning"),
        "output_contract_score": row.get("output_contract_score"),
        "output_tail": row.get("output_tail") or [],
        "recent_action": row.get("recent_action"),
        "recent_action_status": row.get("recent_action_status"),
        "recent_action_message": row.get("recent_action_message"),
        "recent_action_at": row.get("recent_action_at"),
        "summary_quality": row.get("summary_quality") or "ok",
        "summary_warning": row.get("summary_warning") or runtime_health.get("runtime_warning"),
        "runtime_health": runtime_health.get("runtime_health"),
        "runtime_warning": runtime_health.get("runtime_warning"),
        "diagnostic_status": "stale" if is_stale else lane,
        "diagnostic_message": runtime_health.get("runtime_warning") or _subagent_lane_reason(status, lane, "ao"),
        "recovery_recommendation": _ao_recovery_recommendation(status=status, lane=lane, is_stale=is_stale),
        "transcript_available": bool(session.tmux_name) and not is_stale,
        "transcript_tail": row.get("transcript_tail"),
        "transcript_captured_at": row.get("transcript_captured_at"),
        "has_prompt_metadata": bool(prompt_meta),
        "action_unavailable_reason": action_unavailable_reason,
        **group_fields,
        "can_open": True,
        "can_stop": (not is_stale) and lane in {"queued", "running", "needs_input"},
        "can_follow_up": (not is_stale) and lane == "running",
        "can_retry": lane in {"failed", "completed"} and bool(prompt_meta),
        "can_reassign": lane in {"failed", "completed"} and bool(prompt_meta),
    })
    return row


def _ao_recovery_recommendation(*, status: Optional[str], lane: str, is_stale: bool = False) -> str:
    if is_stale:
        return "Runtime is gone. Use Repair Retry to spawn a replacement from the original task context, or Open to inspect the worktree."
    if lane == "running":
        return "Worker is running. Use Resume or Follow-up to steer it, or Stop if it is no longer useful."
    if lane == "needs_input":
        return "Worker needs input. Send a follow-up or open the worker terminal/worktree for more detail."
    if lane == "failed":
        raw = str(status or "").lower()
        if raw in {"killed", "terminated", "cancelled", "canceled"}:
            return "Worker was stopped or terminated. Use Repair Retry to spawn a replacement with the latest diagnostics."
        return "Worker failed. Use transcript tail and action history to diagnose, then Repair Retry if the original prompt is available."
    if lane == "completed":
        return "Worker completed. Review summary and transcript tail; Retry or Reassign if the result needs another pass."
    return "Review diagnostics and choose a recovery action if needed."


def _subagent_board_item_matches(item: Dict[str, Any], params: Any) -> bool:
    runtime = params.get("runtime")
    if runtime:
        wanted = runtime.lower()
        actual = str(item.get("runtime") or "").lower()
        if wanted == "native":
            wanted = "hermes"
        if actual != wanted:
            return False
    status = params.get("status")
    if status:
        wanted_status = status.lower()
        if wanted_status == "needs_attention":
            wanted_status = "needs_input"
        actual_status = str(item.get("status") or "").lower()
        actual_lane = str(item.get("lane") or "").lower()
        if wanted_status not in {actual_status, actual_lane}:
            return False
    lane = params.get("lane")
    if lane and str(item.get("lane") or "").lower() != lane.lower():
        return False
    include_completed = str(params.get("include_completed") or "").lower()
    if include_completed in {"0", "false", "no"} and item.get("lane") == "completed":
        return False
    group_kind = params.get("group_kind")
    if group_kind and item.get("group_kind") != group_kind:
        return False
    group_key = params.get("group_key")
    if group_key and item.get("group_key") != group_key:
        return False
    updated_after = params.get("updated_after")
    if updated_after:
        try:
            if float(item.get("updated_at") or 0) <= float(updated_after):
                return False
        except (TypeError, ValueError):
            return False
    project_id = params.get("project_id")
    if project_id and item.get("ao_project_id") != project_id:
        return False
    session_id = params.get("session_id")
    if session_id and item.get("session_id") != session_id:
        return False
    ao_session_id = params.get("ao_session_id")
    if ao_session_id and item.get("ao_session_id") != ao_session_id:
        return False
    needle = str(params.get("q") or params.get("search") or "").strip().lower()
    if needle:
        haystack = " ".join(str(item.get(key) or "") for key in (
            "goal", "summary", "current_activity", "branch", "ao_project_id", "ao_session_id"
        )).lower()
        if needle not in haystack:
            return False
    return True
