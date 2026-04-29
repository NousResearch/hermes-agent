"""Minimal workflow/task state storage for gateway dispatch."""

from __future__ import annotations

import json
import os
import uuid
import contextlib
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


TASK_STATUSES = frozenset(
    {
        "created",
        "validated",
        "send_pending",
        "sent",
        "relay_enqueued",
        "received",
        "completed",
        "waiting_review",
        "returned_to_pm",
        "failed",
        "cancelled",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _state_path() -> Path:
    configured = os.getenv("HERMES_WORKFLOW_STATE_PATH", "").strip()
    return Path(configured).expanduser() if configured else _repo_root() / "workflow_state" / "active_workflows.json"


def _lock_path() -> Path:
    return _state_path().with_name(f"{_state_path().name}.lock")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _empty_state() -> Dict[str, Any]:
    return {"workflows": []}


@contextlib.contextmanager
def _state_lock() -> Any:
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def load_state() -> Dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return _empty_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_state()
    if not isinstance(payload, dict):
        return _empty_state()
    workflows = payload.get("workflows")
    if not isinstance(workflows, list):
        payload["workflows"] = []
    return payload


def save_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _mutate_state(mutator: Callable[[Dict[str, Any]], Any]) -> Any:
    with _state_lock():
        state = load_state()
        result = mutator(state)
        save_state(state)
        return result


def reset_state() -> None:
    with _state_lock():
        save_state(_empty_state())


def create_workflow(
    *,
    profile_id: str,
    title: str,
    dispatcher_role: str,
    current_node: str = "created",
    next_action: str = "",
    status: str = "running",
) -> Dict[str, Any]:
    now = _now_iso()
    workflow = {
        "workflow_id": _new_id("wf"),
        "profile_id": profile_id,
        "title": title,
        "status": status,
        "current_node": current_node,
        "dispatcher_role": dispatcher_role,
        "pending_tasks": [],
        "completed_tasks": [],
        "blocked_tasks": [],
        "next_action": next_action,
        "updated_at": now,
        "created_at": now,
    }
    def _append(state: Dict[str, Any]) -> None:
        state.setdefault("workflows", []).append(workflow)

    _mutate_state(_append)
    return workflow


def create_task(
    *,
    workflow_id: str,
    to_role: str,
    task_type: str,
    deliverable: str,
    instruction: str,
    return_to: str,
    reviewer_role: str = "",
    deliver_to_role: str = "",
    deliver_after_review_to_role: str = "",
    upstream_role: str = "",
    upstream_summary: str = "",
    deliverable_format: str = "",
    status: str = "created",
) -> Dict[str, Any]:
    now = _now_iso()
    normalized_return_to = str(return_to or "").strip().lower()
    normalized_deliver_to = str(deliver_to_role or normalized_return_to).strip().lower()
    normalized_reviewer = str(reviewer_role or "").strip().lower()
    return {
        "task_id": _new_id("task"),
        "workflow_id": workflow_id,
        "to_role": str(to_role or "").strip().lower(),
        "task_type": str(task_type or "").strip(),
        "deliverable": str(deliverable or "").strip(),
        "instruction": str(instruction or "").strip(),
        "return_to": normalized_return_to,
        "reviewer_role": normalized_reviewer,
        "deliver_to_role": normalized_deliver_to,
        "deliver_after_review_to_role": str(deliver_after_review_to_role or normalized_deliver_to).strip().lower(),
        "upstream_role": str(upstream_role or "").strip().lower(),
        "upstream_summary": str(upstream_summary or "").strip(),
        "deliverable_format": str(deliverable_format or deliverable or "").strip(),
        "status": status,
        "real_sent": False,
        "message_id": None,
        "sent_message_id": None,
        "outbound_id": None,
        "outbound_ids": [],
        "created_at": now,
        "updated_at": now,
        "status_history": [{"status": status, "at": now}],
    }


def add_pending_task(workflow_id: str, task: Dict[str, Any]) -> None:
    add_pending_tasks(workflow_id, [task])


def add_pending_tasks(workflow_id: str, tasks: List[Dict[str, Any]]) -> None:
    def _add(state: Dict[str, Any]) -> None:
        workflow = _find_workflow_in_state(state, workflow_id)
        if not workflow:
            raise ValueError(f"workflow not found: {workflow_id}")
        pending = workflow.setdefault("pending_tasks", [])
        for task in tasks:
            task_id = str((task or {}).get("task_id") or "").strip()
            if not task_id:
                continue
            existing = _find_task_anywhere(workflow, task_id)[1]
            if existing:
                existing.update(task)
            else:
                pending.append(task)
        workflow["updated_at"] = _now_iso()

    _mutate_state(_add)


def update_workflow_progress(
    *,
    workflow_id: str,
    current_node: str = "",
    next_action: str = "",
) -> Optional[Dict[str, Any]]:
    def _update(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        workflow = _find_workflow_in_state(state, workflow_id)
        if not workflow:
            return None
        if current_node:
            workflow["current_node"] = str(current_node)
        if next_action:
            workflow["next_action"] = str(next_action)
        workflow["updated_at"] = _now_iso()
        return dict(workflow)

    return _mutate_state(_update)


def get_active_workflow(profile_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    workflows = load_state().get("workflows", [])
    candidates = [
        workflow
        for workflow in workflows
        if isinstance(workflow, dict)
        and workflow.get("status") == "running"
        and (profile_id is None or workflow.get("profile_id") == profile_id)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: str(item.get("updated_at", "")))


def get_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
    return _find_workflow_in_state(load_state(), workflow_id)


def get_task(*, workflow_id: str, task_id: str) -> Optional[Dict[str, Any]]:
    workflow = _find_workflow_in_state(load_state(), workflow_id)
    if not workflow:
        return None
    _, task = _find_task_anywhere(workflow, task_id)
    return task


def _summarize_result(text: str, *, max_chars: int = 500) -> str:
    summary = " ".join(str(text or "").strip().split())
    if len(summary) > max_chars:
        return summary[: max_chars - 1].rstrip() + "…"
    return summary


def find_waiting_task_for_role(
    *,
    role_id: str,
    profile_id: Optional[str] = None,
    workflow_id: str = "",
    task_id: str = "",
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Return the single pending task currently waiting for this role."""
    role = str(role_id or "").strip().lower()
    if not role:
        return None
    waiting_statuses = {"send_pending", "sent", "relay_enqueued", "received"}
    matches: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for workflow in load_state().get("workflows", []):
        if not isinstance(workflow, dict) or workflow.get("status") != "running":
            continue
        if profile_id and workflow.get("profile_id") != profile_id:
            continue
        if workflow_id and workflow.get("workflow_id") != workflow_id:
            continue
        for task in workflow.get("pending_tasks", []):
            if not isinstance(task, dict):
                continue
            if task_id and task.get("task_id") != task_id:
                continue
            if str(task.get("to_role") or "").strip().lower() != role:
                continue
            if str(task.get("status") or "").strip() not in waiting_statuses:
                continue
            matches.append((workflow, task))
    return matches[0] if len(matches) == 1 else None


def find_waiting_review_for_role(
    *,
    role_id: str,
    profile_id: Optional[str] = None,
    workflow_id: str = "",
    task_id: str = "",
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Return the single task currently waiting for this reviewer."""
    role = str(role_id or "").strip().lower()
    if not role:
        return None
    matches: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for workflow in load_state().get("workflows", []):
        if not isinstance(workflow, dict) or workflow.get("status") != "running":
            continue
        if profile_id and workflow.get("profile_id") != profile_id:
            continue
        if workflow_id and workflow.get("workflow_id") != workflow_id:
            continue
        for task in workflow.get("pending_tasks", []):
            if not isinstance(task, dict):
                continue
            if task_id and task.get("task_id") != task_id:
                continue
            if str(task.get("reviewer_role") or "").strip().lower() != role:
                continue
            if str(task.get("status") or "").strip() != "waiting_review":
                continue
            matches.append((workflow, task))
    return matches[0] if len(matches) == 1 else None


def update_task_status(
    *,
    workflow_id: str,
    task_id: str,
    status: str,
    error: str = "",
    message_id: str = "",
    outbound_id: str = "",
    relay_target_profile: str = "",
    relay_message_id: str = "",
    details: Optional[Dict[str, Any]] = None,
    move_to: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Idempotently update a task status and append status history."""
    normalized_status = str(status or "").strip()
    if normalized_status not in TASK_STATUSES:
        raise ValueError(f"unsupported task status: {status}")

    def _update(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        workflow = _find_workflow_in_state(state, workflow_id)
        if not workflow:
            return None
        bucket, task = _find_task_anywhere(workflow, task_id)
        if not task:
            return None

        now = _now_iso()
        task["status"] = normalized_status
        task["updated_at"] = now
        if error:
            task["error"] = str(error or "")
        if message_id:
            task["message_id"] = str(message_id)
            task["sent_message_id"] = str(message_id)
        if outbound_id:
            task["outbound_id"] = str(outbound_id)
            outbound_ids = task.setdefault("outbound_ids", [])
            if isinstance(outbound_ids, list) and outbound_id not in outbound_ids:
                outbound_ids.append(str(outbound_id))
        if relay_target_profile:
            task["relay_target_profile"] = str(relay_target_profile)
        if relay_message_id:
            task["relay_message_id"] = str(relay_message_id)
        if normalized_status in {"sent", "relay_enqueued", "received", "completed"}:
            task["real_sent"] = bool(task.get("real_sent") or normalized_status in {"sent", "relay_enqueued", "received", "completed"})
        if details:
            extra = task.setdefault("details", {})
            if isinstance(extra, dict):
                extra.update(details)

        _append_status_history(
            task,
            status=normalized_status,
            at=now,
            error=error,
            message_id=message_id,
            outbound_id=outbound_id,
            relay_target_profile=relay_target_profile,
            relay_message_id=relay_message_id,
        )

        if move_to:
            _move_task(workflow, task_id, move_to)
        elif bucket not in {"pending_tasks", "completed_tasks", "blocked_tasks"}:
            workflow.setdefault("pending_tasks", []).append(task)
        workflow["updated_at"] = now
        return dict(task)

    return _mutate_state(_update)


def mark_task_validated(*, workflow_id: str, task_id: str) -> Optional[Dict[str, Any]]:
    return update_task_status(workflow_id=workflow_id, task_id=task_id, status="validated")


def mark_task_send_pending(*, workflow_id: str, task_id: str) -> Optional[Dict[str, Any]]:
    return update_task_status(workflow_id=workflow_id, task_id=task_id, status="send_pending")


def mark_task_sent(
    *,
    workflow_id: str,
    task_id: str,
    message_id: str,
    outbound_id: str = "",
) -> Optional[Dict[str, Any]]:
    return update_task_status(
        workflow_id=workflow_id,
        task_id=task_id,
        status="sent",
        message_id=message_id,
        outbound_id=outbound_id,
    )


def mark_task_relay_enqueued(
    *,
    workflow_id: str,
    task_id: str,
    target_profile: str = "",
    message_id: str = "",
    outbound_id: str = "",
) -> Optional[Dict[str, Any]]:
    return update_task_status(
        workflow_id=workflow_id,
        task_id=task_id,
        status="relay_enqueued",
        relay_target_profile=target_profile,
        relay_message_id=message_id,
        outbound_id=outbound_id,
    )


def mark_task_received(
    *,
    workflow_id: str,
    task_id: str,
    target_profile: str = "",
    relay_message_id: str = "",
) -> Optional[Dict[str, Any]]:
    return update_task_status(
        workflow_id=workflow_id,
        task_id=task_id,
        status="received",
        relay_target_profile=target_profile,
        relay_message_id=relay_message_id,
    )


def mark_task_failed(*, workflow_id: str, task_id: str, error: str) -> Optional[Dict[str, Any]]:
    return update_task_status(
        workflow_id=workflow_id,
        task_id=task_id,
        status="failed",
        error=error,
        move_to="blocked_tasks",
    )


def mark_task_cancelled(*, workflow_id: str, task_id: str, reason: str = "") -> Optional[Dict[str, Any]]:
    return update_task_status(
        workflow_id=workflow_id,
        task_id=task_id,
        status="cancelled",
        error=reason,
        move_to="blocked_tasks",
    )


def mark_task_returned(*, workflow_id: str, task_id: str) -> Optional[Dict[str, Any]]:
    return update_task_status(
        workflow_id=workflow_id,
        task_id=task_id,
        status="completed",
        move_to="completed_tasks",
    )


def record_task_result(
    *,
    workflow_id: str,
    task_id: str,
    result_text: str,
    completed_by_role: str,
    result_summary: str = "",
) -> Optional[Dict[str, Any]]:
    """Record an executor's final answer and move to review or return state."""
    summary = result_summary.strip() if result_summary else _summarize_result(result_text)
    completed_by = str(completed_by_role or "").strip().lower()

    def _record(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        workflow = _find_workflow_in_state(state, workflow_id)
        if not workflow:
            return None
        _, task = _find_task_anywhere(workflow, task_id)
        if not task:
            return None

        now = _now_iso()
        reviewer = str(task.get("reviewer_role") or "").strip().lower()
        deliver_to = str(
            task.get("deliver_to_role")
            or task.get("return_to")
            or workflow.get("dispatcher_role")
            or ""
        ).strip().lower()
        task.update(
            {
                "result_text": str(result_text or "").strip(),
                "result_summary": summary,
                "completed_by_role": completed_by,
                "returned_to_role": reviewer or deliver_to,
                "completed_at": now,
                "updated_at": now,
            }
        )
        _append_status_history(task, status="completed", at=now)

        if reviewer:
            task["status"] = "waiting_review"
            task["review_requested_at"] = now
            workflow["current_node"] = f"waiting_review_{reviewer}"
            workflow["next_action"] = f"wait_for_{reviewer}_review"
            _append_status_history(task, status="waiting_review", at=now)
        else:
            final_status = "returned_to_pm" if deliver_to == "pm" else "completed"
            task["status"] = final_status
            task["returned_at"] = now
            workflow["current_node"] = f"returned_to_{deliver_to}" if deliver_to else "task_returned"
            workflow["next_action"] = f"decide_next_after_{completed_by}_return" if completed_by else "decide_next_after_return"
            _append_status_history(task, status=final_status, at=now)
            _move_task(workflow, task_id, "completed_tasks")

        workflow["updated_at"] = now
        return dict(task)

    return _mutate_state(_record)


def record_review_result(
    *,
    workflow_id: str,
    task_id: str,
    result_text: str,
    reviewed_by_role: str,
    approved: bool = True,
    result_summary: str = "",
) -> Optional[Dict[str, Any]]:
    """Record a reviewer response and return approved work to its deliver target."""
    summary = result_summary.strip() if result_summary else _summarize_result(result_text)
    reviewed_by = str(reviewed_by_role or "").strip().lower()

    def _record(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        workflow = _find_workflow_in_state(state, workflow_id)
        if not workflow:
            return None
        _, task = _find_task_anywhere(workflow, task_id)
        if not task:
            return None

        now = _now_iso()
        deliver_to = str(
            task.get("deliver_after_review_to_role")
            or task.get("deliver_to_role")
            or task.get("return_to")
            or workflow.get("dispatcher_role")
            or ""
        ).strip().lower()
        task.update(
            {
                "review_result_text": str(result_text or "").strip(),
                "review_result_summary": summary,
                "reviewed_by_role": reviewed_by,
                "review_passed": bool(approved),
                "reviewed_at": now,
                "returned_to_role": deliver_to,
                "updated_at": now,
            }
        )
        if approved:
            final_status = "returned_to_pm" if deliver_to == "pm" else "completed"
            task["status"] = final_status
            task["returned_at"] = now
            workflow["current_node"] = f"returned_to_{deliver_to}" if deliver_to else "review_returned"
            workflow["next_action"] = f"decide_next_after_{reviewed_by}_review" if reviewed_by else "decide_next_after_review"
            _append_status_history(task, status=final_status, at=now)
            _move_task(workflow, task_id, "completed_tasks")
        else:
            task["status"] = "received"
            task["review_rejected_at"] = now
            workflow["current_node"] = f"returned_to_{task.get('to_role')}"
            workflow["next_action"] = f"wait_for_{task.get('to_role')}_rework"
            _append_status_history(task, status="received", at=now)
        workflow["updated_at"] = now
        return dict(task)

    return _mutate_state(_record)


def _find_workflow_in_state(state: Dict[str, Any], workflow_id: str) -> Optional[Dict[str, Any]]:
    for workflow in state.get("workflows", []):
        if isinstance(workflow, dict) and workflow.get("workflow_id") == workflow_id:
            return workflow
    return None


def _find_task(tasks: List[Dict[str, Any]], task_id: str) -> Optional[Dict[str, Any]]:
    for task in tasks:
        if isinstance(task, dict) and task.get("task_id") == task_id:
            return task
    return None


def _find_task_anywhere(workflow: Dict[str, Any], task_id: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    for bucket in ("pending_tasks", "completed_tasks", "blocked_tasks"):
        task = _find_task(workflow.get(bucket, []), task_id)
        if task:
            return bucket, task
    return "", None


def _move_task(workflow: Dict[str, Any], task_id: str, target_bucket: str) -> None:
    if target_bucket not in {"pending_tasks", "completed_tasks", "blocked_tasks"}:
        return
    _, task = _find_task_anywhere(workflow, task_id)
    if not task:
        return
    for bucket in ("pending_tasks", "completed_tasks", "blocked_tasks"):
        workflow[bucket] = [
            existing for existing in workflow.get(bucket, []) if existing.get("task_id") != task_id
        ]
    workflow.setdefault(target_bucket, []).append(task)


def _append_status_history(
    task: Dict[str, Any],
    *,
    status: str,
    at: str,
    error: str = "",
    message_id: str = "",
    outbound_id: str = "",
    relay_target_profile: str = "",
    relay_message_id: str = "",
) -> None:
    history = task.setdefault("status_history", [])
    if not isinstance(history, list):
        history = []
        task["status_history"] = history
    entry: Dict[str, Any] = {"status": status, "at": at}
    if error:
        entry["error"] = str(error)
    if message_id:
        entry["message_id"] = str(message_id)
    if outbound_id:
        entry["outbound_id"] = str(outbound_id)
    if relay_target_profile:
        entry["relay_target_profile"] = str(relay_target_profile)
    if relay_message_id:
        entry["relay_message_id"] = str(relay_message_id)
    last = history[-1] if history and isinstance(history[-1], dict) else {}
    comparable = {k: v for k, v in entry.items() if k != "at"}
    last_comparable = {k: v for k, v in last.items() if k != "at"}
    if comparable != last_comparable:
        history.append(entry)
