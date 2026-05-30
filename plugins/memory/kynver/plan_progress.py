"""Plan progress projection and read-back for Hermes `todo` ↔ Kynver AgentOS."""

from __future__ import annotations

import json
import logging
from typing import Any

from .agentos_bridge import KynverAgentOSClient, KynverAgentOSError
from .operating_config import OperatingLinkage
from .pre_transition import (
    PreTransitionError,
    assert_focus_allowed,
    assert_single_in_progress,
    hermes_row_key,
    normalize_hermes_status,
)

logger = logging.getLogger(__name__)

_STATUS_TO_ROW: dict[str, str] = {
    "pending": "todo",
    "in_progress": "in_progress",
    "completed": "partial",
    "cancelled": "blocked",
}

_ROW_TO_HERMES: dict[str, str] = {
    "todo": "pending",
    "in_progress": "in_progress",
    # `running` is harness executor lease only — never Hermes current focus.
    "partial": "completed",
    "blocked": "cancelled",
    "done": "completed",
}


def _row_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("rowKey") or "")
        if key:
            out[key] = row
    return out


def _parse_todo_id(row_key: str) -> str | None:
    if row_key.startswith("hermes-todo:"):
        return row_key[len("hermes-todo:") :]
    return None


def inspect_todo_write(
    client: KynverAgentOSClient,
    linkage: OperatingLinkage,
    todos: list[dict[str, Any]],
    *,
    merge: bool,
) -> str | None:
    """Validate a todo write without mutating Kynver. Returns block message or None."""

    if not linkage.plan_id:
        return None

    assert_single_in_progress(todos)

    plan_path = f"/plans/{linkage.plan_id}"
    rows_payload = client.get(f"{plan_path}/progress-rows")
    items = rows_payload.get("items") if isinstance(rows_payload, dict) else rows_payload
    rows = list(items or [])
    by_key = _row_index(rows)

    for item in todos:
        todo_id = str(item.get("id") or "").strip()
        if not todo_id:
            continue
        row_key = hermes_row_key(todo_id)
        status = normalize_hermes_status(str(item.get("status", "")))
        existing = by_key.get(row_key)
        try:
            assert_focus_allowed(
                row_status=str(existing.get("status")) if existing else None,
                next_hermes_status=status,
            )
        except PreTransitionError as exc:
            return str(exc)
    return None


def project_todo_write(
    client: KynverAgentOSClient,
    linkage: OperatingLinkage,
    todos: list[dict[str, Any]],
    *,
    merge: bool,
) -> dict[str, Any]:
    if not linkage.plan_id:
        return {"projected": False, "reason": "no KYNVER_PLAN_ID"}

    blocked = inspect_todo_write(client, linkage, todos, merge=merge)
    if blocked:
        raise PreTransitionError(blocked)

    plan_path = f"/plans/{linkage.plan_id}"
    rows_payload = client.get(f"{plan_path}/progress-rows")
    items = rows_payload.get("items") if isinstance(rows_payload, dict) else rows_payload
    rows = list(items or [])
    by_key = _row_index(rows)

    upserts: list[dict[str, Any]] = []
    for item in todos:
        todo_id = str(item.get("id") or "").strip()
        if not todo_id:
            continue
        row_key = hermes_row_key(todo_id)
        status = normalize_hermes_status(str(item.get("status", "")))
        existing = by_key.get(row_key)
        upserts.append(
            {
                "rowKey": row_key,
                "title": str(item.get("content") or todo_id)[:500],
                "status": _STATUS_TO_ROW.get(status, "todo"),
                "taskId": linkage.task_id,
            }
        )

    if upserts:
        client.post(f"{plan_path}/progress-rows", {"rows": upserts})

    focus = next(
        (t for t in todos if normalize_hermes_status(str(t.get("status", ""))) == "in_progress"),
        None,
    )
    if focus:
        row_key = hermes_row_key(str(focus.get("id") or ""))
        client.post(
            f"{plan_path}/progress-focus",
            {
                "rowKey": row_key,
                "taskId": linkage.task_id,
                "roleLane": "implementer",
                "executorRef": linkage.executor_ref,
                "note": f"Hermes todo focus: {focus.get('content', '')}"[:500],
            },
        )
    elif not merge:
        client.post(
            f"{plan_path}/progress-focus",
            {
                "rowKey": None,
                "taskId": linkage.task_id,
                "roleLane": "implementer",
                "executorRef": linkage.executor_ref,
                "note": "Hermes cleared todo focus",
            },
        )

    return {"projected": True, "rows": len(upserts), "focus": bool(focus)}


def safe_project_todo_write(
    client: KynverAgentOSClient,
    linkage: OperatingLinkage,
    todos: list[dict[str, Any]],
    *,
    merge: bool,
) -> dict[str, Any]:
    try:
        return project_todo_write(client, linkage, todos, merge=merge)
    except PreTransitionError as exc:
        return {"blocked": True, "error": str(exc)}
    except KynverAgentOSError as exc:
        logger.warning("Kynver todo projection failed: %s", exc)
        return {"projected": False, "error": str(exc)}


def reconcile_todos_from_kynver(
    client: KynverAgentOSClient,
    linkage: OperatingLinkage,
    local_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not linkage.plan_id:
        return [item.copy() for item in local_items]

    plan_path = f"/plans/{linkage.plan_id}"
    plan = client.get(plan_path)
    plan_body = plan.get("plan") if isinstance(plan, dict) else plan
    in_progress_key = plan_body.get("inProgressRowKey") if isinstance(plan_body, dict) else None

    rows_payload = client.get(f"{plan_path}/progress-rows")
    items = rows_payload.get("items") if isinstance(rows_payload, dict) else rows_payload
    remote_rows = [r for r in (items or []) if isinstance(r, dict)]

    by_id = {str(item.get("id")): dict(item) for item in local_items}
    for row in remote_rows:
        row_key = str(row.get("rowKey") or "")
        todo_id = _parse_todo_id(row_key)
        if not todo_id:
            continue
        status = _ROW_TO_HERMES.get(str(row.get("status") or "todo"), "pending")
        if row_key == in_progress_key:
            status = "in_progress"
        title = str(row.get("title") or todo_id)
        existing = by_id.get(todo_id, {"id": todo_id, "content": title, "status": "pending"})
        existing["content"] = title
        existing["status"] = status
        by_id[todo_id] = existing

    if in_progress_key:
        focus_id = _parse_todo_id(in_progress_key)
        if focus_id and focus_id in by_id:
            for item in by_id.values():
                if item["id"] != focus_id and item.get("status") == "in_progress":
                    item["status"] = "pending"
            by_id[focus_id]["status"] = "in_progress"

    return list(by_id.values())


def safe_reconcile_todos_from_kynver(
    client: KynverAgentOSClient,
    linkage: OperatingLinkage,
    local_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        merged = reconcile_todos_from_kynver(client, linkage, local_items)
        return merged, {"reconciled": True}
    except KynverAgentOSError as exc:
        logger.warning("Kynver todo read-back failed: %s", exc)
        return [item.copy() for item in local_items], {"reconciled": False, "error": str(exc)}


def transform_todo_result(result: str, client: KynverAgentOSClient, linkage: OperatingLinkage) -> str | None:
    try:
        payload = json.loads(result) if isinstance(result, str) else result
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    items = payload.get("todos")
    if not isinstance(items, list):
        return None

    merged, meta = safe_reconcile_todos_from_kynver(client, linkage, items)
    payload["todos"] = merged
    payload["kynverReadBack"] = meta
    payload["summary"] = {
        "total": len(merged),
        "pending": sum(1 for i in merged if i.get("status") == "pending"),
        "in_progress": sum(1 for i in merged if i.get("status") == "in_progress"),
        "completed": sum(1 for i in merged if i.get("status") == "completed"),
        "cancelled": sum(1 for i in merged if i.get("status") == "cancelled"),
    }
    return json.dumps(payload, ensure_ascii=False)
