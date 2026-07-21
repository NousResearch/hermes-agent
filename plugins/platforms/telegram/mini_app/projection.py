"""Field-allowlisted projections for local Hermes state."""

from __future__ import annotations

import time
from typing import Any


def project_status(data: Any) -> dict[str, Any]:
    source = data if isinstance(data, dict) else {}
    platforms: dict[str, dict[str, Any]] = {}
    raw_platforms = source.get("gateway_platforms") or {}
    if isinstance(raw_platforms, dict):
        for name, raw in raw_platforms.items():
            if isinstance(raw, dict):
                platforms[str(name)[:80]] = {
                    key: raw.get(key)
                    for key in ("state", "substate", "error_message")
                    if key in raw
                }
    return {
        key: source.get(key)
        for key in (
            "version",
            "release_date",
            "gateway_running",
            "gateway_state",
            "source",
        )
        if key in source
    } | {"gateway_platforms": platforms}


def project_sessions(data: Any) -> dict[str, Any]:
    source = data if isinstance(data, dict) else {}
    raw_sessions = (
        source.get("sessions") if isinstance(source.get("sessions"), list) else []
    )
    fields = (
        "id",
        "title",
        "preview",
        "source",
        "message_count",
        "last_active",
        "started_at",
    )
    sessions = [
        {key: item.get(key) for key in fields if key in item}
        for item in raw_sessions
        if isinstance(item, dict)
    ]
    return {"sessions": sessions} | {
        key: source.get(key) for key in ("total", "limit", "offset") if key in source
    }


def project_catalog(data: Any, collection_key: str) -> list[dict[str, Any]]:
    items = (
        data
        if isinstance(data, list)
        else data.get(collection_key, [])
        if isinstance(data, dict)
        else []
    )
    fields = ("name", "label", "description", "category", "enabled")
    return [
        {key: item.get(key) for key in fields if key in item}
        for item in items
        if isinstance(item, dict)
    ]


def normalize_swarm_board(
    board: dict[str, Any], profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    """Project Kanban board rows into the Mini App's non-secret swarm summary."""
    task_fields = (
        "id",
        "title",
        "body",
        "assignee",
        "priority",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "updated_at",
    )
    statuses = (
        "triage",
        "todo",
        "scheduled",
        "ready",
        "running",
        "blocked",
        "review",
        "done",
        "archived",
    )
    empty_counts = lambda: {status: 0 for status in statuses}
    by_agent: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        name = str(profile.get("name") or profile.get("profile") or "").strip()
        if name:
            by_agent[name] = {
                "name": name,
                "profile": name,
                "active": bool(profile.get("active")),
                "model": profile.get("model") or "",
                "provider": profile.get("provider") or "",
                "gateway": profile.get("gateway") or "unknown",
                "alias": profile.get("alias") or "",
                "status": "idle",
                "assigned_count": 0,
                "running_count": 0,
                "blocked_count": 0,
                "ready_count": 0,
                "done_count": 0,
                "tasks": empty_counts(),
                "last_activity": None,
            }

    columns: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for raw_column in board.get("columns") or []:
        if not isinstance(raw_column, dict):
            continue
        status = str(raw_column.get("name") or raw_column.get("status") or "todo")
        column_tasks: list[dict[str, Any]] = []
        for raw_task in raw_column.get("tasks") or []:
            if not isinstance(raw_task, dict):
                continue
            task = {key: raw_task.get(key) for key in task_fields if key in raw_task}
            task.setdefault("status", status)
            tasks.append(task)
            column_tasks.append(task)
            assignee = str(task.get("assignee") or "").strip()
            if not assignee:
                continue
            agent = by_agent.setdefault(
                assignee,
                {
                    "name": assignee,
                    "profile": assignee,
                    "active": False,
                    "model": "",
                    "provider": "",
                    "gateway": "unknown",
                    "alias": "",
                    "status": "idle",
                    "assigned_count": 0,
                    "running_count": 0,
                    "blocked_count": 0,
                    "ready_count": 0,
                    "done_count": 0,
                    "tasks": empty_counts(),
                    "last_activity": None,
                },
            )
            agent["assigned_count"] += 1
            agent["tasks"][status] = int(agent["tasks"].get(status, 0)) + 1
            counter = {
                "running": "running_count",
                "blocked": "blocked_count",
                "ready": "ready_count",
                "done": "done_count",
            }.get(status)
            if counter:
                agent[counter] += 1
            changed = task.get("updated_at") or task.get("created_at")
            if changed and (
                not agent["last_activity"] or str(changed) > str(agent["last_activity"])
            ):
                agent["last_activity"] = changed
        columns.append({
            "status": status,
            "name": status,
            "tasks": column_tasks,
            "count": len(column_tasks),
        })

    for agent in by_agent.values():
        if agent["running_count"]:
            agent["status"] = "running"
        elif agent["blocked_count"]:
            agent["status"] = "blocked"
        elif agent["ready_count"]:
            agent["status"] = "queued"
        elif str(agent.get("gateway", "")).lower() in {"stopped", "error", "failed"}:
            agent["status"] = "error"
    rank = {"running": 0, "blocked": 1, "queued": 2, "idle": 3, "error": 4}
    agents = sorted(
        by_agent.values(),
        key=lambda item: (
            rank.get(item["status"], 9),
            -item["assigned_count"],
            item["name"],
        ),
    )
    return {
        "columns": columns,
        "agents": agents,
        "summary": {
            "agents": len(agents),
            "running_agents": sum(a["status"] == "running" for a in agents),
            "blocked_agents": sum(a["status"] == "blocked" for a in agents),
            "tasks": len(tasks),
            "running_tasks": sum(t.get("status") == "running" for t in tasks),
            "blocked_tasks": sum(t.get("status") == "blocked" for t in tasks),
            "ready_tasks": sum(t.get("status") == "ready" for t in tasks),
            "triage_tasks": sum(t.get("status") == "triage" for t in tasks),
            "done_tasks": sum(t.get("status") == "done" for t in tasks),
        },
        "assignees": [
            str(value)[:100]
            for value in (board.get("assignees") or [])
            if isinstance(value, (str, int))
        ],
        "tenants": [
            str(value)[:100]
            for value in (board.get("tenants") or [])
            if isinstance(value, (str, int))
        ],
        "latest_event_id": int(board.get("latest_event_id") or 0),
        "now": board.get("now") or int(time.time()),
        "source": "local-kanban-db",
    }
