"""Read-only kanban task routes for the OpenAI-compatible gateway.

The dashboard plugin (``plugins/kanban/dashboard/plugin_api.py``) already serves
the full task API, but it is mounted on the dashboard web server, not on this
gateway. MyMateNico only needs to *read* the board so a project can show its
work and the last run; every write or control operation (create, assign,
dispatch, reclaim, terminate) stays on the dashboard where its transition rules
and tests live. Exposing writes here would let an app-level API key dispatch
work, which is exactly the race this design avoids.

Route table is registered through ``api_server.APIServerAdapter._http_route_table``
and inherits the gateway's API-key middleware.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Optional

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - aiohttp is a gateway dependency
    web = None  # type: ignore[assignment]

from hermes_cli import kanban_db, kanban_db_boards, kanban_db_connect

logger = logging.getLogger("gateway.platforms.api_server")

MAX_LIMIT = 200


def _iso(value: Any) -> Optional[str]:
    """Epoch seconds -> ISO-8601 UTC; tolerant of None/strings/already-ISO."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = int(float(value))
        except ValueError:
            return value
    try:
        return datetime.fromtimestamp(int(value), tz=UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def task_to_dict(task: Any) -> dict[str, Any]:
    """Small, stable projection for an app-facing read model."""
    return {
        "id": task.id,
        "title": task.title,
        "body": task.body,
        "status": task.status,
        "assignee": task.assignee,
        "priority": task.priority,
        "project_id": getattr(task, "project_id", None),
        "current_run_id": getattr(task, "current_run_id", None),
        "branch_name": getattr(task, "branch_name", None),
        "created_at": _iso(task.created_at),
        "started_at": _iso(getattr(task, "started_at", None)),
        "completed_at": _iso(getattr(task, "completed_at", None)),
        "last_heartbeat_at": _iso(getattr(task, "last_heartbeat_at", None)),
        "last_failure_error": getattr(task, "last_failure_error", None),
    }


def event_to_dict(event: Any) -> dict[str, Any]:
    return {
        "id": event.id,
        "kind": event.kind,
        "created_at": _iso(event.created_at),
        "run_id": getattr(event, "run_id", None),
        "payload": getattr(event, "payload", None),
    }


def _connect(board: Optional[str]):
    return kanban_db_connect.connect_closing(board=board or None)


def list_tasks_payload(
    *,
    board: Optional[str] = None,
    status: Optional[str] = None,
    assignee: Optional[str] = None,
    limit: Optional[int] = None,
    include_archived: bool = False,
) -> list[dict[str, Any]]:
    with _connect(board) as conn:
        tasks = kanban_db.list_tasks(
            conn,
            status=status,
            assignee=assignee,
            include_archived=include_archived,
            limit=limit,
        )
        return [task_to_dict(task) for task in tasks]


def get_task_payload(task_id: str, *, board: Optional[str] = None) -> Optional[dict[str, Any]]:
    with _connect(board) as conn:
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            return None
        return {
            "task": task_to_dict(task),
            "events": [event_to_dict(event) for event in kanban_db.list_events(conn, task_id)],
        }


def boards_payload() -> list[dict[str, Any]]:
    try:
        boards = kanban_db_boards.list_boards()
    except Exception:  # a missing/older boards store must not 500 the route
        logger.warning("[api-server] kanban boards unavailable", exc_info=True)
        boards = []
    try:
        current = kanban_db.get_current_board()
    except Exception:
        current = None
    return [
        {
            "slug": board.get("slug"),
            "name": board.get("name") or board.get("slug"),
            "current": board.get("slug") == current,
            "archived": bool(board.get("archived")),
        }
        for board in boards
    ]


def _limit(raw: Optional[str]) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("limit must be an integer") from exc
    return max(0, min(value, MAX_LIMIT))


def _include_archived(raw: Optional[str]) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes"}


async def _handle_list_tasks(request: web.Request) -> web.Response:
    """GET /api/tasks?board=&status=&assignee=&limit=&include_archived="""
    try:
        payload = await asyncio.to_thread(
            list_tasks_payload,
            board=request.query.get("board"),
            status=request.query.get("status"),
            assignee=request.query.get("assignee"),
            limit=_limit(request.query.get("limit")),
            include_archived=_include_archived(request.query.get("include_archived")),
        )
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception:
        logger.exception("[api-server] kanban task list failed")
        return web.json_response({"error": "failed to list tasks"}, status=500)
    return web.json_response({"tasks": payload})


async def _handle_get_task(request: web.Request) -> web.Response:
    """GET /api/tasks/{task_id}?board="""
    task_id = request.match_info.get("task_id", "")
    try:
        payload = await asyncio.to_thread(get_task_payload, task_id, board=request.query.get("board"))
    except Exception:
        logger.exception("[api-server] kanban task read failed")
        return web.json_response({"error": "failed to read task"}, status=500)
    if payload is None:
        return web.json_response({"error": "task not found"}, status=404)
    return web.json_response(payload)


async def _handle_boards(request: web.Request) -> web.Response:
    """GET /api/tasks/boards"""
    try:
        payload = await asyncio.to_thread(boards_payload)
    except Exception:
        logger.exception("[api-server] kanban boards read failed")
        return web.json_response({"error": "failed to list boards"}, status=500)
    return web.json_response({"boards": payload})


def http_routes() -> list[tuple[str, str, Any]]:
    """Read-only rows for ``APIServerAdapter._http_route_table``.

    ``/api/tasks/boards`` is registered before ``/api/tasks/{task_id}`` so the
    static path is not captured as a task id.
    """
    return [
        ("GET", "/api/tasks", _handle_list_tasks),
        ("GET", "/api/tasks/boards", _handle_boards),
        ("GET", "/api/tasks/{task_id}", _handle_get_task),
    ]
