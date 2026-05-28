"""Kynver AgentOS HTTP routes used by the Hermes adapter.

The installed Kynver MCP AgentOS server proxies its tools to resource-style
HTTP routes under ``/api/agent-os/{slug}``. ``KynverAgentOSClient`` adds that
prefix, so constants and helpers here are suffixes below the slug.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from typing import Any, Mapping

MEMORY_PATH = "/memory"
MEMORY_SEARCH_PATH = MEMORY_PATH
MEMORY_WRITE_PATH = MEMORY_PATH

TASKS_PATH = "/tasks"
TASK_CREATE_PATH = TASKS_PATH
TASK_LIST_PATH = TASKS_PATH

SKILLS_PATH = "/skills"
SKILL_LIST_PATH = f"{SKILLS_PATH}?view=manifest"

SESSIONS_PATH = "/sessions"
SESSION_OPEN_PATH = SESSIONS_PATH

TASK_STATUSES = frozenset(
    {
        "ready",
        "running",
        "waiting",
        "scheduled",
        "blocked",
        "needs_input",
        "awaiting_review",
        "done",
        "failed",
        "cancelled",
    }
)
TODO_STATUS_TO_TASK_STATUS = {
    "pending": "ready",
    "in_progress": "running",
    "completed": "done",
    "cancelled": "cancelled",
}
TASK_TERMINAL_STATUSES = frozenset({"done", "failed", "cancelled"})


def _quote(value: str) -> str:
    return urllib.parse.quote(str(value or "").strip(), safe="")


def _query(path: str, params: Mapping[str, Any]) -> str:
    clean = {
        key: value
        for key, value in params.items()
        if value is not None and value != "" and value != []
    }
    if not clean:
        return path
    return f"{path}?{urllib.parse.urlencode(clean, doseq=True)}"


def memory_search_path(*, q: str, k: int | None = None, **params: Any) -> str:
    return _query(MEMORY_PATH, {"q": q, "k": k, **params})


def task_path(task_id: str) -> str:
    return f"{TASKS_PATH}/{_quote(task_id)}"


def task_update_path(task_id: str) -> str:
    return task_path(task_id)


def task_events_path(task_id: str) -> str:
    return f"{task_path(task_id)}/events"


def task_close_path(task_id: str) -> str:
    return f"{task_path(task_id)}/close"


def task_steer_path(task_id: str) -> str:
    return f"{task_path(task_id)}/steer"


def task_list_path(**params: Any) -> str:
    return _query(TASKS_PATH, params)


def session_path(session_id: str) -> str:
    return f"{SESSIONS_PATH}/{_quote(session_id)}"


def session_events_path(session_id: str) -> str:
    return f"{session_path(session_id)}/events"


def skill_get_path(skill_slug: str, *, source: str = "") -> str:
    return _query(f"{SKILLS_PATH}/{_quote(skill_slug)}", {"source": source})


def normalize_task_status(value: Any, *, default: str = "ready") -> str:
    """Return an AgentOS task lifecycle status."""
    status = str(value or "").strip().lower()
    if status == "completed":
        status = "done"
    return status if status in TASK_STATUSES else default


def normalize_terminal_task_status(value: Any, *, default: str = "done") -> str:
    """Return an AgentOS terminal task lifecycle status."""
    status = normalize_task_status(value, default=default)
    return status if status in TASK_TERMINAL_STATUSES else default


def make_idempotency_key(source_id: str, *parts: Any) -> str:
    """Build a deterministic idempotency key for runtime-originated writes."""
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{source_id}:{digest}"


def todo_to_task_record(todo: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    """Translate Hermes todo state into the AgentOS task lifecycle shape."""
    todo_id = str(todo.get("id") or "").strip()
    content = str(todo.get("content") or "").strip()
    status = TODO_STATUS_TO_TASK_STATUS.get(
        str(todo.get("status") or "").strip().lower(),
        "ready",
    )
    return {
        "idempotencyKey": make_idempotency_key(source_id, "todo", todo_id or content),
        "title": content or todo_id or "Hermes todo",
        "description": content,
        "status": status,
        "metadata": {
            "hermesTodoId": todo_id,
            "hermesTodoStatus": str(todo.get("status") or ""),
        },
    }
