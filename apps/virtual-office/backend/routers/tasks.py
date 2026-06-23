from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from backend.models.task import Task
from backend.services.handoff_manager import HandoffManager
from backend.services.json_store import TASKS_PATH, find_by_id, read_list_store, write_list_store

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def _model_dump(model: Task) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@router.get("")
async def get_tasks() -> list[dict[str, Any]]:
    return read_list_store(TASKS_PATH)


@router.post("")
async def create_task(payload: dict[str, Any]) -> dict[str, Any]:
    title = str(payload.get("title") or "").strip()
    goal = str(payload.get("goal") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    if not goal:
        raise HTTPException(status_code=400, detail="goal is required")

    agent = str(payload.get("agent") or "codex")
    if agent not in {"hermes", "codex", "chez", "system"}:
        agent = "codex"

    priority = str(payload.get("priority") or "medium")
    if priority not in {"low", "medium", "high", "urgent"}:
        priority = "medium"

    now = datetime.now(UTC).isoformat()
    task = Task(
        id=str(uuid4()),
        title=title,
        goal=goal,
        context=str(payload.get("context") or ""),
        agent=agent,
        room=str(payload.get("room") or "main-office"),
        priority=priority,
        status="pending",
        tags=[str(tag) for tag in payload.get("tags") or []],
        created_at=now,
        updated_at=now,
    )

    tasks = read_list_store(TASKS_PATH)
    tasks.append(_model_dump(task))
    write_list_store(TASKS_PATH, tasks)
    return tasks[-1]


@router.get("/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    task = find_by_id(read_list_store(TASKS_PATH), task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.patch("/{task_id}")
async def update_task(task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    tasks = read_list_store(TASKS_PATH)
    task = find_by_id(tasks, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    allowed_fields = {
        "title",
        "status",
        "agent",
        "room",
        "priority",
        "goal",
        "context",
        "result",
        "error",
        "handoff_id",
        "tags",
    }
    for key, value in payload.items():
        if key in allowed_fields:
            if key == "agent" and value not in {"hermes", "codex", "chez", "system"}:
                continue
            if key == "priority" and value not in {"low", "medium", "high", "urgent"}:
                continue
            if key == "status" and value not in {"pending", "in_progress", "completed", "failed", "cancelled"}:
                continue
            task[key] = value

    task["updated_at"] = datetime.now(UTC).isoformat()
    write_list_store(TASKS_PATH, tasks)
    return task


@router.delete("/{task_id}")
async def delete_task(task_id: str) -> dict[str, Any]:
    tasks = read_list_store(TASKS_PATH)
    remaining = [task for task in tasks if task.get("id") != task_id]
    if len(remaining) == len(tasks):
        raise HTTPException(status_code=404, detail="Task not found")

    write_list_store(TASKS_PATH, remaining)
    return {"deleted": True, "id": task_id}


@router.post("/{task_id}/run")
async def run_task(task_id: str) -> dict[str, Any]:
    return HandoffManager().run_task(task_id)


@router.post("/{task_id}/retry")
async def retry_task(task_id: str) -> dict[str, Any]:
    return HandoffManager().retry_task(task_id)


@router.post("/{task_id}/requeue")
async def requeue_task(task_id: str) -> dict[str, Any]:
    return HandoffManager().requeue_task(task_id)
