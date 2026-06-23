from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


TaskStatus = Literal["pending", "in_progress", "completed", "failed", "cancelled"]
TaskAgent = Literal["hermes", "codex", "chez", "system"]
TaskPriority = Literal["low", "medium", "high", "urgent"]


class Task(BaseModel):
    id: str
    title: str
    status: TaskStatus = "pending"
    agent: TaskAgent = "codex"
    room: str = "main-office"
    priority: TaskPriority = "medium"
    goal: str = ""
    context: str = ""
    result: str | None = None
    error: str | None = None
    handoff_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    tags: list[str] = Field(default_factory=list)
