from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


HandoffStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class Handoff(BaseModel):
    id: str
    from_agent: str
    to_agent: str
    status: HandoffStatus = "pending"
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    log_refs: list[str] = Field(default_factory=list)
