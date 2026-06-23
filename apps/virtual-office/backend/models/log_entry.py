from typing import Optional

from pydantic import BaseModel


class LogMetadata(BaseModel):
    agent: str
    task_id: Optional[str] = None
    handoff_id: Optional[str] = None


class LogEntry(BaseModel):
    id: str
    level: str
    message: str
    timestamp: str
    metadata: LogMetadata
