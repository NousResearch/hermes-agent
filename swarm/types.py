from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MessageType(Enum):
    result = "result"
    request = "request"
    status = "status"
    error = "error"
    data = "data"


class TaskState(Enum):
    pending = "pending"
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class WorkerState(Enum):
    idle = "idle"
    busy = "busy"
    draining = "draining"
    dead = "dead"


@dataclass
class SwarmTask:
    name: str
    prompt: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    deps: list[str] = field(default_factory=list)
    state: TaskState = TaskState.pending
    result: Any = None
    worker_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retries: int = 0
    max_retries: int = 3
    role: str = "worker"
    model: str = ""
    artifacts: list[ArtifactRef] = field(default_factory=list)


@dataclass
class SwarmWorker:
    name: str
    backend: str = "local"  # local | modal | ssh | k8s
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: WorkerState = WorkerState.idle
    current_task_id: str | None = None
    capabilities: list[str] = field(default_factory=list)
    max_concurrent: int = 1
    address: str = ""


@dataclass
class SwarmConfig:
    max_workers: int = 2
    max_retries: int = 3
    default_model: str = ""
    artifact_dir: str = "./artifacts"
    backends: dict[str, Any] = field(default_factory=dict)
    budget_limit_usd: float = 0.0
    timeout_seconds: float = 600.0


@dataclass
class SwarmResult:
    task_id: str
    success: bool
    output: Any = None
    artifacts: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0


@dataclass
class ArtifactRef:
    task_id: str
    path: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    mime_type: str = "application/octet-stream"
    size_bytes: int = 0
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SwarmMessage:
    from_agent: str
    to_agent: str  # agent id or "broadcast"
    msg_type: MessageType
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=datetime.utcnow)
    in_reply_to: str | None = None
