"""Structured lane request/result contracts for parallel agent work."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LaneStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"
    DEGRADED = "degraded"


@dataclass
class LaneRequest:
    lane_id: str
    agent: str
    prompt: str
    effort: str | None = None
    timeout_s: float = 60.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LaneResult:
    lane_id: str
    agent: str
    status: LaneStatus
    output: str | None
    error: str | None
    duration_s: float
    exit_code: int | None
    log_path: str | None
    artifacts: dict[str, str] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.status in {
            LaneStatus.SUCCEEDED,
            LaneStatus.FAILED,
            LaneStatus.TIMED_OUT,
            LaneStatus.SKIPPED,
            LaneStatus.DEGRADED,
        }

    @classmethod
    def failed(
        cls,
        req: LaneRequest,
        error: str,
        *,
        duration_s: float = 0.0,
        exit_code: int | None = None,
        log_path: str | None = None,
    ) -> "LaneResult":
        return cls(req.lane_id, req.agent, LaneStatus.FAILED, None, error, duration_s, exit_code, log_path)

    @classmethod
    def timed_out(
        cls,
        req: LaneRequest,
        *,
        duration_s: float | None = None,
        log_path: str | None = None,
    ) -> "LaneResult":
        return cls(
            req.lane_id,
            req.agent,
            LaneStatus.TIMED_OUT,
            None,
            f"timed out after {req.timeout_s}s",
            req.timeout_s if duration_s is None else duration_s,
            None,
            log_path,
        )

    @classmethod
    def skipped(cls, req: LaneRequest, reason: str, *, log_path: str | None = None) -> "LaneResult":
        return cls(req.lane_id, req.agent, LaneStatus.SKIPPED, None, reason, 0.0, None, log_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "agent": self.agent,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration_s": self.duration_s,
            "exit_code": self.exit_code,
            "log_path": self.log_path,
            "artifacts": dict(self.artifacts),
        }
