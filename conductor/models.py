from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StepKind(str, Enum):
    IMPLEMENTATION = "implementation"
    JUDGMENT_REVIEW = "judgment_review"
    DETERMINISTIC_GATE = "deterministic_gate"
    OBSERVATION = "observation"
    HUMAN_DECISION = "human_decision"


class WorkerState(str, Enum):
    RUNNING = "RUNNING"
    TERMINAL = "TERMINAL"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Step:
    step_id: str
    kind: StepKind
    prompt: str
    command: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Step":
        return cls(
            step_id=str(value["step_id"]),
            kind=StepKind(value["kind"]),
            prompt=str(value.get("prompt", "")),
            command=[str(part) for part in value.get("command", [])],
        )


@dataclass(frozen=True)
class CampaignPlan:
    campaign_id: str
    cwd: str
    mutable_manifest: list[str]
    steps: list[Step]
    writer: dict[str, Any]
    reviewer: dict[str, Any]
    budgets: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for step in value["steps"]:
            step["kind"] = step["kind"].value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CampaignPlan":
        return cls(
            campaign_id=str(value["campaign_id"]),
            cwd=str(value["cwd"]),
            mutable_manifest=[str(path) for path in value["mutable_manifest"]],
            steps=[Step.from_dict(step) for step in value["steps"]],
            writer=dict(value["writer"]),
            reviewer=dict(value["reviewer"]),
            budgets=dict(value.get("budgets", {})),
        )


@dataclass(frozen=True)
class CampaignRecord:
    campaign_id: str
    plan: CampaignPlan
    state: str
    step_index: int
    conductor_turns: int
    retries: int
    next_retry_at: float
    started_at: float
    blocker_key: str | None


@dataclass(frozen=True)
class WorkerRecord:
    worker_id: str
    campaign_id: str
    step_index: int
    role: str
    cwd: str
    tmux_session: str
    pid: int | None
    start_marker: str | None
    provider: str
    model: str
    prompt_hash: str
    mutable_manifest: list[str]
    launched_at: float
    heartbeat_at: float | None
    progress_evidence: str | None
    state: WorkerState
    output_path: str
    receipt_path: str
    receipt_hash: str | None
    nonce: str
    read_only: bool
