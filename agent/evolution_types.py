from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ProposalStatus = Literal[
    "draft",
    "pending_approval",
    "approved",
    "rejected",
    "applied",
    "verified",
    "failed",
]
ProposalTargetKind = Literal["skill", "prompt", "doc", "code"]
ProposalRiskLevel = Literal["low", "medium", "high"]


def _new_id() -> str:
    return f"ep_{uuid.uuid4().hex}"


@dataclass(slots=True)
class EvolutionProposal:
    proposal_id: str
    source_run_id: str
    source_session_id: str
    created_at: float
    status: ProposalStatus
    target_kind: ProposalTargetKind
    target_ref: str
    problem_summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    change_summary: str = ""
    proposed_patch_summary: str = ""
    verification_plan: str = ""
    risk_level: ProposalRiskLevel = "low"
    requires_human_approval: bool = True

    @classmethod
    def create(cls, **kwargs: Any) -> "EvolutionProposal":
        return cls(
            proposal_id=kwargs.pop("proposal_id", _new_id()),
            created_at=kwargs.pop("created_at", time.time()),
            status=kwargs.pop("status", "draft"),
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
