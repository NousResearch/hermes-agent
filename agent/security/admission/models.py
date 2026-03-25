from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CandidateKind(StrEnum):
    MCP_SERVER = "mcp_server"
    SKILL = "skill"
    PLUGIN = "plugin"


class AdmissionStatus(StrEnum):
    QUARANTINED = "quarantined"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"


class PromotionDecision(StrEnum):
    PROMOTE = "promote"
    HOLD = "hold"
    REJECT = "reject"


@dataclass(slots=True)
class CandidateSource:
    uri: str
    display_name: str
    version: str | None = None
    installer: str | None = None


@dataclass(slots=True)
class IntegrityState:
    algorithm: str
    digest: str
    verified_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class InspectionReport:
    summary: str
    decision: PromotionDecision = PromotionDecision.HOLD
    reasons: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class AdmissionRecord:
    record_id: str
    kind: CandidateKind
    source: CandidateSource
    lineage_id: str | None = None
    parent_record_id: str | None = None
    revision: int = 1
    source_fingerprint: str | None = None
    status: AdmissionStatus = AdmissionStatus.QUARANTINED
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    quarantine_path: str | None = None
    approved_path: str | None = None
    integrity: IntegrityState | None = None
    approved_at: datetime | None = None
    report_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def transition_to(self, new_status: AdmissionStatus) -> None:
        allowed = {
            AdmissionStatus.QUARANTINED: {
                AdmissionStatus.APPROVED,
                AdmissionStatus.REJECTED,
            },
            AdmissionStatus.APPROVED: {AdmissionStatus.REVOKED},
            AdmissionStatus.REJECTED: set(),
            AdmissionStatus.REVOKED: set(),
        }
        if new_status == self.status:
            return
        if new_status not in allowed[self.status]:
            raise ValueError(f"invalid admission transition: {self.status} -> {new_status}")
        self.status = new_status
        self.updated_at = utc_now()
