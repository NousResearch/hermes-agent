"""Conflict, deduplication, and verdict records."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import ArtifactModel, Identifier, NonEmptyText

UnitScore = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]


class ConflictStatus(str, Enum):
    OPEN = "open"
    ADJUDICATED = "adjudicated"
    INSUFFICIENT = "insufficient"


class ConflictRecord(ArtifactModel):
    schema_version: Literal["conflict_record.v1"]
    conflict_id: Identifier
    claim_id: Identifier
    left_provenance_id: Identifier
    right_provenance_id: Identifier
    conflict_type: Identifier
    status: ConflictStatus
    rationale: NonEmptyText

    @model_validator(mode="after")
    def require_distinct_provenance(self) -> "ConflictRecord":
        if self.left_provenance_id == self.right_provenance_id:
            raise ValueError("conflict provenance references must be distinct")
        return self


class DedupCluster(ArtifactModel):
    schema_version: Literal["dedup_cluster.v1"]
    cluster_id: Identifier
    canonical_provenance_id: Identifier
    member_provenance_ids: tuple[Identifier, ...] = Field(
        min_length=2,
        json_schema_extra={"uniqueItems": True},
    )
    method: Literal["exact", "semantic"]
    confidence: UnitScore

    @model_validator(mode="after")
    def validate_members(self) -> "DedupCluster":
        if len(self.member_provenance_ids) != len(set(self.member_provenance_ids)):
            raise ValueError("member_provenance_ids must be unique")
        if self.canonical_provenance_id not in self.member_provenance_ids:
            raise ValueError("canonical provenance must be a cluster member")
        return self


class Verdict(str, Enum):
    SUPPORTED = "supported"
    REJECTED = "rejected"
    MIXED = "mixed"
    INSUFFICIENT = "insufficient"
    BLOCKED = "blocked"


class VerdictRecord(ArtifactModel):
    schema_version: Literal["verdict_record.v1"]
    verdict_id: Identifier
    question_id: Identifier
    verdict: Verdict
    evidence_provenance_ids: tuple[Identifier, ...] = Field(min_length=1)
    rationale: NonEmptyText
    confidence: UnitScore
