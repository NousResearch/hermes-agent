"""PlanV1 and its bounded execution scope."""

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from .base import ArtifactModel, Identifier, NonEmptyText, Sha256, StrictContract


class Capability(str, Enum):
    RESEARCH_COLLECT = "research-collect"
    EVIDENCE_REVIEW = "evidence-review"
    BUILD = "build"
    PUBLISH = "publish"


class ExternalRight(str, Enum):
    NETWORK_READ = "network-read"
    DATABASE_READ = "database-read"
    EXTERNAL_PUBLISH = "external-publish"


class BudgetLimits(StrictContract):
    max_duration_seconds: int = Field(strict=True, gt=0)
    max_executions: int = Field(strict=True, gt=0)
    max_records: int = Field(strict=True, gt=0)
    max_bytes: int = Field(strict=True, gt=0)
    max_external_calls: int = Field(default=0, strict=True, ge=0)


class InputReference(StrictContract):
    input_id: Identifier
    artifact_type: Identifier
    sha256: Sha256


class OutputRequirement(StrictContract):
    artifact_id: Identifier
    artifact_type: Identifier
    required: bool = True


class CapabilityGrant(StrictContract):
    capability: Capability
    input_hashes: tuple[Sha256, ...] = Field(min_length=1)
    external_rights: tuple[ExternalRight, ...] = ()


class PlanV1(ArtifactModel):
    schema_version: Literal["plan.v1"]
    objective: NonEmptyText
    constraints: tuple[NonEmptyText, ...]
    inputs: tuple[InputReference, ...]
    outputs: tuple[OutputRequirement, ...] = Field(min_length=1)
    budgets: BudgetLimits
    capabilities: tuple[CapabilityGrant, ...] = Field(
        min_length=1,
        json_schema_extra={
            "uniqueItems": True,
            "x-hermes-validation": (
                "Pydantic requires capability values to be unique across grants."
            ),
        },
    )

    @model_validator(mode="after")
    def reject_duplicate_capabilities(self) -> "PlanV1":
        values = [grant.capability for grant in self.capabilities]
        if len(values) != len(set(values)):
            raise ValueError("capabilities must be unique")
        return self
