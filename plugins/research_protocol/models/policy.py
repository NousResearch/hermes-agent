"""Strict schemas for versioned policy documents."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from .base import StrictContract

UnitValue = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]


class QualityDimensions(StrictContract):
    authority: UnitValue
    methodology: UnitValue
    recency: UnitValue
    traceability: UnitValue


class QualityThresholds(StrictContract):
    accept: UnitValue
    review: UnitValue


class SourceQualityPolicy(StrictContract):
    schema_version: Literal["source_quality_policy.v1"]
    dimensions: QualityDimensions
    thresholds: QualityThresholds

    @model_validator(mode="after")
    def validate_weights_and_thresholds(self) -> "SourceQualityPolicy":
        total = sum(self.dimensions.model_dump().values())
        if abs(total - 1.0) > 1e-12:
            raise ValueError("quality dimension weights must total 1")
        if self.thresholds.review >= self.thresholds.accept:
            raise ValueError("review threshold must be below accept threshold")
        return self


class ContradictionPolicy(StrictContract):
    schema_version: Literal["contradiction_policy.v1"]
    minimum_confidence: UnitValue
    temporal_window_days: int = Field(strict=True, gt=0)
    require_unit_match: bool


class DedupPolicy(StrictContract):
    schema_version: Literal["dedup_policy.v1"]
    exact_hash_enabled: bool
    semantic_enabled: bool
    semantic_threshold: UnitValue


POLICY_MODELS: dict[str, type[BaseModel]] = {
    "source_quality.v1.yaml": SourceQualityPolicy,
    "contradiction.v1.yaml": ContradictionPolicy,
    "dedup.v1.yaml": DedupPolicy,
}
