"""Evidence and deterministic source-quality contracts."""

from typing import Annotated, Literal

from pydantic import Field

from .base import ArtifactModel, FrozenMapping, Identifier, NonEmptyText, Sha256

UnitScore = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]


class EvidenceSnapshot(ArtifactModel):
    schema_version: Literal["evidence_snapshot.v1"]
    evidence_id: Identifier
    source_id: Identifier
    locator: NonEmptyText
    content_sha256: Sha256
    extract: NonEmptyText
    provenance_id: Identifier


class SourceQuality(ArtifactModel):
    schema_version: Literal["source_quality.v1"]
    source_id: Identifier
    provenance_id: Identifier
    score: UnitScore
    dimensions: FrozenMapping[Identifier, UnitScore]
    rationale: tuple[NonEmptyText, ...]
