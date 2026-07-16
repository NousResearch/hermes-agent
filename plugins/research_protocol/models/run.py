"""Run configuration and immutable manifest records."""

from enum import Enum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator

from .base import (
    ArtifactModel,
    FrozenMapping,
    Identifier,
    Sha256,
)
from .plan import BudgetLimits, Capability

SAFE_RELATIVE_POSIX_PATTERN = (
    r"^(?!\s)(?!.*\s$)(?!/)(?!.*:)(?!.*\\)(?!.*\x00)(?!.*//)"
    r"(?!.*(?:^|/)\.{1,2}(?:/|$))(?!.*\/$).+$"
)
PATH_BOUNDARY_COMMENT = (
    "Pydantic validation is required to enforce exact, normalized POSIX path "
    "semantics without coercion in addition to this JSON Schema pattern."
)


class ArtifactStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    QUARANTINED = "quarantined"


class RunConfig(ArtifactModel):
    schema_version: Literal["run_config.v1"]
    plan_artifact_id: Identifier
    plan_sha256: Sha256
    capability: Capability
    budgets: BudgetLimits
    input_hashes: tuple[Sha256, ...] = Field(min_length=1)
    policy_hashes: FrozenMapping[Identifier, Sha256]
    seed: int = Field(strict=True, ge=0)


class ManifestRecord(ArtifactModel):
    schema_version: Literal["manifest_record.v1"]
    artifact_id: Identifier
    artifact_type: Identifier
    path_relative: str = Field(
        strict=True,
        min_length=1,
        max_length=8192,
        json_schema_extra={
            "pattern": SAFE_RELATIVE_POSIX_PATTERN,
            "$comment": PATH_BOUNDARY_COMMENT,
        },
    )
    sha256: Sha256
    byte_length: int = Field(strict=True, ge=0)
    status: ArtifactStatus
    provenance_ids: tuple[Identifier, ...]

    @field_validator("path_relative")
    @classmethod
    def require_safe_relative_posix_path(cls, value: str) -> str:
        parts = value.split("/")
        if (
            "\x00" in value
            or "\\" in value
            or ":" in value
            or value != value.strip()
            or value.startswith("/")
            or value.endswith("/")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("path_relative must be a safe normalized POSIX path")

        path = PurePosixPath(value)
        if path.is_absolute() or str(path) != value:
            raise ValueError("path_relative must be normalized")
        return value
