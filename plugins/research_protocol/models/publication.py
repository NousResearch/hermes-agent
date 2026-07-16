"""Data-bound publication contracts."""

from enum import Enum
from typing import Literal

from pydantic import Field

from .base import ArtifactModel, FrozenMapping, Identifier, NonEmptyText, Sha256


class FigureKind(str, Enum):
    TABLE = "table"
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"


class FigureSpec(ArtifactModel):
    schema_version: Literal["figure_spec.v1"]
    figure_id: Identifier
    title: NonEmptyText
    kind: FigureKind
    data_artifact_id: Identifier
    data_sha256: Sha256
    x_field: Identifier
    y_fields: tuple[Identifier, ...] = Field(min_length=1)
    units: FrozenMapping[Identifier, NonEmptyText]
    labels: FrozenMapping[Identifier, NonEmptyText]
    palette: tuple[str, ...] = Field(min_length=1)
