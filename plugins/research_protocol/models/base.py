"""Shared strict primitives for versioned research artifacts."""

from collections.abc import Mapping
import json
from types import MappingProxyType
from typing import Annotated, TypeAlias, TypeVar

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    PlainSerializer,
    StringConstraints,
)

Identifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
NonEmptyText = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        pattern=r"^\S(?:[\s\S]*\S)?$",
    ),
]
Sha256 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]

MappingKey = TypeVar("MappingKey")
MappingValue = TypeVar("MappingValue")


def _freeze_mapping(
    value: Mapping[MappingKey, MappingValue],
) -> Mapping[MappingKey, MappingValue]:
    return MappingProxyType(dict(value))


def _serialize_mapping(value: Mapping[object, object]) -> dict[object, object]:
    return dict(value)


FrozenMapping: TypeAlias = Annotated[
    Mapping[MappingKey, MappingValue],
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_mapping, return_type=dict),
]


class StrictContract(BaseModel):
    """Reject unknown fields, coercion, and non-finite numeric values."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
        frozen=True,
    )


class ArtifactModel(StrictContract):
    """Metadata and canonical serialization shared by persisted v1 artifacts."""

    producer_version: NonEmptyText
    run_id: Identifier
    created_at: AwareDatetime

    def to_canonical_json_bytes(self) -> bytes:
        """Revalidate current state and return deterministic UTF-8 JSON bytes."""

        validated = type(self).model_validate_json(self.model_dump_json(), strict=True)
        return json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


ArtifactType = TypeVar("ArtifactType", bound=ArtifactModel)


def load_artifact_from_json_bytes(
    model: type[ArtifactType], payload: bytes
) -> ArtifactType:
    """Validate persisted artifact bytes through Pydantic's strict JSON mode."""

    if not isinstance(payload, bytes):
        raise TypeError("artifact payload must be bytes")
    if not issubclass(model, ArtifactModel):
        raise TypeError("model must inherit from ArtifactModel")
    return model.model_validate_json(payload, strict=True)
