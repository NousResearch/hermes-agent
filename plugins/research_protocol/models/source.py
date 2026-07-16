"""Source census contracts."""

from typing import Literal

from pydantic import AwareDatetime

from .base import ArtifactModel, Identifier, NonEmptyText, StrictContract


class SourceIdentifier(StrictContract):
    kind: Identifier
    value: NonEmptyText


class SourceCandidate(ArtifactModel):
    schema_version: Literal["source_candidate.v1"]
    source_id: Identifier
    adapter_id: Identifier
    locator: NonEmptyText
    title: NonEmptyText
    identifiers: tuple[SourceIdentifier, ...]
    authors: tuple[NonEmptyText, ...]
    published_at: AwareDatetime | None = None
    license: NonEmptyText | None = None
    provenance_id: Identifier
