"""Typed progress-timeline events shared by stream and progress delivery."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DurableContentSource(str, Enum):
    """Delivery paths that create a new persistent chat timeline entry."""

    STREAM_FINALIZED = "stream_finalized"
    STREAM_PERSISTED = "stream_persisted"
    COMMENTARY = "commentary"
    OVERFLOW = "overflow"
    FALLBACK = "fallback"
    FRESH_FINAL = "fresh_final"


@dataclass(frozen=True, slots=True)
class DurableContentBoundary:
    """A confirmed persistent content entry that seals prior progress."""

    boundary_id: str
    source: DurableContentSource
    message_id: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ProvisionalContentBoundary:
    """A preview delivery whose durable/retracted outcome is not known yet."""

    boundary_id: str
    message_id: Optional[str] = None


@dataclass(frozen=True, slots=True)
class RetractedContentBoundary:
    """Resolution indicating that a provisional preview left no chat entry."""

    boundary_id: str


ContentBoundaryEvent = (
    DurableContentBoundary | ProvisionalContentBoundary | RetractedContentBoundary
)
