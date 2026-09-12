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


class ContentBoundaryBuffer:
    """Hold progress behind unresolved previews, preserving producer order."""

    def __init__(self):
        from collections import deque
        self.pending = deque()
        self.resolved = {}
        self.seen = set()

    def feed(self, event):
        if isinstance(event, (ProvisionalContentBoundary, DurableContentBoundary, RetractedContentBoundary)):
            key = (event.boundary_id, type(event))
            if key in self.seen:
                return []
            self.seen.add(key)
            if isinstance(event, ProvisionalContentBoundary):
                self.pending.append(event)
            elif event.boundary_id not in self.resolved:
                self.resolved[event.boundary_id] = event
                if (event.boundary_id, ProvisionalContentBoundary) not in self.seen:
                    self.pending.append(event)
        else:
            self.pending.append(event)
        ready = []
        while self.pending:
            raw = self.pending[0]
            if isinstance(raw, ProvisionalContentBoundary):
                if raw.boundary_id not in self.resolved:
                    break
                raw = self.resolved[raw.boundary_id]
            self.pending.popleft()
            if not isinstance(raw, RetractedContentBoundary):
                ready.append(raw)
        return ready

    def finish(self):
        # Missing acknowledgements may still have left content on screen. Never
        # replay deferred tools above it; release them below a conservative seal.
        for raw in self.pending:
            if isinstance(raw, ProvisionalContentBoundary):
                self.resolved.setdefault(raw.boundary_id, DurableContentBoundary(
                    raw.boundary_id, DurableContentSource.STREAM_PERSISTED, raw.message_id))
        return self.feed(RetractedContentBoundary("cleanup"))
