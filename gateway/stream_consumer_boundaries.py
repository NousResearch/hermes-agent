"""Correlated stream preview lifecycle, independent of transport details."""

import logging
import uuid
from typing import Optional

from gateway.progress_events import (
    ContentBoundaryEvent, DurableContentBoundary, DurableContentSource,
    ProvisionalContentBoundary, RetractedContentBoundary,
)

logger = logging.getLogger("gateway.stream_consumer")


class StreamBoundaryMixin:
    def _publish_content_boundary(self, event: ContentBoundaryEvent) -> None:
        """Publish each lifecycle phase for a boundary at most once."""
        cb = self._on_content_boundary
        if cb is None:
            return
        event_key = (event.boundary_id, type(event).__name__)
        if event_key in self._published_content_boundaries:
            return
        self._published_content_boundaries.add(event_key)
        try:
            cb(event)
        except Exception:
            logger.debug("content boundary callback error", exc_info=True)

    def _notify_content_boundary(
        self,
        source: DurableContentSource,
        *,
        message_id: Optional[str] = None,
    ) -> None:
        """Publish one idempotent boundary for confirmed persistent content."""
        normalized_message_id = str(message_id) if message_id is not None else None
        if normalized_message_id is not None:
            boundary_id = f"{self._turn_id}:message:{normalized_message_id}"
        else:
            self._content_boundary_sequence += 1
            boundary_id = f"{self._turn_id}:boundary:{self._content_boundary_sequence}"
        self._publish_content_boundary(
            DurableContentBoundary(
                boundary_id=boundary_id,
                source=source,
                message_id=normalized_message_id,
            )
        )

    def _open_preview_boundary(self) -> None:
        """Pause later progress before a preview delivery can overtake it."""
        if self._pending_preview_boundary is not None:
            return
        self._content_boundary_sequence += 1
        event = ProvisionalContentBoundary(
            boundary_id=(
                f"{self._turn_id}:preview:{self._content_boundary_sequence}"
            )
        )
        self._pending_preview_boundary = event
        self._publish_content_boundary(event)

    def _record_pending_preview_message_id(self, message_id: str) -> None:
        """Attach the platform id after a provisional send succeeds."""
        pending = self._pending_preview_boundary
        if pending is None:
            return
        self._pending_preview_boundary = ProvisionalContentBoundary(
            boundary_id=pending.boundary_id,
            message_id=str(message_id),
        )

    def _confirm_pending_preview_boundary(
        self,
        *,
        source: DurableContentSource = DurableContentSource.STREAM_FINALIZED,
        message_id: Optional[str] = None,
    ) -> None:
        """Resolve a provisional preview as a persistent timeline entry."""
        pending = self._pending_preview_boundary
        if pending is None:
            return
        self._pending_preview_boundary = None
        self._publish_content_boundary(
            DurableContentBoundary(
                boundary_id=pending.boundary_id,
                source=source,
                message_id=(str(message_id) if message_id is not None else pending.message_id),
            )
        )

    def _confirm_or_notify_content_boundary(
        self,
        source: DurableContentSource,
        *,
        message_id: Optional[str] = None,
    ) -> None:
        """Resolve the active preview, otherwise publish a direct boundary."""
        if self._pending_preview_boundary is not None:
            self._confirm_pending_preview_boundary(
                source=source,
                message_id=message_id,
            )
        else:
            self._notify_content_boundary(source, message_id=message_id)

    def _retract_pending_preview_boundary(self) -> None:
        """Resolve a provisional preview that left no persistent chat entry."""
        pending = self._pending_preview_boundary
        if pending is None:
            return
        self._pending_preview_boundary = None
        self._publish_content_boundary(
            RetractedContentBoundary(boundary_id=pending.boundary_id)
        )


    def _ack_persistent_stream_boundary(self):
        # Persistent native/draft frames are visible now, not only at turn end.
        # Subsequent frames update the same entry and must not split progress.
        if self._persistent_stream_boundary_published:
            self._retract_pending_preview_boundary()
        else:
            self._confirm_or_notify_content_boundary(DurableContentSource.STREAM_PERSISTED)
            self._persistent_stream_boundary_published = True

    def _settle_preview_boundary(self):
        # Draft text alone is not a persistent timeline entry.
        if self._message_id is not None or self._already_sent:
            self._confirm_pending_preview_boundary()
        else:
            self._retract_pending_preview_boundary()

    def _enqueue_content(self, item):
        # Producer and consumer have different boundary slots: the worker can
        # queue several segments while the platform is still sending the first.
        with self._content_boundary_lock:
            if self._producer_boundary is None:
                boundary = ProvisionalContentBoundary(boundary_id=str(uuid.uuid4()))
                self._producer_boundary = boundary
                self._publish_content_boundary(boundary)
                self._queue.put(boundary)
            self._queue.put(item)
