"""Topic-aware routing helpers for threaded messaging platforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gateway.config import Platform
from gateway.session import SessionSource


@dataclass(frozen=True)
class TopicRoute:
    """Normalized routing context for outbound threaded replies."""

    chat_id: str
    thread_id: Optional[str] = None
    topic_name: Optional[str] = None
    boundary: str = "soft"

    @property
    def is_strict(self) -> bool:
        return bool(self.thread_id and self.boundary == "strict")

    def to_metadata(self) -> Optional[dict[str, str]]:
        metadata: dict[str, str] = {}
        if self.thread_id:
            metadata["thread_id"] = self.thread_id
        if self.topic_name:
            metadata["topic_name"] = self.topic_name
        if self.boundary:
            metadata["topic_boundary"] = self.boundary
        return metadata or None


def route_from_session_source(source: Optional[SessionSource]) -> Optional[TopicRoute]:
    """Derive a routing policy from inbound session context.

    Telegram forum topics are strict routing boundaries. Other sources default to
    soft routing so adapters may apply legacy fallback behavior if appropriate.
    """
    if not source or not source.chat_id:
        return None

    boundary = "soft"
    if source.platform == Platform.TELEGRAM and source.thread_id:
        boundary = "strict"

    return TopicRoute(
        chat_id=source.chat_id,
        thread_id=source.thread_id,
        topic_name=source.chat_topic,
        boundary=boundary,
    )
