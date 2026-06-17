"""
Session management for pharma compliance tracking.

Each session tracks a user's pending messages for multimodal merge,
supporting text, voice, and photo message types with 5-minute auto-merge.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Merge window: messages within this many seconds are merged into one task
MERGE_WINDOW_SECONDS = 300  # 5 minutes
# Phrase that triggers manual merge completion
MANUAL_MERGE_PHRASES = ("完了", "就这样", "结束", "好了", "完成", "提交", "合并", "强制合并")


class MessageType(Enum):
    TEXT = "text"
    VOICE = "voice"
    PHOTO = "photo"


@dataclass
class PendingMessage:
    """A message waiting to be merged into a task record."""

    msg_type: MessageType
    content: str  # text content (original text, STT transcript, or OCR result)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisitSession:
    """A single visit session for a representative."""

    user_id: str
    pending_messages: List[PendingMessage] = field(default_factory=list)
    message_timeout: float = 0.0
    task_type: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def add_message(
        self,
        msg_type: MessageType,
        content: str,
        raw_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        msg = PendingMessage(
            msg_type=msg_type,
            content=content,
            raw_data=raw_data or {},
            metadata=metadata or {},
        )
        self.message_timeout = msg.timestamp + MERGE_WINDOW_SECONDS
        self.pending_messages.append(msg)
        logger.debug(
            "Session %s: added %s message, pending count=%d",
            self.user_id, msg_type.value, len(self.pending_messages),
        )

    def is_timed_out(self) -> bool:
        if not self.pending_messages:
            return False
        return time.time() > self.message_timeout

    def should_merge(self) -> bool:
        """Return True if messages should be merged (timeout or manual trigger)."""
        return self.is_timed_out()

    def check_manual_merge(self, text: str) -> bool:
        """Check if the text contains a manual merge trigger phrase."""
        for phrase in MANUAL_MERGE_PHRASES:
            if phrase in text:
                return True
        return False

    def merge_contents(self) -> str:
        """Merge all pending message contents into a single text string."""
        parts: List[str] = []
        for msg in self.pending_messages:
            if msg.msg_type == MessageType.PHOTO:
                parts.append(f"[门头照] {msg.content}")
            elif msg.msg_type == MessageType.VOICE:
                parts.append(f"[语音] {msg.content}")
            else:
                parts.append(msg.content)
        return " ".join(parts)

    def clear(self) -> None:
        self.pending_messages.clear()
        self.message_timeout = 0.0
        self.task_type = None


class SessionManager:
    """Manages all visit sessions, keyed by user_id."""

    def __init__(self):
        self._sessions: Dict[str, VisitSession] = {}
        self._lock = threading.Lock()

    def get_or_create_session(self, user_id: str) -> VisitSession:
        with self._lock:
            if user_id not in self._sessions:
                self._sessions[user_id] = VisitSession(user_id=user_id)
                logger.info("Created new session for user %s", user_id)
            return self._sessions[user_id]

    def get_session(self, user_id: str) -> Optional[VisitSession]:
        return self._sessions.get(user_id)

    def remove_session(self, user_id: str) -> None:
        with self._lock:
            self._sessions.pop(user_id, None)
            logger.info("Removed session for user %s", user_id)

    def check_timeouts(self) -> List[VisitSession]:
        """Return all sessions that have timed out and need auto-merge."""
        timed_out: List[VisitSession] = []
        with self._lock:
            for session in list(self._sessions.values()):
                if session.is_timed_out() and session.pending_messages:
                    timed_out.append(session)
        return timed_out

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "sessions_with_pending": sum(
                    1 for s in self._sessions.values() if s.pending_messages
                ),
            }
