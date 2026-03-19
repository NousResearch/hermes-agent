"""
cosmos Collaborative Sessions - Real-Time Collaboration

Novel Approaches:
1. Shared Context - Multiple users in same session
2. Real-Time Sync - Live updates across participants
3. Conflict Resolution - Handle concurrent modifications
4. Session Replay - Review session history
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable, AsyncIterator
import json
import hashlib

from loguru import logger


class SessionState(Enum):
    """Session lifecycle states."""
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class EventType(Enum):
    """Types of session events."""
    MESSAGE = "message"
    ACTION = "action"
    STATE_CHANGE = "state_change"
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    MEMORY_UPDATE = "memory_update"
    ERROR = "error"


@dataclass
class SessionEvent:
    """An event in a collaborative session."""
    id: str
    session_id: str
    event_type: EventType
    user_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Event data
    content: str = ""
    data: dict = field(default_factory=dict)

    # Metadata
    sequence_number: int = 0
    acknowledged_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "data": self.data,
            "sequence": self.sequence_number,
        }


@dataclass
class SessionParticipant:
    """A participant in a session."""
    user_id: str
    display_name: str
    role: str = "participant"  # "host", "participant", "observer"

    # Status
    is_active: bool = True
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    # Cursor/focus (for collaborative editing)
    cursor_position: Optional[dict] = None
    current_focus: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "role": self.role,
            "is_active": self.is_active,
        }


@dataclass
class CollaborativeSession:
    """A collaborative session between multiple users."""
    id: str
    name: str
    state: SessionState = SessionState.CREATED

    # Host
    host_id: str = ""

    # Participants
    participants: dict[str, SessionParticipant] = field(default_factory=dict)
    max_participants: int = 10

    # Events
    events: list[SessionEvent] = field(default_factory=list)
    event_sequence: int = 0

    # Shared context
    shared_context: dict = field(default_factory=dict)
    shared_memories: list[str] = field(default_factory=list)

    # Settings
    allow_join: bool = True
    require_approval: bool = False
    record_session: bool = True

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "host": self.host_id,
            "participants": len(self.participants),
            "events": len(self.events),
            "created_at": self.created_at.isoformat(),
        }


class SessionManager:
    """
    Collaborative session management.

    Features:
    - Real-time multi-user sessions
    - Event synchronization
    - Shared context and memory
    - Session recording and replay
    """

    def __init__(
        self,
        max_events_per_session: int = 1000,
    ):
        self.max_events = max_events_per_session

        self.sessions: dict[str, CollaborativeSession] = {}
        self.user_sessions: dict[str, list[str]] = {}  # user_id -> session_ids

        # Subscribers for real-time updates
        self._subscribers: dict[str, dict[str, Callable]] = {}  # session_id -> {user_id -> callback}

        # Pending approvals
        self._pending_joins: dict[str, list[str]] = {}  # session_id -> [user_ids]

        self._lock = asyncio.Lock()
        self._event_counter = 0

    async def create_session(
        self,
        name: str,
        host_id: str,
        host_display_name: str,
        require_approval: bool = False,
    ) -> CollaborativeSession:
        """Create a new collaborative session."""
        session_id = hashlib.sha256(
            f"{name}{host_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        session = CollaborativeSession(
            id=session_id,
            name=name,
            host_id=host_id,
            require_approval=require_approval,
        )

        # Add host as participant
        session.participants[host_id] = SessionParticipant(
            user_id=host_id,
            display_name=host_display_name,
            role="host",
        )

        async with self._lock:
            self.sessions[session_id] = session
            self._subscribers[session_id] = {}
            self._pending_joins[session_id] = []

            if host_id not in self.user_sessions:
                self.user_sessions[host_id] = []
            self.user_sessions[host_id].append(session_id)

        logger.info(f"Created session {session_id}: {name}")
        return session

    async def join_session(
        self,
        session_id: str,
        user_id: str,
        display_name: str,
        role: str = "participant",
    ) -> Optional[CollaborativeSession]:
        """Join an existing session."""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        if not session.allow_join:
            return None

        if len(session.participants) >= session.max_participants:
            return None

        if session.require_approval and user_id != session.host_id:
            # Add to pending
            async with self._lock:
                if user_id not in self._pending_joins[session_id]:
                    self._pending_joins[session_id].append(user_id)

            # Notify host
            await self._add_event(
                session_id,
                EventType.USER_JOIN,
                session.host_id,
                f"{display_name} is requesting to join",
                {"user_id": user_id, "pending": True},
            )

            return None

        # Add participant
        async with self._lock:
            session.participants[user_id] = SessionParticipant(
                user_id=user_id,
                display_name=display_name,
                role=role,
            )

            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = []
            self.user_sessions[user_id].append(session_id)

        # Notify others
        await self._add_event(
            session_id,
            EventType.USER_JOIN,
            user_id,
            f"{display_name} joined the session",
        )

        return session

    async def approve_join(
        self,
        session_id: str,
        user_id: str,
        approved_by: str,
        display_name: str,
    ) -> bool:
        """Approve a pending join request."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        if approved_by != session.host_id:
            return False

        if user_id not in self._pending_joins.get(session_id, []):
            return False

        async with self._lock:
            self._pending_joins[session_id].remove(user_id)

        # Complete join
        await self.join_session(session_id, user_id, display_name)
        return True

    async def leave_session(
        self,
        session_id: str,
        user_id: str,
    ) -> bool:
        """Leave a session."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        if user_id not in session.participants:
            return False

        participant = session.participants[user_id]

        async with self._lock:
            del session.participants[user_id]

            if user_id in self.user_sessions:
                self.user_sessions[user_id] = [
                    s for s in self.user_sessions[user_id] if s != session_id
                ]

            if user_id in self._subscribers.get(session_id, {}):
                del self._subscribers[session_id][user_id]

        # Notify others
        await self._add_event(
            session_id,
            EventType.USER_LEAVE,
            user_id,
            f"{participant.display_name} left the session",
        )

        # End session if host leaves and no participants
        if user_id == session.host_id:
            if not session.participants:
                await self.end_session(session_id, user_id)
            else:
                # Transfer host
                new_host = next(iter(session.participants.keys()))
                session.host_id = new_host
                session.participants[new_host].role = "host"

        return True

    async def end_session(
        self,
        session_id: str,
        ended_by: str,
    ) -> bool:
        """End a session."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        if ended_by != session.host_id:
            return False

        async with self._lock:
            session.state = SessionState.ENDED
            session.ended_at = datetime.now()

        # Notify all participants
        await self._add_event(
            session_id,
            EventType.STATE_CHANGE,
            ended_by,
            "Session ended",
            {"new_state": "ended"},
        )

        # Clear subscribers
        async with self._lock:
            self._subscribers[session_id] = {}

        logger.info(f"Session {session_id} ended")
        return True

    async def send_message(
        self,
        session_id: str,
        user_id: str,
        content: str,
        data: Optional[dict] = None,
    ) -> Optional[SessionEvent]:
        """Send a message in a session."""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        if user_id not in session.participants:
            return None

        if session.state != SessionState.ACTIVE:
            return None

        return await self._add_event(
            session_id,
            EventType.MESSAGE,
            user_id,
            content,
            data,
        )

    async def perform_action(
        self,
        session_id: str,
        user_id: str,
        action_type: str,
        action_data: dict,
    ) -> Optional[SessionEvent]:
        """Perform an action in a session."""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        if user_id not in session.participants:
            return None

        return await self._add_event(
            session_id,
            EventType.ACTION,
            user_id,
            f"Action: {action_type}",
            {"action_type": action_type, **action_data},
        )

    async def update_shared_context(
        self,
        session_id: str,
        user_id: str,
        updates: dict,
    ) -> bool:
        """Update shared session context."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        if user_id not in session.participants:
            return False

        async with self._lock:
            session.shared_context.update(updates)

        await self._add_event(
            session_id,
            EventType.STATE_CHANGE,
            user_id,
            "Context updated",
            {"updates": list(updates.keys())},
        )

        return True

    async def add_shared_memory(
        self,
        session_id: str,
        user_id: str,
        memory_id: str,
    ) -> bool:
        """Add a memory to session's shared memories."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        if user_id not in session.participants:
            return False

        async with self._lock:
            if memory_id not in session.shared_memories:
                session.shared_memories.append(memory_id)

        await self._add_event(
            session_id,
            EventType.MEMORY_UPDATE,
            user_id,
            "Memory added to session",
            {"memory_id": memory_id},
        )

        return True

    async def subscribe(
        self,
        session_id: str,
        user_id: str,
        callback: Callable,
    ) -> bool:
        """Subscribe to session events."""
        if session_id not in self.sessions:
            return False

        async with self._lock:
            self._subscribers[session_id][user_id] = callback

        return True

    async def unsubscribe(
        self,
        session_id: str,
        user_id: str,
    ) -> bool:
        """Unsubscribe from session events."""
        if session_id not in self._subscribers:
            return False

        async with self._lock:
            if user_id in self._subscribers[session_id]:
                del self._subscribers[session_id][user_id]

        return True

    async def _add_event(
        self,
        session_id: str,
        event_type: EventType,
        user_id: str,
        content: str,
        data: Optional[dict] = None,
    ) -> SessionEvent:
        """Add an event to a session."""
        session = self.sessions[session_id]

        async with self._lock:
            self._event_counter += 1
            session.event_sequence += 1

        event = SessionEvent(
            id=f"event_{self._event_counter}",
            session_id=session_id,
            event_type=event_type,
            user_id=user_id,
            content=content,
            data=data or {},
            sequence_number=session.event_sequence,
        )

        async with self._lock:
            session.events.append(event)

            # Trim events if over limit
            if len(session.events) > self.max_events:
                session.events = session.events[-self.max_events // 2:]

        # Update participant last active
        if user_id in session.participants:
            session.participants[user_id].last_active = datetime.now()

        # Notify subscribers
        await self._notify_subscribers(session_id, event)

        return event

    async def _notify_subscribers(
        self,
        session_id: str,
        event: SessionEvent,
    ):
        """Notify all session subscribers."""
        if session_id not in self._subscribers:
            return

        for user_id, callback in self._subscribers[session_id].items():
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Subscriber callback failed: {e}")

    async def get_session_history(
        self,
        session_id: str,
        user_id: str,
        since_sequence: int = 0,
        limit: int = 100,
    ) -> list[SessionEvent]:
        """Get session event history."""
        if session_id not in self.sessions:
            return []

        session = self.sessions[session_id]

        if user_id not in session.participants:
            return []

        events = [
            e for e in session.events
            if e.sequence_number > since_sequence
        ]

        return events[:limit]

    async def replay_session(
        self,
        session_id: str,
        user_id: str,
        speed: float = 1.0,
    ) -> AsyncIterator[SessionEvent]:
        """Replay a session's events."""
        if session_id not in self.sessions:
            return

        session = self.sessions[session_id]

        if user_id not in session.participants:
            return

        prev_time = None

        for event in session.events:
            if prev_time:
                delay = (event.timestamp - prev_time).total_seconds() / speed
                if delay > 0:
                    await asyncio.sleep(min(delay, 2.0))  # Cap at 2 seconds

            yield event
            prev_time = event.timestamp

    async def start_session(
        self,
        session_id: str,
        started_by: str,
    ) -> bool:
        """Start an active session (from created state)."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        if started_by != session.host_id:
            return False

        if session.state != SessionState.CREATED:
            return False

        async with self._lock:
            session.state = SessionState.ACTIVE
            session.started_at = datetime.now()

        await self._add_event(
            session_id,
            EventType.STATE_CHANGE,
            started_by,
            "Session started",
            {"new_state": "active"},
        )

        return True

    async def pause_session(
        self,
        session_id: str,
        paused_by: str,
    ) -> bool:
        """Pause an active session."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        if paused_by != session.host_id:
            return False

        async with self._lock:
            session.state = SessionState.PAUSED

        await self._add_event(
            session_id,
            EventType.STATE_CHANGE,
            paused_by,
            "Session paused",
        )

        return True

    async def resume_session(
        self,
        session_id: str,
        resumed_by: str,
    ) -> bool:
        """Resume a paused session."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]

        if resumed_by != session.host_id:
            return False

        if session.state != SessionState.PAUSED:
            return False

        async with self._lock:
            session.state = SessionState.ACTIVE

        await self._add_event(
            session_id,
            EventType.STATE_CHANGE,
            resumed_by,
            "Session resumed",
        )

        return True

    async def get_active_sessions(
        self,
        user_id: str,
    ) -> list[CollaborativeSession]:
        """Get all active sessions for a user."""
        session_ids = self.user_sessions.get(user_id, [])

        return [
            self.sessions[sid]
            for sid in session_ids
            if sid in self.sessions and self.sessions[sid].state in (
                SessionState.ACTIVE, SessionState.PAUSED
            )
        ]

    def get_stats(self) -> dict:
        """Get session manager statistics."""
        active = sum(
            1 for s in self.sessions.values()
            if s.state == SessionState.ACTIVE
        )

        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active,
            "total_events": sum(len(s.events) for s in self.sessions.values()),
            "users_in_sessions": len(self.user_sessions),
        }
