"""
cosmos Episodic Memory - Timeline and Session-Based Memory

Q1 2025 Feature: Episodic Memory Timeline
- Visual timeline of all interactions
- "On this day" memory surfacing
- Session replay capability
- Event-based memory organization
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, Iterator
from collections import defaultdict
from enum import Enum

from loguru import logger


class EventType(Enum):
    """Types of episodic events."""
    CONVERSATION = "conversation"
    MEMORY_STORE = "memory_store"
    MEMORY_RECALL = "memory_recall"
    AGENT_TASK = "agent_task"
    AGENT_RESULT = "agent_result"
    USER_FEEDBACK = "user_feedback"
    SYSTEM_EVENT = "system_event"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    MILESTONE = "milestone"
    ERROR = "error"


@dataclass
class Episode:
    """A single episodic memory event."""
    id: str
    timestamp: datetime
    event_type: EventType
    content: str
    session_id: str

    # Optional fields
    metadata: dict = field(default_factory=dict)
    importance: float = 0.5  # 0-1 scale
    emotional_valence: float = 0.0  # -1 (negative) to 1 (positive)
    duration_seconds: float = 0.0
    parent_episode_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    # For session replay
    context_before: Optional[str] = None
    context_after: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "content": self.content,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "duration_seconds": self.duration_seconds,
            "parent_episode_id": self.parent_episode_id,
            "tags": self.tags,
            "context_before": self.context_before,
            "context_after": self.context_after,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=EventType(data["event_type"]),
            content=data["content"],
            session_id=data["session_id"],
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 0.5),
            emotional_valence=data.get("emotional_valence", 0.0),
            duration_seconds=data.get("duration_seconds", 0.0),
            parent_episode_id=data.get("parent_episode_id"),
            tags=data.get("tags", []),
            context_before=data.get("context_before"),
            context_after=data.get("context_after"),
        )


@dataclass
class Session:
    """A session containing multiple episodes."""
    id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    episode_count: int = 0
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> Optional[timedelta]:
        if self.ended_at:
            return self.ended_at - self.started_at
        return datetime.now() - self.started_at

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "title": self.title,
            "summary": self.summary,
            "episode_count": self.episode_count,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            id=data["id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            title=data.get("title"),
            summary=data.get("summary"),
            episode_count=data.get("episode_count", 0),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TimelineQuery:
    """Query parameters for timeline search."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    event_types: Optional[list[EventType]] = None
    session_ids: Optional[list[str]] = None
    min_importance: float = 0.0
    tags: Optional[list[str]] = None
    search_text: Optional[str] = None
    limit: int = 100
    offset: int = 0


@dataclass
class OnThisDayResult:
    """Result from 'on this day' lookup."""
    episodes: list[Episode]
    date: datetime
    years_ago: int
    notable_events: list[str]


class EpisodicMemory:
    """
    Episodic memory system for timeline-based recall.

    Features:
    - Session-based organization
    - Timeline visualization support
    - "On this day" memory surfacing
    - Session replay capability
    - Event importance scoring
    """

    def __init__(
        self,
        data_dir: str = "./data/episodic",
        max_episodes_per_session: int = 1000,
        auto_session_timeout_minutes: int = 30,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.max_episodes_per_session = max_episodes_per_session
        self.auto_session_timeout = timedelta(minutes=auto_session_timeout_minutes)

        # Storage
        self.sessions: dict[str, Session] = {}
        self.episodes: dict[str, Episode] = {}

        # Indices
        self.episodes_by_session: dict[str, list[str]] = defaultdict(list)
        self.episodes_by_date: dict[str, list[str]] = defaultdict(list)  # "YYYY-MM-DD" -> episode_ids
        self.episodes_by_type: dict[EventType, list[str]] = defaultdict(list)

        # Current session
        self._current_session: Optional[Session] = None
        self._last_activity: datetime = datetime.now()

        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize and load existing data."""
        if self._initialized:
            return

        await self._load_from_disk()
        self._initialized = True
        logger.info(f"Episodic memory initialized with {len(self.sessions)} sessions, {len(self.episodes)} episodes")

    async def start_session(
        self,
        title: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Session:
        """Start a new session."""
        async with self._lock:
            # End current session if exists
            if self._current_session and self._current_session.is_active:
                await self._end_session(self._current_session)

            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            session = Session(
                id=session_id,
                started_at=datetime.now(),
                title=title,
                metadata=metadata or {},
            )

            self.sessions[session_id] = session
            self._current_session = session
            self._last_activity = datetime.now()

            # Record session start event
            await self._record_episode(
                event_type=EventType.SESSION_START,
                content=f"Session started: {title or 'Untitled'}",
                session=session,
                importance=0.7,
            )

            await self._save_session(session)

            return session

    async def end_session(self, summary: Optional[str] = None) -> Optional[Session]:
        """End the current session."""
        async with self._lock:
            if not self._current_session:
                return None

            self._current_session.summary = summary
            await self._end_session(self._current_session)

            session = self._current_session
            self._current_session = None

            return session

    async def _end_session(self, session: Session):
        """Internal session ending."""
        session.ended_at = datetime.now()

        # Record session end event
        await self._record_episode(
            event_type=EventType.SESSION_END,
            content=f"Session ended: {session.title or session.id}",
            session=session,
            importance=0.6,
        )

        await self._save_session(session)

    async def record_episode(
        self,
        event_type: EventType,
        content: str,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        parent_episode_id: Optional[str] = None,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> Episode:
        """Record an episodic event."""
        async with self._lock:
            # Auto-create or resume session
            session = await self._get_or_create_session()

            episode = await self._record_episode(
                event_type=event_type,
                content=content,
                session=session,
                importance=importance,
                emotional_valence=emotional_valence,
                metadata=metadata,
                tags=tags,
                parent_episode_id=parent_episode_id,
                context_before=context_before,
                context_after=context_after,
            )

            self._last_activity = datetime.now()

            return episode

    async def _record_episode(
        self,
        event_type: EventType,
        content: str,
        session: Session,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        metadata: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        parent_episode_id: Optional[str] = None,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> Episode:
        """Internal episode recording."""
        episode_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}_{uuid.uuid4().hex[:6]}"

        episode = Episode(
            id=episode_id,
            timestamp=datetime.now(),
            event_type=event_type,
            content=content,
            session_id=session.id,
            metadata=metadata or {},
            importance=importance,
            emotional_valence=emotional_valence,
            tags=tags or [],
            parent_episode_id=parent_episode_id,
            context_before=context_before,
            context_after=context_after,
        )

        # Store
        self.episodes[episode_id] = episode

        # Update indices
        self.episodes_by_session[session.id].append(episode_id)
        date_key = episode.timestamp.strftime("%Y-%m-%d")
        self.episodes_by_date[date_key].append(episode_id)
        self.episodes_by_type[event_type].append(episode_id)

        # Update session
        session.episode_count += 1

        # Persist
        await self._save_episode(episode)

        return episode

    async def _get_or_create_session(self) -> Session:
        """Get current session or create new one."""
        # Check if we need a new session due to timeout
        if self._current_session:
            time_since_activity = datetime.now() - self._last_activity
            if time_since_activity > self.auto_session_timeout:
                await self._end_session(self._current_session)
                self._current_session = None

        if not self._current_session:
            # Create new session
            return await self.start_session()

        return self._current_session

    async def query_timeline(self, query: TimelineQuery) -> list[Episode]:
        """Query episodes based on timeline parameters."""
        async with self._lock:
            results = []

            for episode in self.episodes.values():
                # Date range filter
                if query.start_date and episode.timestamp < query.start_date:
                    continue
                if query.end_date and episode.timestamp > query.end_date:
                    continue

                # Event type filter
                if query.event_types and episode.event_type not in query.event_types:
                    continue

                # Session filter
                if query.session_ids and episode.session_id not in query.session_ids:
                    continue

                # Importance filter
                if episode.importance < query.min_importance:
                    continue

                # Tags filter
                if query.tags and not any(t in episode.tags for t in query.tags):
                    continue

                # Text search
                if query.search_text:
                    search_lower = query.search_text.lower()
                    if search_lower not in episode.content.lower():
                        continue

                results.append(episode)

            # Sort by timestamp descending
            results.sort(key=lambda e: e.timestamp, reverse=True)

            # Apply pagination
            return results[query.offset:query.offset + query.limit]

    async def get_on_this_day(
        self,
        date: Optional[datetime] = None,
        years_back: int = 5,
    ) -> list[OnThisDayResult]:
        """Get 'on this day' memories from previous years."""
        if date is None:
            date = datetime.now()

        results = []

        for years_ago in range(1, years_back + 1):
            try:
                past_date = date.replace(year=date.year - years_ago)
            except ValueError:
                # Handle Feb 29 on non-leap years
                past_date = date.replace(year=date.year - years_ago, day=28)

            date_key = past_date.strftime("%Y-%m-%d")
            episode_ids = self.episodes_by_date.get(date_key, [])

            if episode_ids:
                episodes = [self.episodes[eid] for eid in episode_ids if eid in self.episodes]

                # Sort by importance
                episodes.sort(key=lambda e: e.importance, reverse=True)

                # Extract notable events
                notable = [
                    e.content[:100] for e in episodes
                    if e.importance >= 0.7
                ][:3]

                results.append(OnThisDayResult(
                    episodes=episodes,
                    date=past_date,
                    years_ago=years_ago,
                    notable_events=notable,
                ))

        return results

    async def get_session_replay(self, session_id: str) -> list[Episode]:
        """Get all episodes for session replay."""
        episode_ids = self.episodes_by_session.get(session_id, [])
        episodes = [self.episodes[eid] for eid in episode_ids if eid in self.episodes]

        # Sort by timestamp for replay order
        episodes.sort(key=lambda e: e.timestamp)

        return episodes

    async def get_session_summary(self, session_id: str) -> dict:
        """Get a summary of a session."""
        session = self.sessions.get(session_id)
        if not session:
            return {}

        episodes = await self.get_session_replay(session_id)

        # Analyze episodes
        event_counts = defaultdict(int)
        total_importance = 0
        total_valence = 0

        for ep in episodes:
            event_counts[ep.event_type.value] += 1
            total_importance += ep.importance
            total_valence += ep.emotional_valence

        avg_importance = total_importance / max(1, len(episodes))
        avg_valence = total_valence / max(1, len(episodes))

        return {
            "session": session.to_dict(),
            "episode_count": len(episodes),
            "event_breakdown": dict(event_counts),
            "average_importance": avg_importance,
            "average_emotional_valence": avg_valence,
            "duration_seconds": session.duration.total_seconds() if session.duration else 0,
            "first_event": episodes[0].content[:100] if episodes else None,
            "last_event": episodes[-1].content[:100] if episodes else None,
        }

    async def get_timeline_segments(
        self,
        granularity: str = "day",  # "hour", "day", "week", "month"
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[dict]:
        """Get timeline segments for visualization."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()

        segments = []

        # Group episodes by granularity
        if granularity == "hour":
            fmt = "%Y-%m-%d %H:00"
            delta = timedelta(hours=1)
        elif granularity == "day":
            fmt = "%Y-%m-%d"
            delta = timedelta(days=1)
        elif granularity == "week":
            fmt = "%Y-W%W"
            delta = timedelta(weeks=1)
        else:  # month
            fmt = "%Y-%m"
            delta = timedelta(days=30)

        current = start_date
        while current <= end_date:
            segment_key = current.strftime(fmt)

            # Find episodes in this segment
            segment_episodes = [
                ep for ep in self.episodes.values()
                if ep.timestamp.strftime(fmt) == segment_key
            ]

            if segment_episodes:
                segments.append({
                    "period": segment_key,
                    "start": current.isoformat(),
                    "episode_count": len(segment_episodes),
                    "event_types": list(set(ep.event_type.value for ep in segment_episodes)),
                    "avg_importance": sum(ep.importance for ep in segment_episodes) / len(segment_episodes),
                    "top_episodes": [
                        {"id": ep.id, "content": ep.content[:50], "type": ep.event_type.value}
                        for ep in sorted(segment_episodes, key=lambda e: e.importance, reverse=True)[:3]
                    ],
                })

            current += delta

        return segments

    async def mark_milestone(
        self,
        content: str,
        metadata: Optional[dict] = None,
    ) -> Episode:
        """Mark a milestone event."""
        return await self.record_episode(
            event_type=EventType.MILESTONE,
            content=content,
            importance=1.0,
            metadata=metadata,
            tags=["milestone"],
        )

    async def _save_episode(self, episode: Episode):
        """Save episode to disk."""
        # Save to date-based file
        date_dir = self.data_dir / "episodes" / episode.timestamp.strftime("%Y/%m")
        date_dir.mkdir(parents=True, exist_ok=True)

        episode_file = date_dir / f"{episode.id}.json"
        episode_file.write_text(json.dumps(episode.to_dict(), indent=2), encoding='utf-8')

    async def _save_session(self, session: Session):
        """Save session to disk."""
        sessions_dir = self.data_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        session_file = sessions_dir / f"{session.id}.json"
        session_file.write_text(json.dumps(session.to_dict(), indent=2), encoding='utf-8')

    async def _load_from_disk(self):
        """Load all data from disk."""
        # Load sessions
        sessions_dir = self.data_dir / "sessions"
        if sessions_dir.exists():
            for session_file in sessions_dir.glob("*.json"):
                try:
                    data = json.loads(session_file.read_text(encoding='utf-8'))
                    session = Session.from_dict(data)
                    self.sessions[session.id] = session
                except Exception as e:
                    logger.error(f"Failed to load session {session_file}: {e}")

        # Load episodes
        episodes_dir = self.data_dir / "episodes"
        if episodes_dir.exists():
            for episode_file in episodes_dir.rglob("*.json"):
                try:
                    data = json.loads(episode_file.read_text(encoding='utf-8'))
                    episode = Episode.from_dict(data)
                    self.episodes[episode.id] = episode

                    # Update indices
                    self.episodes_by_session[episode.session_id].append(episode.id)
                    date_key = episode.timestamp.strftime("%Y-%m-%d")
                    self.episodes_by_date[date_key].append(episode.id)
                    self.episodes_by_type[episode.event_type].append(episode.id)
                except Exception as e:
                    logger.error(f"Failed to load episode {episode_file}: {e}")

    def get_stats(self) -> dict:
        """Get episodic memory statistics."""
        return {
            "total_sessions": len(self.sessions),
            "total_episodes": len(self.episodes),
            "active_session": self._current_session.id if self._current_session else None,
            "episodes_by_type": {
                t.value: len(ids) for t, ids in self.episodes_by_type.items()
            },
            "date_range": {
                "earliest": min(self.episodes_by_date.keys()) if self.episodes_by_date else None,
                "latest": max(self.episodes_by_date.keys()) if self.episodes_by_date else None,
            },
        }
