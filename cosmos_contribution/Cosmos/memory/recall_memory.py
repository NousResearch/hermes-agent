"""
cosmos Recall Memory - Conversation History Search

Novel Approaches:
1. Temporal Chunking - Smart conversation segmentation
2. Topic Threading - Automatic conversation topic tracking
3. Sentiment Indexing - Emotion-aware retrieval
4. Speaker Attribution - Multi-party conversation support
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import deque

from loguru import logger


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    # Analysis
    topics: list[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1 to 1
    intent: str = ""  # question, statement, command, etc.

    # Threading
    thread_id: Optional[str] = None
    references_turns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "topics": self.topics,
            "sentiment": self.sentiment,
            "intent": self.intent,
            "thread_id": self.thread_id,
            "references_turns": self.references_turns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            topics=data.get("topics", []),
            sentiment=data.get("sentiment", 0.0),
            intent=data.get("intent", ""),
            thread_id=data.get("thread_id"),
            references_turns=data.get("references_turns", []),
        )


@dataclass
class ConversationChunk:
    """A chunk of related conversation turns."""
    id: str
    turns: list[ConversationTurn]
    start_time: datetime
    end_time: datetime
    primary_topic: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "turns": [t.to_dict() for t in self.turns],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "primary_topic": self.primary_topic,
            "summary": self.summary,
        }


@dataclass
class RecallResult:
    """Result from recall search."""
    turn: ConversationTurn
    score: float
    context_turns: list[ConversationTurn] = field(default_factory=list)
    match_type: str = "content"  # content, topic, temporal


class RecallMemory:
    """
    Conversation history with intelligent search.

    Features:
    - Automatic conversation chunking by topic/time
    - Multi-dimensional search (content, topic, time, sentiment)
    - Efficient recent history access
    - Persistent storage with lazy loading
    """

    def __init__(
        self,
        data_dir: str = "./data/conversations",
        max_recent_turns: int = 100,
        chunk_time_gap_minutes: int = 30,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.max_recent_turns = max_recent_turns
        self.chunk_time_gap = timedelta(minutes=chunk_time_gap_minutes)

        # In-memory recent history
        self.recent_turns: deque[ConversationTurn] = deque(maxlen=max_recent_turns)

        # All turns (lazy loaded)
        self.all_turns: dict[str, ConversationTurn] = {}

        # Chunks for organization
        self.chunks: list[ConversationChunk] = []
        self.current_chunk_turns: list[ConversationTurn] = []

        # Topic tracking
        self.active_topics: list[str] = []
        self.topic_history: dict[str, list[str]] = {}  # topic -> turn_ids

        # Session tracking
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._turn_counter = 0
        self._lock = asyncio.Lock()

    async def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> ConversationTurn:
        """Add a new conversation turn."""
        async with self._lock:
            self._turn_counter += 1
            turn_id = f"{self.session_id}_{self._turn_counter:06d}"

            # Analyze the turn
            topics = self._extract_topics(content)
            sentiment = self._analyze_sentiment(content)
            intent = self._classify_intent(content)

            # Detect references to previous turns
            references = self._detect_references(content)

            turn = ConversationTurn(
                id=turn_id,
                role=role,
                content=content,
                metadata=metadata or {},
                topics=topics,
                sentiment=sentiment,
                intent=intent,
                references_turns=references,
            )

            # Check if we need to start a new chunk
            if self._should_start_new_chunk(turn):
                await self._finalize_current_chunk()

            # Add to current chunk
            self.current_chunk_turns.append(turn)
            turn.thread_id = self.chunks[-1].id if self.chunks else f"chunk_{self.session_id}_0"

            # Add to memory
            self.recent_turns.append(turn)
            self.all_turns[turn_id] = turn

            # Update topic tracking
            for topic in topics:
                if topic not in self.topic_history:
                    self.topic_history[topic] = []
                self.topic_history[topic].append(turn_id)
            self.active_topics = topics[:3]  # Keep top 3 recent topics

            # Persist
            await self._persist_turn(turn)

            return turn

    def _should_start_new_chunk(self, new_turn: ConversationTurn) -> bool:
        """Determine if a new conversation chunk should start."""
        if not self.current_chunk_turns:
            return False

        last_turn = self.current_chunk_turns[-1]

        # Time gap check
        time_gap = new_turn.timestamp - last_turn.timestamp
        if time_gap > self.chunk_time_gap:
            return True

        # Topic shift check
        if last_turn.topics and new_turn.topics:
            overlap = set(last_turn.topics) & set(new_turn.topics)
            if not overlap and len(self.current_chunk_turns) > 5:
                return True

        # Length check
        if len(self.current_chunk_turns) > 50:
            return True

        return False

    async def _finalize_current_chunk(self):
        """Finalize the current chunk and start a new one."""
        if not self.current_chunk_turns:
            return

        chunk_id = f"chunk_{self.session_id}_{len(self.chunks)}"
        chunk = ConversationChunk(
            id=chunk_id,
            turns=self.current_chunk_turns.copy(),
            start_time=self.current_chunk_turns[0].timestamp,
            end_time=self.current_chunk_turns[-1].timestamp,
            primary_topic=self._get_primary_topic(self.current_chunk_turns),
        )

        self.chunks.append(chunk)
        self.current_chunk_turns = []

        # Persist chunk
        await self._persist_chunk(chunk)

    def _get_primary_topic(self, turns: list[ConversationTurn]) -> str:
        """Get the primary topic from a list of turns."""
        topic_counts: dict[str, int] = {}
        for turn in turns:
            for topic in turn.topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        if topic_counts:
            return max(topic_counts.items(), key=lambda x: x[1])[0]
        return "general"

    def _extract_topics(self, content: str) -> list[str]:
        """Extract topics from content."""
        topics = []

        # Simple keyword-based extraction
        tech_keywords = [
            "code", "programming", "python", "javascript", "api", "database",
            "algorithm", "function", "class", "error", "bug", "test",
        ]
        general_keywords = [
            "help", "explain", "how", "why", "what", "create", "build",
            "fix", "improve", "optimize", "design", "implement",
        ]

        content_lower = content.lower()

        for keyword in tech_keywords:
            if keyword in content_lower:
                topics.append(f"tech:{keyword}")

        for keyword in general_keywords:
            if keyword in content_lower:
                topics.append(f"intent:{keyword}")

        # Extract capitalized terms (potential proper nouns/concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        for term in capitalized[:3]:
            topics.append(f"entity:{term.lower()}")

        return topics[:5]

    def _analyze_sentiment(self, content: str) -> float:
        """Simple sentiment analysis."""
        positive_words = [
            "good", "great", "excellent", "amazing", "perfect", "thanks",
            "helpful", "awesome", "love", "appreciate", "wonderful",
        ]
        negative_words = [
            "bad", "wrong", "error", "fail", "broken", "hate", "terrible",
            "awful", "problem", "issue", "bug", "crash", "frustrating",
        ]

        content_lower = content.lower()
        positive_count = sum(1 for w in positive_words if w in content_lower)
        negative_count = sum(1 for w in negative_words if w in content_lower)

        if positive_count + negative_count == 0:
            return 0.0

        return (positive_count - negative_count) / (positive_count + negative_count)

    def _classify_intent(self, content: str) -> str:
        """Classify the intent of a message."""
        content_lower = content.lower().strip()

        if content_lower.endswith("?") or content_lower.startswith(("what", "how", "why", "when", "where", "who", "can you", "could you")):
            return "question"
        elif content_lower.startswith(("please", "can you", "could you", "i need", "i want")):
            return "request"
        elif content_lower.startswith(("do", "make", "create", "build", "fix", "run", "execute")):
            return "command"
        elif content_lower.startswith(("thank", "great", "good", "nice")):
            return "feedback"
        else:
            return "statement"

    def _detect_references(self, content: str) -> list[str]:
        """Detect references to previous conversation."""
        references = []

        # Check for anaphora
        if any(word in content.lower() for word in ["that", "this", "it", "those", "these", "previous", "earlier", "before"]):
            if self.recent_turns:
                # Reference last few turns
                for turn in list(self.recent_turns)[-3:]:
                    references.append(turn.id)

        return references

    async def search(
        self,
        query: str,
        top_k: int = 10,
        time_range: Optional[tuple[datetime, datetime]] = None,
        role_filter: Optional[str] = None,
        topic_filter: Optional[list[str]] = None,
        include_context: bool = True,
        context_window: int = 2,
    ) -> list[RecallResult]:
        """
        Search conversation history.

        Args:
            query: Search query
            top_k: Number of results
            time_range: Optional (start, end) datetime filter
            role_filter: Filter by role (user/assistant)
            topic_filter: Filter by topics
            include_context: Include surrounding turns
            context_window: Number of turns before/after to include

        Returns:
            List of RecallResult
        """
        async with self._lock:
            results = []

            # Combine all turns
            all_turns = list(self.all_turns.values())

            for turn in all_turns:
                # Apply filters
                if time_range:
                    if not (time_range[0] <= turn.timestamp <= time_range[1]):
                        continue

                if role_filter and turn.role != role_filter:
                    continue

                if topic_filter:
                    if not any(t in turn.topics for t in topic_filter):
                        continue

                # Score the turn
                score = self._score_turn(query, turn)

                if score > 0:
                    results.append((turn, score))

            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]

            # Build RecallResults with context
            recall_results = []
            for turn, score in results:
                context_turns = []
                if include_context:
                    context_turns = self._get_context_turns(turn, context_window)

                recall_results.append(RecallResult(
                    turn=turn,
                    score=score,
                    context_turns=context_turns,
                    match_type="content",
                ))

            return recall_results

    def _score_turn(self, query: str, turn: ConversationTurn) -> float:
        """Score a turn's relevance to a query."""
        query_lower = query.lower()
        content_lower = turn.content.lower()

        score = 0.0

        # Exact phrase match
        if query_lower in content_lower:
            score += 1.0

        # Word overlap
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = query_words & content_words
        if query_words:
            score += len(overlap) / len(query_words) * 0.5

        # Topic match
        query_topics = self._extract_topics(query)
        topic_overlap = set(query_topics) & set(turn.topics)
        score += len(topic_overlap) * 0.2

        # Recency boost
        age_hours = (datetime.now() - turn.timestamp).total_seconds() / 3600
        recency_factor = 1.0 / (1.0 + age_hours / 24)
        score *= (0.8 + 0.2 * recency_factor)

        return score

    def _get_context_turns(self, turn: ConversationTurn, window: int) -> list[ConversationTurn]:
        """Get surrounding context turns."""
        all_turns = list(self.all_turns.values())
        all_turns.sort(key=lambda t: t.timestamp)

        try:
            idx = next(i for i, t in enumerate(all_turns) if t.id == turn.id)
        except StopIteration:
            return []

        start = max(0, idx - window)
        end = min(len(all_turns), idx + window + 1)

        return [t for t in all_turns[start:end] if t.id != turn.id]

    def get_recent(self, count: int = 10) -> list[ConversationTurn]:
        """Get most recent turns."""
        return list(self.recent_turns)[-count:]

    def get_by_topic(self, topic: str, limit: int = 20) -> list[ConversationTurn]:
        """Get turns by topic."""
        if topic not in self.topic_history:
            return []

        turn_ids = self.topic_history[topic][-limit:]
        return [self.all_turns[tid] for tid in turn_ids if tid in self.all_turns]

    def to_context_string(self, max_turns: int = 10, max_length: int = 3000) -> str:
        """Convert recent history to context string."""
        recent = self.get_recent(max_turns)
        parts = ["[Conversation History]"]
        current_length = len(parts[0])

        for turn in recent:
            turn_str = f"\n{turn.role}: {turn.content[:500]}"
            if current_length + len(turn_str) > max_length:
                break
            parts.append(turn_str)
            current_length += len(turn_str)

        return "".join(parts)

    async def _persist_turn(self, turn: ConversationTurn):
        """Persist turn to disk."""
        session_file = self.data_dir / f"{self.session_id}.jsonl"
        line = json.dumps(turn.to_dict()) + "\n"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: session_file.open('a', encoding='utf-8').write(line)
        )

    async def _persist_chunk(self, chunk: ConversationChunk):
        """Persist chunk to disk."""
        chunk_file = self.data_dir / f"chunks/{chunk.id}.json"
        chunk_file.parent.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: chunk_file.write_text(json.dumps(chunk.to_dict()), encoding='utf-8')
        )

    async def load_session(self, session_id: str):
        """Load a previous session."""
        session_file = self.data_dir / f"{session_id}.jsonl"
        if not session_file.exists():
            return

        for line in session_file.read_text(encoding='utf-8').strip().split('\n'):
            if line:
                turn = ConversationTurn.from_dict(json.loads(line))
                self.all_turns[turn.id] = turn

    def get_stats(self) -> dict:
        """Get recall memory statistics."""
        return {
            "total_turns": len(self.all_turns),
            "recent_turns": len(self.recent_turns),
            "chunks": len(self.chunks),
            "active_topics": self.active_topics,
            "session_id": self.session_id,
            "unique_topics": len(self.topic_history),
        }
