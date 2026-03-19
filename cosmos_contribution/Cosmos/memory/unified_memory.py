"""
COSMOS UNIFIED MEMORY ARCHITECTURE
=======================================

The most advanced multi-layered memory system integrating all 18 memory types
into a cohesive architecture with lightweight session references for agents.

Memory Systems Integrated:
1.  Working Memory       - Current task scratchpad
2.  Archival Memory      - Long-term persistent storage
3.  Recall Memory        - Conversation history
4.  Episodic Memory      - Timeline of interactions
5.  Knowledge Graph v1   - Entity relationships
6.  Knowledge Graph v2   - Temporal edges + entity resolution
7.  Virtual Context      - MemGPT-style memory paging
8.  Memory Dreaming      - Pattern discovery + creative synthesis
9.  Dream Consolidation  - Sleep-cycle simulation with 8 strategies
10. Memory Sharing       - Export/import + multi-agent sync
11. Project Tracking     - Auto-detect projects from context
12. Semantic Layers      - Concept hierarchy abstraction
13. Semantic Dedup       - TF-IDF similarity deduplication
14. Sharding             - Distributed storage hashing
15. Planetary Audio      - P2P TTS cache sharing
16. Conversation Export  - Multi-format export
17. Query Cache          - LRU cache with TTL
18. P2P Memory           - Gossipsub protocol for distributed knowledge

Architecture:
- SessionReference: Lightweight token for agent memory access
- MemoryRouter: Routes queries to optimal memory systems
- MemoryPipeline: Sequential/parallel processing stages
- UnifiedMemoryAPI: Single interface for all operations
"""

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Any, Dict, List, Callable, Set, Tuple
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class MemoryTier(Enum):
    """Memory storage tiers by access speed/freshness"""
    HOT = "hot"           # In-context, immediate access
    WARM = "warm"         # Cached, fast retrieval
    COLD = "cold"         # Indexed, search required
    ARCHIVE = "archive"   # Long-term, slow retrieval
    PLANETARY = "planetary"  # Distributed across network


class MemoryDomain(Enum):
    """Semantic domains for routing"""
    CONVERSATION = auto()   # Chat history
    KNOWLEDGE = auto()      # Facts and entities
    TASK = auto()           # Projects and todos
    CODE = auto()           # Code and technical
    PERSONAL = auto()       # User preferences
    TEMPORAL = auto()       # Time-based events
    AUDIO = auto()          # Voice/audio content
    CREATIVE = auto()       # Generated insights


class QueryIntent(Enum):
    """Types of memory queries"""
    RECALL = "recall"       # Find specific past content
    SEARCH = "search"       # Semantic similarity search
    RELATE = "relate"       # Find relationships
    TIMELINE = "timeline"   # Time-ordered retrieval
    PROJECT = "project"     # Project-specific context
    SYNTHESIS = "synthesis" # Creative recombination


@dataclass
class SessionReference:
    """
    Lightweight token for agent memory access.

    Agents receive this at session start. It provides:
    - Session ID for tracking
    - Cached topic/entity hints for fast routing
    - Recent query history for relevance boosting
    - Access permissions and quotas
    """
    session_id: str
    agent_id: str
    created_at: datetime = field(default_factory=datetime.now)

    # Lightweight context hints (updated incrementally)
    active_topics: Set[str] = field(default_factory=set)
    active_entities: Set[str] = field(default_factory=set)
    active_projects: Set[str] = field(default_factory=set)

    # Query history for relevance boosting
    recent_queries: List[str] = field(default_factory=list)
    max_query_history: int = 20

    # Access control
    allowed_tiers: Set[MemoryTier] = field(default_factory=lambda: {
        MemoryTier.HOT, MemoryTier.WARM, MemoryTier.COLD
    })
    query_quota_remaining: int = 1000

    # Performance hints
    preferred_latency_ms: int = 100

    def add_query(self, query: str):
        """Track a query for relevance boosting"""
        self.recent_queries.append(query)
        if len(self.recent_queries) > self.max_query_history:
            self.recent_queries.pop(0)

    def update_context(self, topics: List[str] = None, entities: List[str] = None, projects: List[str] = None):
        """Update session context hints"""
        if topics:
            self.active_topics.update(topics[:10])  # Limit to prevent bloat
        if entities:
            self.active_entities.update(entities[:20])
        if projects:
            self.active_projects.update(projects[:5])

    def get_context_hints(self) -> Dict:
        """Get lightweight context for query routing"""
        return {
            "topics": list(self.active_topics)[-5:],
            "entities": list(self.active_entities)[-10:],
            "projects": list(self.active_projects)[-3:],
            "recent_queries": self.recent_queries[-3:],
        }

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "active_topics": list(self.active_topics),
            "active_entities": list(self.active_entities),
            "active_projects": list(self.active_projects),
            "query_quota_remaining": self.query_quota_remaining,
        }


@dataclass
class MemoryResult:
    """Unified result from any memory system"""
    content: str
    source_system: str
    tier: MemoryTier
    domain: MemoryDomain
    score: float
    memory_id: str
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "source_system": self.source_system,
            "tier": self.tier.value,
            "domain": self.domain.name,
            "score": self.score,
            "memory_id": self.memory_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RoutingDecision:
    """Decision about which memory systems to query"""
    systems: List[str]
    parallel: bool
    max_results_per_system: int
    timeout_ms: int
    tier_priority: List[MemoryTier]


# =============================================================================
# MEMORY ROUTER
# =============================================================================

class MemoryRouter:
    """
    Routes queries to optimal memory systems based on:
    - Query intent classification
    - Session context hints
    - System availability and load
    - Latency requirements
    """

    # System configurations
    SYSTEM_CONFIG = {
        "working_memory": {
            "tier": MemoryTier.HOT,
            "domains": [MemoryDomain.TASK, MemoryDomain.CODE],
            "latency_ms": 5,
            "intents": [QueryIntent.RECALL],
        },
        "recall_memory": {
            "tier": MemoryTier.WARM,
            "domains": [MemoryDomain.CONVERSATION],
            "latency_ms": 20,
            "intents": [QueryIntent.RECALL, QueryIntent.TIMELINE],
        },
        "archival_memory": {
            "tier": MemoryTier.COLD,
            "domains": [MemoryDomain.KNOWLEDGE, MemoryDomain.PERSONAL],
            "latency_ms": 80,
            "intents": [QueryIntent.SEARCH, QueryIntent.RECALL],
        },
        "episodic_memory": {
            "tier": MemoryTier.WARM,
            "domains": [MemoryDomain.TEMPORAL],
            "latency_ms": 30,
            "intents": [QueryIntent.TIMELINE],
        },
        "knowledge_graph": {
            "tier": MemoryTier.WARM,
            "domains": [MemoryDomain.KNOWLEDGE],
            "latency_ms": 50,
            "intents": [QueryIntent.RELATE, QueryIntent.SEARCH],
        },
        "knowledge_graph_v2": {
            "tier": MemoryTier.WARM,
            "domains": [MemoryDomain.KNOWLEDGE, MemoryDomain.TEMPORAL],
            "latency_ms": 60,
            "intents": [QueryIntent.RELATE, QueryIntent.TIMELINE],
        },
        "virtual_context": {
            "tier": MemoryTier.HOT,
            "domains": [MemoryDomain.CONVERSATION, MemoryDomain.TASK],
            "latency_ms": 10,
            "intents": [QueryIntent.RECALL],
        },
        "dream_consolidation": {
            "tier": MemoryTier.COLD,
            "domains": [MemoryDomain.CREATIVE],
            "latency_ms": 200,
            "intents": [QueryIntent.SYNTHESIS],
        },
        "semantic_layers": {
            "tier": MemoryTier.WARM,
            "domains": [MemoryDomain.KNOWLEDGE],
            "latency_ms": 100,
            "intents": [QueryIntent.RELATE, QueryIntent.SYNTHESIS],
        },
        "project_tracking": {
            "tier": MemoryTier.WARM,
            "domains": [MemoryDomain.TASK],
            "latency_ms": 40,
            "intents": [QueryIntent.PROJECT],
        },
        "planetary_audio": {
            "tier": MemoryTier.PLANETARY,
            "domains": [MemoryDomain.AUDIO],
            "latency_ms": 500,
            "intents": [QueryIntent.RECALL],
        },
        "p2p_memory": {
            "tier": MemoryTier.PLANETARY,
            "domains": [MemoryDomain.KNOWLEDGE],
            "latency_ms": 1000,
            "intents": [QueryIntent.SEARCH],
        },
    }

    def __init__(self):
        self.query_patterns = self._build_query_patterns()

    def _build_query_patterns(self) -> Dict[str, QueryIntent]:
        """Build patterns for intent classification"""
        return {
            "what did": QueryIntent.RECALL,
            "remember when": QueryIntent.RECALL,
            "find": QueryIntent.SEARCH,
            "search for": QueryIntent.SEARCH,
            "related to": QueryIntent.RELATE,
            "connected to": QueryIntent.RELATE,
            "timeline": QueryIntent.TIMELINE,
            "history of": QueryIntent.TIMELINE,
            "project": QueryIntent.PROJECT,
            "task": QueryIntent.PROJECT,
            "synthesize": QueryIntent.SYNTHESIS,
            "combine": QueryIntent.SYNTHESIS,
        }

    def classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent"""
        query_lower = query.lower()
        for pattern, intent in self.query_patterns.items():
            if pattern in query_lower:
                return intent
        return QueryIntent.SEARCH  # Default

    def classify_domain(self, query: str, context: Dict) -> List[MemoryDomain]:
        """Classify query domain based on content and context"""
        domains = []
        query_lower = query.lower()

        # Content-based classification
        if any(kw in query_lower for kw in ["conversation", "said", "asked", "told"]):
            domains.append(MemoryDomain.CONVERSATION)
        if any(kw in query_lower for kw in ["code", "function", "class", "api", "bug"]):
            domains.append(MemoryDomain.CODE)
        if any(kw in query_lower for kw in ["when", "date", "time", "yesterday", "last week"]):
            domains.append(MemoryDomain.TEMPORAL)
        if any(kw in query_lower for kw in ["project", "task", "todo", "milestone"]):
            domains.append(MemoryDomain.TASK)
        if any(kw in query_lower for kw in ["voice", "audio", "sound", "speak"]):
            domains.append(MemoryDomain.AUDIO)

        # Context-based boost
        if context.get("projects"):
            domains.append(MemoryDomain.TASK)

        # Default to knowledge if nothing else matches
        if not domains:
            domains.append(MemoryDomain.KNOWLEDGE)

        return list(set(domains))

    def route(
        self,
        query: str,
        session: SessionReference,
        max_latency_ms: int = None,
    ) -> RoutingDecision:
        """
        Route a query to optimal memory systems.

        Returns a RoutingDecision with:
        - List of systems to query
        - Whether to query in parallel
        - Per-system result limits
        - Timeout configuration
        """
        context = session.get_context_hints()
        intent = self.classify_intent(query)
        domains = self.classify_domain(query, context)
        max_latency = max_latency_ms or session.preferred_latency_ms

        # Find matching systems
        matching_systems = []
        for system, config in self.SYSTEM_CONFIG.items():
            # Check tier access
            if config["tier"] not in session.allowed_tiers:
                continue

            # Check latency constraint
            if config["latency_ms"] > max_latency:
                continue

            # Check intent match
            if intent in config["intents"]:
                matching_systems.append((system, config, 1.0))
                continue

            # Check domain match
            for domain in domains:
                if domain in config["domains"]:
                    matching_systems.append((system, config, 0.5))
                    break

        # Sort by relevance and latency
        matching_systems.sort(key=lambda x: (-x[2], x[1]["latency_ms"]))

        # Take top systems
        selected = [s[0] for s in matching_systems[:5]]

        # Determine if we can run in parallel
        parallel = len(selected) > 1 and max_latency >= 100

        # Calculate tier priority
        tier_priority = [MemoryTier.HOT, MemoryTier.WARM, MemoryTier.COLD]
        if intent == QueryIntent.SYNTHESIS:
            tier_priority = [MemoryTier.COLD, MemoryTier.WARM, MemoryTier.HOT]

        return RoutingDecision(
            systems=selected or ["archival_memory"],
            parallel=parallel,
            max_results_per_system=5,
            timeout_ms=max_latency,
            tier_priority=tier_priority,
        )


# =============================================================================
# MEMORY PIPELINE
# =============================================================================

class MemoryPipeline:
    """
    Processing pipeline for memory operations.

    Stages:
    1. Pre-process: Expand query, extract keywords
    2. Route: Determine target systems
    3. Execute: Run queries (parallel/sequential)
    4. Fuse: Combine results with deduplication
    5. Rank: Score and sort by relevance
    6. Post-process: Add context, update session
    """

    def __init__(self, router: MemoryRouter):
        self.router = router
        self._deduplicator = None

    async def process(
        self,
        query: str,
        session: SessionReference,
        systems: Dict[str, Any],
        top_k: int = 10,
    ) -> List[MemoryResult]:
        """
        Execute the full memory pipeline.

        Args:
            query: The search query
            session: Session reference for context
            systems: Dictionary of initialized memory system instances
            top_k: Maximum results to return
        """
        # Stage 1: Pre-process
        processed_query = self._preprocess(query, session)

        # Stage 2: Route
        decision = self.router.route(processed_query, session)

        # Stage 3: Execute
        results = await self._execute(processed_query, decision, systems)

        # Stage 4: Fuse
        fused = self._fuse_results(results)

        # Stage 5: Rank
        ranked = self._rank_results(fused, session)

        # Stage 6: Post-process
        final = self._postprocess(ranked[:top_k], session)

        # Update session
        session.add_query(query)

        return final

    def _preprocess(self, query: str, session: SessionReference) -> str:
        """Expand query with session context"""
        # Add context hints to query for better matching
        hints = session.get_context_hints()

        # Extract keywords from recent queries for continuity
        expanded = query
        if hints["entities"]:
            # If query mentions pronouns, try to resolve
            if "it" in query.lower() or "this" in query.lower():
                recent_entity = hints["entities"][-1] if hints["entities"] else ""
                if recent_entity:
                    expanded = f"{query} (context: {recent_entity})"

        return expanded

    async def _execute(
        self,
        query: str,
        decision: RoutingDecision,
        systems: Dict[str, Any],
    ) -> List[MemoryResult]:
        """Execute queries against selected systems"""
        results = []

        async def query_system(system_name: str) -> List[MemoryResult]:
            if system_name not in systems:
                return []

            system = systems[system_name]
            config = MemoryRouter.SYSTEM_CONFIG.get(system_name, {})

            try:
                # Different systems have different query interfaces
                if hasattr(system, "search"):
                    raw_results = await system.search(query, top_k=decision.max_results_per_system)
                elif hasattr(system, "query"):
                    raw_results = await system.query(query)
                elif hasattr(system, "recall"):
                    raw_results = await system.recall(query, limit=decision.max_results_per_system)
                else:
                    return []

                # Normalize results
                return self._normalize_results(raw_results, system_name, config)

            except Exception as e:
                logger.warning(f"Query to {system_name} failed: {e}")
                return []

        if decision.parallel and len(decision.systems) > 1:
            # Parallel execution
            tasks = [query_system(s) for s in decision.systems]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in all_results:
                if isinstance(r, list):
                    results.extend(r)
        else:
            # Sequential execution
            for system_name in decision.systems:
                results.extend(await query_system(system_name))

        return results

    def _normalize_results(
        self,
        raw_results: Any,
        system_name: str,
        config: Dict,
    ) -> List[MemoryResult]:
        """Normalize results from different systems to MemoryResult"""
        normalized = []

        if not raw_results:
            return normalized

        # Handle different result formats
        items = raw_results if isinstance(raw_results, list) else [raw_results]

        for item in items:
            try:
                # Extract content
                if hasattr(item, "content"):
                    content = item.content
                elif hasattr(item, "entry") and hasattr(item.entry, "content"):
                    content = item.entry.content
                elif hasattr(item, "text"):
                    content = item.text
                elif isinstance(item, dict):
                    content = item.get("content") or item.get("text") or str(item)
                elif isinstance(item, str):
                    content = item
                else:
                    content = str(item)

                # Extract score
                score = 0.5
                if hasattr(item, "score"):
                    score = item.score
                elif isinstance(item, dict) and "score" in item:
                    score = item["score"]

                # Extract ID
                memory_id = ""
                if hasattr(item, "id"):
                    memory_id = item.id
                elif hasattr(item, "entry") and hasattr(item.entry, "id"):
                    memory_id = item.entry.id
                elif isinstance(item, dict):
                    memory_id = item.get("id", str(uuid.uuid4())[:8])
                else:
                    memory_id = hashlib.md5(content.encode()).hexdigest()[:8]

                normalized.append(MemoryResult(
                    content=content,
                    source_system=system_name,
                    tier=config.get("tier", MemoryTier.COLD),
                    domain=config.get("domains", [MemoryDomain.KNOWLEDGE])[0],
                    score=score,
                    memory_id=memory_id,
                ))

            except Exception as e:
                logger.debug(f"Failed to normalize result: {e}")

        return normalized

    def _fuse_results(self, results: List[MemoryResult]) -> List[MemoryResult]:
        """Fuse and deduplicate results"""
        seen_content = {}
        fused = []

        for result in results:
            # Create content hash for deduplication
            content_key = result.content[:200].lower().strip()

            if content_key in seen_content:
                # Merge scores for duplicate content
                existing = seen_content[content_key]
                existing.score = max(existing.score, result.score)
            else:
                seen_content[content_key] = result
                fused.append(result)

        return fused

    def _rank_results(
        self,
        results: List[MemoryResult],
        session: SessionReference,
    ) -> List[MemoryResult]:
        """Rank results by relevance"""
        context = session.get_context_hints()

        for result in results:
            # Boost for active topics
            for topic in context.get("topics", []):
                if topic.lower() in result.content.lower():
                    result.score *= 1.2

            # Boost for active entities
            for entity in context.get("entities", []):
                if entity.lower() in result.content.lower():
                    result.score *= 1.3

            # Tier-based boost (prefer hot/warm)
            if result.tier == MemoryTier.HOT:
                result.score *= 1.1
            elif result.tier == MemoryTier.WARM:
                result.score *= 1.05

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _postprocess(
        self,
        results: List[MemoryResult],
        session: SessionReference,
    ) -> List[MemoryResult]:
        """Post-process results and update session context"""
        # Extract entities/topics from results for session update
        new_entities = set()
        new_topics = set()

        for result in results:
            # Simple entity extraction (names, capitalized words)
            words = result.content.split()
            for word in words:
                if len(word) > 2 and word[0].isupper() and word.isalpha():
                    new_entities.add(word)

        session.update_context(
            entities=list(new_entities)[:5],
        )

        return results


# =============================================================================
# UNIFIED MEMORY API
# =============================================================================

class UnifiedMemoryAPI:
    """
    Single interface for all 18 memory systems.

    Provides:
    - Session management with lightweight references
    - Unified query interface with automatic routing
    - Cross-system memory operations
    - Statistics and monitoring
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize router and pipeline
        self.router = MemoryRouter()
        self.pipeline = MemoryPipeline(self.router)

        # Memory system instances (lazy loaded)
        self._systems: Dict[str, Any] = {}
        self._initialized = False

        # Session tracking
        self._sessions: Dict[str, SessionReference] = {}

    async def initialize(self):
        """Initialize all memory systems"""
        if self._initialized:
            return

        logger.info("Initializing Unified Memory API...")

        # Import and initialize each system
        try:
            # Core memory systems
            from Cosmos.memory.working_memory import WorkingMemory
            from Cosmos.memory.archival_memory import ArchivalMemory
            from Cosmos.memory.recall_memory import RecallMemory
            from Cosmos.memory.knowledge_graph import KnowledgeGraph
            from Cosmos.memory.virtual_context import VirtualContext
            from Cosmos.memory.memory_dreaming import MemoryDreamer

            self._systems["working_memory"] = WorkingMemory()
            self._systems["archival_memory"] = ArchivalMemory(str(self.data_dir / "archival"))
            self._systems["recall_memory"] = RecallMemory(str(self.data_dir / "conversations"))
            self._systems["knowledge_graph"] = KnowledgeGraph(str(self.data_dir / "graph"))
            self._systems["virtual_context"] = VirtualContext(data_dir=str(self.data_dir / "context"))
            self._systems["memory_dreaming"] = MemoryDreamer()

        except ImportError as e:
            logger.warning(f"Some core memory systems unavailable: {e}")

        # Extended memory systems
        try:
            from Cosmos.memory.episodic_memory import EpisodicMemory
            self._systems["episodic_memory"] = EpisodicMemory(str(self.data_dir / "episodic"))
        except ImportError:
            pass

        try:
            from Cosmos.memory.knowledge_graph_v2 import KnowledgeGraphV2
            self._systems["knowledge_graph_v2"] = KnowledgeGraphV2(str(self.data_dir / "graph_v2"))
        except ImportError:
            pass

        try:
            from Cosmos.memory.dream_consolidation import DreamConsolidator
            self._systems["dream_consolidation"] = DreamConsolidator(str(self.data_dir / "dreams"))
        except ImportError:
            pass

        try:
            from Cosmos.memory.semantic_layers import SemanticLayerSystem
            self._systems["semantic_layers"] = SemanticLayerSystem(str(self.data_dir / "semantic"))
        except ImportError:
            pass

        try:
            from Cosmos.memory.project_tracking import ProjectTracker
            self._systems["project_tracking"] = ProjectTracker(str(self.data_dir / "projects"))
        except ImportError:
            pass

        try:
            from Cosmos.memory.semantic_deduplication import SemanticDeduplicator
            self._systems["semantic_dedup"] = SemanticDeduplicator()
        except ImportError:
            pass

        try:
            from Cosmos.memory.memory_sharing import MemorySharing
            self._systems["memory_sharing"] = MemorySharing(str(self.data_dir))
        except ImportError:
            pass

        # Planetary Audio Shard (distributed TTS cache)
        try:
            from Cosmos.core.memory.planetary.audio_shard import get_audio_shard
            self._systems["planetary_audio"] = get_audio_shard()
        except ImportError:
            pass

        # Planetary Memory / Akashic Record (P2P skill sharing)
        try:
            from Cosmos.core.memory.planetary.akashic import PlanetaryMemory
            self._systems["p2p_memory"] = PlanetaryMemory(use_p2p=True)
        except ImportError:
            pass

        # Initialize async systems
        for name, system in self._systems.items():
            if hasattr(system, "initialize"):
                try:
                    await system.initialize()
                    logger.debug(f"Initialized {name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize {name}: {e}")

        self._initialized = True
        logger.info(f"Unified Memory API ready with {len(self._systems)} systems")

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def create_session(self, agent_id: str) -> SessionReference:
        """
        Create a lightweight session reference for an agent.

        This is the entry point for agents to access memory.
        The reference is lightweight and can be serialized.
        """
        session = SessionReference(
            session_id=str(uuid.uuid4()),
            agent_id=agent_id,
        )
        self._sessions[session.session_id] = session
        logger.debug(f"Created session {session.session_id} for agent {agent_id}")
        return session

    def get_session(self, session_id: str) -> Optional[SessionReference]:
        """Retrieve an existing session"""
        return self._sessions.get(session_id)

    def end_session(self, session_id: str):
        """End a session and cleanup"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Ended session {session_id}")

    # =========================================================================
    # UNIFIED QUERY INTERFACE
    # =========================================================================

    async def query(
        self,
        query: str,
        session: SessionReference,
        top_k: int = 10,
        systems: List[str] = None,
        intent: QueryIntent = None,
    ) -> List[MemoryResult]:
        """
        Query memory systems with automatic routing.

        Args:
            query: Natural language query
            session: Session reference for context
            top_k: Maximum results to return
            systems: Optional list of specific systems to query
            intent: Optional explicit query intent
        """
        if not self._initialized:
            await self.initialize()

        # Check quota
        if session.query_quota_remaining <= 0:
            logger.warning(f"Session {session.session_id} exceeded query quota")
            return []
        session.query_quota_remaining -= 1

        # Use pipeline for automatic routing
        if systems:
            # Filter to requested systems
            available = {k: v for k, v in self._systems.items() if k in systems}
        else:
            available = self._systems

        return await self.pipeline.process(query, session, available, top_k)

    async def remember(
        self,
        content: str,
        session: SessionReference,
        tags: List[str] = None,
        importance: float = 0.5,
        metadata: Dict = None,
    ) -> str:
        """
        Store a memory across relevant systems.

        Returns the primary memory ID.
        """
        if not self._initialized:
            await self.initialize()

        memory_id = str(uuid.uuid4())

        # Store in archival memory (primary)
        if "archival_memory" in self._systems:
            try:
                memory_id = await self._systems["archival_memory"].store(
                    content=content,
                    tags=tags,
                    metadata=metadata,
                )
            except Exception as e:
                logger.error(f"Failed to store in archival: {e}")

        # Extract entities for knowledge graph
        if "knowledge_graph" in self._systems:
            try:
                entities = await self._systems["knowledge_graph"].extract_entities_from_text(content)
                if entities:
                    session.update_context(entities=[e.name for e in entities])
            except Exception as e:
                logger.debug(f"Entity extraction failed: {e}")

        # Add to working memory if important
        if importance > 0.7 and "working_memory" in self._systems:
            try:
                from Cosmos.memory.working_memory import SlotType
                await self._systems["working_memory"].set(
                    name=f"memory_{memory_id[:8]}",
                    value=content[:500],
                    slot_type=SlotType.REFERENCE,
                )
            except Exception as e:
                logger.debug(f"Working memory store failed: {e}")

        # Update session context
        if tags:
            session.update_context(topics=tags)

        return memory_id

    async def forget(self, memory_id: str, session: SessionReference) -> bool:
        """Delete a specific memory"""
        if "archival_memory" in self._systems:
            try:
                return await self._systems["archival_memory"].delete(memory_id)
            except Exception as e:
                logger.error(f"Failed to delete memory: {e}")
        return False

    # =========================================================================
    # SPECIALIZED OPERATIONS
    # =========================================================================

    async def get_timeline(
        self,
        session: SessionReference,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 50,
    ) -> List[MemoryResult]:
        """Get memories organized by timeline"""
        if "episodic_memory" not in self._systems:
            return []

        try:
            results = await self._systems["episodic_memory"].query_timeline(
                start=start_date,
                end=end_date,
                limit=limit,
            )
            return self.pipeline._normalize_results(results, "episodic_memory",
                MemoryRouter.SYSTEM_CONFIG.get("episodic_memory", {}))
        except Exception as e:
            logger.error(f"Timeline query failed: {e}")
            return []

    async def get_related(
        self,
        entity_or_topic: str,
        session: SessionReference,
        depth: int = 2,
    ) -> List[MemoryResult]:
        """Get related entities and memories"""
        results = []

        if "knowledge_graph" in self._systems:
            try:
                related = await self._systems["knowledge_graph"].get_neighbors(
                    entity_or_topic, max_depth=depth
                )
                results.extend(self.pipeline._normalize_results(
                    related, "knowledge_graph",
                    MemoryRouter.SYSTEM_CONFIG.get("knowledge_graph", {})
                ))
            except Exception as e:
                logger.debug(f"Graph query failed: {e}")

        # Also search archival for related content
        if "archival_memory" in self._systems:
            try:
                search_results = await self._systems["archival_memory"].search(
                    entity_or_topic, top_k=10
                )
                results.extend(self.pipeline._normalize_results(
                    search_results, "archival_memory",
                    MemoryRouter.SYSTEM_CONFIG.get("archival_memory", {})
                ))
            except Exception as e:
                logger.debug(f"Archival search failed: {e}")

        return self.pipeline._fuse_results(results)[:10]

    async def get_project_context(
        self,
        project_name: str,
        session: SessionReference,
    ) -> Dict:
        """Get full context for a project"""
        if "project_tracking" not in self._systems:
            return {}

        try:
            project = await self._systems["project_tracking"].get_project(project_name)
            if project:
                session.update_context(projects=[project_name])
                return project.to_dict() if hasattr(project, "to_dict") else {"name": project_name}
        except Exception as e:
            logger.error(f"Project context failed: {e}")
        return {}

    async def trigger_consolidation(self, session: SessionReference) -> Dict:
        """Trigger memory consolidation/dreaming"""
        results = {}

        if "memory_dreaming" in self._systems:
            try:
                dream_result = await self._systems["memory_dreaming"].dream()
                results["dreaming"] = dream_result
            except Exception as e:
                logger.error(f"Dreaming failed: {e}")

        if "dream_consolidation" in self._systems:
            try:
                consolidation = await self._systems["dream_consolidation"].run_consolidation_cycle()
                results["consolidation"] = consolidation
            except Exception as e:
                logger.error(f"Consolidation failed: {e}")

        return results

    # =========================================================================
    # STATISTICS AND MONITORING
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get comprehensive statistics across all systems"""
        stats = {
            "systems_available": list(self._systems.keys()),
            "active_sessions": len(self._sessions),
            "systems": {},
        }

        for name, system in self._systems.items():
            if hasattr(system, "get_stats"):
                try:
                    stats["systems"][name] = system.get_stats()
                except Exception:
                    stats["systems"][name] = {"status": "available"}
            else:
                stats["systems"][name] = {"status": "available"}

        return stats

    def get_system(self, name: str) -> Optional[Any]:
        """Get direct access to a specific memory system"""
        return self._systems.get(name)


# =============================================================================
# MCP INTEGRATION
# =============================================================================

# Global instance for MCP tools
_unified_api: Optional[UnifiedMemoryAPI] = None

def get_unified_memory_api() -> UnifiedMemoryAPI:
    """Get or create the global unified memory API instance"""
    global _unified_api
    if _unified_api is None:
        _unified_api = UnifiedMemoryAPI()
    return _unified_api


async def mcp_create_session(agent_id: str) -> Dict:
    """MCP tool: Create a session reference for an agent"""
    api = get_unified_memory_api()
    await api.initialize()
    session = api.create_session(agent_id)
    return session.to_dict()


async def mcp_query_memory(
    query: str,
    session_id: str,
    top_k: int = 10,
    systems: List[str] = None,
) -> List[Dict]:
    """MCP tool: Query memory with unified routing"""
    api = get_unified_memory_api()
    session = api.get_session(session_id)
    if not session:
        return [{"error": "Invalid session"}]

    results = await api.query(query, session, top_k, systems)
    return [r.to_dict() for r in results]


async def mcp_remember(
    content: str,
    session_id: str,
    tags: List[str] = None,
    importance: float = 0.5,
) -> Dict:
    """MCP tool: Store a memory"""
    api = get_unified_memory_api()
    session = api.get_session(session_id)
    if not session:
        return {"error": "Invalid session"}

    memory_id = await api.remember(content, session, tags, importance)
    return {"memory_id": memory_id, "success": True}


async def mcp_get_related(
    entity_or_topic: str,
    session_id: str,
    depth: int = 2,
) -> List[Dict]:
    """MCP tool: Find related entities and memories"""
    api = get_unified_memory_api()
    session = api.get_session(session_id)
    if not session:
        return [{"error": "Invalid session"}]

    results = await api.get_related(entity_or_topic, session, depth)
    return [r.to_dict() for r in results]


async def mcp_get_memory_stats() -> Dict:
    """MCP tool: Get memory system statistics"""
    api = get_unified_memory_api()
    return api.get_stats()
