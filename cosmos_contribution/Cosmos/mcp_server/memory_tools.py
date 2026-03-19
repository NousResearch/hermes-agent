"""
cosmos Memory Tools - MCP Tool Implementations for Memory Operations

Provides detailed memory access capabilities:
- Advanced search with filters
- Memory graph queries
- Working memory access
- Memory statistics
- Unified Memory API with session references
"""

from dataclasses import dataclass
from typing import Optional, Any, List, Dict
from datetime import datetime

from loguru import logger

# Import unified memory system
try:
    from Cosmos.memory.unified_memory import (
        UnifiedMemoryAPI,
        get_unified_memory_api,
        SessionReference,
        MemoryResult,
        QueryIntent,
    )
    UNIFIED_MEMORY_AVAILABLE = True
except ImportError:
    UNIFIED_MEMORY_AVAILABLE = False
    logger.warning("Unified memory API not available")


@dataclass
class MemoryToolResult:
    """Result from a memory tool operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict = None

    def to_dict(self) -> dict:
        result = {"success": self.success}
        if self.data is not None:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class MemoryTools:
    """
    Memory tool implementations for the MCP server.

    Provides advanced memory operations beyond basic remember/recall.
    """

    def __init__(self, memory_system):
        self.memory = memory_system

    async def remember_with_context(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
        context: Optional[dict] = None,
        extract_entities: bool = True,
    ) -> MemoryToolResult:
        """
        Store memory with additional context and entity extraction.
        """
        try:
            # Add context to content if provided
            full_content = content
            if context:
                context_str = "\n".join(f"[{k}]: {v}" for k, v in context.items())
                full_content = f"{content}\n\nContext:\n{context_str}"

            memory_id = await self.memory.remember(
                content=full_content,
                tags=tags,
                importance=importance,
                extract_entities=extract_entities,
            )

            return MemoryToolResult(
                success=True,
                data={"memory_id": memory_id},
                metadata={"extracted_entities": extract_entities},
            )

        except Exception as e:
            logger.error(f"Remember with context failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def advanced_recall(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.3,
        tags_filter: Optional[list[str]] = None,
        time_range: Optional[tuple[str, str]] = None,
        source_filter: Optional[list[str]] = None,
    ) -> MemoryToolResult:
        """
        Advanced memory search with multiple filters.
        """
        try:
            results = await self.memory.recall(
                query=query,
                top_k=limit,
                min_score=min_score,
            )

            # Apply additional filters
            filtered = []
            for r in results:
                # Tag filter
                if tags_filter:
                    if "tags" in r.metadata:
                        if not any(t in r.metadata["tags"] for t in tags_filter):
                            continue

                # Source filter
                if source_filter and r.source not in source_filter:
                    continue

                filtered.append({
                    "content": r.content,
                    "source": r.source,
                    "score": r.score,
                    "metadata": r.metadata,
                })

            return MemoryToolResult(
                success=True,
                data={"count": len(filtered), "results": filtered},
            )

        except Exception as e:
            logger.error(f"Advanced recall failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def query_knowledge_graph(
        self,
        query: str,
        max_entities: int = 10,
        max_hops: int = 2,
    ) -> MemoryToolResult:
        """
        Query the knowledge graph for entities and relationships.
        """
        try:
            result = await self.memory.knowledge_graph.query(
                query=query,
                max_entities=max_entities,
                max_hops=max_hops,
            )

            return MemoryToolResult(
                success=True,
                data={
                    "entities": [
                        {
                            "name": e.name,
                            "type": e.entity_type,
                            "mentions": e.mention_count,
                        }
                        for e in result.entities
                    ],
                    "relationships": [
                        {
                            "source": r.source_id,
                            "target": r.target_id,
                            "type": r.relation_type,
                        }
                        for r in result.relationships
                    ],
                    "paths": result.paths,
                },
            )

        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def get_working_memory(self, slot_name: Optional[str] = None) -> MemoryToolResult:
        """
        Access working memory slots.
        """
        try:
            if slot_name:
                value = await self.memory.get_working_memory(slot_name)
                return MemoryToolResult(
                    success=True,
                    data={slot_name: value},
                )
            else:
                # Return all slots
                status = self.memory.working_memory.get_status()
                return MemoryToolResult(
                    success=True,
                    data=status,
                )

        except Exception as e:
            logger.error(f"Get working memory failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def set_working_memory(
        self,
        name: str,
        value: Any,
        slot_type: str = "scratch",
    ) -> MemoryToolResult:
        """
        Set a value in working memory.
        """
        try:
            from Cosmos.memory.working_memory import SlotType

            type_map = {
                "text": SlotType.TEXT,
                "code": SlotType.CODE,
                "data": SlotType.DATA,
                "task": SlotType.TASK,
                "reference": SlotType.REFERENCE,
                "scratch": SlotType.SCRATCH,
            }

            await self.memory.set_working_memory(
                name=name,
                value=value,
                slot_type=type_map.get(slot_type, SlotType.SCRATCH),
            )

            return MemoryToolResult(
                success=True,
                data={"slot": name, "type": slot_type},
            )

        except Exception as e:
            logger.error(f"Set working memory failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def get_memory_stats(self) -> MemoryToolResult:
        """
        Get comprehensive memory statistics.
        """
        try:
            stats = self.memory.get_stats()
            summary = await self.memory.get_memory_summary()

            return MemoryToolResult(
                success=True,
                data={
                    "stats": stats,
                    "summary": summary,
                },
            )

        except Exception as e:
            logger.error(f"Get memory stats failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def trigger_dreaming(self) -> MemoryToolResult:
        """
        Manually trigger memory consolidation (dreaming).
        """
        try:
            result = await self.memory.trigger_dream()

            return MemoryToolResult(
                success=True,
                data={
                    "memories_processed": result.memories_processed,
                    "clusters_formed": result.clusters_formed,
                    "insights": result.insights_generated,
                    "consolidation_score": result.consolidation_score,
                },
            )

        except Exception as e:
            logger.error(f"Trigger dreaming failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[dict] = None,
    ) -> MemoryToolResult:
        """
        Add an entity to the knowledge graph.
        """
        try:
            entity = await self.memory.add_entity(
                name=name,
                entity_type=entity_type,
                properties=properties,
            )

            return MemoryToolResult(
                success=True,
                data={
                    "entity_id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type,
                },
            )

        except Exception as e:
            logger.error(f"Add entity failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def link_entities(
        self,
        source: str,
        target: str,
        relation_type: str,
    ) -> MemoryToolResult:
        """
        Create a relationship between entities.
        """
        try:
            await self.memory.link_entities(
                source=source,
                target=target,
                relation_type=relation_type,
            )

            return MemoryToolResult(
                success=True,
                data={
                    "source": source,
                    "target": target,
                    "relation": relation_type,
                },
            )

        except Exception as e:
            logger.error(f"Link entities failed: {e}")
            return MemoryToolResult(success=False, error=str(e))


# =============================================================================
# UNIFIED MEMORY TOOLS - Integrates all 18 memory systems
# =============================================================================

class UnifiedMemoryTools:
    """
    Unified memory tools for the MCP server.

    Provides session-based access to all 18 memory systems through
    a single interface with automatic query routing.

    Key features:
    - Lightweight session references for agents
    - Automatic routing to optimal memory systems
    - Cross-system memory operations
    - Context-aware query expansion
    """

    def __init__(self):
        self._api: Optional[UnifiedMemoryAPI] = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Lazily initialize the unified API"""
        if not self._initialized and UNIFIED_MEMORY_AVAILABLE:
            self._api = get_unified_memory_api()
            await self._api.initialize()
            self._initialized = True

    async def create_session(self, agent_id: str) -> MemoryToolResult:
        """
        Create a lightweight session reference for an agent.

        This is the entry point for agents to access the unified memory system.
        The returned session ID should be stored and passed to all subsequent calls.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Session reference with ID, context hints, and quota info
        """
        try:
            await self._ensure_initialized()

            if not UNIFIED_MEMORY_AVAILABLE or not self._api:
                return MemoryToolResult(
                    success=False,
                    error="Unified memory API not available"
                )

            session = self._api.create_session(agent_id)

            return MemoryToolResult(
                success=True,
                data={
                    "session_id": session.session_id,
                    "agent_id": session.agent_id,
                    "created_at": session.created_at.isoformat(),
                    "query_quota": session.query_quota_remaining,
                    "context_hints": session.get_context_hints(),
                },
                metadata={"systems_available": list(self._api._systems.keys())},
            )

        except Exception as e:
            logger.error(f"Create session failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def unified_query(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
        systems: Optional[List[str]] = None,
        intent: Optional[str] = None,
    ) -> MemoryToolResult:
        """
        Query the unified memory system with automatic routing.

        The query is automatically routed to the most relevant memory systems
        based on the query intent and session context.

        Args:
            query: Natural language query
            session_id: Session ID from create_session
            top_k: Maximum number of results
            systems: Optional list of specific systems to query
            intent: Optional explicit intent (recall, search, relate, timeline, project, synthesis)

        Returns:
            List of memory results from across all relevant systems
        """
        try:
            await self._ensure_initialized()

            if not UNIFIED_MEMORY_AVAILABLE or not self._api:
                return MemoryToolResult(
                    success=False,
                    error="Unified memory API not available"
                )

            session = self._api.get_session(session_id)
            if not session:
                return MemoryToolResult(
                    success=False,
                    error=f"Invalid session ID: {session_id}"
                )

            # Parse intent if provided
            query_intent = None
            if intent:
                try:
                    query_intent = QueryIntent(intent)
                except ValueError:
                    pass

            results = await self._api.query(
                query=query,
                session=session,
                top_k=top_k,
                systems=systems,
                intent=query_intent,
            )

            return MemoryToolResult(
                success=True,
                data={
                    "count": len(results),
                    "results": [r.to_dict() for r in results],
                    "context_hints": session.get_context_hints(),
                },
                metadata={
                    "quota_remaining": session.query_quota_remaining,
                },
            )

        except Exception as e:
            logger.error(f"Unified query failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def unified_remember(
        self,
        content: str,
        session_id: str,
        tags: Optional[List[str]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> MemoryToolResult:
        """
        Store a memory across relevant systems.

        The memory is stored in archival memory and optionally:
        - Entities are extracted for the knowledge graph
        - Added to working memory if importance > 0.7
        - Session context is updated with extracted topics

        Args:
            content: Memory content to store
            session_id: Session ID from create_session
            tags: Optional tags for categorization
            importance: Importance score 0.0-1.0
            metadata: Optional additional metadata

        Returns:
            Memory ID and storage confirmation
        """
        try:
            await self._ensure_initialized()

            if not UNIFIED_MEMORY_AVAILABLE or not self._api:
                return MemoryToolResult(
                    success=False,
                    error="Unified memory API not available"
                )

            session = self._api.get_session(session_id)
            if not session:
                return MemoryToolResult(
                    success=False,
                    error=f"Invalid session ID: {session_id}"
                )

            memory_id = await self._api.remember(
                content=content,
                session=session,
                tags=tags,
                importance=importance,
                metadata=metadata,
            )

            return MemoryToolResult(
                success=True,
                data={
                    "memory_id": memory_id,
                    "importance": importance,
                    "tags": tags or [],
                },
            )

        except Exception as e:
            logger.error(f"Unified remember failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def get_related(
        self,
        entity_or_topic: str,
        session_id: str,
        depth: int = 2,
    ) -> MemoryToolResult:
        """
        Find related entities and memories.

        Queries the knowledge graph and archival memory to find
        entities and content related to the given topic.

        Args:
            entity_or_topic: Entity name or topic to find relations for
            session_id: Session ID from create_session
            depth: Maximum relationship depth (1-3)

        Returns:
            List of related entities and memories
        """
        try:
            await self._ensure_initialized()

            if not UNIFIED_MEMORY_AVAILABLE or not self._api:
                return MemoryToolResult(
                    success=False,
                    error="Unified memory API not available"
                )

            session = self._api.get_session(session_id)
            if not session:
                return MemoryToolResult(
                    success=False,
                    error=f"Invalid session ID: {session_id}"
                )

            results = await self._api.get_related(entity_or_topic, session, depth)

            return MemoryToolResult(
                success=True,
                data={
                    "count": len(results),
                    "related": [r.to_dict() for r in results],
                },
            )

        except Exception as e:
            logger.error(f"Get related failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def get_timeline(
        self,
        session_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50,
    ) -> MemoryToolResult:
        """
        Get memories organized by timeline.

        Queries the episodic memory system to retrieve time-ordered
        events and interactions.

        Args:
            session_id: Session ID from create_session
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format)
            limit: Maximum number of events

        Returns:
            Timeline of memory events
        """
        try:
            await self._ensure_initialized()

            if not UNIFIED_MEMORY_AVAILABLE or not self._api:
                return MemoryToolResult(
                    success=False,
                    error="Unified memory API not available"
                )

            session = self._api.get_session(session_id)
            if not session:
                return MemoryToolResult(
                    success=False,
                    error=f"Invalid session ID: {session_id}"
                )

            # Parse dates if provided
            start = datetime.fromisoformat(start_date) if start_date else None
            end = datetime.fromisoformat(end_date) if end_date else None

            results = await self._api.get_timeline(session, start, end, limit)

            return MemoryToolResult(
                success=True,
                data={
                    "count": len(results),
                    "timeline": [r.to_dict() for r in results],
                },
            )

        except Exception as e:
            logger.error(f"Get timeline failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def get_project_context(
        self,
        project_name: str,
        session_id: str,
    ) -> MemoryToolResult:
        """
        Get full context for a project.

        Retrieves project details, tasks, milestones, and related
        memories from the project tracking system.

        Args:
            project_name: Name of the project
            session_id: Session ID from create_session

        Returns:
            Project context with tasks and related memories
        """
        try:
            await self._ensure_initialized()

            if not UNIFIED_MEMORY_AVAILABLE or not self._api:
                return MemoryToolResult(
                    success=False,
                    error="Unified memory API not available"
                )

            session = self._api.get_session(session_id)
            if not session:
                return MemoryToolResult(
                    success=False,
                    error=f"Invalid session ID: {session_id}"
                )

            context = await self._api.get_project_context(project_name, session)

            return MemoryToolResult(
                success=True,
                data=context,
            )

        except Exception as e:
            logger.error(f"Get project context failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def trigger_consolidation(self, session_id: str) -> MemoryToolResult:
        """
        Trigger memory consolidation/dreaming.

        Runs the memory dreaming and consolidation systems to:
        - Cluster related memories
        - Generate insights
        - Prune low-value memories
        - Create abstractions

        Args:
            session_id: Session ID from create_session

        Returns:
            Consolidation results including insights generated
        """
        try:
            await self._ensure_initialized()

            if not UNIFIED_MEMORY_AVAILABLE or not self._api:
                return MemoryToolResult(
                    success=False,
                    error="Unified memory API not available"
                )

            session = self._api.get_session(session_id)
            if not session:
                return MemoryToolResult(
                    success=False,
                    error=f"Invalid session ID: {session_id}"
                )

            results = await self._api.trigger_consolidation(session)

            return MemoryToolResult(
                success=True,
                data=results,
            )

        except Exception as e:
            logger.error(f"Trigger consolidation failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def get_unified_stats(self) -> MemoryToolResult:
        """
        Get comprehensive statistics across all 18 memory systems.

        Returns:
            Statistics including counts, health, and performance metrics
        """
        try:
            await self._ensure_initialized()

            if not UNIFIED_MEMORY_AVAILABLE or not self._api:
                return MemoryToolResult(
                    success=False,
                    error="Unified memory API not available"
                )

            stats = self._api.get_stats()

            return MemoryToolResult(
                success=True,
                data=stats,
            )

        except Exception as e:
            logger.error(f"Get unified stats failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def end_session(self, session_id: str) -> MemoryToolResult:
        """
        End a session and release resources.

        Args:
            session_id: Session ID to end

        Returns:
            Confirmation of session end
        """
        try:
            await self._ensure_initialized()

            if not UNIFIED_MEMORY_AVAILABLE or not self._api:
                return MemoryToolResult(
                    success=False,
                    error="Unified memory API not available"
                )

            self._api.end_session(session_id)

            return MemoryToolResult(
                success=True,
                data={"session_id": session_id, "ended": True},
            )

        except Exception as e:
            logger.error(f"End session failed: {e}")
            return MemoryToolResult(success=False, error=str(e))


# Global unified memory tools instance
_unified_tools: Optional[UnifiedMemoryTools] = None

def get_unified_memory_tools() -> UnifiedMemoryTools:
    """Get or create the global unified memory tools instance"""
    global _unified_tools
    if _unified_tools is None:
        _unified_tools = UnifiedMemoryTools()
    return _unified_tools
