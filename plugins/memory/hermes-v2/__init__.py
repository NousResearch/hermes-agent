"""
Hermes Memory V2 Plugin — SQLite FTS5 memory provider for the pluggable memory architecture.

Implements the MemoryProvider interface with:
- SQLite FTS5 full-text search + optional embedding-based hybrid search
- Tiered memory lifecycle (active -> archived -> superseded)
- Automatic memory extraction from conversations
- Session notes (9-section structured template)
- Periodic consolidation (merge, update, archive)
- Memory tools: hermes_v2_search, hermes_v2_add, hermes_v2_remove, hermes_v2_stats
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


class HermesV2MemoryProvider(MemoryProvider):
    """SQLite FTS5 memory provider with tiered lifecycle and hybrid search."""

    def __init__(self):
        self._engine = None
        self._session_memory = None
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._config: dict = {}
        self._initialized: bool = False
        self._agent_context: str = "primary"
        self._turn_count: int = 0
        self._extract_interval: int = 3  # Extract every N turns

    @property
    def name(self) -> str:
        return "hermes-v2"

    # -- Core lifecycle ------------------------------------------------------

    def is_available(self) -> bool:
        """Check if hermes-v2 is the configured memory provider."""
        try:
            from hermes_config import config
            provider = config.get("memory", {}).get("provider", "")
            return provider == "hermes-v2"
        except ImportError:
            pass

        # Fallback: check environment variable
        return os.environ.get("HERMES_MEMORY_PROVIDER", "") == "hermes-v2"

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize SQLite DB and memory engine."""
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", str(Path.home() / ".hermes"))
        self._agent_context = kwargs.get("agent_context", "primary")

        # Load memory-specific config
        try:
            from hermes_config import config
            self._config = config.get("memory", {})
        except ImportError:
            self._config = {}

        self._extract_interval = self._config.get("extract_interval", 3)

        # Initialize the SQLite memory engine
        from .memory_engine import MemoryEngine

        db_dir = Path(self._hermes_home) / "memories"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / "memory.db"

        self._engine = MemoryEngine(db_path=db_path, config=self._config)

        # Migrate from flat files if this is first use
        try:
            result = self._engine.migrate_from_flat_files(memory_dir=db_dir)
            if result.get("migrated"):
                logger.info(
                    "Migrated %d entries from flat files to SQLite",
                    result.get("count", 0),
                )
        except Exception as e:
            logger.debug("Flat file migration skipped: %s", e)

        # Capture frozen snapshot for system prompt
        self._engine.snapshot()

        # Initialize session memory (structured notes)
        from .session_memory import SessionMemory

        session_config = self._config.get("session_memory", {})
        self._session_memory = SessionMemory(config=session_config)

        # Load extractor cursor
        from .extractor import get_extractor_state

        get_extractor_state().load_cursor(self._engine)

        self._initialized = True
        logger.info(
            "Hermes V2 memory initialized: %s (session=%s)",
            db_path,
            session_id[:8] if session_id else "none",
        )

    def system_prompt_block(self) -> str:
        """Return memory context for the system prompt."""
        if not self._engine:
            return ""

        parts = []

        # Memory entries (from frozen snapshot)
        mem_block = self._engine.get_snapshot("memory")
        if mem_block:
            parts.append(mem_block)

        user_block = self._engine.get_snapshot("user")
        if user_block:
            parts.append(user_block)

        # Session notes (if initialized)
        if self._session_memory and self._session_memory.is_initialized:
            notes = self._session_memory.get_summary()
            if notes and notes.strip():
                parts.append(
                    f"{'═' * 46}\n"
                    f"SESSION NOTES (structured context from this session)\n"
                    f"{'═' * 46}\n"
                    f"{notes}\n"
                )

        return "\n\n".join(parts)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Search memories relevant to the user's query."""
        if not self._engine or not query.strip():
            return ""

        results = self._engine.search(
            query,
            limit=5,
            min_relevance=0.1,
        )

        if not results:
            return ""

        # Reinforce accessed memories
        for mem in results:
            try:
                self._engine.reinforce(mem["id"])
            except Exception:
                pass

        # Format results
        lines = []
        for mem in results:
            score = mem.get("relevance_score", 0)
            content = mem["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            tag = mem.get("type", "gen")[:4]
            lines.append(f"  [{tag}|{score:.2f}] {content}")

        if not lines:
            return ""

        return (
            f"[hermes-v2] Recalled {len(lines)} relevant memories:\n"
            + "\n".join(lines)
        )

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """Extract and store memories from a completed turn."""
        if not self._engine or self._agent_context != "primary":
            return

        self._turn_count += 1

        # Run extraction every N turns (background thread)
        if self._turn_count % self._extract_interval == 0:
            try:
                from .extractor import extract_memories_background, get_extractor_state

                # Build a simple messages list for the extractor.
                # Reset cursor to -1 so the extractor processes this fresh
                # slice instead of skipping it (cursor tracking assumes a
                # growing list, but we pass a per-turn 2-element list).
                get_extractor_state().last_extracted_message_index = -1

                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                extract_memories_background(
                    recent_messages=messages,
                    memory_store=self,  # We expose .engine property
                    session_id=self._session_id,
                )
            except Exception as e:
                logger.debug("Memory extraction skipped: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Expose memory tools: search, add, remove, stats."""
        return [
            {
                "name": "hermes_v2_search",
                "description": (
                    "Search your long-term memory for relevant facts, preferences, "
                    "corrections, and context. Use before making assumptions about "
                    "the user or project."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query",
                        },
                        "target": {
                            "type": "string",
                            "enum": ["memory", "user"],
                            "description": "Search scope: 'memory' for project/general facts, 'user' for user profile",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "hermes_v2_add",
                "description": (
                    "Save a durable fact to long-term memory. Use for preferences, "
                    "corrections, project facts, and important observations that "
                    "should persist across sessions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The fact or observation to remember",
                        },
                        "target": {
                            "type": "string",
                            "enum": ["memory", "user"],
                            "description": "'memory' for project/general facts, 'user' for user profile",
                            "default": "memory",
                        },
                        "type": {
                            "type": "string",
                            "enum": [
                                "general",
                                "preference",
                                "correction",
                                "project",
                                "reference",
                            ],
                            "description": "Memory type for categorization",
                            "default": "general",
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "hermes_v2_remove",
                "description": "Remove a memory by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The memory ID to remove",
                        },
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "hermes_v2_stats",
                "description": "Show memory statistics: counts, tiers, embeddings.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "memory_feedback",
                "description": (
                    "Rate a memory after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
                    "This trains the memory — good memories rise, bad memories sink."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["helpful", "unhelpful"],
                            "description": "Whether the memory was helpful or unhelpful",
                        },
                        "memory_id": {
                            "type": "string",
                            "description": "The memory ID to rate.",
                        },
                    },
                    "required": ["action", "memory_id"],
                },
            },
            {
                "name": "memory_contradictions",
                "description": (
                    "Find potentially contradictory memories — pairs that share entities "
                    "(same subject) but have divergent content (different claims). "
                    "Useful for identifying outdated or conflicting information."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": "Optional: only check memories related to this entity",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max contradictions to return (default: 20)",
                            "default": 20,
                        },
                    },
                },
            },
        ]

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs
    ) -> str:
        """Dispatch memory tool calls."""
        if not self._engine:
            return json.dumps({"error": "Memory engine not initialized"})

        try:
            if tool_name == "hermes_v2_search":
                results = self._engine.search(
                    query=args.get("query", ""),
                    target=args.get("target"),
                    limit=args.get("limit", 10),
                )
                # Reinforce accessed memories
                for mem in results:
                    try:
                        self._engine.reinforce(mem["id"])
                    except Exception:
                        pass

                # Format for display
                formatted = []
                for mem in results:
                    formatted.append({
                        "id": mem["id"][:8],
                        "content": mem["content"],
                        "type": mem.get("type", "general"),
                        "target": mem.get("target", "memory"),
                        "relevance": mem.get("relevance_score", 0),
                        "strength": mem.get("strength", 1.0),
                    })
                return json.dumps({
                    "results": formatted,
                    "count": len(formatted),
                })

            elif tool_name == "hermes_v2_add":
                result = self._engine.add(
                    content=args.get("content", ""),
                    target=args.get("target", "memory"),
                    type=args.get("type", "general"),
                    source="agent",
                    session_id=self._session_id,
                )
                # Notify extractor that agent wrote memory this turn
                try:
                    from .extractor import get_extractor_state
                    get_extractor_state().mark_agent_wrote_memory()
                except Exception:
                    pass
                return json.dumps(result)

            elif tool_name == "hermes_v2_remove":
                result = self._engine.remove(args.get("memory_id", ""))
                return json.dumps(result)

            elif tool_name == "hermes_v2_stats":
                stats = self._engine.stats()
                return json.dumps(stats, indent=2)

            elif tool_name == "memory_feedback":
                return self._handle_memory_feedback(args)

            elif tool_name == "memory_contradictions":
                return self._handle_memory_contradictions(args)

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            logger.error("hermes-v2 tool call failed (%s): %s", tool_name, e)
            return json.dumps({"error": str(e)})

    def _handle_memory_feedback(self, args: dict) -> str:
        """Handle memory_feedback tool call — rate a memory as helpful/unhelpful."""
        try:
            memory_id = args["memory_id"]
            helpful = args["action"] == "helpful"
            result = self._engine.record_feedback(memory_id, helpful=helpful)
            return json.dumps(result)
        except KeyError as exc:
            return json.dumps({"error": f"Missing required argument: {exc}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _handle_memory_contradictions(self, args: dict) -> str:
        """Handle memory_contradictions tool call — find contradictory memories."""
        try:
            entity_name = args.get("entity_name")
            limit = args.get("limit", 20)
            contradictions = self._engine.find_contradictions(
                entity_name=entity_name,
                limit=limit,
            )
            return json.dumps({
                "contradictions": contradictions,
                "count": len(contradictions),
            })
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def shutdown(self) -> None:
        """Close SQLite connections."""
        if self._engine:
            try:
                self._engine.close()
            except Exception as e:
                logger.debug("Engine close error: %s", e)
            self._engine = None
        self._initialized = False

    # -- Optional hooks ------------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Clear per-turn extraction flags."""
        try:
            from .extractor import get_extractor_state
            get_extractor_state().clear_agent_wrote_memory()
        except Exception:
            pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Run consolidation and increment session counter."""
        if not self._engine or self._agent_context != "primary":
            return

        # Increment session count for consolidation gating
        try:
            from .consolidator import increment_session_count
            increment_session_count(self._engine)
        except Exception as e:
            logger.debug("Session count increment failed: %s", e)

        # Attempt consolidation (respects gates)
        try:
            from .consolidator import consolidate_memories

            # Try to get auxiliary_client for LLM-powered consolidation
            aux_client = None
            try:
                from agent.auxiliary_client import AuxiliaryClient
                aux_client = AuxiliaryClient()
            except ImportError:
                pass

            if aux_client:
                result = consolidate_memories(
                    engine=self._engine,
                    auxiliary_client=aux_client,
                    config=self._config,
                )
                if result.get("consolidated"):
                    logger.info(
                        "Consolidation: %d actions, %d archived",
                        result.get("actions", 0),
                        result.get("archived", 0),
                    )
        except Exception as e:
            logger.debug("Consolidation failed: %s", e)

        # Purge dead memories (superseded/archived > 30 days)
        try:
            purged = self._engine.purge_dead()
            if purged:
                logger.info("Purged %d dead memories", purged)
        except Exception as e:
            logger.debug("Purge failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract memories from messages about to be compressed."""
        if not self._engine or self._agent_context != "primary":
            return ""

        try:
            from .extractor import _do_extraction

            # Try to get auxiliary_client
            aux_client = None
            try:
                from agent.auxiliary_client import AuxiliaryClient
                aux_client = AuxiliaryClient()
            except ImportError:
                pass

            if aux_client:
                _do_extraction(
                    recent_messages=messages,
                    engine=self._engine,
                    auxiliary_client=aux_client,
                    session_id=self._session_id,
                )
                return "[hermes-v2] Extracted memories from compressed context."
        except Exception as e:
            logger.debug("Pre-compress extraction failed: %s", e)

        return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to the SQLite store."""
        if not self._engine:
            return

        try:
            if action == "add":
                self._engine.add(
                    content=content,
                    target=target,
                    type="general",
                    source="builtin_mirror",
                    session_id=self._session_id,
                )
            elif action == "remove":
                # Search for matching content and remove
                results = self._engine.search_fts(content, target=target, limit=1)
                if results:
                    self._engine.remove(results[0]["id"])
        except Exception as e:
            logger.debug("Mirror write failed: %s", e)

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Store delegation results as memories."""
        if not self._engine or self._agent_context != "primary":
            return

        try:
            # Extract key facts from delegation result
            summary = result[:500] if len(result) > 500 else result
            self._engine.add(
                content=f"Delegated task result: {task[:100]}... → {summary}",
                target="memory",
                type="project",
                source="delegation",
                session_id=self._session_id,
            )
        except Exception as e:
            logger.debug("Delegation memory failed: %s", e)

    # -- Properties for internal access --------------------------------------

    @property
    def engine(self):
        """Access the underlying MemoryEngine (for extractor compatibility)."""
        return self._engine
