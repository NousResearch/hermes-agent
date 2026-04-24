"""Episodic Memory Provider — plugs the episodic store into the Hermes agent loop.

Implements the MemoryProvider ABC from agent/memory_provider.py.
This is the glue layer between the agent's turn lifecycle and our SQLite/JSONL store.

Phase 3 additions:
  - Extraction: every N turns, pull structured facts via GPT-5.4-mini
  - Merge: ADD/UPDATE/DELETE/NOOP dedup at session end
  - Compress: D0 episode compression at session end
  - Context injection: token-budgeted context via context_injector
  - Recall tools: memory_grep, memory_expand, memory_describe
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from memory.config import (
    ENABLE_LLM_EXTRACTION,
    ENABLE_SESSION_JOURNAL,
    EXTRACT_BATCH_SIZE,
    HEALTH_CHECK_INTERVAL,
    MAX_MEMORY_INJECTION_TOKENS,
    TOP_EPISODES,
    TOP_ENTITIES,
)
from memory.context_injector import assemble_context, assemble_recent_context
from memory.extraction import extract_from_turns
from memory.episodic_store import EpisodicStore
from memory.session_writer import append_turn_jsonl

logger = logging.getLogger(__name__)


class EpisodicMemoryProvider(MemoryProvider):
    """Persistent episodic memory for the Hermes agent.

    Lifecycle:
      1. initialize() — open store, run health check, ensure session
      2. on_turn_start() — append turns to JSONL + SQLite, trigger extraction every N turns
      3. sync_turn() — persist completed turn (called by agent loop)
      4. prefetch() — search for relevant episodes/entities before each API call
      5. on_session_end() — trigger merge/distill
      6. shutdown() — close store
    """

    def __init__(self):
        self._store: Optional[EpisodicStore] = None
        self._session_id: Optional[str] = None
        self._platform: str = ""
        self._turn_count: int = 0
        self._turns_since_extract: int = 0
        self._turn_ids: List[int] = []  # IDs of turns since last extraction
        self._available: bool = False
        self._health_warned: bool = False
        self._prefetch_cache: str = ""
        self._pending_extractions: List[Dict[str, Any]] = []  # Cached extractions for merge

    # ── Core lifecycle ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "episodic"

    def is_available(self) -> bool:
        """Check if the episodic memory store is accessible.

        Unlike Hindsight, we fail LOUDLY — if anything is wrong, we tell
        the user immediately instead of silently disabling.
        """
        try:
            self._store = EpisodicStore()
            status = self._store.health_check()
            if status["round_trip"]:
                self._available = True
                return True
            else:
                logger.error(
                    "Episodic memory health check FAILED: %s. "
                    "Memory will be unavailable for this session.",
                    status.get("error", "round-trip failed"),
                )
                self._available = False
                return False
        except Exception as e:
            logger.error(
                "Episodic memory initialization FAILED: %s. "
                "Memory will be unavailable for this session.",
                e,
            )
            self._available = False
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize for a session. Run health check, ensure session record."""
        self._session_id = session_id
        self._turn_count = 0
        self._turns_since_extract = 0
        self._turn_ids = []
        self._pending_extractions = []
        self._platform = kwargs.get("platform", "") or ""

        if not self._available:
            if not self._store:
                try:
                    self._store = EpisodicStore()
                except Exception as e:
                    logger.error("Episodic memory store open failed: %s", e)
                    return

        try:
            source = kwargs.get("platform", "")
            self._store.ensure_session(session_id, source=source)

            # Health check on first session of the day (cheap)
            health = self._store.get_health()
            if health:
                last_check = health.get("checked_at", 0)
                if time.time() - last_check > 3600:  # More than 1 hour ago
                    status = self._store.health_check()
                    if not status["round_trip"]:
                        logger.warning(
                            "Episodic memory health check degraded: %s",
                            status.get("error", "round-trip failed"),
                        )

            logger.info("Episodic memory initialized for session %s", session_id)
        except Exception as e:
            logger.error("Episodic memory session init failed: %s", e)

    def system_prompt_block(self) -> str:
        """Tell the agent that episodic memory is available."""
        if not self._available:
            return ""
        return (
            "\n[Episodic Memory: Active. You have persistent episodic memory across sessions. "
            "Use memory_search to find past conversations, memory_get_entity to look up "
            "people/projects/tools, memory_get_episode to recall specific discussions, "
            "memory_grep to search raw turns, and memory_expand to drill into a summary. "
            "Your turns are being recorded automatically.]\n"
        )

    # ── Turn persistence ──────────────────────────────────────────────────

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        tool_calls: Optional[list] = None,
    ) -> None:
        """Persist a completed turn to JSONL + SQLite.

        Called after each turn by the agent loop. This is the critical
        path — must be fast and never fail silently.
        """
        if not self._available or not self._store:
            return

        sid = session_id or self._session_id
        if not sid:
            return

        try:
            # Append to immutable JSONL (source of truth)
            append_turn_jsonl(sid, "user", user_content)
            user_tid = self._store.append_turn(sid, "user", user_content)
            self._turn_ids.append(user_tid)

            append_turn_jsonl(sid, "assistant", assistant_content, tool_calls=tool_calls)
            asst_tid = self._store.append_turn(sid, "assistant", assistant_content)
            self._turn_ids.append(asst_tid)

            self._turn_count += 1
            self._turns_since_extract += 1

        except Exception as e:
            logger.error("Episodic memory sync_turn failed: %s", e)
            # DON'T silently pass — log it so we know

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Called at the start of each turn.

        - Periodic health check (every HEALTH_CHECK_INTERVAL turns)
        - Extraction trigger (every EXTRACT_BATCH_SIZE turns)
        """
        if not self._available or not self._store:
            return

        # Periodic health check
        if turn_number > 0 and turn_number % HEALTH_CHECK_INTERVAL == 0:
            try:
                status = self._store.health_check()
                if not status["round_trip"] and not self._health_warned:
                    logger.warning(
                        "⚠️ Episodic memory health check failed at turn %d: %s",
                        turn_number,
                        status.get("error", "round-trip failed"),
                    )
                    self._health_warned = True
            except Exception as e:
                logger.error("Episodic memory health check error: %s", e)

        # Extraction trigger — every EXTRACT_BATCH_SIZE turns
        if ENABLE_LLM_EXTRACTION and self._turns_since_extract >= EXTRACT_BATCH_SIZE:
            self._run_extraction()

    def _run_extraction(self) -> None:
        """Run extraction on accumulated turns since last extraction."""
        if not self._turn_ids or not self._store:
            return

        try:
            # Get the actual turn data
            turns = []
            for tid in self._turn_ids:
                # Fetch from DB — turns table has id, session_id, role, content, ...
                rows = self._store._execute_read(
                    "SELECT id, role, content, tool_name FROM turns WHERE id = ?",
                    (tid,),
                )
                if rows:
                    turns.append(dict(rows[0]))

            if not turns:
                return

            extracted = extract_from_turns(turns)
            if any(extracted.get(k) for k in ("entities", "facts", "events")):
                self._pending_extractions.append(extracted)
                logger.info(
                    "Extraction batch: %d entities, %d facts, %d events (pending: %d)",
                    len(extracted.get("entities", [])),
                    len(extracted.get("facts", [])),
                    len(extracted.get("events", [])),
                    len(self._pending_extractions),
                )

            # Reset accumulation
            self._turn_ids = []
            self._turns_since_extract = 0

        except Exception as e:
            logger.error("Extraction failed: %s", e)
            # Don't reset turn_ids — retry next time

    # ── Recall / Prefetch ─────────────────────────────────────────────────

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Search for relevant episodes and entities before each API call.

        Returns formatted context to inject, or empty string.
        Uses the cache from queue_prefetch() if available.
        """
        if not self._available or not self._store:
            return ""

        if self._prefetch_cache:
            result = self._prefetch_cache
            self._prefetch_cache = ""
            return result

        # On-demand search using context_injector
        return assemble_context(self._store, query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background recall for the next turn.

        Called after each turn. We do the search now and cache it
        so prefetch() returns instantly on the next turn.
        """
        if not self._available or not self._store:
            return

        try:
            self._prefetch_cache = assemble_context(self._store, query)
        except Exception as e:
            logger.debug("Episodic memory queue_prefetch failed: %s", e)

    # ── Tool schemas ──────────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Expose memory search tools to the agent."""
        if not self._available:
            return []

        return [
            {
                "name": "memory_search",
                "description": (
                    "Search episodic memory for past conversations, episodes, and entities. "
                    "Use when the user references something from a past session or you need "
                    "context from previous discussions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query — keywords or phrases to find in past episodes and entities.",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["episodes", "entities", "both"],
                            "description": "What to search. Default: both.",
                            "default": "both",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results to return. Default: 5.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_get_entity",
                "description": "Look up a specific entity (person, project, tool) by ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity ID (e.g., 'proj-hermes', 'person-jefe').",
                        }
                    },
                    "required": ["entity_id"],
                },
            },
            {
                "name": "memory_get_episode",
                "description": "Retrieve a specific episode by ID, including the raw turns it was distilled from.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "episode_id": {
                            "type": "integer",
                            "description": "The episode ID to retrieve.",
                        }
                    },
                    "required": ["episode_id"],
                },
            },
            {
                "name": "memory_grep",
                "description": (
                    "Search raw conversation turns using full-text search. "
                    "Use when you need to find specific phrases, commands, or "
                    "exact wording from past turns. More granular than memory_search."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search phrase to find in raw turn content.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results to return. Default: 10.",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_expand",
                "description": (
                    "Drill down from an episode or DAG summary into the raw source turns. "
                    "Use when a summary mentions something you want to see in detail. "
                    "LCM-inspired drill-down retrieval."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "episode_id": {
                            "type": "integer",
                            "description": "Episode ID to expand.",
                        },
                        "dag_node_id": {
                            "type": "string",
                            "description": "DAG node ID to expand (alternative to episode_id).",
                        },
                    },
                },
            },
            {
                "name": "memory_describe",
                "description": (
                    "Get a human-readable description of an entity or episode, "
                    "including its history and related items. Richer than memory_get_entity."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity ID to describe.",
                        },
                        "episode_id": {
                            "type": "integer",
                            "description": "Episode ID to describe (alternative to entity_id).",
                        },
                    },
                },
            },
            {
                "name": "memory_health",
                "description": "Check the health status of the episodic memory system.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "memory_stats",
                "description": "Get statistics about the episodic memory store (counts, DB size).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "memory_stale_facts",
                "description": (
                    "Find entities or facts that haven't been confirmed recently. "
                    "Use to identify outdated information that may need refreshing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Check staleness for a specific entity. Omit to check all entities.",
                        },
                        "threshold_days": {
                            "type": "integer",
                            "description": "Days since last confirmation. Default: 30.",
                            "default": 30,
                        },
                    },
                },
            },
            {
                "name": "memory_contradictions",
                "description": (
                    "Find potential contradictions — fields that have been updated "
                    "multiple times with different values. Use to identify conflicting "
                    "information that needs resolution."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Check contradictions for a specific entity. Omit to check all.",
                        },
                    },
                },
            },
            {
                "name": "memory_relationships",
                "description": (
                    "Get the relationship graph for an entity — who they're connected to "
                    "and how. Returns nodes and edges for graph traversal."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "Entity ID to get relationships for.",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Graph traversal depth. 1 = direct only, 2 = friends-of-friends. Default: 1.",
                            "default": 1,
                        },
                    },
                    "required": ["entity_id"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch memory tool calls."""
        if not self._store:
            return json.dumps({"error": "Episodic memory store not initialized"})

        try:
            if tool_name == "memory_search":
                return self._tool_search(args)
            elif tool_name == "memory_get_entity":
                return self._tool_get_entity(args)
            elif tool_name == "memory_get_episode":
                return self._tool_get_episode(args)
            elif tool_name == "memory_grep":
                return self._tool_grep(args)
            elif tool_name == "memory_expand":
                return self._tool_expand(args)
            elif tool_name == "memory_describe":
                return self._tool_describe(args)
            elif tool_name == "memory_health":
                return self._tool_health()
            elif tool_name == "memory_stats":
                return self._tool_stats()
            elif tool_name == "memory_stale_facts":
                return self._tool_stale_facts(args)
            elif tool_name == "memory_contradictions":
                return self._tool_contradictions(args)
            elif tool_name == "memory_relationships":
                return self._tool_relationships(args)
            else:
                return json.dumps({"error": f"Unknown memory tool: {tool_name}"})
        except Exception as e:
            logger.error("Memory tool call %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)})

    # ── Tool implementations ──────────────────────────────────────────────

    def _tool_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        search_type = args.get("type", "both")
        limit = args.get("limit", 5)

        results = {}
        if search_type in ("episodes", "both"):
            results["episodes"] = self._store.search_episodes(query, limit=limit)
        if search_type in ("entities", "both"):
            results["entities"] = self._store.search_entities(query, limit=limit)

        return json.dumps(results, ensure_ascii=False, default=str)

    def _tool_get_entity(self, args: Dict[str, Any]) -> str:
        entity_id = args.get("entity_id", "")
        entity = self._store.get_entity(entity_id)
        if entity:
            return json.dumps(entity, ensure_ascii=False, default=str)
        return json.dumps({"error": f"Entity not found: {entity_id}"})

    def _tool_get_episode(self, args: Dict[str, Any]) -> str:
        episode_id = args.get("episode_id")
        episode = self._store.get_episode(episode_id)
        if not episode:
            return json.dumps({"error": f"Episode not found: {episode_id}"})

        # Include the raw turns
        turns = self._store.get_episode_turns(episode_id)
        episode["source_turns"] = turns
        return json.dumps(episode, ensure_ascii=False, default=str)

    def _tool_grep(self, args: Dict[str, Any]) -> str:
        """Search raw conversation turns via SQLite LIKE (simple but effective)."""
        query = args.get("query", "")
        limit = args.get("limit", 10)

        if not query:
            return json.dumps({"error": "query is required"})

        try:
            # Search turns table directly — no FTS5 on turns (too large),
            # but LIKE works fine for grep-style lookups
            rows = self._store._execute_read(
                "SELECT id, session_id, role, content, timestamp "
                "FROM turns WHERE content LIKE ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", limit),
            )
            results = []
            for row in rows:
                d = dict(row)
                # Truncate long content for readability
                if d.get("content") and len(d["content"]) > 500:
                    d["content"] = d["content"][:500] + "..."
                results.append(d)

            return json.dumps({"query": query, "matches": len(results), "results": results},
                              ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_expand(self, args: Dict[str, Any]) -> str:
        """Drill down from an episode or DAG node into source material."""
        episode_id = args.get("episode_id")
        dag_node_id = args.get("dag_node_id")

        if episode_id is not None:
            episode = self._store.get_episode(episode_id)
            if not episode:
                return json.dumps({"error": f"Episode not found: {episode_id}"})

            turns = self._store.get_episode_turns(episode_id)
            return json.dumps({
                "episode": episode,
                "source_turns": turns,
                "turn_count": len(turns),
            }, ensure_ascii=False, default=str)

        if dag_node_id:
            node = self._store.get_dag_node(dag_node_id)
            if not node:
                return json.dumps({"error": f"DAG node not found: {dag_node_id}"})

            # Expand parent chain
            parents = []
            for pid in node.get("parent_ids", []):
                parent = self._store.get_dag_node(pid)
                if parent:
                    parents.append(parent)

            return json.dumps({
                "node": node,
                "parents": parents,
                "parent_count": len(parents),
            }, ensure_ascii=False, default=str)

        return json.dumps({"error": "Provide episode_id or dag_node_id"})

    def _tool_describe(self, args: Dict[str, Any]) -> str:
        """Get a rich description of an entity or episode."""
        entity_id = args.get("entity_id")
        episode_id = args.get("episode_id")

        if entity_id:
            entity = self._store.get_entity(entity_id)
            if not entity:
                return json.dumps({"error": f"Entity not found: {entity_id}"})

            # Find related episodes
            name = entity.get("name", "")
            related_episodes = []
            if name:
                try:
                    related_episodes = self._store.search_episodes(name, limit=5)
                except Exception:
                    pass

            return json.dumps({
                "entity": entity,
                "related_episodes": related_episodes,
                "related_count": len(related_episodes),
            }, ensure_ascii=False, default=str)

        if episode_id is not None:
            episode = self._store.get_episode(episode_id)
            if not episode:
                return json.dumps({"error": f"Episode not found: {episode_id}"})

            turns = self._store.get_episode_turns(episode_id)

            # Find entities mentioned in this episode
            topic = episode.get("topic", "")
            related_entities = []
            if topic:
                try:
                    related_entities = self._store.search_entities(topic, limit=3)
                except Exception:
                    pass

            return json.dumps({
                "episode": episode,
                "source_turns": turns[:20],  # Cap at 20 for readability
                "turn_count": len(turns),
                "related_entities": related_entities,
            }, ensure_ascii=False, default=str)

        return json.dumps({"error": "Provide entity_id or episode_id"})

    def _tool_health(self) -> str:
        status = self._store.health_check()
        health = self._store.get_health()
        return json.dumps(
            {"current_check": status, "last_check": health},
            ensure_ascii=False,
            default=str,
        )

    def _tool_stats(self) -> str:
        return json.dumps(self._store.get_stats(), ensure_ascii=False)

    # ── Temporal tool implementations ─────────────────────────────────────

    def _tool_stale_facts(self, args: Dict[str, Any]) -> str:
        """Find stale entities or facts."""
        from memory.temporal import detect_stale_entities, detect_stale_facts

        entity_id = args.get("entity_id")
        threshold_days = args.get("threshold_days", 30)

        if entity_id:
            stale = detect_stale_facts(self._store, entity_id, threshold_days)
            entity = self._store.get_entity(entity_id)
            entity_name = entity["name"] if entity else entity_id
            return json.dumps({
                "entity": entity_name,
                "stale_facts": stale,
                "count": len(stale),
            }, ensure_ascii=False, default=str)
        else:
            stale = detect_stale_entities(self._store, threshold_days)
            return json.dumps({
                "stale_entities": stale,
                "count": len(stale),
                "threshold_days": threshold_days,
            }, ensure_ascii=False, default=str)

    def _tool_contradictions(self, args: Dict[str, Any]) -> str:
        """Find potential contradictions."""
        from memory.temporal import detect_contradictions

        entity_id = args.get("entity_id")
        contradictions = detect_contradictions(self._store, entity_id=entity_id)
        return json.dumps({
            "contradictions": contradictions,
            "count": len(contradictions),
        }, ensure_ascii=False, default=str)

    def _tool_relationships(self, args: Dict[str, Any]) -> str:
        """Get relationship graph for an entity."""
        entity_id = args.get("entity_id", "")
        depth = args.get("depth", 1)

        if not entity_id:
            return json.dumps({"error": "entity_id is required"})

        entity = self._store.get_entity(entity_id)
        if not entity:
            return json.dumps({"error": f"Entity not found: {entity_id}"})

        graph = self._store.get_entity_relationships_graph(entity_id, depth=depth)
        return json.dumps(graph, ensure_ascii=False, default=str)

    # ── Session lifecycle ─────────────────────────────────────────────────

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Called when session ends.

        Lightweight path: write a tagged markdown journal.
        Heavy LLM pipeline (extraction/merge/compress) is gated by config.
        """
        if not self._available or not self._store:
            return

        sid = self._session_id
        if not sid:
            return

        # ── Lightweight journal (always runs if enabled) ─────────────────────
        if ENABLE_SESSION_JOURNAL:
            try:
                from memory.session_journal import write_session_journal

                journal_path = write_session_journal(
                    session_id=sid,
                    platform=self._platform,
                )
                if journal_path:
                    logger.info("Session journal written: %s", journal_path)
            except Exception as e:
                logger.error("Session journal failed: %s", e)

        # ── Heavy LLM pipeline (gated) ───────────────────────────────────────
        if not ENABLE_LLM_EXTRACTION:
            logger.info(
                "Episodic memory session %s ended: %d turns (LLM pipeline disabled)",
                sid,
                self._turn_count,
            )
            return

        # 1. Final extraction
        if self._turns_since_extract > 0:
            self._run_extraction()

        # 2. Merge all pending extractions
        if self._pending_extractions:
            try:
                from memory.merge import merge_session

                combined = {"entities": [], "facts": [], "events": []}
                for ext in self._pending_extractions:
                    for key in ("entities", "facts", "events"):
                        combined[key].extend(ext.get(key, []))

                merge_stats = merge_session(self._store, sid, combined)
                logger.info("Session %s merge: %s", sid, json.dumps(merge_stats))
            except Exception as e:
                logger.error("Session merge failed: %s", e)

        # 3. Compress session to D0
        try:
            from memory.compress import compress_session_to_d0

            turns = self._store.get_turns_for_session(sid, limit=500)
            if turns:
                d0_node_id = compress_session_to_d0(self._store, sid, turns)
                if d0_node_id:
                    logger.info("Session %s compressed to D0: %s", sid, d0_node_id)
        except Exception as e:
            logger.error("Session compress failed: %s", e)

        logger.info(
            "Episodic memory session %s ended: %d turns, %d extractions merged",
            sid,
            self._turn_count,
            len(self._pending_extractions),
        )

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Before context compression, extract insights from messages about to be discarded."""
        if not self._available:
            return ""

        # Run a quick extraction on the messages being compressed
        try:
            # Convert messages to extraction format
            turns = []
            for msg in messages:
                if isinstance(msg, dict):
                    turns.append({
                        "role": msg.get("role", "unknown"),
                        "content": str(msg.get("content", ""))[:1000],
                        "tool_name": msg.get("tool_name", ""),
                    })

            if turns:
                extracted = extract_from_turns(turns)
                if any(extracted.get(k) for k in ("entities", "facts", "events")):
                    self._pending_extractions.append(extracted)
                    return (
                        f"[Note: Extracted {len(extracted.get('facts', []))} facts "
                        f"and {len(extracted.get('entities', []))} entities from "
                        f"{len(messages)} messages being compressed.]"
                    )
        except Exception as e:
            logger.debug("Pre-compress extraction failed: %s", e)

        return (
            f"[Note: {len(messages)} messages are being compressed. "
            f"Episodic memory has captured {self._turn_count} turns this session. "
            f"Key details should be preserved in the memory store.]"
        )

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs,
    ) -> None:
        """Record delegation observations."""
        if not self._available or not self._store:
            return

        try:
            sid = self._session_id or "unknown"
            self._store.append_turn(
                sid,
                "assistant",
                f"[DELEGATION] Task: {task[:200]}\nResult: {result[:500]}",
                tool_name="delegate_task",
            )
        except Exception as e:
            logger.debug("Episodic memory delegation record failed: %s", e)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to episodic memory.

        When MEMORY.md or USER.md is updated, we note it so we can
        potentially sync episodic entities with the built-in memory.
        """
        if not self._available or not self._store:
            return

        try:
            sid = self._session_id or "unknown"
            self._store.append_turn(
                sid,
                "assistant",
                f"[MEMORY_WRITE] action={action} target={target}\n{content[:500]}",
                tool_name="memory",
            )
        except Exception:
            pass  # Best-effort mirroring

    # ── Shutdown ──────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean shutdown."""
        if self._store:
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None
        logger.info(
            "Episodic memory shutdown: session=%s, turns=%d",
            self._session_id,
            self._turn_count,
        )
