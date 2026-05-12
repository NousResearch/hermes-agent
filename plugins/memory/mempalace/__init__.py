"""MemPalace memory plugin — native MemoryProvider for MemPalace.

Provides cross-session memory using MemPalace v3.3.4: semantic search over
the palace graph (L0 closets + L1 memories), knowledge graph queries, and
automatic diary writing at session end.

MemPalace is a local-first, zero-API-key memory system backed by ChromaDB.
This provider replaces the MCP bridge with direct Python API calls for
lower latency and richer integration.

Exposed tools: mempalace_search, mempalace_kg_query, mempalace_kg_add,
mempalace_diary_read, mempalace_diary_write.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PALACE_PATH = "/root/.mempalace/palace"
PREFETCH_MAX_RESULTS = 5
PREFETCH_MAX_DISTANCE = 1.2  # cosine distance threshold


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "mempalace_search",
    "description": (
        "Semantic search over MemPalace long-term memory. Searches the palace "
        "graph (closets + memories) for content relevant to the query. "
        "Returns verbatim text with similarity scores. "
        "Use this for recalling past facts, decisions, configurations, and conversations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — keywords or a natural-language question.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return (default: 5, max: 20).",
            },
            "wing": {
                "type": "string",
                "description": "Optional: filter by wing (project).",
            },
            "room": {
                "type": "string",
                "description": "Optional: filter by room (aspect: backend, decisions, diary...).",
            },
        },
        "required": ["query"],
    },
}

KG_QUERY_SCHEMA = {
    "name": "mempalace_kg_query",
    "description": (
        "Query the MemPalace Knowledge Graph for facts about an entity. "
        "Returns typed relationships with temporal validity. "
        "Use this to look up structured facts: who manages what, "
        "what belongs to what, project relationships, preferences."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "Entity to query (e.g. 'James Huang', 'docker-ssd', 'MemPalace').",
            },
            "as_of": {
                "type": "string",
                "description": "Date filter — only facts valid at this date (YYYY-MM-DD). Optional.",
            },
        },
        "required": ["entity"],
    },
}

KG_ADD_SCHEMA = {
    "name": "mempalace_kg_add",
    "description": (
        "Add or invalidate a fact in the MemPalace Knowledge Graph. "
        "Use 'add' to record a new fact; use 'invalidate' to mark a fact as no longer true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "'add' to record a new fact, 'invalidate' to mark as no longer true.",
                "enum": ["add", "invalidate"],
            },
            "subject": {
                "type": "string",
                "description": "The entity doing/being something.",
            },
            "predicate": {
                "type": "string",
                "description": "The relationship type (e.g. 'manages', 'belongs_to', 'prefers').",
            },
            "object": {
                "type": "string",
                "description": "The connected entity.",
            },
            "valid_from": {
                "type": "string",
                "description": "When this became true (YYYY-MM-DD). Optional, defaults to today.",
            },
        },
        "required": ["action", "subject", "predicate", "object"],
    },
}

DIARY_READ_SCHEMA = {
    "name": "mempalace_diary_read",
    "description": (
        "Read recent diary entries from MemPalace. "
        "Each agent has their own diary — these are journal entries written "
        "across sessions recording key events, decisions, and observations. "
        "Use this to recall what happened in past sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Agent name (e.g. 'jarvis', 'infra-agent'). Default: 'jarvis'.",
            },
            "last_n": {
                "type": "integer",
                "description": "Number of recent entries (default: 10, max: 50).",
            },
        },
        "required": [],
    },
}

DIARY_WRITE_SCHEMA = {
    "name": "mempalace_diary_write",
    "description": (
        "Write an entry to your MemPalace diary. "
        "Record key events, decisions, and observations after significant work. "
        "Use AAAK format for compression: e.g. 'SESSION:date|built.feature|★★★'. "
        "This persists across sessions — future you will read these entries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_name": {
                "type": "string",
                "description": "Your agent name (e.g. 'jarvis').",
            },
            "entry": {
                "type": "string",
                "description": "Diary entry in AAAK compressed format.",
            },
            "topic": {
                "type": "string",
                "description": "Topic tag (optional, default: 'general').",
            },
        },
        "required": ["agent_name", "entry"],
    },
}


def _get_all_schemas() -> List[Dict[str, Any]]:
    return [SEARCH_SCHEMA, KG_QUERY_SCHEMA, KG_ADD_SCHEMA, DIARY_READ_SCHEMA, DIARY_WRITE_SCHEMA]


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class MemPalaceMemoryProvider(MemoryProvider):
    """Native MemoryProvider backed by MemPalace (ChromaDB + Knowledge Graph)."""

    def __init__(self):
        self._palace_path: str = ""
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._agent_name: str = "jarvis"  # default, overridden in initialize
        self._prefetch_cache: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._turn_counter: int = 0
        self._diary_period: int = 25  # write diary every N turns

    # -- Core lifecycle --------------------------------------------------------

    @property
    def name(self) -> str:
        return "mempalace"

    def is_available(self) -> bool:
        """Available if the palace directory exists and ChromaDB is importable."""
        palace = self._get_palace_path()
        if not palace or not os.path.isdir(palace):
            return False
        try:
            from mempalace import palace as _palace_module  # noqa: F401
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Connect to the palace and warm up."""
        self._hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))
        self._session_id = session_id
        self._palace_path = self._get_palace_path()

        # Derive agent name from profile if available
        agent_identity = kwargs.get("agent_identity", "")
        if agent_identity:
            self._agent_name = agent_identity
        else:
            # Fall back to env or default
            self._agent_name = os.environ.get("MEMPALACE_AGENT_NAME", "jarvis")

        if not self._palace_path or not os.path.isdir(self._palace_path):
            logger.warning(
                "MemPalace path not found: %s. Set MEMPALACE_PATH env var.",
                self._palace_path,
            )
            return

        logger.info(
            "MemPalace provider initialized: palace=%s session=%s",
            self._palace_path,
            session_id,
        )

    def system_prompt_block(self) -> str:
        """Return static instructions for how to use MemPalace tools."""
        if not self._palace_path or not os.path.isdir(self._palace_path):
            return ""

        return (
            f"You have MemPalace long-term memory (v3.3.4) at {self._palace_path}.\n"
            "Use mempalace_search for semantic recall, mempalace_kg_query for "
            "structured facts, mempalace_kg_add to record new facts, "
            "and mempalace_diary_read/write for cross-session journaling.\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached recall context for the upcoming turn."""
        with self._prefetch_lock:
            result = self._prefetch_cache
            self._prefetch_cache = ""
            return result if result else ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background recall for the next turn."""
        if not self._palace_path or not os.path.isdir(self._palace_path):
            return

        def _prefetch() -> None:
            try:
                results = self._do_search(query)
                with self._prefetch_lock:
                    self._prefetch_cache = results
            except Exception:
                logger.warning("Background prefetch failed", exc_info=True)

        self._prefetch_thread = threading.Thread(target=_prefetch, daemon=True)
        self._prefetch_thread.start()

    def _diary_dir(self) -> str:
        """Get the diary directory path (under palace root)."""
        d = os.path.join(self._palace_path, "..", "diary")
        return os.path.abspath(d)

    def _write_diary_entry(self, entry: str) -> None:
        """Append a diary entry to today's file and attempt ChromaDB ingest.

        ChromaDB's SharedSystemClient can KeyError when multiple processes
        access the same palace path — we catch *everything* so the gateway
        never crashes from a diary write.
        """
        try:
            diary_dir = self._diary_dir()
            os.makedirs(diary_dir, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            diary_file = os.path.join(diary_dir, f"{today}.md")
            with open(diary_file, "a") as f:
                f.write(f"\n## {datetime.now(timezone.utc).strftime('%H:%M')}\n{entry}\n")

            # ChromaDB ingest — may fail if MCP server or another gateway
            # instance holds the connection; non-fatal.
            try:
                from mempalace import diary_ingest
                diary_ingest.ingest_diaries(
                    diary_dir=diary_dir,
                    palace_path=self._palace_path,
                    wing="diary",
                )
            except Exception:
                logger.warning("diary ChromaDB ingest failed (non-fatal)", exc_info=True)
        except Exception:
            logger.warning("_write_diary_entry failed", exc_info=True)

    def _format_diary_entry(self, tool_names: list[str], msg_count_user: int, msg_count_total: int) -> str:
        tool_summary = ",".join(tool_names[:10]) if tool_names else "?"
        return (
            f"SESSION:{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H')}|"
            f"MSGS:{msg_count_total}(user:{msg_count_user})|"
            f"TOOLS:{tool_summary}|"
            f"★★★"
        )

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """Archive the turn to the palace in the background.

        Also writes a periodic diary entry every ``self._diary_period`` turns
        so long-running sessions get diary checkpoints even if they never end.
        """
        if not self._palace_path or not os.path.isdir(self._palace_path):
            return

        target_session = session_id or self._session_id

        def _sync() -> None:
            try:
                closet_text = (
                    f"SESSION:{target_session}|USER:{user_content[:500]}|"
                    f"ASSISTANT:{assistant_content[:1000]}"
                )

                closet_id_base = f"hermes_turn_{target_session}"

                import json, subprocess

                _script = os.path.join(
                    os.path.expanduser("~/.hermes/scripts"),
                    "mempalace_sync_turn.py",
                )
                payload = json.dumps(
                    {
                        "palace_path": self._palace_path,
                        "closet_text": closet_text,
                        "closet_id_base": closet_id_base,
                        "metadata": {
                            "source": "hermes_agent",
                            "session_id": target_session,
                            "timestamp": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        },
                    }
                )
                _result = subprocess.run(
                    [sys.executable, _script],
                    input=payload,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if _result.returncode != 0:
                    logger.warning(
                        "sync_turn subprocess failed (exit=%d): %s",
                        _result.returncode,
                        _result.stderr.strip()[-300:],
                    )

            except Exception:
                logger.warning("sync_turn failed", exc_info=True)

        # Turn counter for periodic diary
        self._turn_counter += 1

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True)
        self._sync_thread.start()

        # Periodic diary write every N turns
        if self._turn_counter % self._diary_period == 0:
            entry = self._format_diary_entry(
                tool_names=[],
                msg_count_user=self._turn_counter,
                msg_count_total=self._turn_counter * 2,
            )
            self._write_diary_entry(f"{entry}|PERIODIC|")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Write a diary entry and build KG facts at session end."""
        if not self._palace_path or not os.path.isdir(self._palace_path):
            return

        def _end() -> None:
            try:
                # Count tools used
                tool_names: List[str] = []
                for msg in messages:
                    if msg.get("role") == "tool" and msg.get("name"):
                        tool_name = msg.get("name", "")
                        if tool_name:
                            tool_names.append(tool_name)

                msg_count_user = len([m for m in messages if m.get("role") == "user"])
                msg_count_total = len(messages)

                entry = self._format_diary_entry(
                    tool_names=tool_names,
                    msg_count_user=msg_count_user,
                    msg_count_total=msg_count_total,
                )
                self._write_diary_entry(f"{entry}|SESSION_END|")

                logger.info("Session ended, diary written for %s", self._agent_name)

            except Exception:
                logger.warning("on_session_end failed", exc_info=True)

        threading.Thread(target=_end, daemon=True).start()

    def shutdown(self) -> None:
        """Clean shutdown."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        logger.info("MemPalace provider shut down")

    # -- Tool schemas ----------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return all tool schemas regardless of palace path existence.

        Note: ``register()`` is called before ``initialize()`` in the agent
        lifecycle, so ``self._palace_path`` is empty at registration time.
        We return schemas unconditionally here so the MemoryManager can
        build its ``_tool_to_provider`` routing table correctly. Runtime
        handlers will fail gracefully if the palace is unavailable.
        """
        return _get_all_schemas()

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "mempalace_search":
            return self._handle_search(args)
        elif tool_name == "mempalace_kg_query":
            return self._handle_kg_query(args)
        elif tool_name == "mempalace_kg_add":
            return self._handle_kg_add(args)
        elif tool_name == "mempalace_diary_read":
            return self._handle_diary_read(args)
        elif tool_name == "mempalace_diary_write":
            return self._handle_diary_write(args)
        return tool_error(f"Unknown tool: {tool_name}")

    # -- Config ----------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "palace_path",
                "description": "Path to your MemPalace directory (contains palace/ folder).",
                "default": DEFAULT_PALACE_PATH.replace("/palace", ""),
                "required": False,
            },
            {
                "key": "agent_name",
                "description": "Agent name for diary identification.",
                "default": "jarvis",
                "required": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Save non-secret config to mempalace.json."""
        config_path = os.path.join(hermes_home, "mempalace.json")
        with open(config_path, "w") as f:
            json.dump(values, f, indent=2)

    # -- Internal helpers ------------------------------------------------------

    def _get_palace_path(self) -> str:
        """Get the palace path from env or default."""
        # MEMPALACE_PATH can point to the palace/ directory directly
        env_path = os.environ.get("MEMPALACE_PATH", "")
        if env_path and os.path.isdir(env_path):
            return env_path

        # Or MEMPALACE_HOME pointing to parent of palace/
        env_home = os.environ.get("MEMPALACE_HOME", "")
        if env_home:
            palace_dir = os.path.join(env_home, "palace")
            if os.path.isdir(palace_dir):
                return palace_dir

        # Legacy: check config file
        config_json = os.path.join(self._hermes_home, "mempalace.json")
        if os.path.exists(config_json):
            try:
                with open(config_json) as f:
                    cfg = json.load(f)
                if cfg.get("palace_path"):
                    p = os.path.join(cfg["palace_path"], "palace")
                    if os.path.isdir(p):
                        return p
            except Exception:
                pass

        # Default
        return DEFAULT_PALACE_PATH

    def _do_search(
        self, query: str, wing: str = None, room: str = None, limit: int = None
    ) -> str:
        """Execute a semantic search and return formatted results."""
        from mempalace import searcher

        raw = searcher.search_memories(
            query=query,
            palace_path=self._palace_path,
            wing=wing,
            room=room,
            n_results=limit or PREFETCH_MAX_RESULTS,
            max_distance=PREFETCH_MAX_DISTANCE,
        )

        results = raw.get("results", [])
        if not results:
            return ""

        lines = []
        for i, r in enumerate(results, 1):
            text = r.get("text", "")[:500]
            sim = r.get("similarity", 0)
            wing_name = r.get("wing", "?")
            room_name = r.get("room", "?")
            lines.append(f"[{i}] {wing_name}/{room_name} (sim={sim:.2f}): {text}")

        return "\n".join(lines)

    # -- Tool handlers ---------------------------------------------------------

    def _handle_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        if not query or not query.strip():
            return tool_error("Query is required")

        limit = min(int(args.get("limit", 5)), 20)
        wing = args.get("wing")
        room = args.get("room")

        try:
            result = self._do_search(query, wing=wing, room=room, limit=limit)
            if not result:
                return json.dumps({"results": [], "note": "No results found."})
            return json.dumps({"results": result})
        except Exception as e:
            return tool_error(f"Search failed: {e}")

    def _handle_kg_query(self, args: Dict[str, Any]) -> str:
        entity = args.get("entity", "")
        if not entity or not entity.strip():
            return tool_error("Entity is required")

        as_of = args.get("as_of")

        try:
            from mempalace import knowledge_graph

            kg_path = os.path.join(self._palace_path, "knowledge_graph.sqlite3") if self._palace_path else ""
            kg = knowledge_graph.KnowledgeGraph(kg_path) if kg_path and os.path.exists(kg_path) else knowledge_graph.KnowledgeGraph()
            facts = kg.query_entity(entity)
            if not facts and kg_path and os.path.exists(kg_path):
                kg = knowledge_graph.KnowledgeGraph(kg_path)
                facts = kg.query_entity(entity)

            # Filter by as_of if provided
            if as_of:
                from datetime import date
                try:
                    target = date.fromisoformat(as_of)
                    facts = [
                        f for f in facts
                        if (not f.get("valid_from") or date.fromisoformat(f["valid_from"]) <= target)
                        and (not f.get("ended") or date.fromisoformat(f["ended"]) >= target)
                    ]
                except ValueError:
                    pass

            return json.dumps({
                "entity": entity,
                "facts": [
                    {
                        "subject": f.get("subject", entity),
                        "predicate": f.get("predicate", ""),
                        "object": f.get("object", ""),
                        "valid_from": f.get("valid_from"),
                        "ended": f.get("ended"),
                    }
                    for f in facts
                ],
            })
        except Exception as e:
            return tool_error(f"KG query failed: {e}")

    def _handle_kg_add(self, args: Dict[str, Any]) -> str:
        action = args.get("action", "add")
        subject = args.get("subject", "")
        predicate = args.get("predicate", "")
        obj = args.get("object", "")

        if not subject or not predicate or not obj:
            return tool_error("subject, predicate, and object are required")

        valid_from = args.get("valid_from", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        try:
            from mempalace import knowledge_graph

            kg_path = os.path.join(self._palace_path, "knowledge_graph.sqlite3") if self._palace_path else ""
            kg = knowledge_graph.KnowledgeGraph(kg_path) if kg_path and os.path.exists(kg_path) else knowledge_graph.KnowledgeGraph()
            if action == "invalidate":
                kg.invalidate(subject, predicate, obj)
                return json.dumps(
                    {"action": "invalidate", "subject": subject, "predicate": predicate,
                     "object": obj, "status": "done"}
                )
            else:
                kg.add_triple(subject, predicate, obj, valid_from=valid_from)
                return json.dumps(
                    {"action": "add", "subject": subject, "predicate": predicate,
                     "object": obj, "valid_from": valid_from, "status": "done"}
                )
        except Exception as e:
            return tool_error(f"KG add failed: {e}")

    def _handle_diary_read(self, args: Dict[str, Any]) -> str:
        agent_name = args.get("agent_name", self._agent_name)
        last_n = min(int(args.get("last_n", 10)), 50)

        try:
            wing = f"wing_{agent_name}"
            results = self._do_search(
                agent_name,
                wing=wing,
                room="diary",
                limit=last_n,
            )

            if not results:
                return json.dumps({"agent": agent_name, "entries": []})

            # Parse the formatted text back
            entries = []
            for line in results.split("\n"):
                entries.append({"text": line[:600]})

            return json.dumps({"agent": agent_name, "entries": entries})
        except Exception as e:
            return tool_error(f"Diary read failed: {e}")

    def _handle_diary_write(self, args: Dict[str, Any]) -> str:
        agent_name = args.get("agent_name", self._agent_name)
        entry = args.get("entry", "")
        topic = args.get("topic", "general")

        if not entry or not entry.strip():
            return tool_error("Entry text is required")

        try:
            import hashlib
            from mempalace.palace import get_collection

            wing = f"wing_{agent_name}"
            room = "diary"
            col = get_collection(self._palace_path, create=True)
            if not col:
                return tool_error("No palace connection")

            now = datetime.now(timezone.utc)
            entry_id = (
                f"diary_{wing}_{now.strftime('%Y%m%d_%H%M%S%f')}_"
                f"{hashlib.sha256(entry.encode()).hexdigest()[:12]}"
            )

            col.add(
                ids=[entry_id],
                documents=[entry],
                metadatas=[{
                    "wing": wing,
                    "room": room,
                    "hall": "hall_diary",
                    "topic": topic,
                    "type": "diary_entry",
                    "agent": agent_name,
                    "filed_at": now.isoformat(),
                    "date": now.strftime("%Y-%m-%d"),
                }],
            )

            return json.dumps(
                {"agent": agent_name, "topic": topic, "status": "written",
                 "timestamp": now.isoformat()}
            )
        except Exception as e:
            return tool_error(f"Diary write failed: {e}")


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

_mempalace_provider: Optional[MemPalaceMemoryProvider] = None


def register(ctx) -> None:
    """Register the MemPalace memory provider with the plugin system."""
    global _mempalace_provider
    _mempalace_provider = MemPalaceMemoryProvider()
    ctx.register_memory_provider(_mempalace_provider)
