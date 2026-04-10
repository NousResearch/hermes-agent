"""hermes-mempalace — MemPalace memory provider plugin for Hermes Agent.

Integrates MemPalace (https://github.com/milla-jovovich/mempalace) as a
pluggable memory provider. Stores verbatim conversation chunks in ChromaDB
with the palace architecture (wings/halls/rooms/drawers) and a temporal
knowledge graph in SQLite.

Key features:
  - 4-layer memory stack (L0 identity → L3 deep search)
  - 96.6% LongMemEval recall in raw verbatim mode
  - Local-only: ChromaDB + SQLite, zero API calls, free
  - Temporal knowledge graph with entity resolution
  - Session-end conversation mining
  - Pre-compression memory extraction
  - Built-in memory write mirroring

Requires: pip install mempalace (which pulls in chromadb)

Config in $HERMES_HOME/config.yaml:
  memory:
    provider: mempalace

  plugins:
    mempalace:
      palace_path: ~/.mempalace/palace    # ChromaDB storage
      wing: hermes                        # default wing for conversations
      kg_enabled: true                    # temporal knowledge graph
      identity_path: ~/.mempalace/identity.txt
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — mempalace may not be installed
# ---------------------------------------------------------------------------

_mempalace = None
_import_error = None


def _ensure_mempalace():
    """Lazy-import mempalace. Returns True if available."""
    global _mempalace, _import_error
    if _mempalace is not None:
        return True
    if _import_error is not None:
        return False
    try:
        import mempalace as mp
        _mempalace = mp
        return True
    except ImportError as e:
        _import_error = str(e)
        return False


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PALACE_SEARCH_SCHEMA = {
    "name": "palace_search",
    "description": (
        "Deep semantic search across your entire memory palace. Returns verbatim "
        "stored text — the actual words, never summaries.\n\n"
        "Use this to recall conversations, decisions, preferences, code discussions, "
        "or anything discussed in past sessions. The palace stores everything verbatim "
        "and uses ChromaDB embeddings for semantic matching.\n\n"
        "Optional filters: wing (project/topic), room (sub-topic) to narrow results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for. Natural language works best."},
            "wing": {"type": "string", "description": "Filter by wing (project/topic). Optional."},
            "room": {"type": "string", "description": "Filter by room (sub-topic). Optional."},
            "n_results": {"type": "integer", "description": "Max results (default: 5, max: 20)."},
        },
        "required": ["query"],
    },
}

PALACE_STORE_SCHEMA = {
    "name": "palace_store",
    "description": (
        "Store a memory in the palace. Use for important facts, decisions, preferences, "
        "or insights you want to recall later. Stored verbatim — no summarization.\n\n"
        "Memories are organized by wing (project/topic) and room (sub-topic). "
        "Use alongside the built-in memory tool: memory for always-on compact context, "
        "palace_store for deep archival with semantic search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory to store. Be detailed — verbatim is the point."},
            "wing": {"type": "string", "description": "Wing (project/topic). Default: 'hermes'."},
            "room": {"type": "string", "description": "Room (sub-topic). Default: 'general'."},
            "importance": {"type": "integer", "description": "1-5 importance score. Default: 3."},
        },
        "required": ["content"],
    },
}

PALACE_BROWSE_SCHEMA = {
    "name": "palace_browse",
    "description": (
        "Browse the palace structure — list wings, rooms, and drawer counts. "
        "Use to discover what's stored and navigate the memory architecture."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wing": {"type": "string", "description": "List rooms in this wing. Omit to list all wings."},
        },
        "required": [],
    },
}

PALACE_KG_SCHEMA = {
    "name": "palace_kg",
    "description": (
        "Query the temporal knowledge graph — entity relationships with time validity.\n\n"
        "Actions:\n"
        "• query — All facts about an entity (person, project, tool)\n"
        "• add — Store a relationship triple (subject → predicate → object)\n"
        "• timeline — Temporal view of an entity's history\n"
        "• invalidate — Mark a fact as no longer true (with end date)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["query", "add", "timeline", "invalidate"],
            },
            "entity": {"type": "string", "description": "Entity name for query/timeline."},
            "subject": {"type": "string", "description": "Subject entity for add/invalidate."},
            "predicate": {"type": "string", "description": "Relationship type (e.g. 'works_on', 'prefers', 'child_of')."},
            "object": {"type": "string", "description": "Object entity for add/invalidate."},
            "valid_from": {"type": "string", "description": "When this became true (ISO date). Default: today."},
            "valid_to": {"type": "string", "description": "When this stopped being true (for invalidate)."},
            "as_of": {"type": "string", "description": "Query facts valid at this date."},
        },
        "required": ["action"],
    },
}

PALACE_STATUS_SCHEMA = {
    "name": "palace_status",
    "description": "Show palace status: total drawers, wings, knowledge graph stats, layer info.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        return all_config.get("plugins", {}).get("mempalace", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class MemPalaceProvider(MemoryProvider):
    """MemPalace memory with verbatim storage, semantic search, and knowledge graph."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._palace_path = None
        self._wing = None
        self._collection = None
        self._kg = None
        self._kg_enabled = True
        self._stack = None
        self._session_id = None
        self._turn_count = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "mempalace"

    def is_available(self) -> bool:
        """Check if mempalace is installed."""
        return _ensure_mempalace()

    def initialize(self, session_id: str, **kwargs) -> None:
        if not _ensure_mempalace():
            logger.warning("mempalace not installed: pip install mempalace")
            return

        from mempalace.config import MempalaceConfig
        from mempalace.layers import MemoryStack
        from mempalace.palace import get_collection

        hermes_home = kwargs.get("hermes_home", os.environ.get("HERMES_HOME", ""))

        # Resolve palace path: plugin config > mempalace config > default
        cfg = MempalaceConfig()
        self._palace_path = self._config.get("palace_path", cfg.palace_path)
        self._palace_path = os.path.expanduser(self._palace_path)

        self._wing = self._config.get("wing", "hermes")
        self._kg_enabled = self._config.get("kg_enabled", True)
        self._session_id = session_id
        self._turn_count = 0

        # Initialize ChromaDB collection
        try:
            self._collection = get_collection(self._palace_path)
            logger.info(f"MemPalace connected: {self._palace_path} ({self._collection.count()} drawers)")
        except Exception as e:
            logger.error(f"MemPalace ChromaDB init failed: {e}")
            self._collection = None

        # Initialize memory stack for layered retrieval
        identity_path = self._config.get(
            "identity_path",
            os.path.expanduser("~/.mempalace/identity.txt"),
        )
        try:
            self._stack = MemoryStack(
                palace_path=self._palace_path,
                identity_path=identity_path,
            )
        except Exception as e:
            logger.error(f"MemPalace stack init failed: {e}")
            self._stack = None

        # Initialize knowledge graph
        if self._kg_enabled:
            try:
                from mempalace.knowledge_graph import KnowledgeGraph
                kg_path = self._config.get("kg_path")
                self._kg = KnowledgeGraph(db_path=kg_path) if kg_path else KnowledgeGraph()
                logger.info("MemPalace knowledge graph initialized")
            except Exception as e:
                logger.warning(f"MemPalace KG init failed (non-fatal): {e}")
                self._kg = None

    def system_prompt_block(self) -> str:
        """Inject L0 (identity) + L1 (essential story) into system prompt."""
        if not self._stack:
            return ""
        try:
            wake_text = self._stack.wake_up(wing=self._wing)
            count = self._collection.count() if self._collection else 0
            header = (
                f"\n══ MEMPALACE ({count} memories) ══\n"
                "You have a memory palace with semantic search. Use palace_search to recall "
                "past conversations and decisions. Use palace_store for important new facts.\n"
            )
            return header + wake_text
        except Exception as e:
            logger.error(f"MemPalace system_prompt_block failed: {e}")
            return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """L3 semantic search for the upcoming turn's context."""
        if not self._stack or not query.strip():
            return ""
        try:
            result = self._stack.search(query, wing=None, n_results=3)
            if result and "No results" not in result and "No palace" not in result:
                return f"\n── Palace Recall ──\n{result}\n"
        except Exception as e:
            logger.debug(f"MemPalace prefetch error: {e}")
        return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Store conversation turn as a drawer in the palace."""
        if not self._collection or not user_content.strip():
            return
        self._turn_count += 1

        def _store():
            try:
                # Combine user+assistant as one exchange (MemPalace philosophy: store everything)
                exchange = f"> {user_content.strip()}\n\n{assistant_content.strip()}"

                # Chunk if too long (800 char chunks like MemPalace default)
                chunks = self._chunk_text(exchange)
                timestamp = datetime.now().isoformat()
                source = f"session_{self._session_id or 'unknown'}"

                for i, chunk in enumerate(chunks):
                    drawer_id = (
                        f"drawer_{self._wing}_conversation_"
                        f"{hashlib.sha256((source + str(self._turn_count) + str(i)).encode()).hexdigest()[:24]}"
                    )
                    self._collection.upsert(
                        documents=[chunk],
                        ids=[drawer_id],
                        metadatas=[{
                            "wing": self._wing,
                            "room": "conversation",
                            "source_file": source,
                            "chunk_index": i,
                            "turn": self._turn_count,
                            "added_by": "hermes",
                            "filed_at": timestamp,
                            "importance": 3,
                        }],
                    )
            except Exception as e:
                logger.error(f"MemPalace sync_turn failed: {e}")

        # Non-blocking: fire and forget in background thread
        threading.Thread(target=_store, daemon=True).start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = [PALACE_SEARCH_SCHEMA, PALACE_STORE_SCHEMA, PALACE_BROWSE_SCHEMA, PALACE_STATUS_SCHEMA]
        if self._kg_enabled and self._kg:
            schemas.append(PALACE_KG_SCHEMA)
        return schemas

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "palace_search":
                return self._handle_search(args)
            elif tool_name == "palace_store":
                return self._handle_store(args)
            elif tool_name == "palace_browse":
                return self._handle_browse(args)
            elif tool_name == "palace_kg":
                return self._handle_kg(args)
            elif tool_name == "palace_status":
                return self._handle_status(args)
            else:
                return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"MemPalace tool error ({tool_name}): {e}")
            return tool_error(str(e))

    # -- Optional hooks -------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Mine the full session into the palace on exit."""
        if not self._collection or not messages:
            return
        try:
            # Extract user/assistant exchanges and store as conversation drawers
            exchanges = []
            user_msg = None
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                if role == "user":
                    user_msg = content
                elif role == "assistant" and user_msg:
                    exchanges.append(f"> {user_msg.strip()}\n\n{content.strip()}")
                    user_msg = None

            if not exchanges:
                return

            # Store full session as a consolidated document
            full_session = "\n\n---\n\n".join(exchanges)
            timestamp = datetime.now().isoformat()
            source = f"session_end_{self._session_id or 'unknown'}"

            chunks = self._chunk_text(full_session)
            for i, chunk in enumerate(chunks):
                drawer_id = (
                    f"drawer_{self._wing}_session_"
                    f"{hashlib.sha256((source + str(i)).encode()).hexdigest()[:24]}"
                )
                self._collection.upsert(
                    documents=[chunk],
                    ids=[drawer_id],
                    metadatas=[{
                        "wing": self._wing,
                        "room": "session_archive",
                        "source_file": source,
                        "chunk_index": i,
                        "added_by": "hermes",
                        "filed_at": timestamp,
                        "importance": 4,  # session archives are high-value
                    }],
                )
            logger.info(f"MemPalace: archived session ({len(chunks)} drawers from {len(exchanges)} exchanges)")
        except Exception as e:
            logger.error(f"MemPalace on_session_end failed: {e}")

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract and store memories before context compression discards them."""
        if not self._collection or not messages:
            return ""
        try:
            # Store the about-to-be-compressed messages
            texts = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                if content.strip():
                    texts.append(content.strip())

            if not texts:
                return ""

            combined = "\n\n".join(texts)
            timestamp = datetime.now().isoformat()
            source = f"precompress_{self._session_id or 'unknown'}_{timestamp}"

            chunks = self._chunk_text(combined)
            for i, chunk in enumerate(chunks):
                drawer_id = (
                    f"drawer_{self._wing}_precompress_"
                    f"{hashlib.sha256((source + str(i)).encode()).hexdigest()[:24]}"
                )
                self._collection.upsert(
                    documents=[chunk],
                    ids=[drawer_id],
                    metadatas=[{
                        "wing": self._wing,
                        "room": "precompress",
                        "source_file": source,
                        "chunk_index": i,
                        "added_by": "hermes",
                        "filed_at": timestamp,
                        "importance": 4,
                    }],
                )
            logger.info(f"MemPalace: saved {len(chunks)} drawers before compression")
            return "MemPalace has preserved the compressed context in the palace for future recall."
        except Exception as e:
            logger.error(f"MemPalace on_pre_compress failed: {e}")
            return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to the palace as tagged drawers."""
        if not self._collection or not content.strip():
            return
        if action == "remove":
            return  # Don't store removals

        try:
            timestamp = datetime.now().isoformat()
            drawer_id = (
                f"drawer_{self._wing}_memory_mirror_"
                f"{hashlib.sha256((content + timestamp).encode()).hexdigest()[:24]}"
            )
            self._collection.upsert(
                documents=[f"[{target}] {content}"],
                ids=[drawer_id],
                metadatas=[{
                    "wing": self._wing,
                    "room": f"memory_{target}",  # memory_user or memory_memory
                    "source_file": f"builtin_memory_{action}",
                    "chunk_index": 0,
                    "added_by": "hermes",
                    "filed_at": timestamp,
                    "importance": 5,  # explicit memory writes are highest value
                }],
            )
        except Exception as e:
            logger.error(f"MemPalace on_memory_write failed: {e}")

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Track turns for session scoping."""
        self._turn_count = turn_number

    def shutdown(self) -> None:
        """Clean shutdown."""
        # ChromaDB PersistentClient auto-flushes; KG uses WAL mode.
        pass

    # -- Config schema for `hermes memory setup` ------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "palace_path",
                "description": "Path to MemPalace ChromaDB storage",
                "default": "~/.mempalace/palace",
            },
            {
                "key": "wing",
                "description": "Default wing name for Hermes conversations",
                "default": "hermes",
            },
            {
                "key": "kg_enabled",
                "description": "Enable temporal knowledge graph",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "identity_path",
                "description": "Path to identity.txt (L0 layer)",
                "default": "~/.mempalace/identity.txt",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write config to config.yaml under plugins.mempalace."""
        from pathlib import Path as P
        config_path = P(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            # Convert string booleans
            if "kg_enabled" in values:
                values["kg_enabled"] = values["kg_enabled"] in ("true", True)
            existing["plugins"]["mempalace"] = values
            with open(config_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save mempalace config: {e}")

    # -- Tool handlers --------------------------------------------------------

    def _handle_search(self, args: dict) -> str:
        query = args.get("query", "").strip()
        if not query:
            return tool_error("query is required")
        if not self._stack:
            return tool_error("MemPalace not initialized")

        wing = args.get("wing")
        room = args.get("room")
        n = min(int(args.get("n_results", 5)), 20)

        result = self._stack.l3.search(query, wing=wing, room=room, n_results=n)
        return json.dumps({"results": result})

    def _handle_store(self, args: dict) -> str:
        content = args.get("content", "").strip()
        if not content:
            return tool_error("content is required")
        if not self._collection:
            return tool_error("MemPalace not initialized")

        wing = args.get("wing", self._wing)
        room = args.get("room", "general")
        importance = int(args.get("importance", 3))

        timestamp = datetime.now().isoformat()
        drawer_id = (
            f"drawer_{wing}_{room}_"
            f"{hashlib.sha256((content + timestamp).encode()).hexdigest()[:24]}"
        )

        self._collection.upsert(
            documents=[content],
            ids=[drawer_id],
            metadatas=[{
                "wing": wing,
                "room": room,
                "source_file": "agent_store",
                "chunk_index": 0,
                "added_by": "hermes",
                "filed_at": timestamp,
                "importance": importance,
            }],
        )
        return json.dumps({
            "stored": True,
            "wing": wing,
            "room": room,
            "drawer_id": drawer_id,
        })

    def _handle_browse(self, args: dict) -> str:
        if not self._collection:
            return tool_error("MemPalace not initialized")

        target_wing = args.get("wing")

        # Fetch all metadata to build taxonomy
        try:
            _BATCH = 500
            all_metas = []
            offset = 0
            while True:
                batch = self._collection.get(
                    include=["metadatas"], limit=_BATCH, offset=offset,
                )
                metas = batch.get("metadatas", [])
                if not metas:
                    break
                all_metas.extend(metas)
                offset += len(metas)
                if len(metas) < _BATCH:
                    break
        except Exception as e:
            return tool_error(f"Browse failed: {e}")

        # Build wing → room → count map
        taxonomy: Dict[str, Dict[str, int]] = {}
        for m in all_metas:
            w = m.get("wing", "unknown")
            r = m.get("room", "unknown")
            taxonomy.setdefault(w, {})
            taxonomy[w][r] = taxonomy[w].get(r, 0) + 1

        if target_wing:
            rooms = taxonomy.get(target_wing, {})
            return json.dumps({
                "wing": target_wing,
                "rooms": rooms,
                "total_drawers": sum(rooms.values()),
            })

        # All wings summary
        summary = {}
        for w, rooms in sorted(taxonomy.items()):
            summary[w] = {
                "rooms": len(rooms),
                "drawers": sum(rooms.values()),
            }
        return json.dumps({"wings": summary, "total_drawers": len(all_metas)})

    def _handle_kg(self, args: dict) -> str:
        if not self._kg:
            return tool_error("Knowledge graph not enabled or not initialized")

        action = args.get("action", "")

        if action == "query":
            entity = args.get("entity", "").strip()
            if not entity:
                return tool_error("entity is required for query")
            as_of = args.get("as_of")
            results = self._kg.query_entity(entity, as_of=as_of)
            return json.dumps({"entity": entity, "facts": results})

        elif action == "add":
            subj = args.get("subject", "").strip()
            pred = args.get("predicate", "").strip()
            obj = args.get("object", "").strip()
            if not all([subj, pred, obj]):
                return tool_error("subject, predicate, and object are required")
            valid_from = args.get("valid_from", datetime.now().strftime("%Y-%m-%d"))
            self._kg.add_triple(subj, pred, obj, valid_from=valid_from)
            return json.dumps({"stored": True, "triple": [subj, pred, obj], "valid_from": valid_from})

        elif action == "timeline":
            entity = args.get("entity", "").strip()
            if not entity:
                return tool_error("entity is required for timeline")
            results = self._kg.query_entity(entity)
            return json.dumps({"entity": entity, "timeline": results})

        elif action == "invalidate":
            subj = args.get("subject", "").strip()
            pred = args.get("predicate", "").strip()
            obj = args.get("object", "").strip()
            if not all([subj, pred, obj]):
                return tool_error("subject, predicate, and object are required")
            valid_to = args.get("valid_to", datetime.now().strftime("%Y-%m-%d"))
            self._kg.invalidate(subj, pred, obj, ended=valid_to)
            return json.dumps({"invalidated": True, "triple": [subj, pred, obj], "ended": valid_to})

        return tool_error(f"Unknown kg action: {action}")

    def _handle_status(self, args: dict) -> str:
        status = {"provider": "mempalace", "palace_path": self._palace_path}

        if self._collection:
            status["total_drawers"] = self._collection.count()
        if self._stack:
            try:
                stack_status = self._stack.status()
                status["layers"] = stack_status
            except Exception:
                pass
        if self._kg:
            try:
                status["knowledge_graph"] = self._kg.stats() if hasattr(self._kg, "stats") else {"enabled": True}
            except Exception:
                status["knowledge_graph"] = {"enabled": True}

        status["wing"] = self._wing
        status["session_turns"] = self._turn_count
        return json.dumps(status)

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
        """Split text into overlapping chunks (MemPalace default: 800/100)."""
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks
