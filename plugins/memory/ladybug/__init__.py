"""Ladybug memory plugin — MemoryProvider backed by LadybugMemory.

LadybugMemory is a local columnar embedded graph database (*.lbdb), similar to
DuckDB, with:
  - BM25 keyword search and optional semantic (embedding) search
  - Importance-weighted recall
  - Typed memory entries (general, preference, fact, project, person, ...)
  - Graph edges between entries (link / get_related)
  - Knowledge graph via GLiNER2 entity extraction (optional — gracefully
    skipped when the model is not installed)

The plugin exposes 8 tools to the agent:
  ladybug_store           — store a new memory entry
  ladybug_search          — keyword / BM25 search
  ladybug_recall          — retrieve recent / high-importance memories
  ladybug_update          — update content, importance or metadata
  ladybug_delete          — delete a memory by ID
  ladybug_link            — create a named relationship between two memories
  ladybug_related         — traverse the relationship graph
  ladybug_entity          — entity-level KG queries (needs GLiNER2)

Prefetch runs in a background thread after every turn and injects
the top recalled + query-matched memories before the next API call.

Config in config.yaml (all optional):
  memory:
    provider: ladybug
    ladybug:
      db_path: ~/.hermes/ladybug.lbdb   # default: $HERMES_HOME/ladybug.lbdb
      prefetch_limit: 6                 # memories surfaced before each turn
      min_importance: 3                 # minimum importance for prefetch recall
      auto_link: false                  # auto-link mirrored built-in writes
"""

from __future__ import annotations

import importlib.util
import json
import logging
import threading
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

STORE_SCHEMA = {
    "name": "ladybug_store",
    "description": (
        "Persist a memory entry to Ladybug's local store. "
        "Use whenever the user shares something to remember across sessions: "
        "preferences, facts, decisions, project context, names, recurring tasks. "
        "Returns the assigned memory ID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The memory to store.",
            },
            "memory_type": {
                "type": "string",
                "enum": ["general", "preference", "fact", "project", "person", "event", "task"],
                "description": "Category of the memory (default: general).",
            },
            "importance": {
                "type": "integer",
                "description": "Importance score 1–10 (default: 5). Higher = surfaced more often.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional arbitrary key/value metadata to attach.",
            },
        },
        "required": ["content"],
    },
}

SEARCH_SCHEMA = {
    "name": "ladybug_search",
    "description": (
        "Keyword / full-text search across stored memories. "
        "Fast and cheap — no LLM involved. "
        "Use to find specific past facts by keyword, name, or phrase."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default: 8).",
            },
            "memory_type": {
                "type": "string",
                "description": "Filter by category (e.g. 'preference', 'project').",
            },
        },
        "required": ["query"],
    },
}

RECALL_SCHEMA = {
    "name": "ladybug_recall",
    "description": (
        "Retrieve the most important or recent memories. "
        "Use at conversation start to orient yourself, or when you need a "
        "general overview of what's been stored."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum memories to return (default: 10).",
            },
            "min_importance": {
                "type": "integer",
                "description": "Only return memories at or above this importance score (default: 0).",
            },
            "memory_type": {
                "type": "string",
                "description": "Filter by category.",
            },
        },
        "required": [],
    },
}

UPDATE_SCHEMA = {
    "name": "ladybug_update",
    "description": (
        "Update an existing memory by ID. "
        "Use when a fact changes or the user corrects earlier information. "
        "Omit fields you don't want to change."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "integer",
                "description": "ID of the memory to update (from ladybug_store or ladybug_search).",
            },
            "content": {
                "type": "string",
                "description": "Replacement content (omit to keep current).",
            },
            "importance": {
                "type": "integer",
                "description": "New importance score 1–10 (omit to keep current).",
            },
            "metadata": {
                "type": "object",
                "description": "Metadata to merge in (omit to keep current).",
            },
        },
        "required": ["memory_id"],
    },
}

DELETE_SCHEMA = {
    "name": "ladybug_delete",
    "description": (
        "Delete a memory by ID. "
        "Use when the user asks you to forget something, or when a stored "
        "fact is clearly no longer true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "integer",
                "description": "ID of the memory to delete.",
            },
        },
        "required": ["memory_id"],
    },
}

LINK_SCHEMA = {
    "name": "ladybug_link",
    "description": (
        "Create a named relationship edge between two memories. "
        "Use to make explicit connections: 'Alice works-at Acme', "
        "'project-X depends-on library-Y'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source_id": {
                "type": "integer",
                "description": "ID of the source memory.",
            },
            "target_id": {
                "type": "integer",
                "description": "ID of the target memory.",
            },
            "relation": {
                "type": "string",
                "description": "Relationship label (e.g. 'works-at', 'depends-on', 'related'). Default: 'related'.",
            },
        },
        "required": ["source_id", "target_id"],
    },
}

RELATED_SCHEMA = {
    "name": "ladybug_related",
    "description": (
        "Traverse the memory graph to find related entries. "
        "Use to discover connections: what else is linked to a given memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "integer",
                "description": "ID of the memory to start from.",
            },
            "relation": {
                "type": "string",
                "description": "Filter by relationship label (omit for all relations).",
            },
            "max_depth": {
                "type": "integer",
                "description": "Traversal depth (default: 1).",
            },
        },
        "required": ["memory_id"],
    },
}

ENTITY_SCHEMA = {
    "name": "ladybug_entity",
    "description": (
        "Entity-level knowledge graph queries powered by GLiNER2. "
        "Three actions:\n"
        "• extract — extract entities from a text snippet\n"
        "• search  — find memories mentioning a specific entity\n"
        "• graph   — explore entity relationships by entity ID\n"
        "Requires GLiNER2 to be installed; returns an error if unavailable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["extract", "search", "graph"],
                "description": "What to do.",
            },
            "content": {
                "type": "string",
                "description": "Text to extract entities from (required for 'extract').",
            },
            "entity_name": {
                "type": "string",
                "description": "Entity name to search for (required for 'search').",
            },
            "entity_id": {
                "type": "string",
                "description": "Entity ID to build a graph for (required for 'graph').",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Entity type labels for extraction (optional).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results for 'search' (default: 5).",
            },
            "max_depth": {
                "type": "integer",
                "description": "Graph traversal depth for 'graph' (default: 1).",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry_to_dict(entry) -> dict:
    """Convert a MemoryEntry dataclass to a JSON-serialisable dict."""
    return {
        "id": entry.id,
        "content": entry.content,
        "memory_type": entry.memory_type,
        "importance": entry.importance,
        "metadata": entry.metadata or {},
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
    }


def _result_to_dict(result) -> dict:
    """Convert a MemorySearchResult dataclass to a JSON-serialisable dict."""
    return {
        "score": result.score,
        **_entry_to_dict(result.entry),
    }


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class LadybugMemoryProvider(MemoryProvider):
    """Local LadybugMemory (*.lbdb) columnar embedded graph database provider with BM25 and graph edges."""

    def __init__(self):
        self._db: Optional[Any] = None          # LadybugMemory instance
        self._db_path: str = ""
        self._prefetch_limit: int = 6
        self._min_importance: int = 3
        self._auto_link: bool = False
        self._last_stored_id: Optional[int] = None  # for auto-link chaining

        # Background prefetch state
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None

    # -- Identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "ladybug"

    # -- Availability --------------------------------------------------------

    def is_available(self) -> bool:
        """Return True when the 'memory' package (LadybugMemory) is installed."""
        return importlib.util.find_spec("lbmemory") is not None

    # -- Config schema -------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        from hermes_constants import display_hermes_home
        default_db = f"{display_hermes_home()}/ladybug.lbdb"
        return [
            {
                "key": "db_path",
                "description": "Path to the Ladybug database file",
                "default": default_db,
            },
            {
                "key": "prefetch_limit",
                "description": "Number of memories to surface before each turn",
                "default": "6",
            },
            {
                "key": "min_importance",
                "description": "Minimum importance score for prefetch recall (0–10)",
                "default": "3",
            },
            {
                "key": "auto_link",
                "description": "Auto-link mirrored built-in memory writes",
                "default": "false",
                "choices": ["true", "false"],
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write config to config.yaml under memory.ladybug."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing: dict = {}
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("memory", {})
            existing["memory"].setdefault("ladybug", {})
            existing["memory"]["ladybug"].update(values)
            with open(config_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as e:
            logger.warning("Ladybug save_config failed: %s", e)

    # -- Lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        """Open (or create) the Ladybug database for this session."""
        try:
            from lbmemory import LadybugMemory
        except ImportError:
            logger.warning(
                "Ladybug plugin: 'memory' package not installed. "
                "Run: pip install ladybug-memory"
            )
            return

        # Resolve DB path
        from hermes_constants import get_hermes_home
        hermes_home = kwargs.get("hermes_home", str(get_hermes_home()))
        hermes_home_path = hermes_home  # str

        # Read plugin config from config.yaml (memory.ladybug section)
        plugin_cfg = self._load_plugin_config(hermes_home_path)

        import os
        raw_db_path = plugin_cfg.get("db_path", "")
        if raw_db_path:
            # Expand $HERMES_HOME placeholder in user-supplied paths
            raw_db_path = raw_db_path.replace("$HERMES_HOME", hermes_home_path)
            raw_db_path = raw_db_path.replace("${HERMES_HOME}", hermes_home_path)
            raw_db_path = os.path.expanduser(raw_db_path)
            self._db_path = raw_db_path
        else:
            self._db_path = os.path.join(hermes_home_path, "ladybug.lbdb")

        self._prefetch_limit = int(plugin_cfg.get("prefetch_limit", 6))
        self._min_importance = int(plugin_cfg.get("min_importance", 3))
        self._auto_link = str(plugin_cfg.get("auto_link", "false")).lower() == "true"

        try:
            self._db = LadybugMemory(self._db_path, enable_entity_extraction=True)
            logger.info("Ladybug opened at %s (%d entries)", self._db_path, self._db.count())
        except ImportError:
            # GLiNER2 / extract extra not installed — open without entity extraction.
            # ladybug_entity tool will still exist but will return a clear error message.
            logger.debug(
                "Ladybug: GLiNER2 not installed, opening without entity extraction "
                "(install ladybug-memory[extract] to enable ladybug_entity)"
            )
            try:
                self._db = LadybugMemory(self._db_path, enable_entity_extraction=False)
                logger.info(
                    "Ladybug opened at %s (%d entries, no entity extraction)",
                    self._db_path,
                    self._db.count(),
                )
            except Exception as e:
                logger.warning("Ladybug failed to open %s: %s", self._db_path, e)
                self._db = None
        except Exception as e:
            logger.warning("Ladybug failed to open %s: %s", self._db_path, e)
            self._db = None

    @staticmethod
    def _load_plugin_config(hermes_home: str) -> dict:
        """Read memory.ladybug section from config.yaml."""
        import os
        config_path = os.path.join(hermes_home, "config.yaml")
        try:
            import yaml
            with open(config_path) as f:
                all_cfg = yaml.safe_load(f) or {}
            return all_cfg.get("memory", {}).get("ladybug", {}) or {}
        except Exception:
            return {}

    def shutdown(self) -> None:
        """Wait for any running prefetch thread and release resources."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)
        self._db = None

    # -- System prompt -------------------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._db:
            return ""
        try:
            total = self._db.count()
        except Exception:
            total = 0
        if total == 0:
            return (
                "# Ladybug Memory\n"
                "Active (empty). Use ladybug_store to persist facts, preferences, "
                "decisions, and context the user would expect you to remember across sessions."
            )
        return (
            f"# Ladybug Memory\n"
            f"Active. {total} memories stored locally.\n"
            "Use ladybug_search / ladybug_recall to retrieve context, "
            "ladybug_store to add new memories, ladybug_update to correct old ones."
        )

    # -- Prefetch ------------------------------------------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Kick off a background recall + search for the next turn."""
        if not self._db:
            return

        def _run(q: str) -> None:
            try:
                lines: List[str] = []

                # 1. Importance-weighted recent recall
                recalled = self._db.recall(
                    limit=self._prefetch_limit,
                    min_importance=self._min_importance,
                )
                for entry in recalled:
                    lines.append(f"[{entry.memory_type}:{entry.importance}] {entry.content}")

                # 2. Query-relevant keyword search (deduplicated)
                if q:
                    seen_ids = {e.id for e in recalled}
                    results = self._db.search(q, limit=self._prefetch_limit)
                    for r in results:
                        if r.entry.id not in seen_ids:
                            lines.append(f"[{r.entry.memory_type}:{r.entry.importance}] {r.entry.content}")
                            seen_ids.add(r.entry.id)

                text = "\n".join(f"- {l}" for l in lines) if lines else ""
                with self._prefetch_lock:
                    self._prefetch_result = text
            except Exception as e:
                logger.debug("Ladybug prefetch failed: %s", e)

        # Cancel any still-running prefetch before starting a new one
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)

        self._prefetch_thread = threading.Thread(
            target=_run, args=(query,), daemon=True, name="ladybug-prefetch"
        )
        self._prefetch_thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached prefetch result (collected by the background thread)."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Ladybug Memory\n{result}"

    # -- Turn sync -----------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Ladybug is tool-driven; automatic ingestion is not performed."""

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            STORE_SCHEMA,
            SEARCH_SCHEMA,
            RECALL_SCHEMA,
            UPDATE_SCHEMA,
            DELETE_SCHEMA,
            LINK_SCHEMA,
            RELATED_SCHEMA,
            ENTITY_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._db:
            return json.dumps({"error": "Ladybug database is not initialised."})

        try:
            if tool_name == "ladybug_store":
                return self._tool_store(args)
            elif tool_name == "ladybug_search":
                return self._tool_search(args)
            elif tool_name == "ladybug_recall":
                return self._tool_recall(args)
            elif tool_name == "ladybug_update":
                return self._tool_update(args)
            elif tool_name == "ladybug_delete":
                return self._tool_delete(args)
            elif tool_name == "ladybug_link":
                return self._tool_link(args)
            elif tool_name == "ladybug_related":
                return self._tool_related(args)
            elif tool_name == "ladybug_entity":
                return self._tool_entity(args)
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            logger.exception("Ladybug tool %s failed", tool_name)
            return json.dumps({"error": str(e)})

    # -- Individual tool handlers --------------------------------------------

    def _tool_store(self, args: dict) -> str:
        content = args.get("content", "").strip()
        if not content:
            return json.dumps({"error": "content is required"})

        entry = self._db.store(
            content=content,
            memory_type=args.get("memory_type", "general"),
            importance=int(args.get("importance", 5)),
            metadata=args.get("metadata") or None,
        )
        self._last_stored_id = entry.id
        return json.dumps({"memory_id": entry.id, "status": "stored"})

    def _tool_search(self, args: dict) -> str:
        query = args.get("query", "").strip()
        if not query:
            return json.dumps({"error": "query is required"})

        results = self._db.search(
            query=query,
            limit=int(args.get("limit", 8)),
            memory_type=args.get("memory_type") or None,
        )
        return json.dumps({
            "results": [_result_to_dict(r) for r in results],
            "count": len(results),
        })

    def _tool_recall(self, args: dict) -> str:
        entries = self._db.recall(
            limit=int(args.get("limit", 10)),
            min_importance=int(args.get("min_importance", 0)),
            memory_type=args.get("memory_type") or None,
        )
        return json.dumps({
            "memories": [_entry_to_dict(e) for e in entries],
            "count": len(entries),
        })

    def _tool_update(self, args: dict) -> str:
        memory_id = args.get("memory_id")
        if memory_id is None:
            return json.dumps({"error": "memory_id is required"})

        entry = self._db.update(
            memory_id=str(memory_id),
            content=args.get("content") or None,
            importance=int(args["importance"]) if "importance" in args else None,
            metadata=args.get("metadata") or None,
        )
        if entry is None:
            return json.dumps({"error": f"Memory {memory_id} not found"})
        return json.dumps({"status": "updated", **_entry_to_dict(entry)})

    def _tool_delete(self, args: dict) -> str:
        memory_id = args.get("memory_id")
        if memory_id is None:
            return json.dumps({"error": "memory_id is required"})

        ok = self._db.delete(str(memory_id))
        if not ok:
            return json.dumps({"error": f"Memory {memory_id} not found or already deleted"})
        return json.dumps({"status": "deleted", "memory_id": memory_id})

    def _tool_link(self, args: dict) -> str:
        source_id = args.get("source_id")
        target_id = args.get("target_id")
        if source_id is None or target_id is None:
            return json.dumps({"error": "source_id and target_id are required"})

        ok = self._db.link(
            source_id=str(source_id),
            target_id=str(target_id),
            relation=args.get("relation", "related"),
        )
        if not ok:
            return json.dumps({"error": "Failed to create link (one or both IDs may not exist)"})
        return json.dumps({
            "status": "linked",
            "source_id": source_id,
            "target_id": target_id,
            "relation": args.get("relation", "related"),
        })

    def _tool_related(self, args: dict) -> str:
        memory_id = args.get("memory_id")
        if memory_id is None:
            return json.dumps({"error": "memory_id is required"})

        pairs = self._db.get_related(
            memory_id=str(memory_id),
            relation=args.get("relation") or None,
            max_depth=int(args.get("max_depth", 1)),
        )
        items = []
        for entry, relation in pairs:
            items.append({"relation": relation, **_entry_to_dict(entry)})
        return json.dumps({"related": items, "count": len(items)})

    def _tool_entity(self, args: dict) -> str:
        action = args.get("action", "")
        if not action:
            return json.dumps({"error": "action is required"})

        try:
            if action == "extract":
                content = args.get("content", "")
                if not content:
                    return json.dumps({"error": "content is required for extract"})
                entities = self._db.extract_entities(
                    content=content,
                    labels=args.get("labels") or None,
                )
                # Entity objects may not be JSON-serialisable — coerce to str
                return json.dumps({
                    "entities": [
                        e if isinstance(e, (str, dict)) else str(e)
                        for e in entities
                    ],
                    "count": len(entities),
                })

            elif action == "search":
                entity_name = args.get("entity_name", "")
                if not entity_name:
                    return json.dumps({"error": "entity_name is required for search"})
                memories = self._db.search_by_entity(
                    entity_name=entity_name,
                    limit=int(args.get("limit", 5)),
                )
                return json.dumps({
                    "memories": [
                        m if isinstance(m, dict) else str(m)
                        for m in memories
                    ],
                    "count": len(memories),
                })

            elif action == "graph":
                entity_id = args.get("entity_id", "")
                if not entity_id:
                    return json.dumps({"error": "entity_id is required for graph"})
                graph = self._db.get_entity_graph(
                    entity_id=entity_id,
                    max_depth=int(args.get("max_depth", 1)),
                )
                return json.dumps(graph)

            return json.dumps({"error": f"Unknown action: {action}"})

        except NotImplementedError:
            return json.dumps({
                "error": (
                    "Entity KG features require GLiNER2 to be installed. "
                    "Run: pip install gliner"
                )
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    # -- Setup hook ----------------------------------------------------------

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Called by 'hermes memory setup' after provider selection.

        Writes activation to config.yaml, then attempts to open the DB
        so the user gets immediate feedback on whether the path is valid.
        """
        from hermes_cli.config import save_config
        config.setdefault("memory", {})["provider"] = "ladybug"
        save_config(config)
        print(f"\n  Memory provider: ladybug")
        print(f"  Activation saved to config.yaml")

        # Test DB creation at the configured path
        self.initialize(session_id="setup-test", hermes_home=hermes_home)
        if self._db is not None:
            try:
                count = self._db.count()
            except Exception:
                count = 0
            print(f"  ✓ Ladybug DB opened at {self._db_path} ({count} entries)")
            self.shutdown()
        else:
            print(f"  ✗ Failed to open DB at {self._db_path}")
            print(f"    Check that the path is writable and lbmemory is installed.")
        print()

    # -- Optional hooks ------------------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in MEMORY.md / USER.md writes into Ladybug."""
        if not self._db or not content:
            return

        if action == "add":
            try:
                memory_type = "preference" if target == "user" else "fact"
                entry = self._db.store(
                    content=content,
                    memory_type=memory_type,
                    importance=6,  # slightly above default — explicit user signal
                )
                logger.debug("Ladybug mirrored built-in write → id=%d", entry.id)
            except Exception as e:
                logger.debug("Ladybug on_memory_write failed: %s", e)

        elif action == "replace":
            # Best-effort: search for a close match and update it
            try:
                results = self._db.search(content[:60], limit=1)
                if results:
                    self._db.update(
                        memory_id=str(results[0].entry.id),
                        content=content,
                    )
                else:
                    memory_type = "preference" if target == "user" else "fact"
                    self._db.store(content=content, memory_type=memory_type, importance=6)
            except Exception as e:
                logger.debug("Ladybug on_memory_write (replace) failed: %s", e)

    def on_pre_compress(self, messages) -> str:
        """Surface high-importance memories to include in compression summary."""
        if not self._db:
            return ""
        try:
            entries = self._db.recall(limit=5, min_importance=7)
            if not entries:
                return ""
            lines = [f"- [{e.memory_type}] {e.content}" for e in entries]
            return "Key Ladybug memories (importance ≥ 7):\n" + "\n".join(lines)
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register LadybugMemoryProvider with the plugin system."""
    ctx.register_memory_provider(LadybugMemoryProvider())
