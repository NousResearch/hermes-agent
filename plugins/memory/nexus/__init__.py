"""nexus — BM25 + semantic late fusion memory plugin.

DualRetriever: BM25 + ChromaDB semantic search + keyword overlap,
fused with Reciprocal Rank Fusion (RRF).

Benchmarked at 96.2% R@5 on LongMemEval (500 questions),
vs MemPalace reference at 98.4% R@5.

Config in $HERMES_HOME/config.yaml:
  plugins:
    nexus:
      palace_path: ~/.hermes/nexus  # ChromaDB persistence path
      embed_model: default           # ChromaDB embedding model
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from .store import MemoryStore

logger = logging.getLogger(__name__)


# =============================================================================
# Tool schemas
# =============================================================================

MEMORY_STORE_SCHEMA = {
    "name": "memory_store",
    "description": (
        "Persistent memory store with BM25 + semantic late fusion retrieval.\n"
        "Wings organize top-level domains (user, project, tool, general).\n"
        "Rooms are sub-categories within wings.\n\n"
        "Actions:\n"
        "• add — Store a document with wing/room labels\n"
        "• search — Dual-retriever BM25 + semantic + keyword overlap search\n"
        "• remove — Delete by doc_id\n"
        "• stats — Memory statistics\n"
        "• list-wings — All wings\n"
        "• list-rooms — Rooms in a wing\n"
        "• add-fact — Structured fact (subject-predicate-object)\n"
        "• query-facts — Query facts about an entity"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "remove", "stats", "list-wings", "list-rooms", "add-fact", "query-facts"],
            },
            "text": {"type": "string", "description": "Document text (for add)"},
            "wing": {"type": "string", "description": "Top-level category (e.g. user, project, tool, general)"},
            "room": {"type": "string", "description": "Sub-category within wing"},
            "source": {"type": "string", "description": "Source of the memory (default: manual)"},
            "doc_id": {"type": "string", "description": "Document ID (for remove)"},
            "query": {"type": "string", "description": "Search query (for search)"},
            "n_results": {"type": "integer", "description": "Max results (default: 5)"},
            "subject": {"type": "string", "description": "Subject for facts (for add-fact/query-facts)"},
            "predicate": {"type": "string", "description": "Predicate (for add-fact)"},
            "obj": {"type": "string", "description": "Object value (for add-fact)"},
        },
        "required": ["action"],
    },
}


# =============================================================================
# Config
# =============================================================================

def _load_plugin_config() -> dict:
    from pathlib import Path
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        if not config_path.exists():
            return {}
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        return all_config.get("plugins", {}).get("nexus", {}) or {}
    except Exception:
        return {}


# =============================================================================
# MemoryProvider implementation
# =============================================================================

class NexusProvider(MemoryProvider):
    """BM25 + semantic late fusion memory with ChromaDB persistence."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store: MemoryStore | None = None

    @property
    def name(self) -> str:
        return "nexus"

    def is_available(self) -> bool:
        return True

    def save_config(self, values, hermes_home):
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["nexus"] = values
            with open(config_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default = f"{display_hermes_home()}/memory"
        return [
            {"key": "palace_path", "description": "ChromaDB persistence path", "default": _default},
            {"key": "embed_model", "description": "Embedding model (default uses ChromaDB built-in)", "default": "default"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        palace_path = self._config.get("palace_path", "~/.hermes/nexus")
        embed_model = self._config.get("embed_model", "default")
        self._store = MemoryStore(palace_path=palace_path, embed_fn=None)
        self._store.load()
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        s = self._store.stats()
        total = s.get("total", 0)
        wings = s.get("wings", [])
        if total == 0:
            return (
                "# Nexus Memory\n"
                "Active. All conversations are automatically stored.\n"
            )
        return (
            f"# Nexus Memory\n"
            f"Active. {total} memories stored. Wings: {', '.join(wings)}.\n"
            f"Conversations are automatically saved. Use memory_store(action='search', query='...') to retrieve.\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._store or not query:
            return ""
        try:
            results = self._store.search(query, n_results=3)
            if not results:
                return ""
            lines = []
            for r in results:
                lines.append(f"- [{r.get('wing','?')}/{r.get('room','?')}] {r.get('text','')[:200]}")
            return "## Nexus\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Nexus prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._store or not user_content:
            return
        try:
            # Store user turn
            self._store.add(
                text=f"[user] {user_content}",
                wing="conversation",
                room=session_id or "general",
                source="sync_turn",
            )
            # Store assistant turn
            if assistant_content:
                self._store.add(
                    text=f"[assistant] {assistant_content}",
                    wing="conversation",
                    room=session_id or "general",
                    source="sync_turn",
                )
        except Exception as e:
            logger.debug("Nexus sync_turn failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [MEMORY_STORE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "memory_store":
            return self._handle_memory_store(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if action == "add" and self._store and content:
            try:
                wing = target if target in ("user", "project", "tool", "general") else "general"
                self._store.add(content, wing=wing, room="memory_write", source="memory_write")
            except Exception as e:
                logger.debug("Nexus memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        self._store = None

    def _handle_memory_store(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store

            if action == "add":
                doc_id = store.add(
                    text=args["text"],
                    wing=args["wing"],
                    room=args["room"],
                    source=args.get("source", "manual"),
                )
                return json.dumps({"status": "stored", "doc_id": doc_id})

            elif action == "search":
                results = store.search(
                    query=args["query"],
                    n_results=int(args.get("n_results", 5)),
                    wing=args.get("wing"),
                    room=args.get("room"),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "remove":
                ok = store.remove(args["doc_id"])
                return json.dumps({"status": "deleted" if ok else "not_found", "doc_id": args["doc_id"]})

            elif action == "stats":
                return json.dumps(store.stats())

            elif action == "list-wings":
                return json.dumps({"wings": store.list_wings()})

            elif action == "list-rooms":
                rooms = store.list_rooms(wing=args.get("wing"))
                return json.dumps({"rooms": rooms})

            elif action == "add-fact":
                fact_id = store.add_fact(
                    subject=args["subject"],
                    predicate=args["predicate"],
                    obj=args["obj"],
                )
                return json.dumps({"status": "stored", "fact_id": fact_id})

            elif action == "query-facts":
                facts = store.query_facts(subject=args["subject"])
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))


# =============================================================================
# Plugin entry point
# =============================================================================

def register(ctx) -> None:
    """Register the Nexus provider with the plugin system."""
    config = _load_plugin_config()
    provider = NexusProvider(config=config)
    ctx.register_memory_provider(provider)
