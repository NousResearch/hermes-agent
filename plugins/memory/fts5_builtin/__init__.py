"""
FTS5BuiltinMemory — local FTS5 full-text search memory provider.

Zero-dependency semantic-ish memory for Hermes.  Stripped from the
holographic plugin — no HRR vectors, no entities, no trust scoring,
no memory_banks.  Pure SQLite FTS5 keyword retrieval.

Usage (config.yaml):
  memory:
    provider: fts5_builtin

The provider auto-creates ``fts5_memory.db`` in the Hermes home directory.
Prefetch results are injected into the user message at every turn (NOT
the system prompt — preserving the prompt prefix cache).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# tool schemas -----------------------------------------------------------

FTS5_ADD_SCHEMA = {
    "name": "fts5_add",
    "description": (
        "Store a fact in the FTS5 keyword memory. "
        "Use for things you want to recall later via keyword search. "
        "content: the fact text.  category (optional): 'user_pref' | 'project' | 'tool' | 'general'. "
        "tags (optional): comma-separated tags for filtering."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Fact content to store."},
            "category": {
                "type": "string", "enum": ["user_pref", "project", "tool", "general"],
                "description": "Category for filtering. Default: general.",
            },
            "tags": {"type": "string", "description": "Comma-separated tags."},
        },
        "required": ["content"],
    },
}

FTS5_SEARCH_SCHEMA = {
    "name": "fts5_search",
    "description": (
        "Full-text search across stored facts.  Returns relevant facts "
        "ranked by FTS5 relevance.  Use this before answering questions "
        "that may rely on stored knowledge."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords to search for."},
            "category": {
                "type": "string", "enum": ["user_pref", "project", "tool", "general"],
                "description": "Optional category filter.",
            },
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["query"],
    },
}

FTS5_LIST_SCHEMA = {
    "name": "fts5_list",
    "description": "List recently stored facts, optionally filtered by category.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string", "enum": ["user_pref", "project", "tool", "general"],
            },
            "limit": {"type": "integer", "description": "Max results (default: 50)."},
        },
        "required": [],
    },
}

# -----------------------------------------------------------------------


class FTS5BuiltinProvider(MemoryProvider):
    """Minimal FTS5 keyword memory — zero deps, ~150 lines total."""

    def __init__(self):
        self._store = None
        self._db_path = ""

    # -- MemoryProvider interface ---------------------------------------

    @property
    def name(self) -> str:
        return "fts5_builtin"

    def is_available(self) -> bool:
        return True  # zero deps — always works

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home", "~/.hermes")
        self._db_path = str(Path(hermes_home) / "fts5_memory.db")
        from .store import FTS5Store
        self._store = FTS5Store(self._db_path)

    def system_prompt_block(self) -> str:
        return (
            "## FTS5 Keyword Memory\n"
            "You have a local keyword-searchable memory (fts5_add / fts5_search / fts5_list). "
            "Use fts5_search to recall stored facts before answering relevant questions.\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Run FTS5 search on the user query, return formatted context."""
        if not self._store or not query.strip():
            return ""

        results = self._store.search_facts(query, limit=5)
        if not results:
            return ""

        lines = ["## Recalled facts (FTS5 keyword match)"]
        for r in results:
            cat = r.get("category", "general")
            content = r.get("content", "")
            tags = r.get("tags", "")
            tag_str = f" [{tags}]" if tags else ""
            lines.append(f"- [{cat}]{tag_str} {content}")
        return "\n".join(lines)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FTS5_ADD_SCHEMA, FTS5_SEARCH_SCHEMA, FTS5_LIST_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any],
                         **kwargs) -> str:
        if not self._store:
            return json.dumps({"ok": False, "error": "store not initialized"})

        try:
            if tool_name == "fts5_add":
                fid = self._store.add_fact(
                    content=args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                return json.dumps({"ok": True, "fact_id": fid})

            elif tool_name == "fts5_search":
                results = self._store.search_facts(
                    query=args["query"],
                    category=args.get("category"),
                    limit=args.get("limit", 10),
                )
                return json.dumps({"ok": True, "results": results})

            elif tool_name == "fts5_list":
                results = self._store.list_facts(
                    category=args.get("category"),
                    limit=args.get("limit", 50),
                )
                return json.dumps({"ok": True, "results": results})

            else:
                return json.dumps({"ok": False, "error": f"unknown tool: {tool_name}"})

        except Exception as e:
            logger.debug("fts5_builtin tool error: %s", e)
            return json.dumps({"ok": False, "error": str(e)})

    def shutdown(self) -> None:
        self._store = None
