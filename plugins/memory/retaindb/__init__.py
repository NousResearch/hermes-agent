"""RetainDB memory plugin — MemoryProvider interface.

Cross-session memory via RetainDB cloud API. Durable write-behind queue,
semantic search with deduplication, and user profile retrieval.

Original PR #2732 by Alinxus, adapted to MemoryProvider ABC.

Config via environment variables:
  RETAINDB_API_KEY    — API key (required)
  RETAINDB_BASE_URL   — API endpoint (default: https://api.retaindb.com)
  RETAINDB_PROJECT    — Project identifier (default: hermes)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.retaindb.com"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "retaindb_profile",
    "description": "Get the user's stable profile — preferences, facts, and patterns.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "retaindb_search",
    "description": (
        "Semantic search across stored memories. Returns ranked results "
        "with relevance scores."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 8, max: 20)."},
        },
        "required": ["query"],
    },
}

CONTEXT_SCHEMA = {
    "name": "retaindb_context",
    "description": "Synthesized 'what matters now' context block for the current task.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Current task or question."},
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "retaindb_remember",
    "description": "Persist an explicit fact or preference to long-term memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to remember."},
            "memory_type": {
                "type": "string",
                "enum": ["preference", "fact", "decision", "context"],
                "description": "Category (default: fact).",
            },
            "importance": {
                "type": "number",
                "description": "Importance 0-1 (default: 0.5).",
            },
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA = {
    "name": "retaindb_forget",
    "description": "Delete a specific memory by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID to delete."},
        },
        "required": ["memory_id"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class RetainDBMemoryProvider(MemoryProvider):
    """RetainDB cloud memory with write-behind queue and semantic search."""

    def __init__(self):
        self._api_key = ""
        self._base_url = _DEFAULT_BASE_URL
        self._project = "hermes"
        self._user_id = ""
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None

    @property
    def name(self) -> str:
        return "retaindb"

    def is_available(self) -> bool:
        return bool(os.environ.get("RETAINDB_API_KEY"))

    def get_config_schema(self):
        return [
            {"key": "api_key", "description": "RetainDB API key", "secret": True, "required": True, "env_var": "RETAINDB_API_KEY", "url": "https://retaindb.com"},
            {"key": "base_url", "description": "API endpoint", "default": "https://api.retaindb.com"},
            {"key": "project", "description": "Project identifier", "default": "hermes"},
        ]

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _api(self, method: str, path: str, **kwargs):
        """Make an API call to RetainDB."""
        import requests
        url = f"{self._base_url}{path}"
        resp = requests.request(method, url, headers=self._headers(), timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def initialize(self, session_id: str, **kwargs) -> None:
        self._api_key = os.environ.get("RETAINDB_API_KEY", "")
        self._base_url = os.environ.get("RETAINDB_BASE_URL", _DEFAULT_BASE_URL)
        self._user_id = kwargs.get("user_id", "default")
        self._session_id = session_id

        # Derive profile-scoped project name so different profiles don't
        # share server-side memory.  Explicit RETAINDB_PROJECT always wins.
        explicit_project = os.environ.get("RETAINDB_PROJECT")
        if explicit_project:
            self._project = explicit_project
        else:
            hermes_home = kwargs.get("hermes_home", "")
            profile_name = os.path.basename(hermes_home) if hermes_home else ""
            # Default profile (~/.hermes) → "hermes"; named profiles → "hermes-<name>"
            if profile_name and profile_name != ".hermes":
                self._project = f"hermes-{profile_name}"
            else:
                self._project = "hermes"

    def system_prompt_block(self) -> str:
        return (
            "# RetainDB Memory\n"
            f"Active. Project: {self._project}.\n"
            "Use retaindb_search to find memories, retaindb_remember to store facts, "
            "retaindb_profile for a user overview, retaindb_context for task-relevant context."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## RetainDB Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        def _run():
            try:
                data = self._api("POST", "/v1/recall", json={
                    "project": self._project,
                    "query": query,
                    "user_id": self._user_id,
                    "top_k": 5,
                })
                results = data.get("results", [])
                if results:
                    lines = [r.get("content", "") for r in results if r.get("content")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
            except Exception as e:
                logger.debug("RetainDB prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="retaindb-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Ingest conversation turn in background (non-blocking)."""
        def _sync():
            try:
                self._api("POST", "/v1/ingest", json={
                    "project": self._project,
                    "user_id": self._user_id,
                    "session_id": self._session_id,
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                })
            except Exception as e:
                logger.warning("RetainDB sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="retaindb-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONTEXT_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        try:
            if tool_name == "retaindb_profile":
                data = self._api("GET", f"/v1/profile/{self._project}/{self._user_id}")
                return json.dumps(data)

            elif tool_name == "retaindb_search":
                query = args.get("query", "")
                if not query:
                    return json.dumps({"error": "query is required"})
                data = self._api("POST", "/v1/search", json={
                    "project": self._project,
                    "user_id": self._user_id,
                    "query": query,
                    "top_k": min(int(args.get("top_k", 8)), 20),
                })
                return json.dumps(data)

            elif tool_name == "retaindb_context":
                query = args.get("query", "")
                if not query:
                    return json.dumps({"error": "query is required"})
                data = self._api("POST", "/v1/recall", json={
                    "project": self._project,
                    "user_id": self._user_id,
                    "query": query,
                    "top_k": 5,
                })
                return json.dumps(data)

            elif tool_name == "retaindb_remember":
                content = args.get("content", "")
                if not content:
                    return json.dumps({"error": "content is required"})
                data = self._api("POST", "/v1/remember", json={
                    "project": self._project,
                    "user_id": self._user_id,
                    "content": content,
                    "memory_type": args.get("memory_type", "fact"),
                    "importance": float(args.get("importance", 0.5)),
                })
                return json.dumps(data)

            elif tool_name == "retaindb_forget":
                memory_id = args.get("memory_id", "")
                if not memory_id:
                    return json.dumps({"error": "memory_id is required"})
                data = self._api("DELETE", f"/v1/memory/{memory_id}")
                return json.dumps(data)

            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if action == "add":
            try:
                self._api("POST", "/v1/remember", json={
                    "project": self._project,
                    "user_id": self._user_id,
                    "content": content,
                    "memory_type": "preference" if target == "user" else "fact",
                })
            except Exception as e:
                logger.debug("RetainDB memory bridge failed: %s", e)

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)


def register(ctx) -> None:
    """Register RetainDB as a memory provider plugin."""
    ctx.register_memory_provider(RetainDBMemoryProvider())
