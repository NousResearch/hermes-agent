"""Cyber Memory provider — first-class Hermes memory via a local stdio MCP client."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from .client import CyberMemoryClient, cyber_memory_mcp_available

logger = logging.getLogger(__name__)

_DEFAULT_COMMAND = "cyber-memory"
_CONFIG_NAME = "cyber-memory.json"
_MIN_QUERY_LEN = 8
_REQUIRED_BACKEND_TOOLS = {
    "memory_store",
    "memory_recall",
    "memory_search",
    "memory_relate",
    "memory_graph",
    "memory_update",
    "memory_forget",
    "memory_stats",
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _config_path(hermes_home: str | Path) -> Path:
    return Path(hermes_home) / _CONFIG_NAME


def _default_db_path(hermes_home: str | Path) -> str:
    return str(Path(hermes_home) / "cyber-memory" / "db.sqlite3")


def _load_config(hermes_home: str | Path | None = None) -> dict:
    from hermes_constants import get_hermes_home

    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    config = {
        "command": _DEFAULT_COMMAND,
        "db_path": _default_db_path(home),
    }
    path = _config_path(home)
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config.update(loaded)
        except Exception:
            pass
    return config


def _resolve_command(command: str) -> Optional[str]:
    if not command:
        return None
    if os.path.sep in command:
        path = Path(command).expanduser()
        return str(path) if path.exists() else None
    return shutil.which(command)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

STORE_SCHEMA = {
    "name": "cyber_memory_store",
    "description": (
        "Store content in Cyber Memory. The backend generates embeddings and "
        "stores the memory locally with optional tags, importance, kind, and source."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to store."},
            "kind": {
                "type": "string",
                "enum": ["episodic", "semantic", "procedural"],
                "description": "Memory kind (default: semantic).",
            },
            "importance": {"type": "number", "description": "Importance multiplier (default: 1.0)."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags to attach to the memory.",
            },
            "source": {"type": "string", "description": "Optional source label."},
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "cyber_memory_recall",
    "description": (
        "Recall semantically related memories from Cyber Memory using vector "
        "similarity, recency, and importance scoring."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language recall query."},
            "limit": {"type": "integer", "description": "Maximum results to return (default: 5)."},
            "kind": {
                "type": "string",
                "enum": ["episodic", "semantic", "procedural"],
                "description": "Optional kind filter.",
            },
            "min_score": {"type": "number", "description": "Minimum recall score threshold."},
        },
        "required": ["query"],
    },
}

SEARCH_SCHEMA = {
    "name": "cyber_memory_search",
    "description": "Run keyword/full-text search against locally stored Cyber Memory entries.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keyword query."},
            "limit": {"type": "integer", "description": "Maximum results to return (default: 10)."},
        },
        "required": ["query"],
    },
}

RELATE_SCHEMA = {
    "name": "cyber_memory_relate",
    "description": "Create a typed relation edge between two Cyber Memory entries.",
    "parameters": {
        "type": "object",
        "properties": {
            "src_id": {"type": "integer", "description": "Source memory ID."},
            "dst_id": {"type": "integer", "description": "Destination memory ID."},
            "kind": {
                "type": "string",
                "enum": ["supports", "contradicts", "precedes", "relates_to"],
                "description": "Edge relationship type.",
            },
            "weight": {"type": "number", "description": "Optional edge weight (default: 1.0)."},
        },
        "required": ["src_id", "dst_id", "kind"],
    },
}

GRAPH_SCHEMA = {
    "name": "cyber_memory_graph",
    "description": "Traverse the Cyber Memory knowledge graph from a root memory ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "Root memory ID."},
            "depth": {"type": "integer", "description": "Traversal depth (default: 2)."},
        },
        "required": ["id"],
    },
}

UPDATE_SCHEMA = {
    "name": "cyber_memory_update",
    "description": "Update stored Cyber Memory content, tags, importance, or kind.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "Memory ID to update."},
            "content": {"type": "string", "description": "Replacement content."},
            "kind": {
                "type": "string",
                "enum": ["episodic", "semantic", "procedural"],
                "description": "Optional replacement kind.",
            },
            "importance": {"type": "number", "description": "Optional replacement importance."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional replacement tags.",
            },
        },
        "required": ["id"],
    },
}

FORGET_SCHEMA = {
    "name": "cyber_memory_forget",
    "description": "Delete a Cyber Memory entry and its graph relations.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "Memory ID to delete."},
        },
        "required": ["id"],
    },
}

STATS_SCHEMA = {
    "name": "cyber_memory_stats",
    "description": "Return Cyber Memory database statistics and timestamps.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_TOOL_SCHEMAS = [
    STORE_SCHEMA,
    RECALL_SCHEMA,
    SEARCH_SCHEMA,
    RELATE_SCHEMA,
    GRAPH_SCHEMA,
    UPDATE_SCHEMA,
    FORGET_SCHEMA,
    STATS_SCHEMA,
]


class CyberMemoryProvider(MemoryProvider):
    """First-class Hermes MemoryProvider backed by the cyber-memory binary."""

    def __init__(self):
        self._config: dict = {}
        self._client: Optional[CyberMemoryClient] = None
        self._command: str = _DEFAULT_COMMAND
        self._db_path: str = ""
        self._session_id: str = ""
        self._available_tools: set[str] = set()
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "cyber_memory"

    def is_available(self) -> bool:
        cfg = _load_config()
        return cyber_memory_mcp_available() and bool(_resolve_command(cfg.get("command", _DEFAULT_COMMAND)))

    def get_config_schema(self):
        from hermes_constants import display_hermes_home

        return [
            {
                "key": "command",
                "description": "Cyber Memory command or absolute binary path",
                "default": _DEFAULT_COMMAND,
            },
            {
                "key": "db_path",
                "description": "SQLite database path",
                "default": f"{display_hermes_home()}/cyber-memory/db.sqlite3",
            },
        ]

    def save_config(self, values, hermes_home):
        path = _config_path(hermes_home)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        if not hermes_home:
            from hermes_constants import get_hermes_home
            hermes_home = str(get_hermes_home())

        self._config = _load_config(hermes_home)
        self._command = self._config.get("command", _DEFAULT_COMMAND)
        self._db_path = self._config.get("db_path") or _default_db_path(hermes_home)
        self._session_id = session_id

        resolved = _resolve_command(self._command)
        if not resolved:
            raise RuntimeError(
                f"Cyber Memory command not found: {self._command}. "
                "Install it first or configure memory.cyber_memory.command."
            )

        Path(self._db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CYBER_MEMORY_DB"] = str(Path(self._db_path).expanduser())

        self._client = CyberMemoryClient()
        self._client.start(resolved, [], env)
        self._available_tools = set(self._client.list_tools())
        missing = sorted(_REQUIRED_BACKEND_TOOLS - self._available_tools)
        if missing:
            self.shutdown()
            raise RuntimeError(
                "Cyber Memory binary is missing required MCP tools: "
                + ", ".join(missing)
            )

    def system_prompt_block(self) -> str:
        if not self._client:
            return ""
        tool_count = len(self._available_tools) or len(_TOOL_SCHEMAS)
        return (
            "# Cyber Memory\n"
            f"Active. Local persistent memory at {self._db_path}.\n"
            f"{tool_count} memory tools available for storage, semantic recall, "
            "graph traversal, updates, deletion, and stats."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._client or not query or len(query.strip()) < _MIN_QUERY_LEN:
            return

        def _run():
            data = self._invoke_backend(
                "memory_recall",
                {"query": query.strip()[:5000], "limit": 5, "min_score": 0.25},
            )
            formatted = self._format_prefetch(data)
            if formatted:
                with self._prefetch_lock:
                    self._prefetch_result = formatted

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="cyber-memory-prefetch"
        )
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._client or len(user_content.strip()) < _MIN_QUERY_LEN:
            return

        def _sync():
            content = (
                f"User: {user_content[:2000]}\n"
                f"Assistant: {assistant_content[:2000]}"
            )
            self._invoke_backend(
                "memory_store",
                {
                    "content": content,
                    "kind": "episodic",
                    "importance": 1.0,
                    "tags": ["conversation", "hermes"],
                    "source": "hermes",
                },
            )

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="cyber-memory-sync"
        )
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not tool_name.startswith("cyber_memory_"):
            return json.dumps({"error": f"Unknown Cyber Memory tool: {tool_name}"})
        backend_tool = "memory_" + tool_name[len("cyber_memory_"):]
        result = self._invoke_backend(backend_tool, args)
        return json.dumps(result)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._client or action not in ("add", "replace") or not content:
            return

        def _write():
            tags = ["builtin", target]
            self._invoke_backend(
                "memory_store",
                {
                    "content": content[:4000],
                    "kind": "semantic",
                    "importance": 1.3 if target == "user" else 1.0,
                    "tags": tags,
                    "source": f"hermes_{target}",
                },
            )

        t = threading.Thread(target=_write, daemon=True, name="cyber-memory-mirror")
        t.start()

    def shutdown(self) -> None:
        for t in (self._sync_thread, self._prefetch_thread):
            if t and t.is_alive():
                t.join(timeout=10.0)
        if self._client:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _invoke_backend(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self._client:
            return {"error": "Cyber Memory provider is not initialized"}

        result = self._client.call_tool(tool_name, args)
        if isinstance(result, dict):
            return result
        return {"result": result}

    def _format_prefetch(self, data: Dict[str, Any]) -> str:
        if not data or data.get("error"):
            return ""

        text = data.get("result")
        if isinstance(text, str) and text.strip():
            return f"## Cyber Memory\n{text.strip()}"

        items = None
        for key in ("results", "memories", "items"):
            value = data.get(key)
            if isinstance(value, list):
                items = value
                break
        if not items:
            return ""

        lines = []
        for item in items[:5]:
            if isinstance(item, dict):
                content = str(item.get("content", "")).strip()
                score = item.get("score")
                prefix = f"[{score:.2f}] " if isinstance(score, (int, float)) else ""
                if content:
                    lines.append(f"- {prefix}{content}")
            elif item:
                lines.append(f"- {item}")

        if not lines:
            return ""
        return "## Cyber Memory\n" + "\n".join(lines)


def register(ctx) -> None:
    """Register Cyber Memory as a Hermes memory provider."""
    ctx.register_memory_provider(CyberMemoryProvider())
