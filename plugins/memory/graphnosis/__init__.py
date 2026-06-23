"""Graphnosis memory provider — local encrypted memory via MCP socket."""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .client import GraphnosisMcpClient, GraphnosisMcpError, DEFAULT_SOCKET

logger = logging.getLogger(__name__)

RECALL_SCHEMA = {
    "name": "graphnosis_recall",
    "description": (
        "Search the user's Graphnosis encrypted memory graph. "
        "Use before answering questions about past notes, preferences, projects, or personal context. "
        "Strip conversational framing from the query; pass content words in the user's language."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Dense search query (3-8 content words)."},
            "max_tokens": {
                "type": "integer",
                "description": "Token budget for attached context (default 1500).",
            },
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "graphnosis_remember",
    "description": (
        "Save a durable fact, decision, or note to Graphnosis memory. "
        "Use when the user shares something they would want remembered across sessions. "
        "To update existing memory, use mcp_graphnosis_edit (MCP catalog) — not a second remember."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Content to remember (user's language)."},
            "target_engram": {
                "type": "string",
                "description": "Optional engram name (e.g. 'Work decisions').",
            },
            "label": {"type": "string", "description": "Short label for the Sources list."},
        },
        "required": ["text"],
    },
}

STATS_SCHEMA = {
    "name": "graphnosis_stats",
    "description": "List Graphnosis engrams and memory capacity overview.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _load_config() -> dict:
    from hermes_constants import get_hermes_home

    config = {
        "socket_path": DEFAULT_SOCKET,
        "default_engram": "",
        "prefetch_max_tokens": 1500,
    }
    config_path = get_hermes_home() / "graphnosis.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
        except Exception:
            pass
    return config


class GraphnosisMemoryProvider(MemoryProvider):
    """Graphnosis local memory — prefetch + recall/remember tools over MCP socket."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or _load_config()
        self._client: Optional[GraphnosisMcpClient] = None
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "graphnosis"

    def is_available(self) -> bool:
        path = self._config.get("socket_path", DEFAULT_SOCKET)
        return Path(str(path)).expanduser().exists()

    def get_config_schema(self):
        return [
            {
                "key": "socket_path",
                "description": "Graphnosis MCP socket path",
                "default": DEFAULT_SOCKET,
            },
            {
                "key": "default_engram",
                "description": "Default engram for remember (optional)",
                "default": "",
            },
            {
                "key": "prefetch_max_tokens",
                "description": "Max tokens injected via prefetch per turn",
                "default": "1500",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        config_path = Path(hermes_home) / "graphnosis.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = self._config or _load_config()
        socket_path = self._config.get("socket_path", DEFAULT_SOCKET)
        self._client = GraphnosisMcpClient(socket_path)
        self._session_id = session_id

    def _get_client(self) -> GraphnosisMcpClient:
        if self._client is None:
            self.initialize(session_id="")
        assert self._client is not None
        return self._client

    def system_prompt_block(self) -> str:
        return (
            "# Graphnosis Memory\n"
            "Active — local encrypted memory on this machine. "
            "Relevant context is prefetched each turn; use graphnosis_recall for deeper search. "
            "For edits, forget, cross-engram search, and skills, use mcp_graphnosis_* tools "
            "when the Graphnosis MCP catalog is installed (`hermes mcp install graphnosis`)."
        )

    @staticmethod
    def _strip_query_framing(query: str) -> str:
        q = query.strip()
        q = re.sub(
            r"^(remind me( about| of)?|what did i (say|tell you) about|do you know( if| about)?)\s+",
            "",
            q,
            flags=re.IGNORECASE,
        )
        return q.strip() or query.strip()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Graphnosis Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query:
            return

        def _run() -> None:
            try:
                client = self._get_client()
                max_tokens = int(self._config.get("prefetch_max_tokens", 1500))
                text = client.call_tool(
                    "recall",
                    {
                        "query": self._strip_query_framing(query),
                        "maxTokens": max_tokens,
                        "maxNodes": 15,
                    },
                )
                if text:
                    with self._prefetch_lock:
                        self._prefetch_result = text
            except GraphnosisMcpError as exc:
                logger.debug("Graphnosis prefetch failed: %s", exc)
            except Exception as exc:
                logger.debug("Graphnosis prefetch error: %s", exc)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="graphnosis-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Conservative v1: no automatic extraction every turn.
        pass

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if action != "add" or not content:
            return

        def _mirror() -> None:
            try:
                client = self._get_client()
                note = f"[Hermes {target}] {content}"
                args: Dict[str, Any] = {"text": note[:4000], "kind": "clip"}
                default_engram = self._config.get("default_engram")
                if default_engram:
                    args["target_engram"] = default_engram
                client.call_tool("remember", args)
            except Exception as exc:
                logger.debug("Graphnosis memory_write mirror failed: %s", exc)

        threading.Thread(target=_mirror, daemon=True, name="graphnosis-mirror").start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, REMEMBER_SCHEMA, STATS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            client = self._get_client()
        except GraphnosisMcpError as exc:
            return tool_error(str(exc))

        try:
            if tool_name == "graphnosis_recall":
                query = args.get("query", "")
                if not query:
                    return tool_error("Missing required parameter: query")
                max_tokens = int(args.get("max_tokens", self._config.get("prefetch_max_tokens", 1500)))
                text = client.call_tool(
                    "recall",
                    {
                        "query": self._strip_query_framing(query),
                        "maxTokens": max_tokens,
                        "maxNodes": 20,
                    },
                )
                return json.dumps({"result": text or "No relevant memories found."})

            if tool_name == "graphnosis_remember":
                text = args.get("text", "")
                if not text:
                    return tool_error("Missing required parameter: text")
                payload: Dict[str, Any] = {"text": text, "kind": "clip"}
                if args.get("target_engram"):
                    payload["target_engram"] = args["target_engram"]
                elif self._config.get("default_engram"):
                    payload["target_engram"] = self._config["default_engram"]
                if args.get("label"):
                    payload["label"] = args["label"]
                result = client.call_tool("remember", payload)
                return json.dumps({"result": result or "Saved to Graphnosis."})

            if tool_name == "graphnosis_stats":
                result = client.call_tool("stats", {})
                return json.dumps({"result": result or "No stats returned."})

            return tool_error(f"Unknown tool: {tool_name}")
        except GraphnosisMcpError as exc:
            return tool_error(str(exc))
        except Exception as exc:
            return tool_error(str(exc))

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def register(ctx) -> None:
    ctx.register_memory_provider(GraphnosisMemoryProvider())
