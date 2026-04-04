"""Clude memory plugin — MemoryProvider for the Clude Cortex API.

Clude is a generative memory system with semantic search, importance decay,
emotional valence scoring, and graph-linked recall. Memories accumulate
importance over time and trigger reflective dream cycles that consolidate
insights — the agent's understanding deepens automatically between sessions.

Config via environment variables:
  CLUDE_API_KEY   — Cortex API key (get one at https://clude.io)
  CLUDE_API_URL   — Base URL (default: https://clude.io)

Or via $HERMES_HOME/clude.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://clude.io"
_REGISTER_PATH = "/api/cortex/register"
_STORE_PATH = "/api/cortex/store"
_RECALL_PATH = "/api/cortex/recall"
_STATS_PATH = "/api/cortex/stats"

# Circuit breaker
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/clude.json overrides."""
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "clude.json"
    except Exception:
        config_path = Path.home() / ".hermes" / "clude.json"

    config = {
        "api_key": os.environ.get("CLUDE_API_KEY", ""),
        "api_url": os.environ.get("CLUDE_API_URL", _DEFAULT_API_URL).rstrip("/"),
    }

    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "clude_recall",
    "description": (
        "Semantic search over Clude's memory store. Returns memories ranked by "
        "relevance, recency, importance, and graph connectivity. "
        "Use when you need to find what you know about a topic, person, or past event."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query — what to search for.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of memories to return (default 8, max 20).",
            },
            "memory_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Filter by memory type. Options: episodic, semantic, procedural, self_model. "
                    "Omit to search all types."
                ),
            },
            "min_importance": {
                "type": "number",
                "description": "Only return memories above this importance threshold (0.0–1.0).",
            },
        },
        "required": ["query"],
    },
}

STORE_SCHEMA = {
    "name": "clude_store",
    "description": (
        "Store a fact, insight, or observation in Clude's memory. "
        "Use for anything worth remembering across sessions: user preferences, "
        "project context, decisions made, lessons learned. "
        "Stored memories are semantically indexed and will surface in future recalls."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The full memory content to store.",
            },
            "summary": {
                "type": "string",
                "description": "A short one-line summary (used for indexing and display).",
            },
            "memory_type": {
                "type": "string",
                "description": (
                    "Type of memory: episodic (events), semantic (facts), "
                    "procedural (how-to), self_model (about the user). Default: semantic."
                ),
            },
            "importance": {
                "type": "number",
                "description": "Importance score 0.0–1.0. Higher = retrieved more often. Default: 0.6.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for filtering and grouping.",
            },
        },
        "required": ["content", "summary"],
    },
}

ALL_TOOL_SCHEMAS = [RECALL_SCHEMA, STORE_SCHEMA]


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _post(url: str, api_key: str, payload: dict, timeout: int = 15) -> Optional[dict]:
    """Make an authenticated POST request to the Clude API."""
    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.warning("Clude API HTTP %s: %s", e.code, body[:200])
        return None
    except Exception as e:
        logger.warning("Clude API request failed: %s", e)
        return None


def _get(url: str, api_key: str, timeout: int = 10) -> Optional[dict]:
    """Make an authenticated GET request to the Clude API."""
    import urllib.request
    import urllib.error

    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning("Clude API GET failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class CludeMemoryProvider(MemoryProvider):
    """Clude generative memory with semantic search and importance-driven recall."""

    def __init__(self) -> None:
        self._config: dict = {}
        self._api_key: str = ""
        self._api_url: str = _DEFAULT_API_URL
        self._agent_id: Optional[str] = None
        self._prefetch_cache: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0
        self._breaker_until: float = 0.0

    @property
    def name(self) -> str:
        return "clude"

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("api_key"))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config["api_key"]
        self._api_url = self._config["api_url"]

        # Register this agent to get a stable agent_id
        result = _post(
            f"{self._api_url}{_REGISTER_PATH}",
            self._api_key,
            {
                "agent_id": f"hermes-{session_id[:8]}",
                "name": "Hermes Agent",
                "description": "Hermes Agent — memory-augmented AI assistant by Nous Research",
            },
        )
        if result:
            self._agent_id = result.get("agent_id")
            logger.info("Clude memory initialized (agent_id=%s)", self._agent_id)
        else:
            logger.warning("Clude registration failed — recall/store will still work via API key")

    def system_prompt_block(self) -> str:
        return (
            "You have access to Clude memory — a generative memory system with semantic search, "
            "importance decay, and cross-session recall. Use clude_recall to retrieve relevant "
            "context before responding to complex queries. Use clude_store to persist facts, "
            "decisions, and insights worth remembering across sessions."
        )

    def _is_circuit_open(self) -> bool:
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            if time.time() < self._breaker_until:
                return True
            # Cooldown expired — reset
            self._consecutive_failures = 0
        return False

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_until = time.time() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Clude circuit breaker open for %ds after %d failures",
                _BREAKER_COOLDOWN_SECS,
                _BREAKER_THRESHOLD,
            )

    def _do_recall(self, query: str, limit: int = 8, **kwargs) -> str:
        """Run a recall and return formatted text."""
        if self._is_circuit_open():
            return ""

        payload: dict = {
            "query": query,
            "limit": min(limit, 20),
            "skip_expansion": True,
            "track_access": True,
        }
        if kwargs.get("memory_types"):
            payload["memory_types"] = kwargs["memory_types"]
        if kwargs.get("min_importance") is not None:
            payload["min_importance"] = kwargs["min_importance"]

        result = _post(f"{self._api_url}{_RECALL_PATH}", self._api_key, payload)
        if result is None:
            self._record_failure()
            return ""

        self._record_success()
        memories = result.get("memories", [])
        if not memories:
            return ""

        lines = ["[Clude Memory]"]
        for m in memories:
            score = m.get("_score")
            score_str = f" (score: {score:.2f})" if score else ""
            lines.append(f"• {m['summary']}{score_str}")
            if m.get("content") and m["content"] != m["summary"]:
                # Truncate long content
                content = m["content"]
                if len(content) > 300:
                    content = content[:297] + "..."
                lines.append(f"  {content}")
        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        with self._prefetch_lock:
            result = self._prefetch_cache
            self._prefetch_cache = ""
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        def _run():
            text = self._do_recall(query, limit=6)
            with self._prefetch_lock:
                self._prefetch_cache = text

        self._prefetch_thread = threading.Thread(target=_run, daemon=True)
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Store each turn as a lightweight episodic memory in the background."""
        if not user_content.strip() or not assistant_content.strip():
            return

        def _store():
            if self._is_circuit_open():
                return
            summary = user_content.strip()[:100]
            content = f"User: {user_content.strip()}\nAssistant: {assistant_content.strip()}"
            result = _post(
                f"{self._api_url}{_STORE_PATH}",
                self._api_key,
                {
                    "type": "episodic",
                    "content": content[:4000],
                    "summary": summary,
                    "importance": 0.4,
                    "source": "hermes-turn",
                    "tags": ["hermes", "conversation"],
                },
            )
            if result:
                self._record_success()
            else:
                self._record_failure()

        threading.Thread(target=_store, daemon=True).start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Extract and store key facts from the session."""
        if not messages:
            return

        # Build a brief session summary from the last few exchanges
        recent = messages[-6:] if len(messages) > 6 else messages
        parts = []
        for m in recent:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            if role in ("user", "assistant") and content:
                parts.append(f"{role.capitalize()}: {content[:200]}")

        if not parts:
            return

        summary_text = "\n".join(parts)
        if self._is_circuit_open():
            return

        _post(
            f"{self._api_url}{_STORE_PATH}",
            self._api_key,
            {
                "type": "episodic",
                "content": summary_text[:4000],
                "summary": f"Session ended ({len(messages)} messages)",
                "importance": 0.5,
                "source": "hermes-session",
                "tags": ["hermes", "session"],
            },
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return ALL_TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "clude_recall":
            return self._handle_recall(args)
        if tool_name == "clude_store":
            return self._handle_store(args)
        raise NotImplementedError(f"Unknown tool: {tool_name}")

    def _handle_recall(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "query is required"})

        if self._is_circuit_open():
            return json.dumps({"error": "Clude API temporarily unavailable", "memories": []})

        payload: dict = {
            "query": query,
            "limit": min(int(args.get("limit", 8)), 20),
            "skip_expansion": False,
            "track_access": True,
        }
        if args.get("memory_types"):
            payload["memory_types"] = args["memory_types"]
        if args.get("min_importance") is not None:
            payload["min_importance"] = float(args["min_importance"])

        result = _post(f"{self._api_url}{_RECALL_PATH}", self._api_key, payload)
        if result is None:
            self._record_failure()
            return json.dumps({"error": "Recall failed", "memories": []})

        self._record_success()
        memories = result.get("memories", [])
        return json.dumps({
            "memories": [
                {
                    "id": m.get("id"),
                    "type": m.get("memory_type"),
                    "summary": m.get("summary"),
                    "content": m.get("content"),
                    "importance": m.get("importance"),
                    "created_at": m.get("created_at"),
                    "score": m.get("_score"),
                }
                for m in memories
            ],
            "count": len(memories),
        })

    def _handle_store(self, args: Dict[str, Any]) -> str:
        content = args.get("content", "")
        summary = args.get("summary", "")
        if not content or not summary:
            return json.dumps({"error": "content and summary are required"})

        if self._is_circuit_open():
            return json.dumps({"error": "Clude API temporarily unavailable", "stored": False})

        payload = {
            "type": args.get("memory_type", "semantic"),
            "content": content[:4000],
            "summary": summary[:200],
            "importance": float(args.get("importance", 0.6)),
            "tags": args.get("tags", []) + ["hermes"],
            "source": "hermes-agent",
        }

        result = _post(f"{self._api_url}{_STORE_PATH}", self._api_key, payload)
        if result is None:
            self._record_failure()
            return json.dumps({"stored": False, "error": "Store failed"})

        self._record_success()
        return json.dumps({"stored": result.get("stored", False), "memory_id": result.get("memory_id")})

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "description": "Clude API key",
                "secret": True,
                "required": True,
                "env_var": "CLUDE_API_KEY",
                "url": "https://clude.io",
            },
            {
                "key": "api_url",
                "description": "Clude API base URL",
                "secret": False,
                "required": False,
                "default": _DEFAULT_API_URL,
                "env_var": "CLUDE_API_URL",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "clude.json"
        existing: dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update({k: v for k, v in values.items() if v})
        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        logger.info("Clude config saved to %s", config_path)

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
