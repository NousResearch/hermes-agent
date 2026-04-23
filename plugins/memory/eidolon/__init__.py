"""Eidolon Agent Memory plugin — MemoryProvider interface.

Long-term semantic memory backed by eidolon-agent-memory: fact triples
with pgvector embeddings, episodic memory, companion profiles, and
graceful omission of high-salience crisis content.

Connects to a running eidolon-agent-memory MCP server over HTTP/JSON-RPC.

Config: $HERMES_HOME/eidolon/config.json
  {
    "api_url":      "http://localhost:3100/mcp",
    "companion_id": "<UUID>",
    "recall_intent": "factual",
    "auto_recall":   true,
    "auto_extract":  true
  }

Secret: $HERMES_HOME/.env
  EIDOLON_API_KEY=mnemo-...
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "http://localhost:3100/mcp"
_VALID_INTENTS = {"factual", "emotional", "casual", "recall"}
_RPC_TIMEOUT = 30.0

# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _mcp_initialize(api_url: str, timeout: float = _RPC_TIMEOUT) -> str:
    """Run the MCP initialize handshake and return the session ID.

    FastMCP streamable-http requires an initialize request before any tool call.
    The session ID is returned in the 'mcp-session-id' response header.
    """
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hermes-eidolon-plugin", "version": "1.0"},
        },
        "id": 1,
    }).encode("utf-8")
    req = urllib.request.Request(api_url, data=payload, headers=_MCP_HEADERS, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            session_id = resp.headers.get("mcp-session-id", "")
            resp.read()  # drain body
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Eidolon HTTP {e.code}: {e.reason}") from e
    except Exception as exc:
        raise RuntimeError(f"Eidolon init error: {exc}") from exc
    if not session_id:
        raise RuntimeError("Eidolon MCP server did not return a session ID")
    return session_id


def _parse_sse(raw: str) -> dict:
    """Extract JSON from an SSE-wrapped or plain response body."""
    lines = [l for l in raw.splitlines() if l.startswith("data:") and l.strip() != "data:"]
    if lines:
        return json.loads(lines[-1][5:].strip())
    return json.loads(raw)


def _rpc_call(api_url: str, tool_name: str, arguments: dict,
              timeout: float = _RPC_TIMEOUT,
              session_id: str = "") -> dict:
    """Call an eidolon MCP tool via JSON-RPC 2.0 over streamable-http.

    Requires a pre-established session_id from _mcp_initialize().
    Handles SSE-wrapped responses automatically.
    """
    headers = {**_MCP_HEADERS, "mcp-session-id": session_id}
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
        "id": 1,
    }).encode("utf-8")
    req = urllib.request.Request(api_url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = _parse_sse(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Eidolon HTTP {e.code}: {e.reason}") from e
    except Exception as exc:
        raise RuntimeError(f"Eidolon RPC error: {exc}") from exc

    if "error" in body:
        raise RuntimeError(f"Eidolon RPC error: {body['error']}")

    # FastMCP wraps tool results in content[].text — extract the JSON string
    result = body.get("result", {})
    if isinstance(result, dict) and "content" in result:
        content = result.get("content", [])
        if content and isinstance(content[0], dict) and "text" in content[0]:
            try:
                return json.loads(content[0]["text"])
            except (json.JSONDecodeError, KeyError):
                pass
    return result


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from $HERMES_HOME/eidolon/config.json with env var fallbacks.

    Also explicitly reads $HERMES_HOME/.env to get API keys (not just os.environ,
    which only has env vars if the shell sourced the file).
    """
    # First try os.environ (set by shell if .env was sourced)
    config: dict = {
        "api_url": os.environ.get("EIDOLON_API_URL", ""),
        "api_key": os.environ.get("EIDOLON_API_KEY", ""),
        "companion_id": os.environ.get("EIDOLON_COMPANION_ID", ""),
        "recall_intent": "factual",
        "auto_recall": True,
        "auto_extract": True,
    }

    # Then read .env file explicitly (fallback for when shell didn't source it)
    hermes_home = get_hermes_home()
    env_path = hermes_home / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key == "EIDOLON_API_KEY" and not config.get("api_key"):
                    config["api_key"] = val
                elif key == "EIDOLON_API_URL" and not config.get("api_url"):
                    config["api_url"] = val
                elif key == "EIDOLON_COMPANION_ID" and not config.get("companion_id"):
                    config["companion_id"] = val
        except Exception:
            logger.debug("Failed to parse %s", env_path, exc_info=True)

    # Defaults for missing keys
    if not config.get("api_url"):
        config["api_url"] = _DEFAULT_API_URL

    # Then read eidolon/config.json (non-secret config, takes precedence over .env)
    config_path = hermes_home / "eidolon" / "config.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
        except Exception:
            logger.debug("Failed to parse %s", config_path, exc_info=True)

    return config


def _save_config(values: dict, hermes_home: str) -> None:
    config_dir = Path(hermes_home) / "eidolon"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    existing: dict = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing.update(values)
    config_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Tool schemas exposed to the model
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "eidolon_search",
    "description": (
        "Search long-term memory for facts about the user. Returns semantically "
        "relevant fact triples ranked by importance and relevance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "intent": {
                "type": "string",
                "enum": ["factual", "emotional", "casual", "recall"],
                "description": (
                    "Search mode. 'casual' suppresses high-salience trauma/grief content. "
                    "Default: factual."
                ),
            },
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["query"],
    },
}

STORE_FACT_SCHEMA = {
    "name": "eidolon_store_fact",
    "description": (
        "Store a fact about the user in long-term memory. "
        "Prefer subject→predicate→object triples (e.g. subject='user', "
        "predicate='prefers', object='dark mode')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "The entity the fact is about."},
            "predicate": {"type": "string", "description": "The relationship or attribute."},
            "object_value": {"type": "string", "description": "The value or target of the relationship."},
            "fact_text": {"type": "string", "description": "Human-readable fact sentence."},
            "emotional_salience": {
                "type": "string",
                "enum": ["LOW", "MED", "HIGH"],
                "description": "Emotional weight. HIGH for grief/trauma (will be gracefully omitted in casual contexts).",
            },
        },
        "required": ["subject", "predicate", "object_value", "fact_text"],
    },
}

GET_CONTEXT_SCHEMA = {
    "name": "eidolon_context",
    "description": (
        "Get a pre-formatted memory context block for the current query. "
        "Returns a ready-to-use text summary of relevant facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The current user query or topic."},
            "intent": {
                "type": "string",
                "enum": ["factual", "emotional", "casual", "recall"],
                "description": "Search intent (default: factual).",
            },
        },
        "required": ["query"],
    },
}


JOURNAL_SCHEMA = {
    "name": "eidolon_journal",
    "description": (
        "Write a journal entry, reflection, diary note, or dream record to long-term memory. "
        "Use when the user shares something worth preserving as a narrative — a dream, a meaningful "
        "event, a reflection, or a diary-style update."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The journal/diary/dream content to store."},
            "memory_type": {
                "type": "string",
                "enum": ["journal", "reflection", "diary", "dream", "musing", "narrative", "conversation"],
                "description": "Type of entry. Use 'dream' for dreams, 'diary' for diary-style, 'reflection' for reflections.",
            },
            "importance": {
                "type": "number",
                "description": "Importance score between 0.0 and 1.0 (default: 0.5).",
            },
        },
        "required": ["text"],
    },
}

GET_JOURNAL_SCHEMA = {
    "name": "eidolon_get_journal",
    "description": (
        "Retrieve the companion's current journal — a curated narrative of who the user is, "
        "their key insights, and preferences. Use at session start or when asked about the user's history."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

GENERATE_INSIGHTS_SCHEMA = {
    "name": "eidolon_generate_insights",
    "description": (
        "Synthesize psychological insights and growth patterns from stored facts. "
        "Use periodically or when the user asks for self-reflection, patterns, or personal analysis. "
        "Avoid calling on every turn — this triggers an LLM analysis pass."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

GENERATE_MUSING_SCHEMA = {
    "name": "eidolon_generate_musing",
    "description": (
        "Generate a short spontaneous reflection or musing based on stored memories. "
        "Use during idle moments or when the user wants a thoughtful, unprompted observation. "
        "Do not call inside active response generation."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

LOOKUP_FACT_SCHEMA = {
    "name": "eidolon_lookup_fact",
    "description": (
        "Direct lookup of facts by subject and optional predicate. "
        "Use for precise fact questions (e.g. 'what is user's job?') rather than broad semantic search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "The entity to look up (e.g. 'user', 'Luna', 'Cisco')."},
            "predicate": {"type": "string", "description": "Optional relationship to filter by (e.g. 'prefers', 'owns', 'IS_NAMED')."},
        },
        "required": ["subject"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class EidolonMemoryProvider(MemoryProvider):
    """Eidolon Agent Memory — semantic fact storage with pgvector and companion profiles."""

    def __init__(self) -> None:
        self._api_url = _DEFAULT_API_URL
        self._api_key = ""
        self._companion_id = ""
        self._recall_intent = "factual"
        self._auto_recall = True
        self._auto_extract = True
        self._session_turns: list[tuple[str, str]] = []
        self._session_id = ""
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        # MCP session (established lazily on first _call)
        self._mcp_session_id = ""
        self._mcp_session_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "eidolon"

    # ------------------------------------------------------------------
    # Auth / RPC helpers
    # ------------------------------------------------------------------

    def _get_mcp_session(self) -> str:
        """Return a valid MCP session ID, initializing one if needed."""
        with self._mcp_session_lock:
            if not self._mcp_session_id:
                self._mcp_session_id = _mcp_initialize(self._api_url)
            return self._mcp_session_id

    def _call(self, tool_name: str, arguments: dict, timeout: float = _RPC_TIMEOUT) -> dict:
        """Call the eidolon MCP server, always injecting api_key and using the MCP session."""
        arguments = {"api_key": self._api_key, **arguments}
        try:
            return _rpc_call(self._api_url, tool_name, arguments,
                             timeout=timeout, session_id=self._get_mcp_session())
        except RuntimeError as exc:
            # Session may have expired — reset and retry once
            if "session" in str(exc).lower() or "400" in str(exc):
                with self._mcp_session_lock:
                    self._mcp_session_id = ""
                return _rpc_call(self._api_url, tool_name, arguments,
                                 timeout=timeout, session_id=self._get_mcp_session())
            raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            cfg = _load_config()
            return bool(cfg.get("api_key") and cfg.get("companion_id") and cfg.get("api_url"))
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        cfg = _load_config()
        self._api_url = str(cfg.get("api_url") or _DEFAULT_API_URL).rstrip("/")
        self._api_key = str(cfg.get("api_key") or "").strip()
        self._companion_id = str(cfg.get("companion_id") or "").strip()
        intent = str(cfg.get("recall_intent", "factual")).strip().lower()
        self._recall_intent = intent if intent in _VALID_INTENTS else "factual"
        self._auto_recall = bool(cfg.get("auto_recall", True))
        self._auto_extract = bool(cfg.get("auto_extract", True))
        self._session_id = str(session_id or "").strip()
        self._session_turns = []
        self._prefetch_result = ""
        self._mcp_session_id = ""  # reset MCP session on each agent session
        logger.info(
            "Eidolon initialized: api_url=%s, companion=%s, intent=%s, auto_recall=%s, auto_extract=%s",
            self._api_url, self._companion_id, self._recall_intent, self._auto_recall, self._auto_extract,
        )

    def system_prompt_block(self) -> str:
        return (
            "# Eidolon Long-Term Memory\n"
            "Active. Relevant facts are automatically retrieved before each turn.\n"
            "Use eidolon_search to look up specific memories, eidolon_store_fact to "
            "persist important information, or eidolon_context for a formatted context block."
        )

    # ------------------------------------------------------------------
    # Prefetch (background recall before each turn)
    # ------------------------------------------------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._auto_recall or not self._companion_id:
            return

        def _run() -> None:
            try:
                result = self._call("get_context", {
                    "companion_id": self._companion_id,
                    "query": query,
                    "intent": self._recall_intent,
                })
                context = str(result.get("context", "")).strip()
                if context:
                    with self._prefetch_lock:
                        self._prefetch_result = context
            except Exception as exc:
                logger.debug("Eidolon prefetch failed: %s", exc)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="eidolon-prefetch")
        self._prefetch_thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return (
            "# Eidolon Memory (long-term facts about the user)\n"
            "Use this to inform your response. Do not look up information already present here.\n\n"
            f"{result}"
        )

    # ------------------------------------------------------------------
    # Turn syncing
    # ------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Buffer the turn for batch extraction at session end."""
        if self._auto_extract:
            self._session_turns.append((user_content, assistant_content))

    # ------------------------------------------------------------------
    # Tool schemas + dispatch
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            SEARCH_SCHEMA,
            STORE_FACT_SCHEMA,
            GET_CONTEXT_SCHEMA,
            JOURNAL_SCHEMA,
            GET_JOURNAL_SCHEMA,
            GENERATE_INSIGHTS_SCHEMA,
            GENERATE_MUSING_SCHEMA,
            LOOKUP_FACT_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "eidolon_search":
            return self._handle_search(args)
        if tool_name == "eidolon_store_fact":
            return self._handle_store_fact(args)
        if tool_name == "eidolon_context":
            return self._handle_get_context(args)
        if tool_name == "eidolon_journal":
            return self._handle_journal(args)
        if tool_name == "eidolon_get_journal":
            return self._handle_get_journal()
        if tool_name == "eidolon_generate_insights":
            return self._handle_generate_insights()
        if tool_name == "eidolon_generate_musing":
            return self._handle_generate_musing()
        if tool_name == "eidolon_lookup_fact":
            return self._handle_lookup_fact(args)
        return tool_error(f"Unknown eidolon tool: {tool_name}")

    def _handle_search(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return tool_error("query is required")
        intent = str(args.get("intent", self._recall_intent)).lower()
        if intent not in _VALID_INTENTS:
            intent = self._recall_intent
        limit = int(args.get("limit", 10))
        try:
            result = self._call("search_memory", {
                "companion_id": self._companion_id,
                "query": query,
                "intent": intent,
                "limit": limit,
            })
            facts = result.get("facts", [])
            if not facts:
                return json.dumps({"memories": [], "count": 0})
            formatted = [
                {
                    "fact": f.get("fact_text", ""),
                    "predicate": f.get("predicate", ""),
                    "salience": f.get("emotional_salience", ""),
                    "importance": f.get("importance", 0),
                    "score": f.get("score", 0),
                }
                for f in facts
            ]
            return json.dumps({"memories": formatted, "count": len(formatted)})
        except Exception as exc:
            logger.warning("eidolon_search failed: %s", exc)
            return tool_error(str(exc))

    def _handle_store_fact(self, args: dict) -> str:
        subject = str(args.get("subject", "")).strip()
        predicate = str(args.get("predicate", "")).strip()
        object_value = str(args.get("object_value", "")).strip()
        fact_text = str(args.get("fact_text", "")).strip()
        if not (subject and predicate and object_value and fact_text):
            return tool_error("subject, predicate, object_value, and fact_text are all required")
        salience = str(args.get("emotional_salience", "LOW")).upper()
        if salience not in ("LOW", "MED", "HIGH"):
            salience = "LOW"
        try:
            result = self._call("store_fact", {
                "companion_id": self._companion_id,
                "subject": subject,
                "predicate": predicate,
                "object_value": object_value,
                "fact_text": fact_text,
                "emotional_salience": salience,
            })
            return json.dumps({"stored": True, "fact_id": result.get("fact_id", "")})
        except Exception as exc:
            logger.warning("eidolon_store_fact failed: %s", exc)
            return tool_error(str(exc))

    def _handle_get_context(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return tool_error("query is required")
        intent = str(args.get("intent", self._recall_intent)).lower()
        if intent not in _VALID_INTENTS:
            intent = self._recall_intent
        try:
            result = self._call("get_context", {
                "companion_id": self._companion_id,
                "query": query,
                "intent": intent,
            })
            return json.dumps({
                "context": result.get("context", ""),
                "fact_count": result.get("fact_count", 0),
            })
        except Exception as exc:
            logger.warning("eidolon_context failed: %s", exc)
            return tool_error(str(exc))

    def _handle_journal(self, args: dict) -> str:
        text = str(args.get("text", "")).strip()
        if not text:
            return tool_error("text is required")
        memory_type = str(args.get("memory_type", "journal")).lower()
        valid_types = {"journal", "reflection", "diary", "dream", "musing", "narrative", "conversation"}
        if memory_type not in valid_types:
            memory_type = "journal"
        importance = float(args.get("importance", 0.5))
        importance = max(0.0, min(1.0, importance))
        try:
            result = self._call("store_episodic", {
                "companion_id": self._companion_id,
                "text": text,
                "memory_type": memory_type,
                "importance": importance,
            })
            return json.dumps({"stored": True, "memory_id": result.get("memory_id", ""), "type": memory_type})
        except Exception as exc:
            logger.warning("eidolon_journal failed: %s", exc)
            return tool_error(str(exc))

    def _handle_get_journal(self) -> str:
        try:
            result = self._call("get_journal", {
                "companion_id": self._companion_id,
            })
            return json.dumps({
                "journal": result.get("journal", ""),
                "version": result.get("version", 0),
                "top_insights": result.get("top_insights", []),
                "preferences": result.get("preferences", {}),
            })
        except Exception as exc:
            logger.warning("eidolon_get_journal failed: %s", exc)
            return tool_error(str(exc))

    def _handle_generate_insights(self) -> str:
        try:
            result = self._call("generate_insights", {
                "companion_id": self._companion_id,
            }, timeout=60.0)
            insights = result.get("insights", [])
            return json.dumps({"insights": insights, "count": result.get("count", len(insights))})
        except Exception as exc:
            logger.warning("eidolon_generate_insights failed: %s", exc)
            return tool_error(str(exc))

    def _handle_generate_musing(self) -> str:
        try:
            result = self._call("generate_musing", {
                "companion_id": self._companion_id,
            }, timeout=60.0)
            return json.dumps({
                "musing": result.get("text", ""),
                "memory_id": result.get("memory_id", ""),
            })
        except Exception as exc:
            logger.warning("eidolon_generate_musing failed: %s", exc)
            return tool_error(str(exc))

    def _handle_lookup_fact(self, args: dict) -> str:
        subject = str(args.get("subject", "")).strip()
        if not subject:
            return tool_error("subject is required")
        predicate = str(args.get("predicate", "")).strip()
        try:
            result = self._call("lookup_fact", {
                "companion_id": self._companion_id,
                "subject": subject,
                "predicate": predicate,
            })
            facts = result.get("facts", [])
            formatted = [
                {
                    "fact": f.get("fact_text", ""),
                    "predicate": f.get("predicate", ""),
                    "salience": f.get("emotional_salience", ""),
                    "importance": f.get("importance", 0),
                    "confidence": f.get("confidence", 0),
                }
                for f in facts
            ]
            return json.dumps({"facts": formatted, "count": len(formatted)})
        except Exception as exc:
            logger.warning("eidolon_lookup_fact failed: %s", exc)
            return tool_error(str(exc))

    # ------------------------------------------------------------------
    # Session end — extract facts from full conversation
    # ------------------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._auto_extract or not self._companion_id or not self._session_turns:
            return

        def _run() -> None:
            try:
                import uuid
                transcript = "\n".join(
                    f"User: {u}\nAssistant: {a}"
                    for u, a in self._session_turns
                    if u or a
                )
                if not transcript.strip():
                    return
                # Generate a proper UUID for the extraction call (session_id may be non-UUID string)
                extraction_session_id = str(uuid.uuid4())
                result = self._call(
                    "extract_session_facts",
                    {
                        "companion_id": self._companion_id,
                        "conversation_text": transcript,
                        "session_id": extraction_session_id,
                    },
                    timeout=60.0,
                )
                logger.info("Eidolon: session fact extraction complete (%d turns). Result: %s",
                            len(self._session_turns), result)
            except Exception as exc:
                logger.warning("Eidolon: session fact extraction failed: %s", exc, exc_info=True)

        t = threading.Thread(target=_run, daemon=True, name="eidolon-extract")
        t.start()
        t.join(timeout=65.0)

    # ------------------------------------------------------------------
    # Setup wizard
    # ------------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_url",
                "description": "Eidolon MCP server URL",
                "default": _DEFAULT_API_URL,
            },
            {
                "key": "api_key",
                "description": "Eidolon API key (mnemo-...)",
                "secret": True,
                "env_var": "EIDOLON_API_KEY",
                "required": True,
            },
            {
                "key": "companion_id",
                "description": "Companion UUID (run hermes memory setup to create one)",
                "required": True,
            },
            {
                "key": "recall_intent",
                "description": "Default search intent",
                "default": "factual",
                "choices": ["factual", "emotional", "casual", "recall"],
            },
            {
                "key": "auto_recall",
                "description": "Automatically retrieve memories before each turn",
                "default": True,
            },
            {
                "key": "auto_extract",
                "description": "Extract and store facts from conversations at session end",
                "default": True,
            },
        ]

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Interactive setup: provision user, create/pick companion, save config."""
        import getpass
        import sys
        from hermes_cli.config import save_config

        print("\n  Configuring Eidolon Agent Memory:\n")

        api_url = input(f"  MCP server URL [{_DEFAULT_API_URL}]: ").strip() or _DEFAULT_API_URL
        api_url = api_url.rstrip("/")

        # Check connectivity
        try:
            sid = _mcp_initialize(api_url, timeout=5.0)
            print("  ✓ Server reachable")
        except Exception as exc:
            print(f"  ⚠ Could not reach server: {exc}")
            print("  Continuing anyway — check docker-compose is running.\n")
            sid = ""

        def _setup_call(tool_name: str, args: dict) -> dict:
            nonlocal sid
            if not sid:
                try:
                    sid = _mcp_initialize(api_url)
                except Exception:
                    pass
            return _rpc_call(api_url, tool_name, args, session_id=sid)

        # API key: provision new user or use existing
        existing_key = os.environ.get("EIDOLON_API_KEY", "")
        if existing_key:
            masked = f"mnemo-...{existing_key[-4:]}" if len(existing_key) > 4 else "(set)"
            choice = input(f"  Existing key detected ({masked}). (k) Keep it  (n) New user  (e) Enter different [k]: ").strip().lower() or "k"
        else:
            choice = input("  (n) Provision new user  (e) Enter existing API key [n]: ").strip().lower() or "n"

        api_key = existing_key

        if choice == "n":
            name = input("  Your name (optional): ").strip()
            email = input("  Email (optional): ").strip()
            try:
                result = _setup_call("provision_user", {
                    "name": name, "email": email, "timezone": "UTC",
                })
                api_key = result.get("api_key", "")
                if not api_key:
                    print("  ✗ provision_user did not return an API key.")
                    sys.exit(1)
                print(f"\n  ✓ User created.")
                print(f"  API key: {api_key}")
                print("  ⚠ This key will NOT be shown again — write it down!\n")
            except Exception as exc:
                print(f"  ✗ Failed to provision user: {exc}")
                sys.exit(1)

        elif choice == "e":
            sys.stdout.write("  API key (mnemo-...): ")
            sys.stdout.flush()
            api_key = getpass.getpass(prompt="") if sys.stdin.isatty() else sys.stdin.readline().strip()

        if not api_key:
            print("  ✗ No API key. Aborting setup.")
            sys.exit(1)

        # List existing companions
        companions: list[dict] = []
        try:
            result = _setup_call("list_companions", {"api_key": api_key})
            companions = result.get("companions", [])
        except Exception as exc:
            logger.debug("list_companions failed: %s", exc)

        companion_id = ""
        if companions:
            print("  Existing companions:")
            for i, c in enumerate(companions):
                print(f"    [{i}] {c['name']}  (id: {c['companion_id']})")
            print("    [n] Create new companion")
            pick = input("  Select [0]: ").strip()
            if pick.lower() == "n":
                companion_id = self._wizard_create_companion(_setup_call, api_key)
            else:
                try:
                    companion_id = companions[int(pick or 0)]["companion_id"]
                except (IndexError, ValueError):
                    companion_id = companions[0]["companion_id"]
        else:
            print("  No companions found. Let's create one.")
            companion_id = self._wizard_create_companion(_setup_call, api_key)

        if not companion_id:
            print("  ✗ No companion selected. Aborting setup.")
            sys.exit(1)

        # Recall intent
        intents = ["factual", "emotional", "casual", "recall"]
        print(f"\n  Default recall intent ({'/'.join(intents)}):")
        intent_pick = input("  [factual]: ").strip().lower() or "factual"
        if intent_pick not in intents:
            intent_pick = "factual"

        # Save config (non-secret keys → config.json)
        _save_config({
            "api_url": api_url,
            "companion_id": companion_id,
            "recall_intent": intent_pick,
            "auto_recall": True,
            "auto_extract": True,
        }, hermes_home)

        # Save api_key → .env (secret, never in config.json)
        env_path = Path(hermes_home) / ".env"
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        lines = [l for l in lines if not l.startswith("EIDOLON_API_KEY=")]
        lines.append(f"EIDOLON_API_KEY={api_key}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Update main config to activate this provider
        config["memory"]["provider"] = "eidolon"
        save_config(config)

        print(f"\n  ✓ Eidolon memory configured.")
        print(f"  Companion: {companion_id}")
        print(f"  API key saved to .env")
        print(f"  Start a new session to activate.\n")

    @staticmethod
    def _wizard_create_companion(call_fn, api_key: str) -> str:
        """Interactively create a companion and return its UUID."""
        name = input("  Companion name [Hermes]: ").strip() or "Hermes"
        persona = input("  Persona description (optional, e.g. 'warm, curious, empathetic'): ").strip()
        pronouns = input("  Pronouns (optional, e.g. 'they/them'): ").strip()
        try:
            result = call_fn("create_companion", {
                "api_key": api_key,
                "name": name,
                "persona": persona,
                "pronouns": pronouns,
            })
            companion_id = result.get("companion_id", "")
            if companion_id:
                print(f"  ✓ Companion '{name}' created (id: {companion_id})")
            else:
                print("  ✗ create_companion did not return a companion_id.")
            return companion_id
        except Exception as exc:
            print(f"  ✗ Failed to create companion: {exc}")
            return ""

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
