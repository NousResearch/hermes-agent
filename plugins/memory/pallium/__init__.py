"""Pallium memory plugin — MemoryProvider interface.

Local-first memory sidecar with automatic fact extraction, hybrid retrieval
(lexical + vector RRF), and evidence provenance. Two extraction packages run
in parallel over every ingested turn:

  - Work continuity: decisions, investigation outcomes, task checkpoints
  - Factual recall: names, dates, preferences, events, relationships

Multilingual by design — memory is stored and recalled in the original language.
Cross-language recall works natively (query in one language, retrieve another).

Requires a running Pallium server. See https://github.com/rotemhermon/pallium
or start with: python -m app.run serve --host 127.0.0.1 --port 8000

Config via $HERMES_HOME/pallium.json:
  base_url    — Pallium server URL (default: http://localhost:8000)
  actor_ref   — stable user identifier for cross-session scoping (default: hermes-user)
  container_ref — memory container (default: hermes)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# Circuit breaker: pause API calls after consecutive failures
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# Prefetch: join background thread before LLM call (max wait)
_PREFETCH_TIMEOUT_SECS = 4.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(hermes_home: str = "") -> dict:
    """Load config from $HERMES_HOME/pallium.json, with defaults."""
    config = {
        "base_url": "http://localhost:8000",
        "actor_ref": "hermes-user",
        "container_ref": "hermes",
    }

    if hermes_home:
        config_path = Path(hermes_home) / "pallium.json"
    else:
        try:
            from hermes_constants import get_hermes_home
            config_path = get_hermes_home() / "pallium.json"
        except Exception:
            return config

    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

QUERY_SCHEMA = {
    "name": "pallium_query",
    "description": (
        "Search Pallium's persistent memory for relevant context. Returns compact "
        "evidence-backed cards: decisions, investigation outcomes, task checkpoints, "
        "and extracted facts. Use for any question where past context would help — "
        "'why did we choose X?', 'where did we leave off?', 'what do we know about Y?'"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "pallium_remember",
    "description": (
        "Store an important piece of information in Pallium's persistent memory. "
        "Use for decisions, investigation findings, or facts worth keeping across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to remember."},
            "kind": {
                "type": "string",
                "description": "Type of memory: 'note' (default), 'decision', 'finding'.",
                "enum": ["note", "decision", "finding"],
            },
        },
        "required": ["content"],
    },
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_post(url: str, payload: dict | list, timeout: float = 8.0) -> dict:
    """POST JSON to Pallium. Returns parsed response or raises."""
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx not installed. Run: pip install httpx")

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class PalliumMemoryProvider(MemoryProvider):
    """Pallium — local-first memory sidecar with hybrid retrieval and evidence provenance."""

    def __init__(self):
        self._base_url = "http://localhost:8000"
        self._actor_ref = "hermes-user"
        self._container_ref = "hermes"
        self._session_id = ""
        self._turn_count = 0

        # Background prefetch
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None

        # Background sync
        self._sync_thread: Optional[threading.Thread] = None

        # Circuit breaker
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "pallium"

    # -- Availability --------------------------------------------------------

    def is_available(self) -> bool:
        """True if pallium.json exists with a base_url, or default localhost is configured.

        Does NOT make a network call — just checks config presence.
        """
        cfg = _load_config()
        return bool(cfg.get("base_url"))

    # -- Config wizard -------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "base_url",
                "description": "Pallium server URL",
                "default": "http://localhost:8000",
                "required": False,
            },
            {
                "key": "actor_ref",
                "description": "Stable user identifier for cross-session memory scoping",
                "default": "hermes-user",
                "required": False,
            },
            {
                "key": "container_ref",
                "description": "Memory container identifier",
                "default": "hermes",
                "required": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "pallium.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # -- Lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home", "")
        cfg = _load_config(hermes_home)
        self._base_url = cfg["base_url"].rstrip("/")
        self._actor_ref = cfg["actor_ref"]
        self._container_ref = cfg["container_ref"]
        self._session_id = session_id
        self._turn_count = 0

    def system_prompt_block(self) -> str:
        return (
            "# Pallium Memory\n"
            f"Active. Container: {self._container_ref}. User: {self._actor_ref}.\n"
            "Use pallium_query to search past decisions, findings, and facts. "
            "Use pallium_remember to store important information."
        )

    # -- Retrieval -----------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached prefetch result queued after the previous turn."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=_PREFETCH_TIMEOUT_SECS)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Pallium Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background Pallium query for the next turn."""
        if self._is_breaker_open():
            return

        def _run():
            try:
                data = _http_post(
                    f"{self._base_url}/query",
                    {
                        "text": query,
                        "container_ref": self._container_ref,
                        "actor_ref": self._actor_ref,
                        "visibility": "private",
                    },
                    timeout=6.0,
                )
                blocks = data.get("injectable_blocks", [])
                results = data.get("results", [])
                lines = []
                if blocks:
                    for block in blocks:
                        title = block.get("title", "")
                        text = block.get("text", "")
                        if text:
                            lines.append(f"[{title}] {text}" if title else text)
                elif results:
                    # Fall back to source excerpts when no blocks yet
                    for r in results[:5]:
                        excerpt = r.get("excerpt", "")
                        if excerpt:
                            lines.append(excerpt)
                if lines:
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Pallium prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="pallium-prefetch")
        self._prefetch_thread.start()

    # -- Ingest --------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Ingest the completed turn into Pallium (non-blocking)."""
        if self._is_breaker_open():
            return

        self._turn_count += 1
        turn = self._turn_count
        sid = session_id or self._session_id

        def _sync():
            try:
                items = []
                for role, content in (("user", user_content), ("assistant", assistant_content)):
                    items.append({
                        "source_type": "hermes_turn",
                        "source_id": f"{sid}:{turn}:{role}",
                        "content_type": "text/plain",
                        "content": content,
                        "role": role,
                        "artifact_kind": "message",
                        "container_ref": self._container_ref,
                        "thread_ref": sid,
                        "actor_ref": self._actor_ref,
                        "visibility": "private",
                    })
                _http_post(f"{self._base_url}/items", items, timeout=10.0)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Pallium sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="pallium-sync")
        self._sync_thread.start()

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in MEMORY.md / USER.md writes to Pallium."""
        if action not in ("add", "replace") or not content:
            return

        label = "user-profile" if target == "user" else "agent-memory"

        def _write():
            try:
                _http_post(
                    f"{self._base_url}/items",
                    [{
                        "source_type": f"hermes_{label}",
                        "source_id": f"{self._container_ref}:{label}:{hash(content) & 0xFFFFFFFF}",
                        "content_type": "text/plain",
                        "content": content,
                        "role": "user" if target == "user" else "assistant",
                        "container_ref": self._container_ref,
                        "actor_ref": self._actor_ref,
                        "visibility": "private",
                    }],
                    timeout=10.0,
                )
            except Exception as e:
                logger.debug("Pallium memory mirror failed: %s", e)

        t = threading.Thread(target=_write, daemon=True, name="pallium-memwrite")
        t.start()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Ingest messages about to be discarded before context compression."""
        if not messages:
            return ""

        sid = self._session_id

        def _flush():
            try:
                for i, msg in enumerate(messages):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if not isinstance(content, str) or not content.strip():
                        continue
                    if role not in ("user", "assistant"):
                        continue
                    _http_post(
                        f"{self._base_url}/items",
                        [{
                            "source_type": "hermes_compression_flush",
                            "source_id": f"{sid}:compress:{i}:{role}",
                            "content_type": "text/plain",
                            "content": content,
                            "role": role,
                            "artifact_kind": "message",
                            "container_ref": self._container_ref,
                            "actor_ref": self._actor_ref,
                            "visibility": "private",
                        }],
                        timeout=10.0,
                    )
            except Exception as e:
                logger.debug("Pallium pre-compress flush failed: %s", e)

        t = threading.Thread(target=_flush, daemon=True, name="pallium-compress")
        t.start()
        return ""

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [QUERY_SCHEMA, REMEMBER_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Pallium temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        if tool_name == "pallium_query":
            return self._tool_query(args)
        elif tool_name == "pallium_remember":
            return self._tool_remember(args)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _tool_query(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "query is required"})

        try:
            data = _http_post(
                f"{self._base_url}/query",
                {
                    "text": query,
                    "container_ref": self._container_ref,
                    "actor_ref": self._actor_ref,
                    "visibility": "private",
                },
            )
            self._record_success()
        except Exception as e:
            self._record_failure()
            return json.dumps({"error": f"Pallium query failed: {e}"})

        blocks = data.get("injectable_blocks", [])
        results = data.get("results", [])

        if blocks:
            items = [
                {"type": b.get("memory_type", b.get("block_type", "")), "text": b.get("text", "")}
                for b in blocks if b.get("text")
            ]
            return json.dumps({"results": items, "count": len(items)})

        if results:
            # Fall back to source excerpts when memory objects not yet extracted
            items = [
                {"type": r.get("result_kind", "source_hit"), "text": r.get("excerpt", "")}
                for r in results[:5] if r.get("excerpt")
            ]
            if items:
                return json.dumps({"results": items, "count": len(items)})

        return json.dumps({"result": "No relevant memories found."})

    def _tool_remember(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return json.dumps({"error": "content is required"})

        kind = args.get("kind", "note")
        role_map = {"decision": "assistant", "finding": "assistant", "note": "user"}
        role = role_map.get(kind, "user")

        try:
            _http_post(
                f"{self._base_url}/items",
                [{
                    "source_type": "hermes_explicit_memory",
                    "source_id": f"{self._container_ref}:explicit:{hash(content) & 0xFFFFFFFF}",
                    "content_type": "text/plain",
                    "content": content,
                    "role": role,
                    "container_ref": self._container_ref,
                    "actor_ref": self._actor_ref,
                    "visibility": "private",
                }],
            )
            self._record_success()
            return json.dumps({"result": "Stored."})
        except Exception as e:
            self._record_failure()
            return json.dumps({"error": f"Failed to store: {e}"})

    # -- Shutdown ------------------------------------------------------------

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)

    # -- Circuit breaker -----------------------------------------------------

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Pallium circuit breaker tripped after %d consecutive failures. "
                "Pausing for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register Pallium as a memory provider plugin."""
    ctx.register_memory_provider(PalliumMemoryProvider())
