"""OpenViking memory plugin — full bidirectional MemoryProvider interface.

Context database by Volcengine (ByteDance) that organizes agent knowledge
into a filesystem hierarchy (viking:// URIs) with tiered context loading,
automatic memory extraction, and session management.

Original PR #3369 by Mibayy, rewritten to use the full OpenViking session
lifecycle instead of read-only search endpoints.

Config via environment variables (profile-scoped via each profile's .env):
  OPENVIKING_ENDPOINT  — Server URL (default: http://127.0.0.1:1933)
  OPENVIKING_API_KEY   — API key (required for authenticated servers)
  OPENVIKING_ACCOUNT   — Tenant account (default: default)
  OPENVIKING_USER      — Tenant user (default: hermes)
  OPENVIKING_AGENT     — Tenant agent (default: hermes)

Capabilities:
  - Automatic memory extraction on session commit (6 categories)
  - Tiered context: L0 (~100 tokens), L1 (~2k), L2 (full)
  - Semantic search with hierarchical directory retrieval
  - Filesystem-style browsing via viking:// URIs
  - Resource ingestion (URLs, docs, code)
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://127.0.0.1:1933"
_TIMEOUT = 30.0
_EXPLICIT_MEMORY_ROOT = "viking://resources/hermes_explicit_memories"


# ---------------------------------------------------------------------------
# Process-level atexit safety net — ensures pending sessions are committed
# even if shutdown_memory_provider is never called (e.g. gateway crash,
# SIGKILL, or exception in _async_flush_memories preventing shutdown).
# ---------------------------------------------------------------------------
_last_active_provider: Optional["OpenVikingMemoryProvider"] = None


def _atexit_commit_sessions():
    """Fire on_session_end for the last active provider on process exit."""
    global _last_active_provider
    provider = _last_active_provider
    if provider is None:
        return
    _last_active_provider = None
    try:
        provider.on_session_end([])
    except Exception:
        pass  # best-effort at shutdown time


atexit.register(_atexit_commit_sessions)


# ---------------------------------------------------------------------------
# HTTP helper — uses httpx to avoid requiring the openviking SDK
# ---------------------------------------------------------------------------

def _get_httpx():
    """Lazy import httpx."""
    try:
        import httpx
        return httpx
    except ImportError:
        return None


class _VikingClient:
    """Thin HTTP client for the OpenViking REST API."""

    def __init__(self, endpoint: str, api_key: str = "",
                 account: str = "", user: str = "", agent: str = ""):
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._account = account or os.environ.get("OPENVIKING_ACCOUNT", "default")
        self._user = user or os.environ.get("OPENVIKING_USER", "hermes")
        self._agent = agent or os.environ.get("OPENVIKING_AGENT", "hermes")
        self._httpx = _get_httpx()
        if self._httpx is None:
            raise ImportError("httpx is required for OpenViking: pip install httpx")

    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "X-OpenViking-Account": self._account,
            "X-OpenViking-User": self._user,
            "X-OpenViking-Agent": self._agent,
        }
        if self._api_key:
            h["X-API-Key"] = self._api_key
        return h

    def _url(self, path: str) -> str:
        return f"{self._endpoint}{path}"

    def get(self, path: str, **kwargs) -> dict:
        resp = self._httpx.get(
            self._url(path), headers=self._headers(), timeout=_TIMEOUT, **kwargs
        )
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, payload: dict = None, **kwargs) -> dict:
        resp = self._httpx.post(
            self._url(path), json=payload or {}, headers=self._headers(),
            timeout=_TIMEOUT, **kwargs
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        try:
            resp = self._httpx.get(
                self._url("/health"), timeout=3.0
            )
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "viking_search",
    "description": (
        "Semantic search over the OpenViking knowledge base. "
        "Returns ranked results with viking:// URIs for deeper reading. "
        "Use mode='deep' for complex queries that need reasoning across "
        "multiple sources, 'fast' for simple lookups."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "mode": {
                "type": "string", "enum": ["auto", "fast", "deep"],
                "description": "Search depth (default: auto).",
            },
            "scope": {
                "type": "string",
                "description": "Viking URI prefix to scope search (e.g. 'viking://resources/docs/').",
            },
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["query"],
    },
}

READ_SCHEMA = {
    "name": "viking_read",
    "description": (
        "Read content at a viking:// URI. Three detail levels:\n"
        "  abstract — ~100 token summary (L0)\n"
        "  overview — ~2k token key points (L1)\n"
        "  full — complete content (L2)\n"
        "Start with abstract/overview, only use full when you need details."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "uri": {"type": "string", "description": "viking:// URI to read."},
            "level": {
                "type": "string", "enum": ["abstract", "overview", "full"],
                "description": "Detail level (default: overview).",
            },
        },
        "required": ["uri"],
    },
}

BROWSE_SCHEMA = {
    "name": "viking_browse",
    "description": (
        "Browse the OpenViking knowledge store like a filesystem.\n"
        "  list — show directory contents\n"
        "  tree — show hierarchy\n"
        "  stat — show metadata for a URI"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string", "enum": ["tree", "list", "stat"],
                "description": "Browse action.",
            },
            "path": {
                "type": "string",
                "description": "Viking URI path (default: viking://). Examples: 'viking://resources/', 'viking://user/memories/'.",
            },
        },
        "required": ["action"],
    },
}

REMEMBER_SCHEMA = {
    "name": "viking_remember",
    "description": (
        "Explicitly store a fact or memory in the OpenViking knowledge base. "
        "Use for important information the agent should remember long-term. "
        "The system automatically categorizes and indexes the memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to remember."},
            "category": {
                "type": "string",
                "enum": ["preference", "entity", "event", "case", "pattern"],
                "description": "Memory category (default: auto-detected).",
            },
        },
        "required": ["content"],
    },
}

ADD_RESOURCE_SCHEMA = {
    "name": "viking_add_resource",
    "description": (
        "Add a URL or document to the OpenViking knowledge base. "
        "Supports web pages, GitHub repos, PDFs, markdown, code files. "
        "The system automatically parses, indexes, and generates summaries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL or path of the resource to add."},
            "reason": {
                "type": "string",
                "description": "Why this resource is relevant (improves search).",
            },
        },
        "required": ["url"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class OpenVikingMemoryProvider(MemoryProvider):
    """Full bidirectional memory via OpenViking context database."""

    def __init__(self):
        self._client: Optional[_VikingClient] = None
        self._endpoint = ""
        self._api_key = ""
        self._account = "default"
        self._user = "hermes"
        self._agent = "hermes"
        self._session_id = ""
        self._turn_count = 0
        self._sync_thread: Optional[threading.Thread] = None
        self._memwrite_thread: Optional[threading.Thread] = None
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "openviking"

    def is_available(self) -> bool:
        """Check if OpenViking endpoint is configured. No network calls."""
        return bool(os.environ.get("OPENVIKING_ENDPOINT"))

    def get_config_schema(self):
        return [
            {
                "key": "endpoint",
                "description": "OpenViking server URL",
                "required": True,
                "default": _DEFAULT_ENDPOINT,
                "env_var": "OPENVIKING_ENDPOINT",
            },
            {
                "key": "api_key",
                "description": "OpenViking API key",
                "secret": True,
                "env_var": "OPENVIKING_API_KEY",
            },
            {
                "key": "account",
                "description": "OpenViking tenant account ID ([default], used when local mode, OPENVIKING_API_KEY is empty)",
                "default": "default",
                "env_var": "OPENVIKING_ACCOUNT",
            },
            {
                "key": "user",
                "description": "OpenViking user ID within the account ([hermes], used when local mode, OPENVIKING_API_KEY is empty)",
                "default": "hermes",
                "env_var": "OPENVIKING_USER",
            },
            {
                "key": "agent",
                "description": "OpenViking agent ID within the account ([hermes], useful in multi-agent mode)",
                "default": "hermes",
                "env_var": "OPENVIKING_AGENT",
            },
        ]

    def _ensure_session(self) -> bool:
        if not self._client or not self._session_id:
            return False
        try:
            self._client.get(
                f"/api/v1/sessions/{self._session_id}",
                params={"auto_create": "true"},
            )
            return True
        except Exception as e:
            logger.warning("OpenViking session ensure failed for %s: %s", self._session_id, e)
            return False

    def _build_client(self) -> _VikingClient:
        return _VikingClient(
            self._endpoint,
            self._api_key,
            account=self._account,
            user=self._user,
            agent=self._agent,
        )

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value or "memory").strip("_")
        return slug[:64] or "memory"

    def _fallback_roots_on_disk(self) -> List[Path]:
        base = Path.home() / ".openviking" / "data" / "viking"
        roots = []
        account = getattr(self, "_account", "default") or "default"
        for candidate in dict.fromkeys([account, "default", "root"]):
            root = base / candidate / "resources" / "hermes_explicit_memories"
            if root.exists():
                roots.append(root)
        return roots

    def _search_explicit_fallback(self, query: str, *, limit: int = 5) -> List[Dict[str, Any]]:
        needle = (query or "").strip().lower()
        if not needle:
            return []

        matches = []
        for root in self._fallback_roots_on_disk():
            for path in root.rglob("*.md"):
                if path.name in {".abstract.md", ".overview.md"}:
                    continue
                try:
                    content = path.read_text(errors="ignore")
                except Exception:
                    continue
                lower = content.lower()
                if needle not in lower:
                    continue
                idx = lower.index(needle)
                start = max(0, idx - 120)
                end = min(len(content), idx + 220)
                snippet = content[start:end].replace("\n", " ").strip()
                rel = path.relative_to(root.parent).as_posix()
                score = 1.0 if needle in path.name.lower() else 0.97
                matches.append({
                    "uri": f"viking://resources/{rel}",
                    "type": "resource",
                    "score": score,
                    "abstract": snippet[:280],
                    "_mtime": path.stat().st_mtime,
                })

        matches.sort(key=lambda item: (item["score"], item["_mtime"]), reverse=True)
        return [{k: v for k, v in item.items() if k != "_mtime"} for item in matches[:limit]]

    def _store_explicit_memory_resource(self, target: str, content: str) -> Optional[str]:
        if not self._client or not content:
            return None

        try:
            note = (
                "# Hermes explicit memory\n\n"
                f"Target: {target}\n"
                f"Session: {self._session_id or 'n/a'}\n"
                f"Recorded at ms: {int(time.time() * 1000)}\n\n"
                f"Content: {content}\n"
            ).encode("utf-8")
            headers = self._client._headers().copy()
            headers.pop("Content-Type", None)
            upload = self._client._httpx.post(
                self._client._url("/api/v1/resources/temp_upload"),
                headers=headers,
                files={"file": ("explicit-memory.md", note, "text/markdown")},
                data={"telemetry": "false"},
                timeout=_TIMEOUT,
            )
            upload.raise_for_status()
            temp_file_id = upload.json().get("result", {}).get("temp_file_id")
            if not temp_file_id:
                return None

            uri = (
                f"{_EXPLICIT_MEMORY_ROOT}/"
                f"{self._slugify(target)}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            )
            resp = self._client.post(
                "/api/v1/resources",
                {
                    "temp_file_id": temp_file_id,
                    "to": uri,
                    "reason": "Hermes explicit memory fallback",
                    "wait": False,
                },
            )
            return resp.get("result", {}).get("root_uri", uri)
        except Exception as e:
            logger.debug("OpenViking explicit memory fallback failed: %s", e)
            return None

    def _is_directory_uri(self, uri: str) -> bool:
        try:
            resp = self._client.get("/api/v1/fs/stat", params={"uri": uri})
            result = resp.get("result", {})
            return bool(result.get("isDir"))
        except Exception:
            return uri.endswith("/")

    def initialize(self, session_id: str, **kwargs) -> None:
        self._endpoint = os.environ.get("OPENVIKING_ENDPOINT", _DEFAULT_ENDPOINT)
        self._api_key = os.environ.get("OPENVIKING_API_KEY", "")
        self._account = os.environ.get("OPENVIKING_ACCOUNT", "default")
        self._user = os.environ.get("OPENVIKING_USER", "hermes")
        self._agent = os.environ.get("OPENVIKING_AGENT", "hermes")
        self._session_id = session_id
        self._turn_count = 0

        try:
            self._client = self._build_client()
            if not self._client.health():
                logger.warning("OpenViking server at %s is not reachable", self._endpoint)
                self._client = None
            else:
                self._ensure_session()
        except ImportError:
            logger.warning("httpx not installed — OpenViking plugin disabled")
            self._client = None

        # Register as the last active provider for atexit safety net
        global _last_active_provider
        _last_active_provider = self

    def system_prompt_block(self) -> str:
        if not self._client:
            return ""
        # Provide brief info about the knowledge base
        try:
            # Check what's in the knowledge base via a root listing
            resp = self._client.get("/api/v1/fs/ls", params={"uri": "viking://"})
            result = resp.get("result", [])
            children = len(result) if isinstance(result, list) else 0
            if children == 0:
                return ""
            return (
                "# OpenViking Knowledge Base\n"
                f"Active. Endpoint: {self._endpoint}\n"
                "Use viking_search to find information, viking_read for details "
                "(abstract/overview/full), viking_browse to explore.\n"
                "Use viking_remember to store facts, viking_add_resource to index URLs/docs."
            )
        except Exception as e:
            logger.warning("OpenViking system_prompt_block failed: %s", e)
            return (
                "# OpenViking Knowledge Base\n"
                f"Active. Endpoint: {self._endpoint}\n"
                "Use viking_search, viking_read, viking_browse, "
                "viking_remember, viking_add_resource."
            )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched results from the background thread."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## OpenViking Context\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire a background search to pre-load relevant context."""
        if not self._client or not query:
            return

        def _run():
            try:
                client = self._build_client()
                resp = client.post("/api/v1/search/find", {
                    "query": query,
                    "top_k": 5,
                })
                result = resp.get("result", {})
                parts = []
                for ctx_type in ("memories", "resources"):
                    items = result.get(ctx_type, [])
                    for item in items[:3]:
                        uri = item.get("uri", "")
                        abstract = item.get("abstract", "")
                        score = item.get("score", 0)
                        if abstract:
                            parts.append(f"- [{score:.2f}] {abstract} ({uri})")
                for item in self._search_explicit_fallback(query, limit=3):
                    parts.append(
                        f"- [{item['score']:.2f}] {item.get('abstract', '')} ({item['uri']})"
                    )
                if parts:
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(parts)
            except Exception as e:
                logger.debug("OpenViking prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="openviking-prefetch"
        )
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Record the conversation turn in OpenViking's session (non-blocking)."""
        if not self._client:
            return

        self._turn_count += 1

        def _sync():
            try:
                client = self._build_client()
                sid = self._session_id
                client.get(f"/api/v1/sessions/{sid}", params={"auto_create": "true"})

                # Add user message
                client.post(f"/api/v1/sessions/{sid}/messages", {
                    "role": "user",
                    "content": user_content[:4000],  # trim very long messages
                })
                # Add assistant message
                client.post(f"/api/v1/sessions/{sid}/messages", {
                    "role": "assistant",
                    "content": assistant_content[:4000],
                })
            except Exception as e:
                logger.debug("OpenViking sync_turn failed: %s", e)

        # Wait for any previous sync to finish before starting a new one
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="openviking-sync"
        )
        self._sync_thread.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Commit the session to trigger memory extraction.

        OpenViking automatically extracts 6 categories of memories:
        profile, preferences, entities, events, cases, and patterns.
        """
        if not self._client:
            return

        # Wait for any pending sync to finish first — do this before the
        # turn_count check so the last turn's messages are flushed even if
        # the count hasn't been incremented yet.
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)
        if self._memwrite_thread and self._memwrite_thread.is_alive():
            self._memwrite_thread.join(timeout=10.0)

        if self._turn_count == 0:
            return

        try:
            if not self._ensure_session():
                return
            self._client.post(f"/api/v1/sessions/{self._session_id}/commit")
            logger.info("OpenViking session %s committed (%d turns)", self._session_id, self._turn_count)
        except Exception as e:
            logger.warning("OpenViking session commit failed: %s", e)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to OpenViking as explicit memories."""
        if not self._client or action != "add" or not content:
            return

        def _write():
            try:
                client = self._build_client()
                client.get(
                    f"/api/v1/sessions/{self._session_id}",
                    params={"auto_create": "true"},
                )
                # Add as a user message with memory context so the commit
                # picks it up as an explicit memory during extraction
                client.post(f"/api/v1/sessions/{self._session_id}/messages", {
                    "role": "user",
                    "parts": [
                        {"type": "text", "text": f"[Memory note — {target}] {content}"},
                    ],
                })
            except Exception as e:
                logger.debug("OpenViking memory mirror failed: %s", e)
            finally:
                uri = self._store_explicit_memory_resource(target, content)
                if uri:
                    logger.info("OpenViking explicit memory fallback stored at %s", uri)

        if self._memwrite_thread and self._memwrite_thread.is_alive():
            self._memwrite_thread.join(timeout=5.0)

        self._memwrite_thread = threading.Thread(
            target=_write, daemon=True, name="openviking-memwrite"
        )
        self._memwrite_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, READ_SCHEMA, BROWSE_SCHEMA, REMEMBER_SCHEMA, ADD_RESOURCE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if not self._client:
            return tool_error("OpenViking server not connected")

        try:
            if tool_name == "viking_search":
                return self._tool_search(args)
            elif tool_name == "viking_read":
                return self._tool_read(args)
            elif tool_name == "viking_browse":
                return self._tool_browse(args)
            elif tool_name == "viking_remember":
                return self._tool_remember(args)
            elif tool_name == "viking_add_resource":
                return self._tool_add_resource(args)
            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            return tool_error(str(e))

    def shutdown(self) -> None:
        # Wait for background threads to finish
        for t in (self._sync_thread, self._memwrite_thread, self._prefetch_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        # Clear atexit reference so it doesn't double-commit
        global _last_active_provider
        if _last_active_provider is self:
            _last_active_provider = None

    # -- Tool implementations ------------------------------------------------

    def _tool_search(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")

        payload: Dict[str, Any] = {"query": query}
        mode = args.get("mode", "auto")
        if mode != "auto":
            payload["mode"] = mode
        if args.get("scope"):
            payload["target_uri"] = args["scope"]
        if args.get("limit"):
            payload["top_k"] = args["limit"]

        resp = self._client.post("/api/v1/search/find", payload)
        result = resp.get("result", {})

        # Format results for the model — keep it concise
        scored_entries = []
        for ctx_type in ("memories", "resources", "skills"):
            items = result.get(ctx_type, [])
            for item in items:
                raw_score = item.get("score")
                sort_score = raw_score if raw_score is not None else 0.0
                entry = {
                    "uri": item.get("uri", ""),
                    "type": ctx_type.rstrip("s"),
                    "score": round(raw_score, 3) if raw_score is not None else 0.0,
                    "abstract": item.get("abstract", ""),
                }
                if item.get("relations"):
                    entry["related"] = [r.get("uri") for r in item["relations"][:3]]
                scored_entries.append((sort_score, entry))

        scored_entries.sort(key=lambda x: x[0], reverse=True)
        formatted = [entry for _, entry in scored_entries]
        seen = {entry["uri"] for entry in formatted}
        for item in self._search_explicit_fallback(query, limit=args.get("limit") or 5):
            if item["uri"] in seen:
                continue
            formatted.append(item)
            seen.add(item["uri"])
        formatted.sort(key=lambda entry: entry.get("score", 0), reverse=True)

        return json.dumps({
            "results": formatted,
            "total": max(result.get("total", 0), len(formatted)),
        }, ensure_ascii=False)

    def _tool_read(self, args: dict) -> str:
        uri = args.get("uri", "")
        if not uri:
            return tool_error("uri is required")

        level = args.get("level", "overview")
        if level != "full" and not self._is_directory_uri(uri):
            level = "full"
        # Map our level names to OpenViking GET endpoints
        if level == "abstract":
            resp = self._client.get("/api/v1/content/abstract", params={"uri": uri})
        elif level == "full":
            resp = self._client.get("/api/v1/content/read", params={"uri": uri})
        else:  # overview
            resp = self._client.get("/api/v1/content/overview", params={"uri": uri})

        result = resp.get("result", "")
        # result is a plain string from the content endpoints
        content = result if isinstance(result, str) else result.get("content", "")

        # Truncate very long content to avoid flooding the context
        if len(content) > 8000:
            content = content[:8000] + "\n\n[... truncated, use a more specific URI or abstract level]"

        return json.dumps({
            "uri": uri,
            "level": level,
            "content": content,
        }, ensure_ascii=False)

    def _tool_browse(self, args: dict) -> str:
        action = args.get("action", "list")
        path = args.get("path", "viking://")

        # Map action to the correct fs endpoint (all GET with uri= param)
        endpoint_map = {"tree": "/api/v1/fs/tree", "list": "/api/v1/fs/ls", "stat": "/api/v1/fs/stat"}
        endpoint = endpoint_map.get(action, "/api/v1/fs/ls")
        resp = self._client.get(endpoint, params={"uri": path})
        result = resp.get("result", {})

        # Format list/tree results for readability
        if action in ("list", "tree") and isinstance(result, list):
            entries = []
            for e in result[:50]:  # cap at 50 entries
                entries.append({
                    "name": e.get("rel_path", e.get("name", "")),
                    "uri": e.get("uri", ""),
                    "type": "dir" if e.get("isDir") else "file",
                    "abstract": e.get("abstract", ""),
                })
            return json.dumps({"path": path, "entries": entries}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)

    def _tool_remember(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return tool_error("content is required")

        # Store as a session message that will be extracted during commit.
        # The category hint helps OpenViking's extraction classify correctly.
        category = args.get("category", "")
        text = f"[Remember] {content}"
        if category:
            text = f"[Remember — {category}] {content}"

        if not self._ensure_session():
            return tool_error("OpenViking session unavailable")

        self._client.post(f"/api/v1/sessions/{self._session_id}/messages", {
            "role": "user",
            "parts": [
                {"type": "text", "text": text},
            ],
        })
        self._client.post(f"/api/v1/sessions/{self._session_id}/commit")
        fallback_uri = self._store_explicit_memory_resource(category or "memory", content)

        return json.dumps({
            "status": "stored",
            "message": "Memory recorded. Will be extracted and indexed on session commit.",
            "fallback_uri": fallback_uri,
        })

    def _tool_add_resource(self, args: dict) -> str:
        url = args.get("url", "")
        if not url:
            return tool_error("url is required")

        if os.path.exists(url):
            headers = self._client._headers().copy()
            headers.pop("Content-Type", None)
            with open(url, "rb") as f:
                upload = self._client._httpx.post(
                    self._client._url("/api/v1/resources/temp_upload"),
                    headers=headers,
                    files={"file": (os.path.basename(url), f, "application/octet-stream")},
                    data={"telemetry": "false"},
                    timeout=_TIMEOUT,
                )
            upload.raise_for_status()
            temp_file_id = upload.json().get("result", {}).get("temp_file_id")
            if not temp_file_id:
                return tool_error("OpenViking temp upload failed")

            payload: Dict[str, Any] = {"temp_file_id": temp_file_id, "wait": False}
            if args.get("reason"):
                payload["reason"] = args["reason"]
            resp = self._client.post("/api/v1/resources", payload)
        else:
            payload = {"path": url}
            if args.get("reason"):
                payload["reason"] = args["reason"]
            resp = self._client.post("/api/v1/resources", payload)
        result = resp.get("result", {})

        return json.dumps({
            "status": "added",
            "root_uri": result.get("root_uri", ""),
            "message": "Resource queued for processing. Use viking_search after a moment to find it.",
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register OpenViking as a memory provider plugin."""
    ctx.register_memory_provider(OpenVikingMemoryProvider())
