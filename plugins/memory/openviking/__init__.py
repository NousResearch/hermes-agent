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
  OPENVIKING_USER      — Tenant user (default: default)
  OPENVIKING_AGENT   — Tenant agent (default: hermes)

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
import mimetypes
import os
import queue
import sqlite3
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

# Standalone session registry — zero-dependency module used by all
# code paths (CLI, gateway, cron, batch) to track session lifecycle.
from .registry import (
    register_session as _register_session,
    update_state as _update_registry_state,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://127.0.0.1:1933"
_TIMEOUT = 30.0
_REMOTE_RESOURCE_PREFIXES = ("http://", "https://", "git@", "ssh://", "git://")

# Cache and persistence limits
_MAX_CACHED_MESSAGES = int(os.environ.get("OPENVIKING_CACHE_SIZE", "10000"))
_RECOVERY_DIR = os.path.join(os.path.expanduser("~"), ".hermes", "openviking-recovery")

# Maps the viking_remember `category` enum to a viking:// subdirectory.
# Keep in sync with REMEMBER_SCHEMA.parameters.properties.category.enum.
_CATEGORY_SUBDIR_MAP = {
    "preference": "preferences",
    "entity": "entities",
    "event": "events",
    "case": "cases",
    "pattern": "patterns",
}
_DEFAULT_MEMORY_SUBDIR = "preferences"

# Maps the built-in memory tool's `target` ("user" vs "memory") to a subdir
# for on_memory_write mirroring. User profile facts → preferences; agent
# notes / observations → patterns. Anything unknown falls back to the default.
_MEMORY_WRITE_TARGET_SUBDIR_MAP = {
    "user": "preferences",
    "memory": "patterns",
}


# ---------------------------------------------------------------------------
# Process-level atexit safety net — ensures pending sessions are committed
# even if shutdown_memory_provider is never called (e.g. gateway crash,
# SIGKILL, or exception in the session expiry watcher preventing shutdown).
# ---------------------------------------------------------------------------
_last_active_provider: Optional["OpenVikingMemoryProvider"] = None


def _atexit_commit_sessions():
    """Fire on_session_end for the last active provider on process exit.

    This is the LAST line of defence. The normal path is:
    shutdown_memory_provider() → on_session_end() → commit.

    If that path was taken and SUCCEEDED, the _commit_state is 2 and
    _last_active_provider was cleared by shutdown(), so this is a no-op.

    If that path was taken but FAILED, _commit_state is 3 and the
    provider's _last_active_provider was cleared — we fall through to
    the persistent recovery file check below.

    If shutdown_memory_provider was NEVER called (SIGKILL recovery,
    atexit-only exit), _last_active_provider is still set and we
    attempt the commit directly.
    """
    global _last_active_provider
    provider = _last_active_provider

    # Priority 1: Live provider with cached messages
    if provider is not None:
        # Set to None now so we don't retry — the _commit_state
        # guard in on_session_end/force_commit prevents double-commit
        _last_active_provider = None
        try:
            messages = list(getattr(provider, "_cached_messages", []))
            if messages:
                provider.on_session_end(messages)
                return
        except Exception:
            pass  # fall through to recovery files

    # Priority 2: Persistent recovery files
    # Catches cases where shutdown() cleared _last_active_provider
    # before the commit completed, or where the process died before
    # any cleanup ran.
    try:
        marker_path = os.path.join(_RECOVERY_DIR, ".pending")
        if os.path.exists(marker_path):
            with open(marker_path) as f:
                pending_sids = [line.strip() for line in f if line.strip()]
            for sid in pending_sids:
                snap_path = os.path.join(_RECOVERY_DIR, f"{sid}.json")
                if os.path.exists(snap_path):
                    try:
                        with open(snap_path) as f:
                            snap = json.load(f)
                        msgs = snap.get("messages", [])
                        if msgs:
                            _atexit_force_commit(sid, msgs)
                    except Exception:
                        pass
    except Exception:
        pass


def _atexit_force_commit(session_id: str, messages: list) -> None:
    """Standalone OV commit for atexit — no provider dependency.

    Creates its own HTTP client so it works even if the provider's
    client was destroyed by shutdown(). Safe to call from any context.
    """
    if not messages or not session_id:
        return
    endpoint = os.environ.get("OPENVIKING_ENDPOINT", _DEFAULT_ENDPOINT)
    api_key = os.environ.get("OPENVIKING_API_KEY", "")
    try:
        import httpx
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        with httpx.Client(timeout=15.0) as client:
            for msg in messages[-2000:]:  # last 2000 in case of huge sessions
                role = msg.get("role", "user")
                content = (msg.get("content") or "")[:4000]
                client.post(
                    f"{endpoint}/api/v1/sessions/{session_id}/messages",
                    json={"role": role, "content": content},
                    headers=headers,
                )
            client.post(
                f"{endpoint}/api/v1/sessions/{session_id}/commit",
                headers=headers,
            )
    except Exception:
        pass  # best-effort at shutdown time


atexit.register(_atexit_commit_sessions)

# Queue worker sentinel types — these are put on the message queue
# to signal lifecycle events to the single daemon worker thread.
_MSG = "msg"         # A conversation message: (sid, role, content)
_FLUSH = "flush"     # Drain all pending messages before continuing
_SHUTDOWN = "exit"   # Terminate the worker thread


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
        self._user = user or os.environ.get("OPENVIKING_USER", "default")
        self._agent = agent or os.environ.get("OPENVIKING_AGENT", "hermes")
        self._httpx = _get_httpx()
        if self._httpx is None:
            raise ImportError("httpx is required for OpenViking: pip install httpx")

    def _headers(self) -> dict:
        # Always send tenant headers when account/user are configured.
        # OpenViking 0.3.x requires X-OpenViking-Account and X-OpenViking-User
        # for ROOT API key requests to tenant-scoped APIs — omitting them
        # causes INVALID_ARGUMENT errors even when account="default".
        # User-level keys can omit them (server derives tenancy from the key),
        # but ROOT keys must always include them explicitly.
        h = {
            "Content-Type": "application/json",
            "X-OpenViking-Agent": self._agent,
        }
        if self._account:
            h["X-OpenViking-Account"] = self._account
        if self._user:
            h["X-OpenViking-User"] = self._user
        if self._api_key:
            h["X-API-Key"] = self._api_key
            h["Authorization"] = "Bearer " + self._api_key
        return h

    def _url(self, path: str) -> str:
        return f"{self._endpoint}{path}"

    def _multipart_headers(self) -> dict:
        headers = self._headers()
        headers.pop("Content-Type", None)
        return headers

    def _parse_response(self, resp) -> dict:
        try:
            data = resp.json()
        except Exception:
            data = None

        if resp.status_code >= 400:
            if isinstance(data, dict):
                error = data.get("error")
                if isinstance(error, dict):
                    code = error.get("code", "HTTP_ERROR")
                    message = error.get("message", resp.text)
                    raise RuntimeError(f"{code}: {message}")
                if data.get("status") == "error":
                    raise RuntimeError(str(data))
            resp.raise_for_status()

        if isinstance(data, dict) and data.get("status") == "error":
            error = data.get("error")
            if isinstance(error, dict):
                code = error.get("code", "OPENVIKING_ERROR")
                message = error.get("message", "")
                raise RuntimeError(f"{code}: {message}")
            raise RuntimeError(str(data))

        if data is None:
            return {}
        return data

    def get(self, path: str, **kwargs) -> dict:
        resp = self._httpx.get(
            self._url(path), headers=self._headers(), timeout=_TIMEOUT, **kwargs
        )
        return self._parse_response(resp)

    def post(self, path: str, payload: dict = None, **kwargs) -> dict:
        resp = self._httpx.post(
            self._url(path), json=payload or {}, headers=self._headers(),
            timeout=_TIMEOUT, **kwargs
        )
        return self._parse_response(resp)

    def upload_temp_file(self, file_path: Path) -> str:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        with file_path.open("rb") as f:
            resp = self._httpx.post(
                self._url("/api/v1/resources/temp_upload"),
                files={"file": (file_path.name, f, mime_type)},
                headers=self._multipart_headers(),
                timeout=_TIMEOUT,
            )
        data = self._parse_response(resp)
        result = data.get("result", {})
        temp_file_id = result.get("temp_file_id", "")
        if not temp_file_id:
            raise RuntimeError("OpenViking temp upload did not return temp_file_id")
        return temp_file_id

    def health(self) -> bool:
        try:
            resp = self._httpx.get(
                self._url("/health"), headers=self._headers(), timeout=3.0
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
        "Add a remote URL or local file/directory to the OpenViking knowledge base. "
        "Remote resources must be public http(s), git, or ssh URLs. "
        "Local files are uploaded first using OpenViking temp_upload. "
        "The system automatically parses, indexes, and generates summaries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Remote URL or local file/directory path to add."},
            "reason": {
                "type": "string",
                "description": "Why this resource is relevant (improves search).",
            },
            "to": {
                "type": "string",
                "description": "Optional target viking:// URI for the resource.",
            },
            "parent": {
                "type": "string",
                "description": "Optional parent viking:// URI. Cannot be used with to.",
            },
            "instruction": {
                "type": "string",
                "description": "Optional processing instruction for semantic extraction.",
            },
            "wait": {
                "type": "boolean",
                "description": "Whether to wait for processing to complete.",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds when wait is true.",
            },
        },
        "required": ["url"],
    },
}


def _zip_directory(dir_path: Path) -> Path:
    """Create a temporary zip file containing a directory tree."""
    root = dir_path.resolve()
    zip_path = Path(tempfile.gettempdir()) / f"openviking_upload_{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in dir_path.rglob("*"):
            if file_path.is_symlink():
                continue
            if file_path.is_file():
                try:
                    file_path.resolve().relative_to(root)
                except ValueError:
                    continue
                arcname = str(file_path.relative_to(dir_path)).replace("\\", "/")
                zipf.write(file_path, arcname=arcname)
    return zip_path


def _is_windows_absolute_path(value: str) -> bool:
    return (
        len(value) >= 3
        and value[0].isalpha()
        and value[1] == ":"
        and value[2] in {"/", "\\"}
    )


def _is_remote_resource_source(value: str) -> bool:
    return value.startswith(_REMOTE_RESOURCE_PREFIXES)


def _is_local_path_reference(value: str) -> bool:
    if not value or "\n" in value or "\r" in value:
        return False
    if _is_remote_resource_source(value):
        return False
    if _is_windows_absolute_path(value):
        return True
    return (
        value.startswith(("/", "./", "../", "~/", ".\\", "..\\", "~\\"))
        or "/" in value
        or "\\" in value
    )


def _path_from_file_uri(uri: str) -> Path | str:
    parsed = urlparse(uri)
    if parsed.netloc not in {"", "localhost"}:
        return f"Unsupported non-local file URI: {uri}"
    return Path(url2pathname(parsed.path)).expanduser()


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class OpenVikingMemoryProvider(MemoryProvider):
    """Full bidirectional memory via OpenViking context database."""

    def __init__(self):
        self._client: Optional[_VikingClient] = None
        self._endpoint = ""
        self._api_key = ""
        self._session_id = ""
        self._turn_count = 0
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        # Message cache for atexit safety net — sync_turn() stores a copy
        # here so _atexit_commit_sessions() has real data to commit even
        # if the background sync thread never finished.
        self._cached_messages: List[Dict[str, Any]] = []
        # Queue-backed worker thread — replaces thread-per-turn pattern.
        # sync_turn pushes messages onto the queue; a single daemon worker
        # drains them and POSTs to OpenViking. Eliminates zombie threads
        # from timeouts and guarantees in-order message delivery.
        self._msg_queue: queue.Queue = queue.Queue(maxsize=0)
        self._worker_thread: Optional[threading.Thread] = None
        self._flush_event = threading.Event()
        # Journal directory for local fallback when OpenViking is unreachable.
        self._journal_dir: Optional[str] = None
        # Commit state guard: 0=uncommitted, 1=committing, 2=committed,
        # 3=failed. Prevents double-commit when both shutdown_memory_provider
        # and the atexit handler call on_session_end().
        self._commit_state = 0

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
                "description": "OpenViking API key (leave blank for local dev mode)",
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
                "description": "OpenViking user ID within the account ([default], used when local mode, OPENVIKING_API_KEY is empty)",
                "default": "default",
                "env_var": "OPENVIKING_USER",
            },
            {
                "key": "agent",
                "description": "OpenViking agent ID within the account ([hermes], useful in multi-agent mode)",
                "default": "hermes",
                "env_var": "OPENVIKING_AGENT",
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._endpoint = os.environ.get("OPENVIKING_ENDPOINT", _DEFAULT_ENDPOINT)
        self._api_key = os.environ.get("OPENVIKING_API_KEY", "")
        self._account = os.environ.get("OPENVIKING_ACCOUNT", "default")
        self._user = os.environ.get("OPENVIKING_USER", "default")
        self._agent = os.environ.get("OPENVIKING_AGENT", "hermes")
        self._session_id = session_id
        self._turn_count = 0
        self._cached_messages = []
        self._commit_state = 0
        self._update_state_db("CREATED")
        # Register in the standalone session registry — this is the only
        # place all code paths converge. The registry provides a process-
        # independent audit trail for the finalizer.
        _register_session(session_id, source="ov-plugin")
        try:
            self._client = _VikingClient(
                self._endpoint, self._api_key,
                account=self._account, user=self._user, agent=self._agent,
            )
            if not self._client.health():
                logger.warning("OpenViking server at %s is not reachable", self._endpoint)
                self._client = None
            else:
                logger.info(
                    "OpenViking client initialized for session %s at %s",
                    session_id, self._endpoint,
                )
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
                client = _VikingClient(
                    self._endpoint, self._api_key,
                    account=self._account, user=self._user, agent=self._agent,
                )
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
        """Record the conversation turn in OpenViking's session (non-blocking).

        Pushes messages onto a single-worker queue instead of spawning a
        thread per turn. The daemon worker drains the queue in order and
        POSTs to OpenViking with up to 3 retries.
        """
        if not self._client:
            return

        self._turn_count += 1

        # Cache messages for atexit safety net immediately — always have
        # the complete turn log regardless of worker thread state.
        self._cached_messages.append({"role": "user", "content": user_content[:4000]})
        self._cached_messages.append({"role": "assistant", "content": assistant_content[:4000]})
        # Cap at 5000 message pairs to avoid unbounded memory growth
        # in long-running gateway session with thousands of turns.
        if len(self._cached_messages) > _MAX_CACHED_MESSAGES:
            self._cached_messages = self._cached_messages[-(_MAX_CACHED_MESSAGES):]

        # Queue the messages — capture session_id at enqueue time so the
        # worker writes to the correct session even if on_session_switch
        # is called before the worker processes this item.
        sid = self._session_id
        self._start_worker()
        self._msg_queue.put((_MSG, sid, "user", user_content[:4000]))
        self._msg_queue.put((_MSG, sid, "assistant", assistant_content[:4000]))
        self._update_state_db("IN_SYNC")

    def _start_worker(self) -> None:
        """Start the daemon queue worker if it is not already running."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._queue_worker,
                daemon=True,
                name="openviking-queue-worker",
            )
            self._worker_thread.start()

    def _queue_worker(self) -> None:
        """Daemon worker: drain the message queue, POST to OpenViking with
        retries, handle lifecycle sentinels.

        Three sentinel types:
          (_MSG, sid, role, content)  — POST a conversation message
          (_FLUSH, "", "", "")         — signal the flush event then continue
          (_SHUTDOWN, "", "", "")      — exit the loop

        The worker is daemon=True so it does not prevent process exit.
        If the worker dies (unhandled exception), the next sync_turn()
        call restarts it via _start_worker().
        """
        while True:
            try:
                item = self._msg_queue.get()
            except Exception:
                # Queue corrupted or interrupted — exit the loop.
                break

            try:
                msg_type = item[0]

                if msg_type == _SHUTDOWN:
                    self._msg_queue.task_done()
                    break

                if msg_type == _FLUSH:
                    self._flush_event.set()
                    self._msg_queue.task_done()
                    continue

                if msg_type == _MSG:
                    _sid, role, content = item[1], item[2], item[3]
                    # Retry chain: up to 3 attempts with backoff
                    last_error = None
                    for attempt in range(3):
                        try:
                            client = _VikingClient(
                                self._endpoint, self._api_key or "",
                                account=self._account, user=self._user,
                                agent=self._agent,
                            )
                            client.post(f"/api/v1/sessions/{_sid}/messages", {
                                "role": role,
                                "content": content[:4000],
                            })
                            last_error = None
                            break
                        except Exception as e:
                            last_error = e
                            if attempt < 2:
                                time.sleep(1.0 + attempt)  # 1s, then 2s backoff

                    if last_error is not None:
                        logger.error(
                            "OpenViking queue worker: failed to POST %s message for "
                            "session %s after 3 retries: %s",
                            role, _sid, last_error,
                        )
                        self._journal_message(_sid, role, content)
            except Exception as e:
                logger.error(
                    "OpenViking queue worker: unhandled error processing %s: %s",
                    item[0] if isinstance(item, tuple) else type(item).__name__, e,
                )
            finally:
                try:
                    self._msg_queue.task_done()
                except Exception:
                    pass

    def _journal_message(self, session_id: str, role: str, content: str) -> None:
        """Write a failed message to the local journal for later repair.

        The journal file is a JSONL file at:
          ~/.hermes/openviking-repair/<session_id>.jsonl

        This feeds into the Phase 4 cron finalizer which discovers
        uncommitted sessions and attempts recovery.
        """
        try:
            repair_dir = os.path.join(
                os.path.expanduser("~"), ".hermes", "openviking-repair",
            )
            os.makedirs(repair_dir, exist_ok=True)
            journal_path = os.path.join(repair_dir, f"{session_id}.jsonl")
            with open(journal_path, "a") as f:
                f.write(json.dumps({
                    "session_id": session_id,
                    "role": role,
                    "content": content[:4000],
                }) + "\n")
        except Exception:
            pass  # best-effort, can't do much if journal write fails

    def _write_repair_marker(self) -> None:
        """Write a repair marker for sessions that failed to commit.

        The Phase 4 cron finalizer discovers these markers and attempts
        to recover the session using data from the Hermes session DB.
        """
        try:
            repair_dir = os.path.join(
                os.path.expanduser("~"), ".hermes", "openviking-repair",
            )
            os.makedirs(repair_dir, exist_ok=True)
            marker_path = os.path.join(repair_dir, f"{self._session_id}.json")
            with open(marker_path, "w") as f:
                json.dump({
                    "session_id": self._session_id,
                    "cached_messages": self._cached_messages,
                }, f)
        except Exception:
            pass

    def _persist_snapshot(self) -> None:
        """Persist session data to a crash-safe JSON file before shutdown.

        This file survives process death (SIGKILL, crash) and is discovered
        by the atexit handler and the background finalizer. Written atomically
        via tmp + os.replace to prevent partial-write corruption.
        """
        try:
            os.makedirs(_RECOVERY_DIR, exist_ok=True)
            path = os.path.join(_RECOVERY_DIR, f"{self._session_id}.json")
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump({
                    "session_id": self._session_id,
                    "messages": self._cached_messages,
                    "turn_count": self._turn_count,
                    "commit_state": self._commit_state,
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }, f)
            os.replace(tmp_path, path)
        except Exception:
            pass  # best-effort

    def _write_recovery_marker(self) -> None:
        """Write a lightweight marker that the finalizer discovers quickly.

        Appends the session_id to a .pending file that the finalizer
        scans on each run. Avoids scanning the full recovery directory.
        """
        try:
            os.makedirs(_RECOVERY_DIR, exist_ok=True)
            marker_path = os.path.join(_RECOVERY_DIR, ".pending")
            with open(marker_path, "a") as f:
                f.write(f"{self._session_id}\n")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # SQLite session state database — delegated to .registry module
    # ------------------------------------------------------------------ #
    # Uses the standalone registry.py which has zero agent dependencies
    # and can be imported from any code path (cron, batch, gateway).
    # The registry is the single source of truth for session lifecycle.
    # ------------------------------------------------------------------ #

    def _update_state_db(
        self,
        state: str,
        turn_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Upsert the current session's state into the SQLite registry.

        Args:
            state: One of CREATED, IN_SYNC, FINALIZING, COMMITTED, FAILED.
            turn_count: Current turn count (defaults to self._turn_count).
            error: Optional error message for FAILED state.
        """
        sid = self._session_id
        if not sid:
            return
        tc = turn_count if turn_count is not None else self._turn_count
        _update_registry_state(
            sid, state,
            turn_count=tc,
            messages=self._cached_messages,
            error=error,
        )

    def force_commit(self, messages: List[Dict[str, Any]]) -> None:
        """Synchronous fallback: create a fresh client, POST all messages, commit.

        This is the breaker bar for known-session-end scenarios where the
        background sync thread may still be running (or have timed out).
        Creates its own ``_VikingClient`` so it does not depend on ``self._client``,
        which may have been destroyed by a gateway crash or race with shutdown.

        Used as a fallback by ``on_session_end()`` when the sync thread join
        times out, and directly by ``_atexit_commit_sessions()``.
        """
        if not messages:
            return
        # Guard: prevent double-commit when both shutdown_memory_provider
        # and the atexit handler fire (force_commit is the fallback path
        # for on_session_end, so it should also respect the guard).
        if self._commit_state >= 2:
            return
        self._commit_state = 1
        self._update_state_db("FINALIZING", turn_count=len(messages) // 2)
        endpoint = self._endpoint or _DEFAULT_ENDPOINT
        api_key = self._api_key or ""
        try:
            client = _VikingClient(
                endpoint, api_key,
                account=self._account, user=self._user, agent=self._agent,
            )
            sid = self._session_id
            for msg in messages:
                role = msg.get("role", "user")
                content = (msg.get("content") or "")[:4000]
                client.post(f"/api/v1/sessions/{sid}/messages", {
                    "role": role,
                    "content": content,
                })
            client.post(f"/api/v1/sessions/{sid}/commit")
            self._commit_state = 2
            self._update_state_db("COMMITTED")
            logger.info(
                "OpenViking force_commit completed for session %s (%d messages)",
                sid, len(messages),
            )
        except Exception as e:
            self._commit_state = 3
            self._update_state_db("FAILED", error=str(e))
            logger.error(
                'OpenViking force_commit failed for session %s: %s',
                self._session_id, e,
            )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Commit the session to trigger memory extraction.

        OpenViking automatically extracts 6 categories of memories:
        profile, preferences, entities, events, cases, and patterns.

        First flushes the queue worker so all pending messages are written
        to OpenViking, then POSTs /commit. If the flush times out, falls
        back to force_commit(cached_messages) — the synchronous breaker bar.
        """
        if not self._client:
            return
        # Guard: prevent double-commit when both shutdown_memory_provider
        # and the atexit handler fire.
        if self._commit_state >= 2:
            return
        self._commit_state = 1
        self._update_state_db("FINALIZING")

        # Flush the queue — wait for the worker to drain all pending messages
        # before committing. Uses the flush sentinel + event pattern so the
        # worker stays alive for future flushes.
        flush_ok = self._flush_queue(timeout=15.0)

        if self._turn_count == 0:
            # No sync_turn calls were made — check if the caller passed
            # messages directly (e.g. from gateway _session_messages or
            # CLI atexit cleanup). Use them as a fallback.
            if messages:
                logger.info(
                    "OpenViking on_session_end: _turn_count=0 but caller passed %d messages — "
                    "using force_commit fallback",
                    len(messages),
                )
                self.force_commit(list(messages))
                self._commit_state = 2
                return
            self._commit_state = 2
            return

        if not flush_ok:
            # Queue flush timed out — the worker may be stuck or dead.
            # Fall back to synchronous force_commit with the message cache,
            # or the messages passed by the caller if the cache is empty.
            source = list(self._cached_messages) if self._cached_messages else list(messages or [])
            logger.warning(
                "OpenViking on_session_end: queue flush timed out for session %s "
                "— falling back to force_commit (%d msgs from %s)",
                self._session_id, len(source),
                "cache" if self._cached_messages else "caller messages",
            )
            if source:
                self.force_commit(source)
            else:
                logger.warning(
                    "OpenViking on_session_end: no messages available for session %s — "
                    "cannot commit",
                    self._session_id,
                )
                self._commit_state = 3
                self._update_state_db("FAILED", error="no messages available")
            return

        try:
            self._client.post(f"/api/v1/sessions/{self._session_id}/commit")
            self._commit_state = 2
            self._update_state_db("COMMITTED")
            logger.info("OpenViking session %s committed (%d turns)", self._session_id, self._turn_count)
        except Exception as e:
            self._commit_state = 3
            self._update_state_db("FAILED", error=str(e))
            logger.warning("OpenViking session commit failed: %s", e)
            self._write_repair_marker()

    def _flush_queue(self, timeout: float = 15.0) -> bool:
        """Put a flush sentinel on the queue and wait for the worker to drain.

        Returns True if the worker drained and set the flush event within
        *timeout* seconds. Returns False if the flush timed out (worker
        may be stuck or dead).
        """
        if not self._worker_thread or not self._worker_thread.is_alive():
            # Worker not running — nothing to flush.
            return True
        self._flush_event.clear()
        self._msg_queue.put((_FLUSH, "", "", ""))
        return self._flush_event.wait(timeout=timeout)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """Update cached session state when the agent rotates session_id.

        The base ``MemoryProvider.on_session_switch`` is a no-op, but we
        cache ``_session_id`` and ``_turn_count`` in ``initialize()`` and
        ``sync_turn()`` — if we don't refresh them here, all subsequent
        writes and the final commit target the **original** session_id
        instead of the new one.

        Called on ``/new``, ``/reset``, ``/resume``, ``/branch``, context
        compression — any path that rotates ``AIAgent.session_id`` without
        tearing the provider down.
        """
        if not new_session_id:
            return
        self._session_id = new_session_id
        if reset:
            # Genuinely new conversation — reset the turn counter so
            # ``on_session_end()`` starts fresh.  The previous session's
            # commit was already triggered by ``commit_memory_session()``
            # before this switch.
            self._turn_count = 0
            self._cached_messages = []
            self._commit_state = 0

    def _build_memory_uri(self, subdir: str) -> str:
        """Build a viking:// memory URI under the configured user/agent/subdir."""
        slug = uuid.uuid4().hex[:12]
        return f"viking://user/{self._user}/agent/{self._agent}/memories/{subdir}/mem_{slug}.md"

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes to OpenViking via content/write."""
        if not self._client or action != "add" or not content:
            return

        subdir = _MEMORY_WRITE_TARGET_SUBDIR_MAP.get(target, _DEFAULT_MEMORY_SUBDIR)
        uri = self._build_memory_uri(subdir)

        try:
            client = _VikingClient(
                self._endpoint, self._api_key,
                account=self._account, user=self._user, agent=self._agent,
            )
            client.post("/api/v1/content/write", {
                "uri": uri,
                "content": content,
                "mode": "create",
            })
        except Exception as e:
            logger.debug("OpenViking memory mirror failed: %s", e)

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
        # Phase 1: Persist snapshot to disk BEFORE clearing the atexit ref.
        # This ensures recovery files exist even if the commit below fails.
        if self._session_id and self._cached_messages:
            self._persist_snapshot()
        # Phase 2: Attempt synchronous force_commit as a last write.
        # force_commit creates its own HTTP client and is independent
        # of the worker thread. If this succeeds, the session is safe.
        if self._commit_state < 2 and self._cached_messages:
            try:
                self.force_commit(list(self._cached_messages))
            except Exception:
                pass
        # Phase 3: Shut down worker threads
        if self._worker_thread and self._worker_thread.is_alive():
            self._msg_queue.put((_SHUTDOWN, "", "", ""))
            self._worker_thread.join(timeout=5.0)
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5.0)
        # Phase 4: Clear atexit ref ONLY after everything above ran.
        # The atexit handler will check persistent recovery files
        # as a fallback if this commit failed.
        global _last_active_provider
        if _last_active_provider is self:
            _last_active_provider = None
        # Phase 5: Write recovery marker for the finalizer
        if self._commit_state < 2 and self._session_id:
            self._write_recovery_marker()

    # -- Tool implementations ------------------------------------------------

    @staticmethod
    def _unwrap_result(resp: Any) -> Any:
        """Return OpenViking payload body regardless of wrapped/unwrapped shape."""
        if isinstance(resp, dict) and "result" in resp:
            return resp.get("result")
        return resp

    @staticmethod
    def _normalize_summary_uri(uri: str) -> str:
        """Map pseudo summary files to their parent directory URI for L0/L1 reads."""
        if not uri:
            return uri
        for suffix in ("/.abstract.md", "/.overview.md", "/.read.md", "/.full.md"):
            if uri.endswith(suffix):
                return uri[: -len(suffix)] or "viking://"
        return uri

    def _is_directory_uri(self, uri: str) -> bool | None:
        """Probe fs/stat to decide if a URI is a directory.

        Returns True/False when the server answers cleanly, and None when the
        probe itself fails (network error, unexpected shape). Callers should
        treat None as "unknown" and fall back to the exception-based path.
        """
        try:
            resp = self._client.get("/api/v1/fs/stat", params={"uri": uri})
        except Exception:
            return None
        result = self._unwrap_result(resp)
        if isinstance(result, dict):
            if "isDir" in result:
                return bool(result.get("isDir"))
            if "is_dir" in result:
                return bool(result.get("is_dir"))
            if result.get("type") == "dir":
                return True
            if result.get("type") == "file":
                return False
        return None

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

        return json.dumps({
            "results": formatted,
            "total": result.get("total", len(formatted)),
        }, ensure_ascii=False)

    def _tool_read(self, args: dict) -> str:
        uri = args.get("uri", "")
        if not uri:
            return tool_error("uri is required")

        level = args.get("level", "overview")

        summary_level = level in {"abstract", "overview"}
        # OpenViking expects directory URIs for pseudo summary files
        # (e.g. viking://user/hermes/.overview.md).
        resolved_uri = self._normalize_summary_uri(uri) if summary_level else uri
        used_fallback = False

        # abstract/overview endpoints are directory-only on OpenViking
        # (v0.3.x returns 500/412 for file URIs). When the caller asks for a
        # summary level on a non-pseudo URI, probe fs/stat first and route
        # file URIs straight to /content/read instead of eating a failing
        # round-trip. The pseudo-URI path already points at a directory, so
        # skip the probe there.
        if summary_level and resolved_uri == uri:
            is_dir = self._is_directory_uri(uri)
            if is_dir is False:
                resolved_uri = uri
                used_fallback = True

        # Map our level names to OpenViking GET endpoints.
        endpoint = "/api/v1/content/read"
        if not used_fallback:
            if level == "abstract":
                endpoint = "/api/v1/content/abstract"
            elif level == "overview":
                endpoint = "/api/v1/content/overview"

        try:
            resp = self._client.get(endpoint, params={"uri": resolved_uri})
        except Exception:
            # OpenViking may return HTTP 500 for abstract/overview reads on normal
            # file URIs (mem_*.md). For those, gracefully fallback to full read.
            if not summary_level or resolved_uri != uri or used_fallback:
                raise
            resp = self._client.get("/api/v1/content/read", params={"uri": uri})
            used_fallback = True

        result = self._unwrap_result(resp)
        # Content endpoints may return either plain strings or objects.
        if isinstance(result, str):
            content = result
        elif isinstance(result, dict):
            content = result.get("content", "") or result.get("text", "")
        else:
            content = ""

        # Truncate long content to avoid flooding context.
        max_len = 8000
        if level == "overview":
            max_len = 4000
        elif level == "abstract":
            max_len = 1200

        if len(content) > max_len:
            content = content[:max_len] + "\n\n[... truncated, use a more specific URI or full level]"

        payload = {
            "uri": uri,
            "resolved_uri": resolved_uri,
            "level": level,
            "content": content,
        }
        if used_fallback:
            payload["fallback"] = "content/read"

        return json.dumps(payload, ensure_ascii=False)

    def _tool_browse(self, args: dict) -> str:
        action = args.get("action", "list")
        path = args.get("path", "viking://")

        # Map action to the correct fs endpoint (all GET with uri= param)
        endpoint_map = {"tree": "/api/v1/fs/tree", "list": "/api/v1/fs/ls", "stat": "/api/v1/fs/stat"}
        endpoint = endpoint_map.get(action, "/api/v1/fs/ls")
        resp = self._client.get(endpoint, params={"uri": path})
        result = self._unwrap_result(resp)

        # Format list/tree results for readability
        if action in {"list", "tree"}:
            raw_entries = result
            if isinstance(result, dict):
                raw_entries = result.get("entries") or result.get("items") or result.get("children") or []

            if isinstance(raw_entries, list):
                entries = []
                for e in raw_entries[:50]:  # cap at 50 entries
                    uri = e.get("uri", "")
                    name = e.get("rel_path") or e.get("name") or (uri.rsplit("/", 1)[-1] if uri else "")
                    is_dir = bool(e.get("isDir") or e.get("is_dir") or e.get("type") == "dir")
                    entries.append({
                        "name": name,
                        "uri": uri,
                        "type": "dir" if is_dir else "file",
                        "abstract": e.get("abstract", ""),
                    })
                return json.dumps({"path": path, "entries": entries}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)

    def _tool_remember(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return tool_error("content is required")

        category = args.get("category", "")
        subdir = _CATEGORY_SUBDIR_MAP.get(category, _DEFAULT_MEMORY_SUBDIR)
        uri = self._build_memory_uri(subdir)

        # Write directly via content/write API.
        # This creates the file, stores the content, and queues vector indexing
        # in a single call — no dependency on session commit / VLM extraction.
        try:
            result = self._client.post("/api/v1/content/write", {
                "uri": uri,
                "content": content,
                "mode": "create",
            })
            written = result.get("result", {}).get("written_bytes", 0)
            return json.dumps({
                "status": "stored",
                "message": f"Memory stored ({written}b) and queued for vector indexing.",
            })
        except Exception as e:
            logger.error("OpenViking content/write failed: %s", e)
            return tool_error(f"Failed to store memory: {e}")

    def _tool_add_resource(self, args: dict) -> str:
        url = args.get("url", "")
        if not url:
            return tool_error("url is required")

        if args.get("to") and args.get("parent"):
            return tool_error("Cannot specify both 'to' and 'parent'")

        payload: Dict[str, Any] = {}
        for key in ("reason", "to", "parent", "instruction", "wait", "timeout"):
            if key in args and args[key] not in {None, ""}:
                payload[key] = args[key]

        parsed_url = urlparse(url)
        if _is_remote_resource_source(url):
            source_path = None
        elif parsed_url.scheme == "file":
            source_path = _path_from_file_uri(url)
            if isinstance(source_path, str):
                return tool_error(source_path)
        elif parsed_url.scheme and not _is_windows_absolute_path(url):
            source_path = None
        else:
            source_path = Path(url).expanduser()

        cleanup_path: Optional[Path] = None
        try:
            if source_path is not None:
                if source_path.exists():
                    if source_path.is_dir():
                        payload["source_name"] = source_path.name
                        cleanup_path = _zip_directory(source_path)
                        upload_path = cleanup_path
                    elif source_path.is_file():
                        payload["source_name"] = source_path.name
                        upload_path = source_path
                    else:
                        return tool_error(f"Unsupported local resource path: {url}")
                    payload["temp_file_id"] = self._client.upload_temp_file(upload_path)
                elif _is_local_path_reference(url):
                    return tool_error(f"Local resource path does not exist: {url}")
                else:
                    payload["path"] = url
            else:
                payload["path"] = url

            resp = self._client.post("/api/v1/resources", payload)
            result = resp.get("result", {})
        finally:
            if cleanup_path:
                cleanup_path.unlink(missing_ok=True)

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
