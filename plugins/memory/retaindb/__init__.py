"""RetainDB memory plugin — MemoryProvider interface.

Cross-session memory via RetainDB cloud API.

Features:
- Correct API routes for all operations
- Durable SQLite write-behind queue (crash-safe, async ingest)
- Semantic search + user profile retrieval
- Context query with deduplication overlay
- Dialectic synthesis (LLM-powered user understanding, prefetched each turn)
- Agent self-model (persona + instructions from SOUL.md, prefetched each turn)
- Shared file store tools (upload, list, read, ingest, delete)
- Explicit memory tools (profile, search, context, remember, forget)

Config (env vars or hermes config.yaml under retaindb:):
  RETAINDB_API_KEY     — API key (required)
  RETAINDB_BASE_URL    — API endpoint (default: https://api.retaindb.com)
  RETAINDB_PROJECT     — Project identifier (optional — defaults to "default")
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.retaindb.com"
_ASYNC_SHUTDOWN = object()


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "retaindb_profile",
    "description": "Get the user's stable profile — preferences, facts, and patterns recalled from long-term memory.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "retaindb_search",
    "description": "Semantic search across stored memories. Returns ranked results with relevance scores.",
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
    "description": "Synthesized context block — what matters most for the current task, pulled from long-term memory.",
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
    "description": "Persist an explicit fact, preference, or decision to long-term memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to remember."},
            "memory_type": {
                "type": "string",
                "enum": ["factual", "preference", "goal", "instruction", "event", "opinion"],
                "description": "Category (default: factual).",
            },
            "importance": {"type": "number", "description": "Importance 0-1 (default: 0.7)."},
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

FILE_UPLOAD_SCHEMA = {
    "name": "retaindb_upload_file",
    "description": "Upload a file to the shared RetainDB file store. Returns an rdb:// URI any agent can reference.",
    "parameters": {
        "type": "object",
        "properties": {
            "local_path": {"type": "string", "description": "Local file path to upload."},
            "remote_path": {"type": "string", "description": "Destination path, e.g. /reports/q1.pdf"},
            "scope": {"type": "string", "enum": ["USER", "PROJECT", "ORG"], "description": "Access scope (default: PROJECT)."},
            "ingest": {"type": "boolean", "description": "Also extract memories from file after upload (default: false)."},
        },
        "required": ["local_path"],
    },
}

FILE_LIST_SCHEMA = {
    "name": "retaindb_list_files",
    "description": "List files in the shared file store.",
    "parameters": {
        "type": "object",
        "properties": {
            "prefix": {"type": "string", "description": "Path prefix to filter by, e.g. /reports/"},
            "limit": {"type": "integer", "description": "Max results (default: 50)."},
        },
        "required": [],
    },
}

FILE_READ_SCHEMA = {
    "name": "retaindb_read_file",
    "description": "Read the text content of a stored file by its file ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_id": {"type": "string", "description": "File ID returned from upload or list."},
        },
        "required": ["file_id"],
    },
}

FILE_INGEST_SCHEMA = {
    "name": "retaindb_ingest_file",
    "description": "Chunk, embed, and extract memories from a stored file. Makes its contents searchable.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_id": {"type": "string", "description": "File ID to ingest."},
        },
        "required": ["file_id"],
    },
}

FILE_DELETE_SCHEMA = {
    "name": "retaindb_delete_file",
    "description": "Delete a stored file.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_id": {"type": "string", "description": "File ID to delete."},
        },
        "required": ["file_id"],
    },
}


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class _Client:
    def __init__(self, api_key: str, base_url: str, project: str):
        self.api_key = api_key
        self.base_url = re.sub(r"/+$", "", base_url)
        self.project = project

    def _headers(self, path: str) -> dict:
        token = self.api_key.replace("Bearer ", "").strip()
        h = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-sdk-runtime": "hermes-plugin",
        }
        if path.startswith(("/v1/memory", "/v1/context")):
            h["X-API-Key"] = token
        return h

    def request(self, method: str, path: str, *, params=None, json_body=None, timeout: float = 8.0) -> Any:
        import requests
        url = f"{self.base_url}{path}"
        resp = requests.request(
            method.upper(), url,
            params=params,
            json=json_body if method.upper() not in {"GET", "DELETE"} else None,
            headers=self._headers(path),
            timeout=timeout,
        )
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        if not resp.ok:
            msg = ""
            if isinstance(payload, dict):
                msg = str(payload.get("message") or payload.get("error") or "")
            raise RuntimeError(f"RetainDB {method} {path} failed ({resp.status_code}): {msg or payload}")
        return payload

    # ── Memory ────────────────────────────────────────────────────────────────

    def query_context(self, user_id: str, session_id: str, query: str, max_tokens: int = 1200) -> dict:
        return self.request("POST", "/v1/context/query", json_body={
            "project": self.project,
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "include_memories": True,
            "max_tokens": max_tokens,
        })

    def search(self, user_id: str, session_id: str, query: str, top_k: int = 8) -> dict:
        return self.request("POST", "/v1/memory/search", json_body={
            "project": self.project,
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "top_k": top_k,
            "include_pending": True,
        })

    def get_profile(self, user_id: str) -> dict:
        try:
            return self.request("GET", f"/v1/memory/profile/{quote(user_id, safe='')}", params={"project": self.project, "include_pending": "true"})
        except Exception:
            return self.request("GET", "/v1/memories", params={"project": self.project, "user_id": user_id, "limit": "200"})

    def add_memory(self, user_id: str, session_id: str, content: str, memory_type: str = "factual", importance: float = 0.7) -> dict:
        try:
            return self.request("POST", "/v1/memory", json_body={
                "project": self.project, "content": content, "memory_type": memory_type,
                "user_id": user_id, "session_id": session_id, "importance": importance, "write_mode": "sync",
            }, timeout=5.0)
        except Exception:
            return self.request("POST", "/v1/memories", json_body={
                "project": self.project, "content": content, "memory_type": memory_type,
                "user_id": user_id, "session_id": session_id, "importance": importance,
            }, timeout=5.0)

    def delete_memory(self, memory_id: str) -> dict:
        try:
            return self.request("DELETE", f"/v1/memory/{quote(memory_id, safe='')}", timeout=5.0)
        except Exception:
            return self.request("DELETE", f"/v1/memories/{quote(memory_id, safe='')}", timeout=5.0)

    def ingest_session(self, user_id: str, session_id: str, messages: list, timeout: float = 15.0) -> dict:
        return self.request("POST", "/v1/memory/ingest/session", json_body={
            "project": self.project, "session_id": session_id, "user_id": user_id,
            "messages": messages, "write_mode": "sync",
        }, timeout=timeout)

    def ask_user(self, user_id: str, query: str, reasoning_level: str = "low") -> dict:
        return self.request("POST", f"/v1/memory/profile/{quote(user_id, safe='')}/ask", json_body={
            "project": self.project, "query": query, "reasoning_level": reasoning_level,
        }, timeout=8.0)

    def get_agent_model(self, agent_id: str) -> dict:
        return self.request("GET", f"/v1/memory/agent/{quote(agent_id, safe='')}/model", params={"project": self.project}, timeout=4.0)

    def seed_agent_identity(self, agent_id: str, content: str, source: str = "soul_md") -> dict:
        return self.request("POST", f"/v1/memory/agent/{quote(agent_id, safe='')}/seed", json_body={
            "project": self.project, "content": content, "source": source,
        }, timeout=20.0)

    # ── Files ─────────────────────────────────────────────────────────────────

    def upload_file(self, data: bytes, filename: str, remote_path: str, mime_type: str, scope: str, project_id: str | None) -> dict:
        import io
        import requests
        url = f"{self.base_url}/v1/files"
        token = self.api_key.replace("Bearer ", "").strip()
        headers = {"Authorization": f"Bearer {token}", "x-sdk-runtime": "hermes-plugin"}
        fields = {"path": remote_path, "scope": scope.upper()}
        if project_id:
            fields["project_id"] = project_id
        resp = requests.post(url, files={"file": (filename, io.BytesIO(data), mime_type)}, data=fields, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def list_files(self, prefix: str | None = None, limit: int = 50) -> dict:
        params: dict = {"limit": limit}
        if prefix:
            params["prefix"] = prefix
        return self.request("GET", "/v1/files", params=params)

    def get_file(self, file_id: str) -> dict:
        return self.request("GET", f"/v1/files/{quote(file_id, safe='')}")

    def read_file_content(self, file_id: str) -> bytes:
        import requests
        token = self.api_key.replace("Bearer ", "").strip()
        url = f"{self.base_url}/v1/files/{quote(file_id, safe='')}/content"
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}", "x-sdk-runtime": "hermes-plugin"}, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        return resp.content

    def ingest_file(self, file_id: str, user_id: str | None = None, agent_id: str | None = None) -> dict:
        body: dict = {}
        if user_id:
            body["user_id"] = user_id
        if agent_id:
            body["agent_id"] = agent_id
        return self.request("POST", f"/v1/files/{quote(file_id, safe='')}/ingest", json_body=body, timeout=60.0)

    def delete_file(self, file_id: str) -> dict:
        return self.request("DELETE", f"/v1/files/{quote(file_id, safe='')}", timeout=5.0)


# ---------------------------------------------------------------------------
# Durable write-behind queue
# ---------------------------------------------------------------------------

class _WriteQueue:
    """SQLite-backed async write queue. Survives crashes — pending rows replay on startup."""

    def __init__(self, client: _Client, db_path: Path):
        self._client = client
        self._db_path = db_path
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="retaindb-writer", daemon=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Thread-local connection cache — one connection per thread, reused.
        self._local = threading.local()
        self._init_db()
        self._thread.start()
        # Replay any rows left from a previous crash
        for row_id, user_id, session_id, msgs_json in self._pending_rows():
            self._q.put((row_id, user_id, session_id, json.loads(msgs_json)))

    def _get_conn(self) -> sqlite3.Connection:
        """Return a cached connection for the current thread."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), timeout=30)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""CREATE TABLE IF NOT EXISTS pending (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT, session_id TEXT, messages_json TEXT,
            created_at TEXT, last_error TEXT
        )""")
        conn.commit()

    def _pending_rows(self) -> list:
        conn = self._get_conn()
        return conn.execute("SELECT id, user_id, session_id, messages_json FROM pending ORDER BY id ASC LIMIT 200").fetchall()

    def enqueue(self, user_id: str, session_id: str, messages: list) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO pending (user_id, session_id, messages_json, created_at) VALUES (?,?,?,?)",
            (user_id, session_id, json.dumps(messages, ensure_ascii=False), now),
        )
        row_id = cur.lastrowid
        conn.commit()
        self._q.put((row_id, user_id, session_id, messages))

    def _flush_row(self, row_id: int, user_id: str, session_id: str, messages: list) -> None:
        try:
            self._client.ingest_session(user_id, session_id, messages)
            conn = self._get_conn()
            conn.execute("DELETE FROM pending WHERE id = ?", (row_id,))
            conn.commit()
        except Exception as exc:
            logger.warning("RetainDB ingest failed (will retry): %s", exc)
            conn = self._get_conn()
            conn.execute("UPDATE pending SET last_error = ? WHERE id = ?", (str(exc), row_id))
            conn.commit()
            time.sleep(2)

    def _loop(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=5)
                if item is _ASYNC_SHUTDOWN:
                    break
                self._flush_row(*item)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("RetainDB writer error: %s", exc)

    def shutdown(self) -> None:
        self._q.put(_ASYNC_SHUTDOWN)
        self._thread.join(timeout=10)


# ---------------------------------------------------------------------------
# Overlay formatter
# ---------------------------------------------------------------------------

def _build_overlay(profile: dict, query_result: dict, local_entries: list[str] | None = None) -> str:
    def _compact(s: str) -> str:
        return re.sub(r"\s+", " ", str(s or "")).strip()[:320]

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", _compact(s).lower())

    seen: list[str] = [_norm(e) for e in (local_entries or []) if _norm(e)]
    profile_items: list[str] = []
    for m in list((profile or {}).get("memories") or [])[:5]:
        c = _compact((m or {}).get("content") or "")
        n = _norm(c)
        if c and n not in seen:
            seen.append(n)
            profile_items.append(c)

    query_items: list[str] = []
    for r in list((query_result or {}).get("results") or [])[:5]:
        c = _compact((r or {}).get("content") or "")
        n = _norm(c)
        if c and n not in seen:
            seen.append(n)
            query_items.append(c)

    if not profile_items and not query_items:
        return ""

    lines = ["[RetainDB Context]", "Profile:"]
    lines += [f"- {i}" for i in profile_items] or ["- None"]
    lines.append("Relevant memories:")
    lines += [f"- {i}" for i in query_items] or ["- None"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main plugin class
# ---------------------------------------------------------------------------

class RetainDBMemoryProvider(MemoryProvider):
    """RetainDB cloud memory — durable queue, semantic search, dialectic synthesis, shared files."""

    def __init__(self):
        self._client: _Client | None = None
        self._queue: _WriteQueue | None = None
        self._user_id = "default"
        self._session_id = ""
        self._agent_id = "hermes"
        self._lock = threading.Lock()

        # Prefetch caches
        self._context_result = ""
        self._dialectic_result = ""
        self._agent_model: dict = {}

        # Prefetch thread tracking — prevents accumulation on rapid calls
        self._prefetch_threads: list[threading.Thread] = []

    # ── Core identity ──────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "retaindb"

    def is_available(self) -> bool:
        return bool(os.environ.get("RETAINDB_API_KEY"))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "api_key", "description": "RetainDB API key", "secret": True, "required": True, "env_var": "RETAINDB_API_KEY", "url": "https://retaindb.com"},
            {"key": "base_url", "description": "API endpoint", "default": _DEFAULT_BASE_URL},
            {"key": "project", "description": "Project identifier (optional — uses 'default' project if not set)", "default": ""},
        ]

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def initialize(self, session_id: str, **kwargs) -> None:
        api_key = os.environ.get("RETAINDB_API_KEY", "")
        base_url = re.sub(r"/+$", "", os.environ.get("RETAINDB_BASE_URL", _DEFAULT_BASE_URL))

        # Project resolution: RETAINDB_PROJECT > hermes-<profile> > "default"
        # If unset, the API auto-creates and uses the "default" project — no config required.
        explicit = os.environ.get("RETAINDB_PROJECT")
        if explicit:
            project = explicit
        else:
            hermes_home = str(kwargs.get("hermes_home", ""))
            profile_name = os.path.basename(hermes_home) if hermes_home else ""
            project = f"hermes-{profile_name}" if (profile_name and profile_name not in {"", ".hermes"}) else "default"

        self._client = _Client(api_key, base_url, project)
        self._session_id = session_id
        self._user_id = kwargs.get("user_id", "default") or "default"
        self._agent_id = kwargs.get("agent_id", "hermes") or "hermes"

        from hermes_constants import get_hermes_home
        hermes_home_path = get_hermes_home()
        db_path = hermes_home_path / "retaindb_queue.db"
        self._queue = _WriteQueue(self._client, db_path)

        # Seed agent identity from SOUL.md in background
        soul_path = hermes_home_path / "SOUL.md"
        if soul_path.exists():
            soul_content = soul_path.read_text(encoding="utf-8", errors="replace").strip()
            if soul_content:
                threading.Thread(
                    target=self._seed_soul,
                    args=(soul_content,),
                    name="retaindb-soul-seed",
                    daemon=True,
                ).start()

    def _seed_soul(self, content: str) -> None:
        try:
            self._client.seed_agent_identity(self._agent_id, content, source="soul_md")
        except Exception as exc:
            logger.debug("RetainDB soul seed failed: %s", exc)

    def system_prompt_block(self) -> str:
        project = self._client.project if self._client else "retaindb"
        return (
            "# RetainDB Memory\n"
            f"Active. Project: {project}.\n"
            "Use retaindb_search to find memories, retaindb_remember to store facts, "
            "retaindb_profile for a user overview, retaindb_context for current-task context."
        )

    # ── Background prefetch (fires at turn-end, consumed next turn-start) ──

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire context + dialectic + agent model prefetches in background."""
        if not self._client:
            return
        # Wait for any still-running prefetch threads before spawning new ones.
        # Prevents thread accumulation if turns fire faster than prefetches complete.
        for t in self._prefetch_threads:
            t.join(timeout=2.0)
        threads = [
            threading.Thread(target=self._prefetch_context, args=(query,), name="retaindb-ctx", daemon=True),
            threading.Thread(target=self._prefetch_dialectic, args=(query,), name="retaindb-dialectic", daemon=True),
            threading.Thread(target=self._prefetch_agent_model, name="retaindb-agent-model", daemon=True),
        ]
        self._prefetch_threads = threads
        for t in threads:
            t.start()

    def _prefetch_context(self, query: str) -> None:
        try:
            query_result = self._client.query_context(self._user_id, self._session_id, query)
            profile = self._client.get_profile(self._user_id)
            overlay = _build_overlay(profile, query_result)
            with self._lock:
                self._context_result = overlay
        except Exception as exc:
            logger.debug("RetainDB context prefetch failed: %s", exc)

    def _prefetch_dialectic(self, query: str) -> None:
        try:
            result = self._client.ask_user(self._user_id, query, reasoning_level=self._reasoning_level(query))
            answer = str(result.get("answer") or "")
            if answer:
                with self._lock:
                    self._dialectic_result = answer
        except Exception as exc:
            logger.debug("RetainDB dialectic prefetch failed: %s", exc)

    def _prefetch_agent_model(self) -> None:
        try:
            model = self._client.get_agent_model(self._agent_id)
            if model.get("memory_count", 0) > 0:
                with self._lock:
                    self._agent_model = model
        except Exception as exc:
            logger.debug("RetainDB agent model prefetch failed: %s", exc)

    @staticmethod
    def _reasoning_level(query: str) -> str:
        n = len(query)
        if n < 120:
            return "low"
        if n < 400:
            return "medium"
        return "high"

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Consume prefetched results and return them as a context block."""
        with self._lock:
            context = self._context_result
            dialectic = self._dialectic_result
            agent_model = self._agent_model
            self._context_result = ""
            self._dialectic_result = ""
            self._agent_model = {}

        parts: list[str] = []
        if context:
            parts.append(context)
        if dialectic:
            parts.append(f"[RetainDB User Synthesis]\n{dialectic}")
        if agent_model and agent_model.get("memory_count", 0) > 0:
            model_lines: list[str] = []
            if agent_model.get("persona"):
                model_lines.append(f"Persona: {agent_model['persona']}")
            if agent_model.get("persistent_instructions"):
                model_lines.append("Instructions:\n" + "\n".join(f"- {i}" for i in agent_model["persistent_instructions"]))
            if agent_model.get("working_style"):
                model_lines.append(f"Working style: {agent_model['working_style']}")
            if model_lines:
                parts.append("[RetainDB Agent Self-Model]\n" + "\n".join(model_lines))

        return "\n\n".join(parts)

    # ── Turn sync ──────────────────────────────────────────────────────────

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Queue turn for async ingest. Returns immediately."""
        if not self._queue or not user_content:
            return
        now = datetime.now(timezone.utc).isoformat()
        self._queue.enqueue(
            self._user_id,
            session_id or self._session_id,
            [
                {"role": "user", "content": user_content, "timestamp": now},
                {"role": "assistant", "content": assistant_content, "timestamp": now},
            ],
        )

    # ── Tools ──────────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            PROFILE_SCHEMA, SEARCH_SCHEMA, CONTEXT_SCHEMA,
            REMEMBER_SCHEMA, FORGET_SCHEMA,
            FILE_UPLOAD_SCHEMA, FILE_LIST_SCHEMA, FILE_READ_SCHEMA,
            FILE_INGEST_SCHEMA, FILE_DELETE_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if not self._client:
            return json.dumps({"error": "RetainDB not initialized"})
        try:
            return json.dumps(self._dispatch(tool_name, args))
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _dispatch(self, tool_name: str, args: dict) -> Any:
        c = self._client

        if tool_name == "retaindb_profile":
            return c.get_profile(self._user_id)

        if tool_name == "retaindb_search":
            query = args.get("query", "")
            if not query:
                return {"error": "query is required"}
            return c.search(self._user_id, self._session_id, query, top_k=min(int(args.get("top_k", 8)), 20))

        if tool_name == "retaindb_context":
            query = args.get("query", "")
            if not query:
                return {"error": "query is required"}
            query_result = c.query_context(self._user_id, self._session_id, query)
            profile = c.get_profile(self._user_id)
            overlay = _build_overlay(profile, query_result)
            return {"context": overlay, "raw": query_result}

        if tool_name == "retaindb_remember":
            content = args.get("content", "")
            if not content:
                return {"error": "content is required"}
            return c.add_memory(
                self._user_id, self._session_id, content,
                memory_type=args.get("memory_type", "factual"),
                importance=float(args.get("importance", 0.7)),
            )

        if tool_name == "retaindb_forget":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return {"error": "memory_id is required"}
            return c.delete_memory(memory_id)

        # ── File tools ──────────────────────────────────────────────────────

        if tool_name == "retaindb_upload_file":
            local_path = args.get("local_path", "")
            if not local_path:
                return {"error": "local_path is required"}
            path_obj = Path(local_path)
            if not path_obj.exists():
                return {"error": f"File not found: {local_path}"}
            data = path_obj.read_bytes()
            import mimetypes
            mime = mimetypes.guess_type(path_obj.name)[0] or "application/octet-stream"
            remote_path = args.get("remote_path") or f"/{path_obj.name}"
            result = c.upload_file(data, path_obj.name, remote_path, mime, args.get("scope", "PROJECT"), None)
            if args.get("ingest") and result.get("file", {}).get("id"):
                ingest = c.ingest_file(result["file"]["id"], user_id=self._user_id, agent_id=self._agent_id)
                result["ingest"] = ingest
            return result

        if tool_name == "retaindb_list_files":
            return c.list_files(prefix=args.get("prefix"), limit=int(args.get("limit", 50)))

        if tool_name == "retaindb_read_file":
            file_id = args.get("file_id", "")
            if not file_id:
                return {"error": "file_id is required"}
            meta = c.get_file(file_id)
            file_info = meta.get("file") or {}
            mime = (file_info.get("mime_type") or "").lower()
            raw = c.read_file_content(file_id)
            if not (mime.startswith("text/") or any(file_info.get("name", "").endswith(e) for e in (".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".xml", ".html"))):
                return {"file_id": file_id, "rdb_uri": file_info.get("rdb_uri"), "name": file_info.get("name"), "content": None, "note": "Binary file — use retaindb_ingest_file to extract text into memory."}
            text = raw.decode("utf-8", errors="replace")
            return {"file_id": file_id, "rdb_uri": file_info.get("rdb_uri"), "name": file_info.get("name"), "content": text[:32000], "truncated": len(text) > 32000}

        if tool_name == "retaindb_ingest_file":
            file_id = args.get("file_id", "")
            if not file_id:
                return {"error": "file_id is required"}
            return c.ingest_file(file_id, user_id=self._user_id, agent_id=self._agent_id)

        if tool_name == "retaindb_delete_file":
            file_id = args.get("file_id", "")
            if not file_id:
                return {"error": "file_id is required"}
            return c.delete_file(file_id)

        return {"error": f"Unknown tool: {tool_name}"}

    # ── Optional hooks ─────────────────────────────────────────────────────

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to RetainDB."""
        if action != "add" or not content or not self._client:
            return
        try:
            memory_type = "preference" if target == "user" else "factual"
            self._client.add_memory(self._user_id, self._session_id, content, memory_type=memory_type)
        except Exception as exc:
            logger.debug("RetainDB memory mirror failed: %s", exc)

    def shutdown(self) -> None:
        for t in self._prefetch_threads:
            t.join(timeout=3.0)
        if self._queue:
            self._queue.shutdown()


def register(ctx) -> None:
    """Register RetainDB as a memory provider plugin."""
    ctx.register_memory_provider(RetainDBMemoryProvider())
