"""RetainDB memory plugin — MemoryProvider interface.

Cross-session memory via the RetainDB cloud API: durable SQLite write-behind queue, semantic
search + profile, context overlay, dialectic/agent self-model prefetch, shared file store tools.
Config: RETAINDB_API_KEY (required, scoped secret), RETAINDB_BASE_URL (default https://api.retaindb.com),
RETAINDB_PROJECT (optional; defaults to "default"); the non-secret two also read config.yaml ``memory.retaindb``.
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
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret
from agent.file_safety import raise_if_read_blocked
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.retaindb.com"
_ASYNC_SHUTDOWN = object()
_TEXT_EXTS = (".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".xml", ".html")


def _load_retaindb_config() -> dict[str, Any]:
    """``memory.retaindb`` block from config.yaml (empty on error): Dashboard-persisted base_url/project; api_key stays in scoped secrets."""
    try:
        from hermes_cli.config import load_config_readonly
        block = load_config_readonly().get("memory", {}).get("retaindb", {})
    except Exception:
        block = None
    return dict(block) if isinstance(block, dict) else {}


def _q(s: str) -> str:
    return quote(s, safe="")


def _quiet(label: str, fn: Callable[[], Any]) -> Any:
    """Run *fn*; on any exception log "RetainDB <label> failed" at debug and return None."""
    try:
        return fn()
    except Exception as exc:
        logger.debug("RetainDB %s failed: %s", label, exc)
        return None


def _schema(name: str, description: str, properties: dict | None = None, required: tuple = ()) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": list(required)}}


def _p(description: str, type_: str = "string", **extra) -> dict:
    return {"type": type_, **extra, "description": description}


_SCHEMAS = (
    _schema("retaindb_profile", "Get the user's stable profile — preferences, facts, and patterns recalled from long-term memory."),
    _schema("retaindb_search", "Semantic search across stored memories. Returns ranked results with relevance scores.",
            {"query": _p("What to search for."), "top_k": _p("Max results (default: 8, max: 20).", "integer")}, ("query",)),
    _schema("retaindb_context", "Synthesized context block — what matters most for the current task, pulled from long-term memory.",
            {"query": _p("Current task or question.")}, ("query",)),
    _schema("retaindb_remember", "Persist an explicit fact, preference, or decision to long-term memory.",
            {"content": _p("The fact to remember."),
             "memory_type": _p("Category (default: factual).", enum=["factual", "preference", "goal", "instruction", "event", "opinion"]),
             "importance": _p("Importance 0-1 (default: 0.7).", "number")}, ("content",)),
    _schema("retaindb_forget", "Delete a specific memory by ID.", {"memory_id": _p("Memory ID to delete.")}, ("memory_id",)),
    _schema("retaindb_upload_file", "Upload a file to the shared RetainDB file store. Returns an rdb:// URI any agent can reference.",
            {"local_path": _p("Local file path to upload."), "remote_path": _p("Destination path, e.g. /reports/q1.pdf"),
             "scope": _p("Access scope (default: PROJECT).", enum=["USER", "PROJECT", "ORG"]),
             "ingest": _p("Also extract memories from file after upload (default: false).", "boolean")}, ("local_path",)),
    _schema("retaindb_list_files", "List files in the shared file store.",
            {"prefix": _p("Path prefix to filter by, e.g. /reports/"), "limit": _p("Max results (default: 50).", "integer")}),
    _schema("retaindb_read_file", "Read the text content of a stored file by its file ID.",
            {"file_id": _p("File ID returned from upload or list.")}, ("file_id",)),
    _schema("retaindb_ingest_file", "Chunk, embed, and extract memories from a stored file. Makes its contents searchable.",
            {"file_id": _p("File ID to ingest.")}, ("file_id",)),
    _schema("retaindb_delete_file", "Delete a stored file.", {"file_id": _p("File ID to delete.")}, ("file_id",)),
)


class _Client:
    """Thin HTTP client over the RetainDB REST API (lazy ``requests`` import)."""

    def __init__(self, api_key: str, base_url: str, project: str):
        self.api_key, self.base_url, self.project = api_key, re.sub(r"/+$", "", base_url), project

    def _headers(self, path: str, json_body: bool = True) -> dict:
        token = self.api_key.replace("Bearer ", "").strip()
        return {"Authorization": f"Bearer {token}", "x-sdk-runtime": "hermes-plugin",
                **({"Content-Type": "application/json"} if json_body else {}),
                **({"X-API-Key": token} if path.startswith(("/v1/memory", "/v1/context")) else {})}  # memory/context also accept X-API-Key

    def _http(self, method: str, path: str, *, json_body: bool = True, timeout: float = 30, **kwargs):
        import requests
        return requests.request(method, f"{self.base_url}{path}", headers=self._headers(path, json_body), timeout=timeout, **kwargs)

    def request(self, method: str, path: str, *, params=None, json_body=None, timeout: float = 8.0) -> Any:
        """JSON request; raises RuntimeError carrying the server message on a non-2xx response."""
        method = method.upper()
        resp = self._http(method, path, params=params, json=json_body if method not in {"GET", "DELETE"} else None, timeout=timeout)
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        if not resp.ok:
            msg = str(payload.get("message") or payload.get("error") or "") if isinstance(payload, dict) else ""
            raise RuntimeError(f"RetainDB {method} {path} failed ({resp.status_code}): {msg or payload}")
        return payload

    def _raw(self, method: str, path: str, **kwargs) -> Any:
        """Non-JSON request (multipart upload / binary download); raises on HTTP error."""
        resp = self._http(method, path, json_body=False, **kwargs)
        return resp.raise_for_status() or resp

    @staticmethod
    def _with_fallback(primary: Callable[[], dict], fallback: Callable[[], dict]) -> dict:
        """Try the current API route; on any error retry via the legacy route."""
        try:
            return primary()
        except Exception:
            return fallback()

    def _scoped(self, user_id: str, session_id: str, **extra) -> dict:
        return {"project": self.project, "user_id": user_id, "session_id": session_id, **extra}

    # Memory routes (one endpoint per method; bodies are the wire payloads)
    def query_context(self, user_id: str, session_id: str, query: str, max_tokens: int = 1200) -> dict:
        return self.request("POST", "/v1/context/query", json_body=self._scoped(user_id, session_id, query=query, include_memories=True, max_tokens=max_tokens))
    def search(self, user_id: str, session_id: str, query: str, top_k: int = 8) -> dict:
        return self.request("POST", "/v1/memory/search", json_body=self._scoped(user_id, session_id, query=query, top_k=top_k, include_pending=True))
    def get_profile(self, user_id: str) -> dict:
        return self._with_fallback(
            lambda: self.request("GET", f"/v1/memory/profile/{_q(user_id)}", params={"project": self.project, "include_pending": "true"}),
            lambda: self.request("GET", "/v1/memories", params={"project": self.project, "user_id": user_id, "limit": "200"}))
    def add_memory(self, user_id: str, session_id: str, content: str, memory_type: str = "factual", importance: float = 0.7) -> dict:
        body = self._scoped(user_id, session_id, content=content, memory_type=memory_type, importance=importance)
        return self._with_fallback(
            lambda: self.request("POST", "/v1/memory", json_body={**body, "write_mode": "sync"}, timeout=5.0),
            lambda: self.request("POST", "/v1/memories", json_body=body, timeout=5.0))
    def delete_memory(self, memory_id: str) -> dict:
        return self._with_fallback(
            lambda: self.request("DELETE", f"/v1/memory/{_q(memory_id)}", timeout=5.0),
            lambda: self.request("DELETE", f"/v1/memories/{_q(memory_id)}", timeout=5.0))
    def ingest_session(self, user_id: str, session_id: str, messages: list, timeout: float = 15.0) -> dict:
        return self.request("POST", "/v1/memory/ingest/session", json_body=self._scoped(user_id, session_id, messages=messages, write_mode="sync"), timeout=timeout)
    def ask_user(self, user_id: str, query: str, reasoning_level: str = "low") -> dict:
        return self.request("POST", f"/v1/memory/profile/{_q(user_id)}/ask", json_body={"project": self.project, "query": query, "reasoning_level": reasoning_level}, timeout=8.0)
    def get_agent_model(self, agent_id: str) -> dict:
        return self.request("GET", f"/v1/memory/agent/{_q(agent_id)}/model", params={"project": self.project}, timeout=4.0)
    def seed_agent_identity(self, agent_id: str, content: str, source: str = "soul_md") -> dict:
        return self.request("POST", f"/v1/memory/agent/{_q(agent_id)}/seed", json_body={"project": self.project, "content": content, "source": source}, timeout=20.0)

    # File routes
    def upload_file(self, data: bytes, filename: str, remote_path: str, mime_type: str, scope: str, project_id: str | None) -> dict:
        import io
        fields = {"path": remote_path, "scope": scope.upper(), **({"project_id": project_id} if project_id else {})}
        return self._raw("POST", "/v1/files", files={"file": (filename, io.BytesIO(data), mime_type)}, data=fields).json()
    def list_files(self, prefix: str | None = None, limit: int = 50) -> dict:
        return self.request("GET", "/v1/files", params={"limit": limit, **({"prefix": prefix} if prefix else {})})
    def get_file(self, file_id: str) -> dict:
        return self.request("GET", f"/v1/files/{_q(file_id)}")
    def read_file_content(self, file_id: str) -> bytes:
        return self._raw("GET", f"/v1/files/{_q(file_id)}/content", allow_redirects=True).content
    def ingest_file(self, file_id: str, user_id: str | None = None, agent_id: str | None = None) -> dict:
        body = {k: v for k, v in (("user_id", user_id), ("agent_id", agent_id)) if v}
        return self.request("POST", f"/v1/files/{_q(file_id)}/ingest", json_body=body, timeout=60.0)
    def delete_file(self, file_id: str) -> dict:
        return self.request("DELETE", f"/v1/files/{_q(file_id)}", timeout=5.0)


class _WriteQueue:
    """SQLite-backed async write queue. Survives crashes — pending rows replay on startup."""

    def __init__(self, client: _Client, db_path: Path):
        self._client, self._db_path, self._q = client, db_path, queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="retaindb-writer", daemon=True)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()  # one cached connection per thread, all tracked in _connections
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock, self._shutdown_lock, self._shutdown = threading.Lock(), threading.Lock(), False
        conn = self._execute("CREATE TABLE IF NOT EXISTS pending (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, "
                             "session_id TEXT, messages_json TEXT, created_at TEXT, last_error TEXT)").connection
        self._thread.start()
        replay = conn.execute("SELECT id, user_id, session_id, messages_json FROM pending ORDER BY id ASC LIMIT 200").fetchall()
        for row_id, user_id, session_id, msgs_json in replay:  # rows left from a previous crash
            self._q.put((row_id, user_id, session_id, json.loads(msgs_json)))

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = sqlite3.connect(str(self._db_path), timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            with self._connections_lock:
                self._connections.add(conn)
        return conn

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        cur = self._get_conn().execute(sql, params)
        cur.connection.commit()
        return cur

    def _close(self, *conns: sqlite3.Connection) -> None:
        for conn in conns:
            with self._connections_lock:
                self._connections.discard(conn)
            with suppress(Exception):
                conn.close()

    def _close_thread_conn(self) -> None:
        conn, self._local.conn = getattr(self._local, "conn", None), None
        self._close(*([conn] if conn is not None else []))

    def enqueue(self, user_id: str, session_id: str, messages: list) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._shutdown_lock:
            if self._shutdown:
                return
            cur = self._execute("INSERT INTO pending (user_id, session_id, messages_json, created_at) VALUES (?,?,?,?)",
                                (user_id, session_id, json.dumps(messages, ensure_ascii=False), now))
            self._q.put((cur.lastrowid, user_id, session_id, messages))

    def _flush_row(self, row_id: int, user_id: str, session_id: str, messages: list) -> None:
        try:
            self._client.ingest_session(user_id, session_id, messages)
            self._execute("DELETE FROM pending WHERE id = ?", (row_id,))
        except Exception as exc:
            logger.warning("RetainDB ingest failed (will retry): %s", exc)
            self._execute("UPDATE pending SET last_error = ? WHERE id = ?", (str(exc), row_id))
            time.sleep(2)

    def _loop(self) -> None:
        try:
            while (item := self._q.get()) is not _ASYNC_SHUTDOWN:
                try:
                    self._flush_row(*item)
                except Exception as exc:
                    logger.error("RetainDB writer error: %s", exc)
        finally:
            self._close_thread_conn()  # sqlite3 connections must close on their owning thread

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._q.put(_ASYNC_SHUTDOWN)
        self._close_thread_conn()  # caller thread owns the connection opened in __init__
        self._thread.join(timeout=10)
        if not self._thread.is_alive():  # exited executor workers may have left tracked handles (check_same_thread=False)
            with self._connections_lock:
                stragglers = list(self._connections)
            self._close(*stragglers)


def _compact(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()[:320]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", _compact(s).lower())


def _build_overlay(profile: dict, query_result: dict, local_entries: list[str] | None = None) -> str:
    """Profile + query memories (5 each, compacted, deduped against each other and *local_entries*)."""
    seen = {_norm(e) for e in (local_entries or []) if _norm(e)}

    def _dedupe(items) -> list[str]:
        out: list[str] = []
        for m in list(items or [])[:5]:
            c = _compact((m or {}).get("content") or "")
            if c and _norm(c) not in seen:
                seen.add(_norm(c))
                out.append(c)
        return out

    profile_items, query_items = _dedupe((profile or {}).get("memories")), _dedupe((query_result or {}).get("results"))
    if not profile_items and not query_items:
        return ""
    return "\n".join(["[RetainDB Context]", "Profile:"] + ([f"- {i}" for i in profile_items] or ["- None"])
                     + ["Relevant memories:"] + ([f"- {i}" for i in query_items] or ["- None"]))


# Agent self-model keys -> prefetch line formatter, in display order.
_AGENT_MODEL_FIELDS = (
    ("persona", lambda v: f"Persona: {v}"),
    ("persistent_instructions", lambda v: "Instructions:\n" + "\n".join(f"- {i}" for i in v)),
    ("working_style", lambda v: f"Working style: {v}"),
)


class RetainDBMemoryProvider(MemoryProvider):
    """RetainDB cloud memory — durable queue, semantic search, dialectic synthesis, shared files."""

    def __init__(self):
        self._client: _Client | None = None
        self._queue: _WriteQueue | None = None
        self._user_id, self._session_id, self._agent_id = "default", "", "hermes"
        self._lock = threading.Lock()  # guards the prefetch caches below
        self._context_result, self._dialectic_result, self._agent_model = "", "", {}
        self._prefetch_threads: list[threading.Thread] = []  # tracked so rapid turns don't pile up threads

    @property
    def name(self) -> str:
        return "retaindb"

    def is_available(self) -> bool:
        return bool(get_secret("RETAINDB_API_KEY"))

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {"key": "api_key", "description": "RetainDB API key", "secret": True, "required": True, "env_var": "RETAINDB_API_KEY", "url": "https://retaindb.com"},
            {"key": "base_url", "description": "API endpoint", "default": _DEFAULT_BASE_URL},
            {"key": "project", "description": "Project identifier (optional — uses 'default' project if not set)", "default": ""},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        # Non-secret fields resolve env -> config.yaml (written by the Dashboard) -> default.
        cfg = {k: v.strip() for k, v in _load_retaindb_config().items() if isinstance(v, str)}
        base_url = re.sub(r"/+$", "", os.environ.get("RETAINDB_BASE_URL") or cfg.get("base_url") or _DEFAULT_BASE_URL)
        # Project: RETAINDB_PROJECT > config.yaml > hermes-<profile> > "default" (API auto-creates "default").
        project = os.environ.get("RETAINDB_PROJECT") or cfg.get("project")
        if not project:
            profile_name = os.path.basename(str(kwargs.get("hermes_home", "")))
            project = f"hermes-{profile_name}" if profile_name not in {"", ".hermes"} else "default"
        self._client = _Client(get_secret("RETAINDB_API_KEY", "") or "", base_url, project)
        self._session_id, self._user_id = session_id, kwargs.get("user_id", "default") or "default"
        self._agent_id = kwargs.get("agent_id", "hermes") or "hermes"
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
        self._queue = _WriteQueue(self._client, home / "retaindb_queue.db")
        soul = (home / "SOUL.md").read_text(encoding="utf-8", errors="replace").strip() if (home / "SOUL.md").exists() else ""
        if soul:  # seed agent identity from SOUL.md in background
            seed = lambda: self._client.seed_agent_identity(self._agent_id, soul, source="soul_md")  # noqa: E731
            threading.Thread(target=_quiet, args=("soul seed", seed), name="retaindb-soul-seed", daemon=True).start()

    def system_prompt_block(self) -> str:
        project = self._client.project if self._client else "retaindb"
        return (f"# RetainDB Memory\nActive. Project: {project}.\nUse retaindb_search to find memories, retaindb_remember to store facts, "
                "retaindb_profile for a user overview, retaindb_context for current-task context.")

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire context + dialectic + agent model prefetches in background (turn-end); prefetch() consumes them next turn."""
        if not self._client:
            return
        for t in self._prefetch_threads:  # wait for the previous batch so threads don't accumulate on rapid turns
            t.join(timeout=2.0)
        if any(t.is_alive() for t in self._prefetch_threads):
            logger.debug("RetainDB prefetch still running; skipping new batch")
            return
        jobs = (  # (thread name, log label, cache attr, fetch) — fetch returns None to leave the cache untouched
            ("retaindb-ctx", "context", "_context_result", lambda: self._context_overlay(query)["context"]),
            ("retaindb-dialectic", "dialectic", "_dialectic_result", lambda: str(
                self._client.ask_user(self._user_id, query, reasoning_level=self._reasoning_level(query)).get("answer") or "") or None),
            ("retaindb-agent-model", "agent model", "_agent_model", lambda: self._agent_model_or_none(self._client.get_agent_model(self._agent_id))),
        )
        self._prefetch_threads = [threading.Thread(target=self._store, args=(label, attr, fetch), name=name, daemon=True)
                                  for name, label, attr, fetch in jobs]
        for t in self._prefetch_threads:
            t.start()

    def _context_overlay(self, query: str) -> dict:
        query_result = self._client.query_context(self._user_id, self._session_id, query)
        return {"context": _build_overlay(self._client.get_profile(self._user_id), query_result), "raw": query_result}

    @staticmethod
    def _agent_model_or_none(model: dict) -> dict | None:
        return model if model.get("memory_count", 0) > 0 else None

    def _store(self, label: str, attr: str, fetch: Callable[[], Any]) -> None:
        """Run one prefetch job; cache its value under the lock unless None (failures log at debug)."""
        value = _quiet(f"{label} prefetch", fetch)
        if value is not None:
            with self._lock:
                setattr(self, attr, value)

    @staticmethod
    def _reasoning_level(query: str) -> str:
        return "low" if len(query) < 120 else "medium" if len(query) < 400 else "high"

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Consume prefetched results and return them as a context block."""
        with self._lock:
            context, dialectic, agent_model = self._context_result, self._dialectic_result, self._agent_model
            self._context_result, self._dialectic_result, self._agent_model = "", "", {}
        model_lines = [fmt(agent_model[k]) for k, fmt in _AGENT_MODEL_FIELDS if agent_model.get(k)] if agent_model.get("memory_count", 0) > 0 else []
        parts = [context, dialectic and f"[RetainDB User Synthesis]\n{dialectic}",
                 model_lines and "[RetainDB Agent Self-Model]\n" + "\n".join(model_lines)]
        return "\n\n".join(p for p in parts if p)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Queue turn for async ingest. Returns immediately."""
        if not self._queue or not user_content:
            return
        now = datetime.now(timezone.utc).isoformat()
        self._queue.enqueue(self._user_id, session_id or self._session_id,
                            [{"role": "user", "content": user_content, "timestamp": now},
                             {"role": "assistant", "content": assistant_content, "timestamp": now}])

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return list(_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if not self._client:
            return tool_error("RetainDB not initialized")
        try:
            return json.dumps(self._dispatch(tool_name, args))
        except Exception as exc:
            return tool_error(str(exc))

    def _dispatch(self, tool_name: str, args: dict) -> Any:
        required, handler = _TOOLS.get(tool_name, (None, None))
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        value = args.get(required, "") if required else None
        return {"error": f"{required} is required"} if required and not value else handler(self, args, value)

    def _tool_upload_file(self, args: dict, local_path: str) -> Any:
        path_obj = Path(local_path)
        if not path_obj.exists():
            return {"error": f"File not found: {local_path}"}
        try:
            raise_if_read_blocked(str(path_obj))
        except ValueError as exc:
            return {"error": str(exc)}
        import mimetypes
        result = self._client.upload_file(path_obj.read_bytes(), path_obj.name, args.get("remote_path") or f"/{path_obj.name}",
                                          mimetypes.guess_type(path_obj.name)[0] or "application/octet-stream", args.get("scope", "PROJECT"), None)
        if args.get("ingest") and result.get("file", {}).get("id"):
            result["ingest"] = self._ingest(result["file"]["id"])
        return result

    def _tool_read_file(self, args: dict, file_id: str) -> Any:
        file_info = self._client.get_file(file_id).get("file") or {}
        raw = self._client.read_file_content(file_id)
        out = {"file_id": file_id, "rdb_uri": file_info.get("rdb_uri"), "name": file_info.get("name")}
        if not ((file_info.get("mime_type") or "").lower().startswith("text/") or file_info.get("name", "").endswith(_TEXT_EXTS)):
            return {**out, "content": None, "note": "Binary file — use retaindb_ingest_file to extract text into memory."}
        text = raw.decode("utf-8", errors="replace")
        return {**out, "content": text[:32000], "truncated": len(text) > 32000}

    def _ingest(self, file_id: str) -> Any:
        return self._client.ingest_file(file_id, user_id=self._user_id, agent_id=self._agent_id)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to RetainDB."""
        if action != "add" or not content or not self._client:
            return
        _quiet("memory mirror", lambda: self._client.add_memory(
            self._user_id, self._session_id, content, memory_type="preference" if target == "user" else "factual"))

    def shutdown(self) -> None:
        for t in self._prefetch_threads:
            t.join(timeout=3.0)
        queue_obj, self._prefetch_threads, self._queue, self._client = self._queue, [], None, None
        if queue_obj:
            queue_obj.shutdown()


# tool name -> (required arg or None, handler(provider, args, required_value)); missing arg -> "<arg> is required"
_TOOLS: dict[str, tuple[str | None, Callable[..., Any]]] = {
    "retaindb_profile": (None, lambda p, a, _: p._client.get_profile(p._user_id)),
    "retaindb_search": ("query", lambda p, a, q: p._client.search(p._user_id, p._session_id, q, top_k=min(int(a.get("top_k", 8)), 20))),
    "retaindb_context": ("query", lambda p, a, q: p._context_overlay(q)),
    "retaindb_remember": ("content", lambda p, a, c: p._client.add_memory(
        p._user_id, p._session_id, c, memory_type=a.get("memory_type", "factual"), importance=float(a.get("importance", 0.7)))),
    "retaindb_forget": ("memory_id", lambda p, a, m: p._client.delete_memory(m)),
    "retaindb_upload_file": ("local_path", RetainDBMemoryProvider._tool_upload_file),
    "retaindb_list_files": (None, lambda p, a, _: p._client.list_files(prefix=a.get("prefix"), limit=int(a.get("limit", 50)))),
    "retaindb_read_file": ("file_id", RetainDBMemoryProvider._tool_read_file),
    "retaindb_ingest_file": ("file_id", lambda p, a, f: p._ingest(f)),
    "retaindb_delete_file": ("file_id", lambda p, a, f: p._client.delete_file(f)),
}


def register(ctx) -> None:
    """Register RetainDB as a memory provider plugin."""
    ctx.register_memory_provider(RetainDBMemoryProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Dict  # noqa: F401,E402
from typing import List  # noqa: F401,E402

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

PROFILE_SCHEMA = {
    "name": "retaindb_profile",
    "description": "Get the user's stable profile — preferences, facts, and patterns recalled from long-term memory.",
    "parameters": {"type": "object", "properties": {}, "required": []},
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
# ---- END PLUGIN-COMPAT ----
