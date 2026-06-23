"""Turso memory provider for Hermes.

Local-first, profile-scoped semantic memory using SQLite-compatible storage.
The provider is intentionally additive: built-in MEMORY.md / USER.md remains
the compact always-on memory, while Turso stores deeper searchable recall.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 6
_DEFAULT_MIN_SIMILARITY = 0.15
_DEFAULT_DIM = 64
_MIN_CAPTURE_CHARS = 40


SEARCH_SCHEMA = {
    "name": "turso_memory_search",
    "description": "Search Turso long-term memory. Use for deeper recall beyond compact built-in memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "top_k": {"type": "integer", "description": "Maximum results, default 6, max 20."},
            "kind": {"type": "string", "description": "Optional memory kind filter."},
        },
        "required": ["query"],
    },
}

ADD_SCHEMA = {
    "name": "turso_memory_add",
    "description": "Store a durable fact, preference, decision, or note in Turso memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Memory content to store."},
            "kind": {
                "type": "string",
                "enum": ["fact", "preference", "decision", "instruction", "conversation", "note"],
                "description": "Memory category, default fact.",
            },
            "source": {"type": "string", "description": "Optional provenance label."},
            "weight": {"type": "number", "description": "Importance weight 0.1-3.0, default 1.0."},
        },
        "required": ["content"],
    },
}

UPDATE_SCHEMA = {
    "name": "turso_memory_update",
    "description": "Update a Turso memory by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID."},
            "content": {"type": "string", "description": "Replacement content."},
            "kind": {"type": "string", "description": "Optional replacement kind."},
            "weight": {"type": "number", "description": "Optional replacement weight."},
        },
        "required": ["memory_id", "content"],
    },
}

DELETE_SCHEMA = {
    "name": "turso_memory_delete",
    "description": "Delete a Turso memory by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID to delete."},
        },
        "required": ["memory_id"],
    },
}

SYNC_SCHEMA = {
    "name": "turso_memory_sync",
    "description": "Show or trigger best-effort Turso memory sync status.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["status", "sync"], "description": "Default status."},
        },
        "required": [],
    },
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _now() -> int:
    return int(time.time())


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _default_config(hermes_home: str) -> dict:
    return {
        "db_path": str(Path(hermes_home) / "turso-memory.db"),
        "sync_enabled": False,
        "top_k": _DEFAULT_TOP_K,
        "min_similarity": _DEFAULT_MIN_SIMILARITY,
        "auto_capture": False,
        "embedding_dim": _DEFAULT_DIM,
    }


def _load_config(hermes_home: str) -> dict:
    config = _default_config(hermes_home)
    path = Path(hermes_home) / "turso.json"
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                config.update({k: v for k, v in raw.items() if v is not None})
        except Exception:
            logger.debug("Failed to parse %s", path, exc_info=True)
    config["sync_enabled"] = _as_bool(config.get("sync_enabled"), False)
    config["auto_capture"] = _as_bool(config.get("auto_capture"), False)
    try:
        config["top_k"] = max(1, min(20, int(config.get("top_k", _DEFAULT_TOP_K))))
    except Exception:
        config["top_k"] = _DEFAULT_TOP_K
    try:
        config["min_similarity"] = max(0.0, min(1.0, float(config.get("min_similarity", _DEFAULT_MIN_SIMILARITY))))
    except Exception:
        config["min_similarity"] = _DEFAULT_MIN_SIMILARITY
    try:
        config["embedding_dim"] = max(16, min(512, int(config.get("embedding_dim", _DEFAULT_DIM))))
    except Exception:
        config["embedding_dim"] = _DEFAULT_DIM
    db_path = str(config.get("db_path") or "")
    config["db_path"] = db_path.replace("$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
    return config


def _save_config(values: dict, hermes_home: str) -> None:
    path = Path(hermes_home) / "turso.json"
    existing: dict = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
        except Exception:
            existing = {}
    cleaned = dict(values)
    cleaned.pop("auth_token", None)
    cleaned.pop("api_key", None)
    cleaned.pop("database_url", None)
    existing.update(cleaned)
    atomic_json_write(path, existing, mode=0o600, sort_keys=True)


class _LocalEmbedder:
    """Deterministic lexical embedding.

    This keeps local-only mode and tests useful without making LLM or network
    calls. It is not a replacement for model embeddings, but it gives stable
    semantic-ish retrieval based on normalized token features.
    """

    def __init__(self, dim: int = _DEFAULT_DIM):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-zA-Z0-9_./-]+", (text or "").lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


def _keyword_score(query: str, content: str) -> float:
    q = set(re.findall(r"[a-zA-Z0-9_./-]+", query.lower()))
    c = set(re.findall(r"[a-zA-Z0-9_./-]+", content.lower()))
    if not q or not c:
        return 0.0
    return len(q & c) / len(q)


class TursoMemoryProvider(MemoryProvider):
    """Local-first Turso memory provider."""

    def __init__(self, config: Optional[dict] = None):
        self._config_override = config
        self._config: dict = config or {}
        self._conn: sqlite3.Connection | None = None
        self._embedder = _LocalEmbedder()
        self._session_id = ""
        self._hermes_home = ""
        self._last_sync_error = ""
        self._last_sync_at = 0

    @property
    def name(self) -> str:
        return "turso"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self):
        from hermes_constants import display_hermes_home

        return [
            {"key": "db_path", "description": "Local Turso/SQLite database path", "default": f"{display_hermes_home()}/turso-memory.db"},
            {"key": "sync_enabled", "description": "Enable Turso Cloud sync", "default": "false", "choices": ["false", "true"]},
            {"key": "database_url", "description": "Turso database URL", "secret": True, "required": False, "env_var": "TURSO_DATABASE_URL"},
            {"key": "auth_token", "description": "Turso auth token", "secret": True, "required": False, "env_var": "TURSO_AUTH_TOKEN"},
            {"key": "top_k", "description": "Automatic prefetch result count", "default": str(_DEFAULT_TOP_K)},
            {"key": "min_similarity", "description": "Minimum automatic prefetch score", "default": str(_DEFAULT_MIN_SIMILARITY)},
            {"key": "auto_capture", "description": "Capture completed turns", "default": "false", "choices": ["false", "true"]},
        ]

    def save_config(self, values, hermes_home):
        _save_config(values, hermes_home)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = str(kwargs.get("hermes_home") or "")
        if not self._hermes_home:
            from hermes_constants import get_hermes_home
            self._hermes_home = str(get_hermes_home())
        self._config = dict(self._config_override or _load_config(self._hermes_home))
        self._embedder = _LocalEmbedder(int(self._config.get("embedding_dim", _DEFAULT_DIM)))
        db_path = Path(str(self._config["db_path"])).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def system_prompt_block(self) -> str:
        count = 0
        if self._conn:
            try:
                count = int(self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
            except Exception:
                count = 0
        return (
            "# Turso Memory\n"
            f"Active. {count} searchable memories stored. "
            "Use turso_memory_search for deep recall and the built-in memory tool for compact always-on facts."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not self._conn:
            return ""
        results = self._search(query, top_k=int(self._config.get("top_k", _DEFAULT_TOP_K)))
        min_score = float(self._config.get("min_similarity", _DEFAULT_MIN_SIMILARITY))
        results = [r for r in results if float(r.get("score", 0.0)) >= min_score]
        if not results:
            return ""
        lines = [
            f"- [{item['score']:.2f}] ({item['kind']}) {item['content']}"
            for item in results
        ]
        return "## Turso Memory\n" + "\n".join(lines)

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
        if not _as_bool(self._config.get("auto_capture"), False):
            return
        user = _normalize_text(str(user_content or ""))
        assistant = _normalize_text(str(assistant_content or ""))
        if len(user) + len(assistant) < _MIN_CAPTURE_CHARS:
            return
        content = f"User: {user}\nAssistant: {assistant}"
        self._add_memory(
            content,
            kind="conversation",
            source="turn",
            session_id=session_id or self._session_id,
            metadata={"captured_from": "sync_turn"},
        )

    def on_memory_write(self, action, target, content, metadata=None):
        if action == "remove":
            self._mark_removed_by_content(content or str((metadata or {}).get("old_text", "")), target)
            return
        if action == "replace":
            old = str((metadata or {}).get("old_text", ""))
            if old:
                self._mark_removed_by_content(old, target)
        if content:
            self._add_memory(
                str(content),
                kind="preference" if target == "user" else "fact",
                source=f"builtin:{action}",
                target=target,
                session_id=str((metadata or {}).get("session_id") or self._session_id),
                metadata=metadata or {},
            )

    def on_session_end(self, messages):
        self._sync_if_configured()

    def shutdown(self):
        self._sync_if_configured()
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_tool_schemas(self):
        return [SEARCH_SCHEMA, ADD_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA, SYNC_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "turso_memory_search":
                query = str(args.get("query") or "").strip()
                if not query:
                    return tool_error("query is required")
                top_k = max(1, min(20, int(args.get("top_k") or self._config.get("top_k", _DEFAULT_TOP_K))))
                return json.dumps({"success": True, "results": self._search(query, top_k=top_k, kind=args.get("kind"))})
            if tool_name == "turso_memory_add":
                content = str(args.get("content") or "").strip()
                if not content:
                    return tool_error("content is required")
                item = self._add_memory(
                    content,
                    kind=str(args.get("kind") or "fact"),
                    source=str(args.get("source") or "tool"),
                    weight=float(args.get("weight") or 1.0),
                    session_id=str(kwargs.get("session_id") or self._session_id),
                )
                return json.dumps({"success": True, "memory": item})
            if tool_name == "turso_memory_update":
                memory_id = str(args.get("memory_id") or "").strip()
                content = str(args.get("content") or "").strip()
                if not memory_id or not content:
                    return tool_error("memory_id and content are required")
                return json.dumps(self._update_memory(memory_id, content, kind=args.get("kind"), weight=args.get("weight")))
            if tool_name == "turso_memory_delete":
                memory_id = str(args.get("memory_id") or "").strip()
                if not memory_id:
                    return tool_error("memory_id is required")
                return json.dumps(self._delete_memory(memory_id))
            if tool_name == "turso_memory_sync":
                action = str(args.get("action") or "status")
                if action == "sync":
                    self._sync_if_configured()
                return json.dumps({"success": True, "sync": self._sync_status()})
            return tool_error(f"Unknown Turso memory tool: {tool_name}")
        except Exception as e:
            logger.debug("Turso tool %s failed", tool_name, exc_info=True)
            return tool_error(str(e))

    def backup_paths(self) -> list[str]:
        db_path = self._config.get("db_path")
        return [str(db_path)] if db_path else []

    def _migrate(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'fact',
                source TEXT NOT NULL DEFAULT 'tool',
                target TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                embedding_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                last_retrieved INTEGER NOT NULL DEFAULT 0,
                retrieval_count INTEGER NOT NULL DEFAULT 0,
                weight REAL NOT NULL DEFAULT 1.0,
                deleted_at INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_turso_memories_kind ON memories(kind)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_turso_memories_deleted ON memories(deleted_at)")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_events (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                action TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    def _add_memory(
        self,
        content: str,
        *,
        kind: str = "fact",
        source: str = "tool",
        target: str = "",
        session_id: str = "",
        weight: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> dict:
        assert self._conn is not None
        clean = _normalize_text(content)
        memory_id = uuid.uuid4().hex
        ts = _now()
        embedding = self._embedder.embed(clean)
        weight = max(0.1, min(3.0, float(weight)))
        self._conn.execute(
            """
            INSERT INTO memories
            (id, content, kind, source, target, session_id, embedding_json, metadata_json,
             created_at, updated_at, weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                clean,
                kind or "fact",
                source or "tool",
                target or "",
                session_id or "",
                json.dumps(embedding),
                json.dumps(metadata or {}, sort_keys=True),
                ts,
                ts,
                weight,
            ),
        )
        self._event(memory_id, "add", metadata or {})
        self._conn.commit()
        return {"id": memory_id, "content": clean, "kind": kind or "fact", "source": source or "tool", "weight": weight}

    def _search(self, query: str, *, top_k: int, kind: Any = None) -> list[dict]:
        assert self._conn is not None
        q_embedding = self._embedder.embed(query)
        params: list[Any] = []
        where = "deleted_at = 0"
        if kind:
            where += " AND kind = ?"
            params.append(str(kind))
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY updated_at DESC LIMIT 500",
            params,
        ).fetchall()
        scored: list[dict] = []
        for row in rows:
            try:
                embedding = json.loads(row["embedding_json"])
            except Exception:
                embedding = []
            score = max(_cosine(q_embedding, embedding), _keyword_score(query, row["content"])) * float(row["weight"])
            if score <= 0:
                continue
            scored.append({
                "id": row["id"],
                "content": row["content"],
                "kind": row["kind"],
                "source": row["source"],
                "target": row["target"],
                "session_id": row["session_id"],
                "score": round(float(score), 4),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "retrieval_count": row["retrieval_count"],
            })
        scored.sort(key=lambda item: item["score"], reverse=True)
        selected = scored[:top_k]
        if selected:
            ts = _now()
            self._conn.executemany(
                "UPDATE memories SET last_retrieved = ?, retrieval_count = retrieval_count + 1 WHERE id = ?",
                [(ts, item["id"]) for item in selected],
            )
            self._conn.commit()
        return selected

    def _update_memory(self, memory_id: str, content: str, *, kind: Any = None, weight: Any = None) -> dict:
        assert self._conn is not None
        clean = _normalize_text(content)
        row = self._conn.execute("SELECT * FROM memories WHERE id = ? AND deleted_at = 0", (memory_id,)).fetchone()
        if not row:
            return {"success": False, "error": f"memory not found: {memory_id}"}
        new_kind = str(kind or row["kind"])
        new_weight = float(weight if weight is not None else row["weight"])
        self._conn.execute(
            "UPDATE memories SET content = ?, kind = ?, weight = ?, embedding_json = ?, updated_at = ? WHERE id = ?",
            (clean, new_kind, max(0.1, min(3.0, new_weight)), json.dumps(self._embedder.embed(clean)), _now(), memory_id),
        )
        self._event(memory_id, "update", {"kind": new_kind})
        self._conn.commit()
        return {"success": True, "memory": {"id": memory_id, "content": clean, "kind": new_kind}}

    def _delete_memory(self, memory_id: str) -> dict:
        assert self._conn is not None
        cur = self._conn.execute("UPDATE memories SET deleted_at = ?, updated_at = ? WHERE id = ? AND deleted_at = 0", (_now(), _now(), memory_id))
        if cur.rowcount <= 0:
            return {"success": False, "error": f"memory not found: {memory_id}"}
        self._event(memory_id, "delete", {})
        self._conn.commit()
        return {"success": True, "deleted": memory_id}

    def _mark_removed_by_content(self, content: str, target: str = "") -> None:
        if not content or not self._conn:
            return
        pattern = f"%{content}%"
        self._conn.execute(
            "UPDATE memories SET deleted_at = ?, updated_at = ? WHERE deleted_at = 0 AND content LIKE ? AND (? = '' OR target = ?)",
            (_now(), _now(), pattern, target or "", target or ""),
        )
        self._conn.commit()

    def _event(self, memory_id: str, action: str, metadata: dict) -> None:
        assert self._conn is not None
        self._conn.execute(
            "INSERT INTO memory_events (id, memory_id, action, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, memory_id, action, json.dumps(metadata or {}, sort_keys=True), _now()),
        )

    def _sync_status(self) -> dict:
        enabled = _as_bool(self._config.get("sync_enabled"), False)
        return {
            "enabled": enabled,
            "configured": bool(os.environ.get("TURSO_DATABASE_URL") and os.environ.get("TURSO_AUTH_TOKEN")),
            "last_sync_at": self._last_sync_at,
            "last_error": self._last_sync_error,
        }

    def _sync_if_configured(self) -> None:
        if not _as_bool(self._config.get("sync_enabled"), False):
            return
        if not (os.environ.get("TURSO_DATABASE_URL") and os.environ.get("TURSO_AUTH_TOKEN")):
            self._last_sync_error = "TURSO_DATABASE_URL and TURSO_AUTH_TOKEN are required for sync"
            return
        try:
            # Placeholder for pyturso/libSQL sync once the runtime dependency
            # is installed and configured. Local memory remains fully usable.
            __import__("turso")
            self._last_sync_at = _now()
            self._last_sync_error = ""
        except Exception as e:
            self._last_sync_error = f"Turso sync unavailable: {e}"


def register(ctx) -> None:
    ctx.register_memory_provider(TursoMemoryProvider())
