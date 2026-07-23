"""
Float16 semantic memory provider — local vector storage with float16 embeddings.

Captures conversation turns as sentence-transformer embeddings stored in
float16 precision (half the memory of float32, minimal accuracy loss for
sematic similarity). Provides automatic turn capture and cosine-similarity
semantic search.

Persistence: SQLite-backed. Embeddings stored as numpy float16 byte blobs.
No external API or network calls after the model is loaded.

Config in $HERMES_HOME/config.yaml (profile-scoped)::

    memory:
      float16:
        model_name: all-MiniLM-L6-v2    # sentence-transformers model
        db_path: ~/.hermes/float16_memory.db
        max_memories: 2000               # trim oldest when exceeded
        recall_limit: 5                  # max results returned by prefetch
        recall_threshold: 0.25           # min cosine similarity for recall
        auto_capture: true               # capture turns automatically
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Default config
_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_DEFAULT_RECALL_LIMIT = 5
_DEFAULT_RECALL_THRESHOLD = 0.25
_DEFAULT_MAX_MEMORIES = 2000
_DEFAULT_AUTO_CAPTURE = True
_MIN_CONTENT_LEN = 10
_PREFETCH_CACHE_LOCK = threading.Lock()

# SQL to create/init the memory store
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL DEFAULT '',
    role        TEXT    NOT NULL DEFAULT 'user',
    content     TEXT    NOT NULL,
    embedding   BLOB,                          -- numpy float16 bytes
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default


def _load_plugin_config() -> dict:
    """Read float16-specific config from config.yaml."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        memory_config = config.get("memory", {})
        if not isinstance(memory_config, dict):
            return {}
        provider_cfg = memory_config.get("float16", {})
        if isinstance(provider_cfg, dict):
            return dict(provider_cfg)
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Embedding model singleton (thread-safe, lazy-loaded)
# ---------------------------------------------------------------------------

_embedding_lock = threading.Lock()
_embedding_model = None
_embedding_dim = 384  # default for all-MiniLM-L6-v2


def _get_embedding_model(model_name: str):
    """Lazy-load the sentence-transformer model (thread-safe singleton)."""
    global _embedding_model, _embedding_dim
    if _embedding_model is not None:
        return _embedding_model

    with _embedding_lock:
        if _embedding_model is not None:
            return _embedding_model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading float16 embedding model: %s", model_name)
        model = SentenceTransformer(model_name)
        _embedding_model = model
        _embedding_dim = model.get_sentence_embedding_dimension() or 384
        logger.info(
            "Float16 embedding model loaded (dim=%d)", _embedding_dim
        )
        return model


def _normalize(vec):
    """L2-normalize a numpy array in-place (or return a normalized copy)."""
    import numpy as np
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _embed_text(text: str, model_name: str = _DEFAULT_MODEL_NAME) -> bytes:
    """Embed text and return float16 bytes."""
    import numpy as np
    model = _get_embedding_model(model_name)
    vec = model.encode(text, normalize_embeddings=True)
    vec_f16 = np.asarray(vec, dtype=np.float16)
    return vec_f16.tobytes()


def _cosine_similarity_f16(query_bytes: bytes, stored_bytes: bytes) -> float:
    """Compute cosine similarity between two float16 byte blobs."""
    import numpy as np
    q = np.frombuffer(query_bytes, dtype=np.float16)
    s = np.frombuffer(stored_bytes, dtype=np.float16)
    # Normalize stored vector (should already be normalized, but be safe)
    norm_s = np.linalg.norm(s)
    if norm_s == 0:
        return 0.0
    s = s / norm_s
    dot = float(np.dot(q, s))
    return max(-1.0, min(1.0, dot))


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "float16_search",
    "description": (
        "Semantic search over conversation memory using float16 vector embeddings. "
        "Returns passages semantically similar to the query, ranked by relevance. "
        "Use when you need to recall details from past conversations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in past memory.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return (default: 5).",
            },
        },
        "required": ["query"],
    },
}

STATUS_SCHEMA = {
    "name": "float16_status",
    "description": "Check float16 memory store status — total entries, model info, db path.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

CLEAR_SCHEMA = {
    "name": "float16_clear",
    "description": "Clear all float16 memory entries for the current session.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Float16MemoryProvider(MemoryProvider):
    """Float16 semantic memory provider with automatic turn capture.

    Uses sentence-transformers for embeddings, numpy float16 for storage,
    and SQLite for persistence. No external API dependencies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or _load_plugin_config()
        self._model_name: str = str(
            self._config.get("model_name", _DEFAULT_MODEL_NAME)
        )
        self._db_path: Optional[Path] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._turn_count: int = 0
        self._prefetch_cache: List[Dict[str, Any]] = []
        self._max_memories: int = int(
            self._config.get("max_memories", _DEFAULT_MAX_MEMORIES)
        )
        self._recall_limit: int = max(
            1, min(20, int(self._config.get("recall_limit", _DEFAULT_RECALL_LIMIT)))
        )
        self._recall_threshold: float = max(
            0.0, min(1.0, float(self._config.get("recall_threshold", _DEFAULT_RECALL_THRESHOLD)))
        )
        self._auto_capture: bool = _coerce_bool(
            self._config.get("auto_capture"), _DEFAULT_AUTO_CAPTURE
        )
        self._prefetched_context: str = ""

    # -- Lifecycle -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "float16"

    def is_available(self) -> bool:
        """Check if numpy and sentence-transformers are importable."""
        try:
            import numpy as np  # noqa: F401
            np.dtype("float16")  # verify float16 support
        except ImportError:
            return False
        except TypeError:
            return False
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError:
            return False
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "model_name",
                "description": "sentence-transformers model name for embeddings (default: all-MiniLM-L6-v2)",
                "default": _DEFAULT_MODEL_NAME,
            },
            {
                "key": "auto_capture",
                "description": "Automatically capture conversation turns as memories",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "max_memories",
                "description": "Maximum stored memories before trimming oldest",
                "default": str(_DEFAULT_MAX_MEMORIES),
            },
            {
                "key": "recall_limit",
                "description": "Default number of memories returned by prefetch",
                "default": str(_DEFAULT_RECALL_LIMIT),
            },
            {
                "key": "recall_threshold",
                "description": "Minimum cosine similarity for recall (0.0-1.0)",
                "default": str(_DEFAULT_RECALL_THRESHOLD),
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write float16 memory config to config.yaml."""
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("memory", {})
            existing["memory"]["float16"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", "")
        self._turn_count = 0

        # Resolve DB path
        raw_path = self._config.get("db_path", "")
        if raw_path:
            db_path = Path(str(raw_path))
        else:
            base = Path(self._hermes_home) if self._hermes_home else Path("~/.hermes").expanduser()
            db_path = base / "float16_memory.db"

        # Expand vars
        db_str = str(db_path)
        if self._hermes_home:
            db_str = db_str.replace("$HERMES_HOME", self._hermes_home)
            db_str = db_str.replace("${HERMES_HOME}", self._hermes_home)
        db_str = str(Path(db_str).expanduser().resolve())
        self._db_path = Path(db_str)

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open SQLite connection (thread-safe mode)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

        # Warm up the embedding model (lazy, triggers on first use)
        try:
            _get_embedding_model(self._model_name)
        except Exception as e:
            logger.warning("Float16 embedding model failed to load: %s", e)

        logger.info(
            "Float16 memory initialized (db=%s, model=%s)",
            self._db_path, self._model_name,
        )

    def system_prompt_block(self) -> str:
        if not self._conn:
            return ""
        try:
            with self._db_lock:
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM memories"
                ).fetchone()[0]
        except Exception:
            count = 0

        lines = [
            "# Float16 Semantic Memory",
            "Active. Automatic turn capture enabled.",
        ]
        if count > 0:
            lines.append(f"{count} memories stored (float16 vectors, local SQLite).")
            lines.append(
                "Use float16_search to query past conversation context semantically."
            )
        else:
            lines.append(
                "Memory store is empty — will capture turns automatically."
            )
        return "\n".join(lines)

    # -- Prefetch / recall ---------------------------------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background semantic search for the next turn."""
        if not query or len(query.strip()) < _MIN_CONTENT_LEN or not self._conn:
            return

        def _bg_search():
            try:
                result = self._semantic_search(query, limit=self._recall_limit)
                with _PREFETCH_CACHE_LOCK:
                    self._prefetch_cache = result
            except Exception as e:
                logger.debug("Float16 background prefetch failed: %s", e)

        t = threading.Thread(target=_bg_search, daemon=True, name="f16-prefetch")
        t.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return semantically relevant context for the upcoming turn."""
        # If we have a cached result from queue_prefetch, use it
        with _PREFETCH_CACHE_LOCK:
            cached = list(self._prefetch_cache)
            self._prefetch_cache = []

        if cached:
            return self._format_results(cached, "## Float16 Memory")

        # Fallback: run search synchronously
        results = self._semantic_search(query, limit=self._recall_limit)
        if not results:
            return ""
        return self._format_results(results, "## Float16 Memory")

    def _format_results(self, results: list, heading: str) -> str:
        parts = [heading]
        for r in results:
            score = r.get("score", 0)
            content = r.get("content", "")
            role = r.get("role", "user")
            created = r.get("created_at", "")
            line = f"- [{score:.2f}] ({role}, {created}) {content[:300]}"
            parts.append(line)
        return "\n".join(parts)

    # -- Turn capture --------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Automatically capture the conversation turn as a memory."""
        if not self._auto_capture or not self._conn:
            return

        self._turn_count += 1
        sid = session_id or self._session_id

        def _store(content: str, role: str):
            if not content or len(content.strip()) < _MIN_CONTENT_LEN:
                return
            try:
                embedding = _embed_text(content.strip(), self._model_name)
                with self._db_lock:
                    self._conn.execute(
                        "INSERT INTO memories (session_id, role, content, embedding) "
                        "VALUES (?, ?, ?, ?)",
                        (sid, role, content.strip()[:2000], embedding),
                    )
                    self._conn.commit()
                    self._maybe_trim()
            except Exception as e:
                logger.debug("Float16 sync_turn store failed: %s", e)

        # Store user message
        _store(user_content, "user")
        # Store assistant message
        _store(assistant_content, "assistant")

    def _maybe_trim(self) -> None:
        """Remove oldest memories if we exceed max_memories."""
        if self._max_memories <= 0:
            return
        try:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()[0]
            if count > self._max_memories:
                excess = count - self._max_memories
                self._conn.execute(
                    "DELETE FROM memories WHERE id IN ("
                    "SELECT id FROM memories ORDER BY created_at ASC LIMIT ?"
                    ")",
                    (excess,),
                )
                self._conn.commit()
        except Exception:
            pass

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, STATUS_SCHEMA, CLEAR_SCHEMA]

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs
    ) -> str:
        if tool_name == "float16_search":
            return self._tool_search(args)
        elif tool_name == "float16_status":
            return self._tool_status()
        elif tool_name == "float16_clear":
            return self._tool_clear()
        return tool_error(f"Unknown tool: {tool_name}")

    def _tool_search(self, args: dict) -> str:
        query = args.get("query", "").strip()
        if not query:
            return tool_error("query is required")
        limit = max(1, min(50, int(args.get("limit", self._recall_limit))))
        results = self._semantic_search(query, limit=limit)
        if not results:
            return json.dumps({"results": [], "count": 0, "message": "No relevant memories found."})
        return json.dumps({"results": results, "count": len(results)})

    def _tool_status(self) -> str:
        if not self._conn:
            return json.dumps({"error": "Not initialized"})
        try:
            with self._db_lock:
                total = self._conn.execute(
                    "SELECT COUNT(*) FROM memories"
                ).fetchone()[0]
                sessions = self._conn.execute(
                    "SELECT COUNT(DISTINCT session_id) FROM memories"
                ).fetchone()[0]
                latest = self._conn.execute(
                    "SELECT created_at FROM memories ORDER BY id DESC LIMIT 1"
                ).fetchone()
            return json.dumps({
                "status": "ok",
                "total_memories": total,
                "distinct_sessions": sessions,
                "model": self._model_name,
                "dimension": _embedding_dim,
                "dtype": "float16",
                "db_path": str(self._db_path),
                "auto_capture": self._auto_capture,
                "last_entry": latest[0] if latest else None,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_clear(self) -> str:
        if not self._conn:
            return json.dumps({"error": "Not initialized"})
        sid = self._session_id
        try:
            with self._db_lock:
                self._conn.execute(
                    "DELETE FROM memories WHERE session_id = ?", (sid,)
                )
                self._conn.commit()
            return json.dumps({"message": f"Cleared all memories for session {sid}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # -- Semantic search -----------------------------------------------------

    def _semantic_search(
        self, query: str, *, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search memories by cosine similarity with float16 embeddings."""
        if not self._conn or not query or len(query.strip()) < 2:
            return []

        try:
            query_embedding = _embed_text(query.strip(), self._model_name)
        except Exception as e:
            logger.debug("Float16 embed failed: %s", e)
            return []

        with self._db_lock:
            rows = self._conn.execute(
                "SELECT id, session_id, role, content, embedding, created_at "
                "FROM memories ORDER BY id DESC"
            ).fetchall()

        if not rows:
            return []

        import numpy as np
        q = np.frombuffer(query_embedding, dtype=np.float16)

        scored = []
        for row in rows:
            row_id, sid, role, content, emb_blob, created = row
            if emb_blob is None:
                continue
            try:
                s = np.frombuffer(emb_blob, dtype=np.float16)
                # Normalize stored vector
                norm_s = np.linalg.norm(s)
                if norm_s > 0:
                    s = s / norm_s
                dot = float(np.dot(q, s))
                score = max(-1.0, min(1.0, dot))
            except Exception:
                continue

            if score >= self._recall_threshold:
                scored.append({
                    "id": row_id,
                    "session_id": sid,
                    "role": role,
                    "content": content,
                    "score": round(score, 4),
                    "created_at": created,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # -- Shutdown ------------------------------------------------------------

    def shutdown(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        global _embedding_model
        with _embedding_lock:
            _embedding_model = None

    # -- Optional hooks ------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Optionally do a final flush of remaining messages at session end."""
        pass

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes as float16 embeddings."""
        if not self._auto_capture or not self._conn or not content:
            return
        if action not in {"add", "replace"}:
            return
        try:
            label = f"[memory:{target}]"
            embedding = _embed_text(f"{label} {content}", self._model_name)
            with self._db_lock:
                self._conn.execute(
                    "INSERT INTO memories (session_id, role, content, embedding) "
                    "VALUES (?, ?, ?, ?)",
                    (self._session_id, "system", f"{label} {content[:2000]}", embedding),
                )
                self._conn.commit()
        except Exception as e:
            logger.debug("Float16 on_memory_write failed: %s", e)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the float16 memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = Float16MemoryProvider(config=config)
    ctx.register_memory_provider(provider)
