"""KVMemoryProvider — Model-native semantic memory via hidden states / KV-cache.

Implements the MemoryProvider ABC for Hermes Agent's memory plugin system.

Mode B (current): Hidden-state embeddings via sentence-transformers.
  - Each turn's user+assistant text is encoded into a fixed-size embedding.
  - Embeddings are Q4-quantized and stored in SQLite.
  - Retrieval uses cosine similarity with temporal decay and MMR diversity.

Mode A (Phase 2): Direct KV-cache capture from vLLM.
  - Requires GPU and vLLM integration.
  - KV-cache states are extracted post-inference, Q4-quantized, and stored.
  - Retrieved KV blocks are injected directly into the model's attention.

Architecture:
  ┌─────────────┐     ┌──────────────┐     ┌───────────┐
  │ MemoryProvider│────▶│ capture.py   │────▶│ quantize  │
  │  (this file) │     │ (embeddings) │     │  (Q4)     │
  └──────┬──────┘     └──────────────┘     └─────┬─────┘
         │                                       │
         ▼                                       ▼
  ┌──────────────┐                        ┌───────────┐
  │ retrieval.py │◀───────────────────────│ storage.py │
  │  (search)    │                        │ (SQLite)   │
  └──────────────┘                        └───────────┘
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from agent.memory_provider import MemoryProvider
from .capture import create_embedding_backend
from .config import KVMemoryConfig, load_config
from .quantize import quantize_q4_per_channel, compute_q4_size
from .retrieval import KVRetriever
from .storage import KVMemoryDB

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool schemas
# ═══════════════════════════════════════════════════════════════════════════════

KV_MEMORY_SEARCH_SCHEMA = {
    "name": "kv_memory_search",
    "description": (
        "Search your model-native semantic memory for relevant past conversations. "
        "Unlike keyword search, this finds conceptually related memories even when "
        "the vocabulary doesn't match. Use this to recall past decisions, user "
        "preferences, or project context from previous sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in natural language.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

KV_MEMORY_STATUS_SCHEMA = {
    "name": "kv_memory_status",
    "description": (
        "Show statistics about the model-native memory store: total turns stored, "
        "database size, embedding model in use, and storage mode."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Provider
# ═══════════════════════════════════════════════════════════════════════════════

class KVMemoryProvider(MemoryProvider):
    """Model-native semantic memory via hidden states and KV-cache states."""

    def __init__(self, config: KVMemoryConfig | None = None):
        self._config = config or load_config()
        self._db: Optional[KVMemoryDB] = None
        self._retriever: Optional[KVRetriever] = None
        self._backend = None  # EmbeddingBackend
        self._session_id: str = ""
        self._turn_number: int = 0
        self._model_id: str = ""
        self._prefetch_cache: str = ""  # cached result from queue_prefetch
        self._prefetch_lock = threading.Lock()
        self._initialized = False

    # ── MemoryProvider ABC ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "kv-memory"

    def is_available(self) -> bool:
        """Check if any embedding backend is available."""
        try:
            backend = create_embedding_backend(
                self._config.embedding_backend,
                self._config.embedding_model,
            )
            return True
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize the kv-memory provider for a session.

        Sets up the embedding backend, SQLite database, and retriever.
        Called once at agent startup.
        """
        self._session_id = session_id
        self._turn_number = 0
        self._prefetch_cache = ""

        # Extract model info
        self._model_id = kwargs.get("model", "")

        # Resolve db_path
        db_path = self._config.db_path
        if not db_path:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "kv_memory.db")
        # Expand $HERMES_HOME
        if "$HERMES_HOME" in db_path:
            from hermes_constants import get_hermes_home
            db_path = db_path.replace("$HERMES_HOME", str(get_hermes_home()))

        # Initialize embedding backend
        self._backend = create_embedding_backend(
            self._config.embedding_backend,
            self._config.embedding_model,
        )
        logger.info(
            "KV-Memory backend: %s (dim=%d)",
            self._backend.backend_name, self._backend.embedding_dim,
        )

        # Initialize database
        self._db = KVMemoryDB(db_path)
        self._db.initialize_schema()
        self._db.migrate_schema()
        self._db.ensure_session(session_id, metadata={
            "platform": kwargs.get("platform", ""),
            "agent_context": kwargs.get("agent_context", ""),
            "model_id": self._model_id,
        })

        # Initialize retriever
        self._retriever = KVRetriever(self._db, self._config)

        # Governance: prune old turns on init
        if self._config.retention_days > 0:
            try:
                pruned = self._db.prune_old_turns(self._config.retention_days)
                if pruned:
                    logger.info("KV-Memory pruned %d old turns", pruned)
            except Exception as e:
                logger.debug("KV-Memory prune failed: %s", e)

        # Check if compaction is needed
        turn_count = self._db.count_turns()
        if turn_count > self._config.max_stored_turns:
            logger.info(
                "KV-Memory: %d turns exceeds max %d, compacting...",
                turn_count, self._config.max_stored_turns,
            )
            self._compact_all()

        self._initialized = True

        stats = self._db.get_stats()
        logger.info(
            "KV-Memory initialized: %d turns, %d sessions, %.1fMB DB",
            stats["total_turns"], stats["total_sessions"], stats["db_size_mb"],
        )

    def system_prompt_block(self) -> str:
        """Return static system prompt text."""
        if not self._initialized or not self._db:
            return ""

        try:
            stats = self._db.get_stats()
        except Exception:
            return ""

        if stats["total_turns"] == 0:
            return (
                "# KV Model-Native Memory\n"
                "Active. Empty memory store — conversations you have will be "
                "automatically stored as model-native embeddings for semantic recall. "
                "Use kv_memory_search to query past conversations, or kv_memory_status "
                "to see storage statistics."
            )

        return (
            "# KV Model-Native Memory\n"
            f"Active. {stats['total_turns']} turns stored across "
            f"{stats['total_sessions']} sessions ({stats['db_size_mb']}MB).\n"
            f"Embedding backend: {self._backend.backend_name if self._backend else 'unknown'}. "
            f"Use kv_memory_search to query with semantic understanding, "
            f"or kv_memory_status for details."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached recall context for the upcoming turn.

        The actual retrieval happens in queue_prefetch() after the previous
        turn. This just returns the cached result.
        """
        if not self._initialized or not self._retriever:
            return ""

        with self._prefetch_lock:
            cached = self._prefetch_cache
            self._prefetch_cache = ""
        return cached

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background retrieval for the next turn.

        This runs the embedding generation and similarity search in the
        background, caching the result for prefetch() to return.
        """
        if not self._initialized or not self._retriever or not query:
            return

        try:
            # Generate query embedding
            query_embedding = self._backend.encode(query)

            # Retrieve relevant context
            context_text = self._retriever.retrieve_text(
                query_embedding,
                current_session_id=session_id or self._session_id,
            )

            with self._prefetch_lock:
                self._prefetch_cache = context_text
        except Exception as e:
            logger.debug("KV-Memory queue_prefetch failed: %s", e)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist a completed turn's embedding to storage.

        Encodes the combined user+assistant text, Q4-quantizes, and stores
        in SQLite. Runs synchronously — embedding generation is fast enough
        (<50ms) that backgrounding isn't needed.
        """
        if not self._initialized or not self._backend or not self._db:
            return

        sid = session_id or self._session_id

        try:
            # Combine user + assistant for the embedding
            combined = f"User: {user_content}\nAssistant: {assistant_content}"

            # Generate embedding
            embedding = self._backend.encode(combined)
            embedding_dim = embedding.shape[0]

            # Determine channel size for Q4 quantization
            channel_size = self._config.q4_channel_size or 128  # head_dim default

            # Q4 quantize
            q4_packed, q4_scales = quantize_q4_per_channel(embedding, channel_size)

            # Generate a lightweight summary (first ~200 chars of user content)
            summary = user_content[:200] if user_content else ""

            # Extract tool calls from messages if available
            tool_calls_list = []
            if messages:
                for msg in messages:
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        tool_calls_list = [
                            {"name": tc.get("function", {}).get("name", ""),
                             "args": tc.get("function", {}).get("arguments", "")}
                            for tc in msg["tool_calls"]
                        ]
                        break

            # Store turn
            self._turn_number += 1
            turn_id = self._db.store_turn(
                session_id=sid,
                turn_number=self._turn_number,
                embedding=embedding,
                q4_embedding=q4_packed,
                q4_scales=q4_scales,
                summary_text=summary,
                tool_calls=tool_calls_list,
                model_id=self._model_id,
                head_dim=channel_size,
                num_layers=0,  # not applicable for sentence-transformers
                num_kv_heads=embedding_dim // channel_size,
                metadata={
                    "backend": self._backend.backend_name,
                    "storage_mode": "q4",
                },
                store_fp16=self._config.store_fp16_fidelity_check,
            )

            logger.debug("KV-Memory stored turn %s (dim=%d)", turn_id, embedding_dim)

        except Exception as e:
            logger.warning("KV-Memory sync_turn failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [KV_MEMORY_SEARCH_SCHEMA, KV_MEMORY_STATUS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "kv_memory_search":
            return self._handle_search(args)
        elif tool_name == "kv_memory_status":
            return self._handle_status()
        raise NotImplementedError(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        """Clean shutdown — close database connection."""
        if self._db:
            try:
                self._db.close()
            except Exception as e:
                logger.debug("KV-Memory shutdown close failed: %s", e)
        self._db = None
        self._retriever = None
        self._backend = None
        self._initialized = False

    # ── Optional hooks ─────────────────────────────────────────────────

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Run compaction on session end if configured."""
        if not self._initialized or not self._config.auto_compact:
            return
        try:
            sid = self._session_id
            removed = self._db.compact_session(sid)
            if removed:
                logger.info("KV-Memory compacted session %s: removed %d turns", sid, removed)
        except Exception as e:
            logger.debug("KV-Memory compaction failed: %s", e)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Handle session transitions.

        On reset/rewind: link sessions, reset turn counter.
        On resume: continue counting from where we left off.
        """
        if reset:
            # Link old → new session
            if parent_session_id and self._db:
                try:
                    self._db.link_sessions(
                        parent_session_id, new_session_id,
                        similarity=1.0, link_type="continuation",
                    )
                except Exception:
                    pass
            self._turn_number = 0
        elif rewound:
            self._turn_number = 0
        else:
            # Resuming — preserve turn counter, ensure session record
            self._session_id = new_session_id
            if self._db:
                self._db.ensure_session(new_session_id)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes as stored turns."""
        if action != "add" or not content or not self._initialized:
            return
        try:
            embedding = self._backend.encode(content)
            channel_size = self._config.q4_channel_size or 128
            q4_packed, q4_scales = quantize_q4_per_channel(embedding, channel_size)
            self._db.store_turn(
                session_id=self._session_id,
                turn_number=self._turn_number + 1,
                embedding=embedding,
                q4_embedding=q4_packed,
                q4_scales=q4_scales,
                summary_text=content[:200],
                model_id=self._model_id,
                head_dim=channel_size,
                num_kv_heads=embedding.shape[0] // channel_size,
                metadata={
                    "source": "builtin_memory_tool",
                    "target": target,
                    "backend": self._backend.backend_name,
                    **(metadata or {}),
                },
            )
            self._turn_number += 1
        except Exception as e:
            logger.debug("KV-Memory memory_write mirror failed: %s", e)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Return config schema for 'hermes memory setup'."""
        return [
            {
                "key": "embedding_backend",
                "description": "Embedding backend: 'auto', 'sentence-transformers', 'vllm', or 'api'",
                "default": "auto",
                "choices": ["auto", "sentence-transformers", "vllm", "api"],
            },
            {
                "key": "embedding_model",
                "description": "Model name for the embedding backend (empty = default)",
                "default": "",
            },
            {
                "key": "top_k",
                "description": "Number of results to return from memory search",
                "default": "5",
            },
            {
                "key": "min_similarity",
                "description": "Minimum cosine similarity threshold (0.0 - 1.0)",
                "default": "0.5",
            },
            {
                "key": "retention_days",
                "description": "Days to retain memory before auto-pruning",
                "default": "90",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write config values to config.yaml under plugins.kv-memory."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["kv-memory"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    # ── Tool handlers ──────────────────────────────────────────────────

    def _handle_search(self, args: dict) -> str:
        """Handle kv_memory_search tool call."""
        if not self._initialized or not self._retriever or not self._backend:
            return json.dumps({"error": "KV-Memory provider not initialized"})

        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "query is required"})

        limit = int(args.get("limit", self._config.top_k))

        try:
            query_embedding = self._backend.encode(query)
            results = self._retriever.retrieve(
                query_embedding,
                current_session_id=self._session_id,
                k=limit,
            )

            formatted = []
            for r in results:
                formatted.append({
                    "similarity": r["similarity"],
                    "text": r["summary_text"],
                    "session_id": r["session_id"][:8] + "..." if len(r["session_id"]) > 8 else r["session_id"],
                })

            return json.dumps({
                "results": formatted,
                "count": len(formatted),
                "backend": self._backend.backend_name,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _handle_status(self) -> str:
        """Handle kv_memory_status tool call."""
        if not self._initialized or not self._db:
            return json.dumps({"error": "KV-Memory provider not initialized"})

        try:
            stats = self._db.get_stats()
            stats["backend"] = self._backend.backend_name if self._backend else "none"
            stats["embedding_dim"] = self._backend.embedding_dim if self._backend else 0
            stats["storage_mode"] = self._config.storage_mode
            stats["mode"] = self._config.mode
            return json.dumps(stats)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Internal ───────────────────────────────────────────────────────

    def _compact_all(self) -> None:
        """Compact all sessions that have >2 turns."""
        if not self._db:
            return
        try:
            # Get all sessions
            rows = self._db.conn.execute(
                "SELECT DISTINCT session_id FROM turns"
            ).fetchall()
            total_removed = 0
            for row in rows:
                removed = self._db.compact_session(row["session_id"])
                total_removed += removed
            if total_removed:
                logger.info("KV-Memory compacted %d turns across all sessions", total_removed)
        except Exception as e:
            logger.debug("KV-Memory compact_all failed: %s", e)
