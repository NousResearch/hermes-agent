"""Brain RAG — local advanced hybrid retrieval memory provider.

Hybrid BM25 + vector search with RRF fusion, MMR reranking, and document
ingestion. No external API keys required — runs entirely on-device.

Config in config.yaml:
  memory:
    provider: brain_rag
  plugins:
    brain_rag:
      db_path: ~/.hermes/brain-rag.db   # optional
      prefetch_top_k: 6
      context_max_chars: 3000
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from hermes_cli.config import cfg_get
from tools.registry import tool_error

from .retrieval import BrainRAGRetriever
from .store import BrainRAGStore

logger = logging.getLogger(__name__)


BRAIN_RAG_SEARCH_SCHEMA = {
    "name": "brain_rag_search",
    "description": (
        "Search the AI Brain knowledge base using hybrid RAG "
        "(BM25 + vector + reranking). Use before answering questions "
        "about prior work, documents, or stored facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default 8)."},
        },
        "required": ["query"],
    },
}

BRAIN_RAG_INGEST_SCHEMA = {
    "name": "brain_rag_ingest",
    "description": (
        "Ingest text or a file into the knowledge base for future retrieval. "
        "Chunks and indexes content automatically."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Raw text to ingest."},
            "file_path": {"type": "string", "description": "Local file path to ingest."},
            "title": {"type": "string", "description": "Optional document title."},
            "source": {"type": "string", "description": "Source label (default: manual)."},
        },
        "required": [],
    },
}

BRAIN_RAG_REMEMBER_SCHEMA = {
    "name": "brain_rag_remember",
    "description": "Store an explicit fact, preference, or decision in long-term memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to remember."},
            "category": {
                "type": "string",
                "enum": ["general", "preference", "project", "code", "task"],
                "description": "Category (default: general).",
            },
        },
        "required": ["content"],
    },
}


def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "brain_rag", default={}) or {}
    except Exception:
        return {}


class BrainRAGProvider(MemoryProvider):
    """Local advanced RAG memory provider."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store: Optional[BrainRAGStore] = None
        self._retriever: Optional[BrainRAGRetriever] = None
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._session_id = ""

    @property
    def name(self) -> str:
        return "brain_rag"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home

        hermes_home = Path(kwargs.get("hermes_home") or get_hermes_home())
        db_path = self._config.get("db_path")
        if db_path:
            db = Path(str(db_path).replace("~", str(Path.home())))
        else:
            db = hermes_home / "brain-rag.db"
        self._store = BrainRAGStore(db)
        self._retriever = BrainRAGRetriever(self._store)
        self._session_id = session_id
        logger.debug("Brain RAG initialized at %s", db)

    def system_prompt_block(self) -> str:
        stats = self._store.stats() if self._store else {}
        return (
            "You have an AI Brain knowledge base with hybrid RAG retrieval "
            "(BM25 + vector search + reranking). Before answering questions "
            "about prior work, code, documents, or user preferences, call "
            "`brain_rag_search`. Store durable facts with `brain_rag_remember` "
            "and ingest documents with `brain_rag_ingest`.\n"
            f"Knowledge base: {stats.get('chunks', 0)} chunks, "
            f"{stats.get('memories', 0)} memories across "
            f"{stats.get('documents', 0)} documents."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query or not self._retriever:
            return

        def _run():
            try:
                top_k = int(self._config.get("prefetch_top_k", 6))
                max_chars = int(self._config.get("context_max_chars", 3000))
                hits = self._retriever.search(query, limit=top_k)
                ctx = self._retriever.format_context(hits, max_chars=max_chars)
                with self._prefetch_lock:
                    self._prefetch_result = ctx
            except Exception as e:
                logger.debug("Brain RAG prefetch failed: %s", e)

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return
        self._prefetch_thread = threading.Thread(target=_run, daemon=True)
        self._prefetch_thread.start()

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        **kwargs,
    ) -> None:
        # Queue prefetch for the next turn based on the latest user message
        self.queue_prefetch(user_content, session_id=session_id)

    def get_tool_schemas(self) -> List[Dict]:
        return [
            BRAIN_RAG_SEARCH_SCHEMA,
            BRAIN_RAG_INGEST_SCHEMA,
            BRAIN_RAG_REMEMBER_SCHEMA,
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any]) -> str:
        if not self._store or not self._retriever:
            return tool_error("Brain RAG not initialized")

        if name == "brain_rag_search":
            query = (args.get("query") or "").strip()
            if not query:
                return tool_error("query is required")
            top_k = min(int(args.get("top_k") or 8), 20)
            hits = self._retriever.search(query, limit=top_k)
            return json.dumps({"success": True, "results": self._serialize_hits(hits)})

        if name == "brain_rag_ingest":
            text = (args.get("text") or "").strip()
            file_path = (args.get("file_path") or "").strip()
            title = (args.get("title") or "").strip()
            source = (args.get("source") or "manual").strip()
            if file_path:
                result = self._store.ingest_file(file_path, title=title)
            elif text:
                result = self._store.ingest_text(text, source=source, title=title or source)
            else:
                return tool_error("Provide text or file_path")
            return json.dumps(result)

        if name == "brain_rag_remember":
            content = (args.get("content") or "").strip()
            if not content:
                return tool_error("content is required")
            category = (args.get("category") or "general").strip()
            result = self._store.remember(content, category=category)
            return json.dumps(result)

        return tool_error(f"Unknown tool: {name}")

    def shutdown(self) -> None:
        if self._store:
            self._store.close()
            self._store = None

    def backup_paths(self) -> List[str]:
        if not self._store:
            return []
        return [str(self._store.db_path)]

    @staticmethod
    def _serialize_hits(hits: List[Dict]) -> List[Dict]:
        out = []
        for h in hits:
            entry = {
                "id": h.get("id"),
                "kind": h.get("kind", "chunk"),
                "content": h.get("content"),
                "score": h.get("score"),
                "source": h.get("source") or h.get("category"),
            }
            if h.get("title"):
                entry["title"] = h["title"]
            out.append(entry)
        return out


_provider = BrainRAGProvider()


def register(ctx):
    ctx.register_memory_provider(_provider)


def get_provider() -> BrainRAGProvider:
    return _provider
