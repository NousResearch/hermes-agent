"""Mem0 memory manager — persistent memory via Mem0 Platform.

Handles memory ingestion (fact extraction from conversations),
retrieval (semantic search), and prefetch (background context
loading for system prompt injection).

Mem0 operates on discrete memories — extracted facts scoped to
user_id. No local message buffer or async writer thread needed:
Mem0's client.add() handles async processing server-side.

SDK method signatures (Platform MemoryClient, v2 API):
  client.add(messages, user_id=..., run_id=..., ...)
  client.search(query, version="v2", filters={"OR": [{"user_id": ...}]}, ...)
  client.get_all(version="v2", filters={"OR": [{"user_id": ...}]}, ...)

The v2 API requires entity IDs (user_id, agent_id) inside the
`filters` dict wrapped in OR/AND logical operators.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mem0 import MemoryClient
    from mem0_integration.client import Mem0ClientConfig

logger = logging.getLogger(__name__)


def _build_v2_filters(user_id: str) -> dict[str, Any]:
    """Build v2 API filters for a user.

    Mem0 scopes records per-entity: a record stored with both user_id
    and run_id won't match a filter that only specifies user_id (the
    platform treats missing fields as "must be null").  Using run_id="*"
    (wildcard) matches any non-null run_id, so we OR both cases to
    cover records with and without a run_id.
    """
    return {
        "OR": [
            {"user_id": user_id},
            {"AND": [{"user_id": user_id}, {"run_id": "*"}]},
        ]
    }


class Mem0MemoryManager:
    """Persistent memory layer via Mem0 Platform."""

    def __init__(self, client: MemoryClient, config: Mem0ClientConfig):
        self._client = client
        self._config = config
        self._prefetch_cache: dict[tuple, str] = {}  # (user_id, run_id) -> context
        self._prefetch_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Memory ingestion
    # ------------------------------------------------------------------

    def add(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        run_id: str | None = None,
    ) -> None:
        """Send conversation messages to Mem0 for automatic fact extraction.

        Only user_id is passed — agent_id is not used since these are
        user memories, not assistant memories.
        """
        kwargs: dict[str, Any] = {
            "user_id": user_id,
            "custom_instructions": self._config.custom_instructions,
        }
        if run_id:
            kwargs["run_id"] = run_id

        try:
            self._client.add(messages, **kwargs)
        except Exception as e:
            logger.warning("Mem0 add failed (non-fatal): %s", e)

    def store_fact(self, fact: str, user_id: str) -> dict:
        """Store an explicit fact without LLM extraction (infer=False).

        Used by mem0_conclude tool. Adds source metadata to distinguish
        agent-written facts from user utterances.

        Warning: infer=False skips duplicate detection per Mem0 docs.
        """
        try:
            return self._client.add(
                [{"role": "user", "content": fact}],
                user_id=user_id,
                infer=False,
                metadata={"source": "hermes_conclude"},
            )
        except Exception as e:
            logger.warning("Mem0 store_fact failed: %s", e)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Memory retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        rerank: bool = False,
    ) -> list[dict]:
        """Semantic search over user's memories using v2 API."""
        kwargs: dict[str, Any] = {
            "version": "v2",
            "filters": _build_v2_filters(user_id),
            "keyword_search": self._config.keyword_search,
            "top_k": min(top_k, 50),
        }
        if rerank:
            kwargs["rerank"] = True

        try:
            result = self._client.search(query, **kwargs)
            if isinstance(result, list):
                return result
            return result.get("results", result.get("memories", []))
        except Exception as e:
            logger.warning("Mem0 search failed: %s", e)
            return []

    def get_profile(self, user_id: str, page_size: int = 20) -> list[dict]:
        """Get all stored memories for a user using v2 API."""
        try:
            result = self._client.get_all(
                version="v2",
                filters=_build_v2_filters(user_id),
                page_size=page_size,
            )
            if isinstance(result, list):
                return result
            return result.get("results", result.get("memories", []))
        except Exception as e:
            logger.warning("Mem0 get_profile failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Prefetch for system prompt injection
    # ------------------------------------------------------------------

    def prefetch(
        self,
        user_id: str,
        query: str,
        run_id: str | None = None,
    ) -> None:
        """Fire background search, cache result for next turn's pop_prefetch()."""
        cache_key = (user_id, run_id)

        def _run():
            try:
                result = self._client.search(
                    query or "What do you know about this user?",
                    version="v2",
                    filters=_build_v2_filters(user_id),
                    keyword_search=self._config.keyword_search,
                    top_k=10,
                )
                memories = result if isinstance(result, list) else result.get("results", result.get("memories", []))

                if not memories:
                    return

                context = self._format_prefetch(memories)
                with self._prefetch_lock:
                    self._prefetch_cache[cache_key] = context

            except Exception as e:
                logger.debug("Mem0 prefetch failed (non-fatal): %s", e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def pop_prefetch(self, user_id: str, run_id: str | None = None) -> str | None:
        """Consume cached prefetch result. Returns formatted markdown or None."""
        cache_key = (user_id, run_id)
        with self._prefetch_lock:
            return self._prefetch_cache.pop(cache_key, None)

    def _format_prefetch(self, memories: list[dict]) -> str:
        """Format prefetch results as markdown for system prompt injection."""
        parts = [
            "# Mem0 Memory (persistent cross-session context)",
            "Use this to answer questions about the user and prior sessions.",
            "Do not call tools to look up information already present here.",
            "",
        ]

        if memories:
            parts.append("## User Memories")
            for m in memories:
                parts.append(f"- {m.get('memory', '')}")
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """No-op — Mem0's client.add() handles async processing server-side."""
        pass
