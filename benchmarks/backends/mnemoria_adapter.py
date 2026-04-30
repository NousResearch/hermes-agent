"""Benchmark adapter for the standalone Mnemoria Python package."""

from __future__ import annotations

from typing import Any, Optional

from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import BenchmarkableStore

BACKEND_NAME = "mnemoria"
BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,
    time_simulation=True,
    access_rehearsal=True,
    consolidation=True,
    scopes=True,
    typed_facts=True,
    supersession=True,
    reward_learning=True,
    exploration=True,
    forgetting=True,
)


class MnemoriaBenchmarkAdapter(BenchmarkableStore):
    """Expose MnemoriaStore through the benchmark interface.

    The adapter intentionally uses an in-memory SQLite database and Mnemoria's
    virtual clock so benchmark scenarios are isolated, fast, and deterministic.
    """

    def __init__(self, **kwargs: Any) -> None:
        try:
            from mnemoria import MnemoriaConfig, MnemoriaStore
        except ImportError as exc:  # pragma: no cover - exercised by environment setup
            raise RuntimeError(
                "Mnemoria backend requires the standalone package. "
                "Install with: python -m pip install -e /path/to/mnemoria"
            ) from exc

        profile = kwargs.get("profile", "balanced")
        embedding_model = kwargs.get("embedding_model", "tfidf")
        contradiction_llm_model = kwargs.get("contradiction_llm_model")

        config = MnemoriaConfig.from_profile(profile)
        config.db_path = ":memory:"
        config.embedding_model = embedding_model
        config.contradiction_llm_model = contradiction_llm_model
        # Scale/capacity scenarios intentionally store hundreds of facts. Gauge
        # pressure is a production feature, but it archives benchmark needles.
        config.enable_pressure = False

        self._store = MnemoriaStore(config=config, db_path=":memory:")
        self._store.enable_virtual_clock()

    @staticmethod
    def _format_result(scored_fact: Any) -> str:
        fact = scored_fact.fact
        content = fact.content
        target = getattr(fact, "target", "general") or "general"
        if target != "general":
            target_words = target.replace(".", " ").replace("_", " ")
            return f"{target_words}: {content}"
        return content

    def store(self, content: str, category: str = "factual",
              scope: str = "global", importance: float = 0.5) -> None:
        self._store.store(content, category=category, scope=scope, importance=importance)

    def recall(self, query: str, top_k: int = 10,
               scope: Optional[str] = None) -> list[str]:
        results = self._store.recall(query=query, top_k=top_k, scope=scope)
        return [self._format_result(result) for result in results[:top_k]]

    def recall_with_ids(self, query: str, top_k: int = 10,
                        scope: Optional[str] = None) -> list[tuple[str, str]]:
        raw = self._store.recall_with_ids(query=query, top_k=top_k, scope=scope)
        # Benchmark Suite G expects (content, memory_id) tuples; Mnemoria's API
        # returns (memory_id, content, score).
        return [(content, memory_id) for memory_id, content, _score in raw]

    def simulate_time(self, days: float) -> None:
        self._store.simulate_time(days)

    def simulate_access(self, content_substring: str) -> None:
        self._store.simulate_access(content_substring)

    def consolidate(self) -> None:
        self._store.consolidate()

    def reward_memory(self, memory_id: str, signal: float) -> None:
        self._store.reward_memory(memory_id, signal)

    def explore(self, query: str, top_k: int = 20,
                scope: Optional[str] = None) -> list[str]:
        results = self._store.explore(query=query, top_k=top_k, scope=scope)
        return [self._format_result(result) for result in results[:top_k]]

    def forget(self, content_substring: str | None = None, scope: str | None = None) -> None:
        if not content_substring:
            return
        self._store.forget_by_content(content_substring, reason="benchmark privacy_forgetting")

    def get_stats(self) -> dict[str, Any]:
        return self._store.get_stats()

    def reset(self) -> None:
        self._store.reset()


BACKEND_CLASS = MnemoriaBenchmarkAdapter
