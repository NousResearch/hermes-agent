"""Benchmark adapter for the holographic memory plugin.

This adapter loads the holographic plugin implementation from the main branch's
plugin source when it is not present in the current benchmark branch.

The goal is to benchmark the real plugin behavior as honestly as possible while
keeping capability declarations conservative.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import BenchmarkableStore

BACKEND_NAME = "holographic"
BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,
)

_MODULE_CACHE: dict[str, Any] = {}

_STOP_WORDS = frozenset({
    "what", "which", "who", "where", "when", "why", "how",
    "does", "did", "is", "are", "was", "were", "the", "a", "an",
    "to", "of", "for", "in", "on", "at", "by", "from", "with",
    "we", "our", "you", "your", "i", "me", "my",
})


def _git_show(pathspec: str) -> str:
    result = subprocess.run(
        ["git", "show", pathspec],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _load_holographic_modules() -> tuple[Any, Any]:
    cached = _MODULE_CACHE.get("modules")
    if cached:
        return cached

    try:
        store_mod = importlib.import_module("plugins.memory.holographic.store")
        retrieval_mod = importlib.import_module("plugins.memory.holographic.retrieval")
        _MODULE_CACHE["modules"] = (store_mod, retrieval_mod)
        return store_mod, retrieval_mod
    except Exception:
        pass

    temp_root = Path(tempfile.mkdtemp(prefix="benchmark-holographic-"))
    package_name = "benchmark_holographic_runtime"
    package_dir = temp_root / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("")

    files = {
        "holographic.py": "main:plugins/memory/holographic/holographic.py",
        "store.py": "main:plugins/memory/holographic/store.py",
        "retrieval.py": "main:plugins/memory/holographic/retrieval.py",
    }
    for name, source in files.items():
        (package_dir / name).write_text(_git_show(source))

    importlib.invalidate_caches()
    sys.path.insert(0, str(temp_root))
    store_mod = importlib.import_module(f"{package_name}.store")
    retrieval_mod = importlib.import_module(f"{package_name}.retrieval")
    _MODULE_CACHE["modules"] = (store_mod, retrieval_mod)
    _MODULE_CACHE["temp_root"] = temp_root
    return store_mod, retrieval_mod


class HolographicBenchmarkAdapter(BenchmarkableStore):
    """Adapter exposing the holographic plugin through BenchmarkableStore."""

    def __init__(self, **kwargs):
        self._tempdir = Path(tempfile.mkdtemp(prefix="holographic-bench-db-"))
        self._db_path = self._tempdir / "memory_store.db"
        self._default_trust = float(kwargs.get("default_trust", 0.5))
        self._hrr_dim = int(kwargs.get("hrr_dim", 1024))
        self._store = None
        self._retriever = None
        self._init_store()

    def _init_store(self) -> None:
        store_mod, retrieval_mod = _load_holographic_modules()
        self._store = store_mod.MemoryStore(
            db_path=str(self._db_path),
            default_trust=self._default_trust,
            hrr_dim=self._hrr_dim,
        )
        self._retriever = retrieval_mod.FactRetriever(
            store=self._store,
            temporal_decay_half_life=0,
            hrr_weight=0.3,
            hrr_dim=self._hrr_dim,
        )

    def _fts5_or_query(self, query: str) -> str:
        tokens = []
        cleaned = []
        for raw in query.lower().replace("?", " ").split():
            token = "".join(ch for ch in raw if ch.isalnum())
            if len(token) <= 1 or token in _STOP_WORDS:
                continue
            cleaned.append(token)
        for token in cleaned[:8]:
            tokens.append(f"{token}*")
        return " OR ".join(tokens)

    def store(self, content: str, category: str = "factual",
              scope: str = "global", importance: float = 0.5) -> None:
        del scope, importance
        cat = category if category in {"user_pref", "project", "tool", "general", "factual"} else "general"
        if cat == "factual":
            cat = "general"
        self._store.add_fact(content, category=cat)

    def recall(self, query: str, top_k: int = 10,
               scope: Optional[str] = None) -> list[str]:
        del scope
        results = self._retriever.search(query, min_trust=0.0, limit=top_k)
        if not results:
            fts_query = self._fts5_or_query(query)
            if fts_query:
                results = self._store.search_facts(fts_query, min_trust=0.0, limit=top_k)
        return [r["content"] for r in results[:top_k]]

    def simulate_time(self, days: float) -> None:
        del days
        # Holographic has no native time-simulation hook in benchmark terms.
        return None

    def simulate_access(self, content_substring: str) -> None:
        del content_substring
        # No dedicated rehearsal API; leave as no-op.
        return None

    def consolidate(self) -> None:
        # No consolidation cycle in the plugin's native design.
        return None

    def get_stats(self) -> dict[str, Any]:
        row = self._store._conn.execute("SELECT COUNT(*) AS c FROM facts").fetchone()
        return {
            "fact_count": int(row[0]) if row else 0,
            "db_path": str(self._db_path),
        }

    def reset(self) -> None:
        try:
            if self._store is not None:
                self._store.close()
        except Exception:
            pass
        if self._db_path.exists():
            self._db_path.unlink()
        wal = Path(str(self._db_path) + "-wal")
        shm = Path(str(self._db_path) + "-shm")
        if wal.exists():
            wal.unlink()
        if shm.exists():
            shm.unlink()
        self._init_store()


BACKEND_CLASS = HolographicBenchmarkAdapter
