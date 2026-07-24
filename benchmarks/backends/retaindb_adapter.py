"""Benchmark adapter for RetainDB cloud memory service.

RetainDB is a hosted memory-as-a-service API.  All operations are performed
over HTTPS using a Bearer token.

Required environment variables:
    RETAINDB_API_KEY   Bearer token issued by the RetainDB dashboard.

Optional environment variables:
    RETAINDB_BASE_URL  Override the default API base URL
                       (default: https://api.retaindb.com).
    RETAINDB_PROJECT   Project identifier (default: "benchmark").
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

import requests

from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import BenchmarkableStore

BACKEND_NAME = "retaindb"
BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,
)


class RetainDBBenchmarkAdapter(BenchmarkableStore):
    """Adapter exposing RetainDB cloud memory through BenchmarkableStore.

    Reset isolation is achieved by rotating to a fresh user_id UUID on each
    reset() call — no deletion needed, no cross-scenario contamination.
    """

    def __init__(self, **kwargs: Any) -> None:
        api_key = os.environ.get("RETAINDB_API_KEY")
        if not api_key:
            raise RuntimeError(
                "RETAINDB_API_KEY environment variable is not set. "
                "Obtain a token from the RetainDB dashboard and export it "
                "before running the benchmark."
            )
        self._api_key: str = api_key
        self._base_url: str = os.environ.get(
            "RETAINDB_BASE_URL", "https://api.retaindb.com"
        ).rstrip("/")
        self._project: str = kwargs.get(
            "project", os.environ.get("RETAINDB_PROJECT", "benchmark")
        )
        self._user_id: str = f"bench-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _api(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(kwargs.pop("headers", {}))
        response = requests.request(
            method, url, headers=headers, timeout=30, **kwargs
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # BenchmarkableStore interface
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        category: str = "factual",
        scope: str = "global",
        importance: float = 0.5,
    ) -> None:
        """Persist a piece of content in RetainDB."""
        del scope
        self._api(
            "POST",
            "/v1/memory",
            json={
                "project": self._project,
                "user_id": self._user_id,
                "content": content,
                "memory_type": "other",
                "write_mode": "sync",
            },
        )

    def recall(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> list[str]:
        """Retrieve the most relevant memories for *query* from RetainDB."""
        del scope
        data = self._api(
            "POST",
            "/v1/memory/search",
            json={
                "project": self._project,
                "user_id": self._user_id,
                "query": query,
                "top_k": top_k,
                "include_pending": True,
                "profile": "fast",
            },
        )
        return [
            r["memory"]["content"]
            for r in data.get("results", [])[:top_k]
            if r.get("memory", {}).get("content")
        ]

    def reset(self) -> None:
        """Rotate to a fresh user_id — guarantees clean state for next scenario."""
        self._user_id = f"bench-{uuid.uuid4().hex[:12]}"

    def simulate_time(self, days: float) -> None:
        """No-op: RetainDB has no time-simulation hook."""

    def simulate_access(self, content_substring: str) -> None:
        """No-op: RetainDB has no explicit rehearsal API."""

    def consolidate(self) -> None:
        """No-op: RetainDB manages consolidation server-side."""

    def get_stats(self) -> dict[str, Any]:
        return {
            "backend": "retaindb",
            "project": self._project,
            "user_id": self._user_id,
        }


BACKEND_CLASS = RetainDBBenchmarkAdapter
