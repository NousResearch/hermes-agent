"""Benchmark adapter for RetainDB cloud memory service.

RetainDB is a hosted memory-as-a-service API.  All operations are performed
over HTTPS using a Bearer token.  Because the project state lives in the cloud
there is no meaningful way to reset or time-travel within a benchmark run; those
operations are therefore no-ops.

Required environment variables:
    RETAINDB_API_KEY   Bearer token issued by the RetainDB dashboard.

Optional environment variables:
    RETAINDB_BASE_URL  Override the default API base URL
                       (default: https://api.retaindb.com).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import BenchmarkableStore

BACKEND_NAME = "retaindb"
BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,
)


class RetainDBBenchmarkAdapter(BenchmarkableStore):
    """Adapter exposing RetainDB cloud memory through BenchmarkableStore."""

    def __init__(self, **kwargs):  # noqa: ANN003
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
        self._project: str = kwargs.get("project", "benchmark")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _api(self, method: str, path: str, **kwargs: Any) -> Any:
        """Make an authenticated HTTP request and return the parsed JSON body.

        Args:
            method: HTTP verb (e.g. "POST").
            path:   API path starting with "/" (e.g. "/v1/remember").
            **kwargs: Extra arguments forwarded to ``requests.request``.

        Returns:
            Parsed JSON response as a Python object.

        Raises:
            requests.HTTPError: If the server returns a 4xx or 5xx status.
        """
        url = f"{self._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Allow callers to extend or override headers if ever needed.
        caller_headers = kwargs.pop("headers", {})
        headers.update(caller_headers)

        response = requests.request(
            method,
            url,
            headers=headers,
            timeout=30,
            **kwargs,
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
        """Persist a piece of content in the RetainDB project."""
        del scope  # RetainDB scoping is handled at the project level.
        self._api(
            "POST",
            "/v1/remember",
            json={
                "project": self._project,
                "user_id": "benchmark-user",
                "content": content,
                "memory_type": "fact",
                "importance": importance,
            },
        )

    def recall(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> list[str]:
        """Retrieve the most relevant memories for *query* from RetainDB."""
        del scope  # RetainDB scoping is handled at the project level.
        data = self._api(
            "POST",
            "/v1/search",
            json={
                "project": self._project,
                "user_id": "benchmark-user",
                "query": query,
                "top_k": top_k,
            },
        )
        return [r["content"] for r in data.get("results", [])[:top_k]]

    def simulate_time(self, days: float) -> None:
        """No-op: RetainDB is a cloud service with no time-simulation hook."""
        del days
        return None

    def simulate_access(self, content_substring: str) -> None:
        """No-op: RetainDB has no explicit rehearsal / access-bump API."""
        del content_substring
        return None

    def consolidate(self) -> None:
        """No-op: RetainDB manages consolidation server-side."""
        return None

    def get_stats(self) -> dict[str, Any]:
        """Return lightweight diagnostic information about this adapter."""
        return {
            "backend": "retaindb",
            "project": self._project,
            "configured": True,
        }

    def reset(self) -> None:
        """No-op: resetting a shared cloud project during benchmarks is unsafe."""
        # Intentionally left as a no-op.  Wiping a remote project could
        # destroy data belonging to other users or benchmark runs sharing the
        # same credentials.
        return None


BACKEND_CLASS = RetainDBBenchmarkAdapter
