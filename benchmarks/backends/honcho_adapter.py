"""Benchmark adapter for the Honcho memory plugin (v3 REST API).

This adapter talks directly to a Honcho v3 server via httpx, bypassing the
honcho-ai SDK which targets an older API version.  It creates a workspace and
session, stores messages, and searches via the v3 endpoints.

Environment requirements
------------------------
HONCHO_BASE_URL : str
    The base URL of a running Honcho v3 server (e.g. ``http://localhost:8000``).
HONCHO_API_KEY : str, optional
    API key for authenticated deployments.  Defaults to ``"local"`` when
    HONCHO_BASE_URL points to localhost.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

import httpx

from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import BenchmarkableStore

BACKEND_NAME = "honcho"
BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,
)


class HonchoBenchmarkAdapter(BenchmarkableStore):
    """Adapter exposing a Honcho v3 server through BenchmarkableStore."""

    def __init__(self, **kwargs):
        base_url = kwargs.get("base_url") or os.environ.get("HONCHO_BASE_URL", "")
        api_key = kwargs.get("api_key") or os.environ.get("HONCHO_API_KEY", "")

        if not base_url:
            raise RuntimeError(
                "Honcho benchmark adapter requires HONCHO_BASE_URL to be set "
                "(e.g. http://localhost:8000)."
            )

        if not api_key:
            api_key = "local" if "localhost" in base_url or "127.0.0.1" in base_url else ""
        if not api_key:
            raise RuntimeError(
                "HONCHO_API_KEY is required for non-localhost Honcho servers."
            )

        self._base = base_url.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self._client = httpx.Client(timeout=30.0)
        self._workspace_id = kwargs.get("workspace_id", "hermes-benchmark")
        self._session_id = f"bench-{uuid.uuid4().hex[:8]}"

        self._peer_id = "benchmark-user"
        self._ensure_workspace()
        self._ensure_peer()
        self._ensure_session()

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        url = f"{self._base}/v3{path}"
        resp = self._client.request(method, url, headers=self._headers, **kwargs)
        return resp

    def _ensure_workspace(self) -> None:
        resp = self._request("POST", "/workspaces", json={"id": self._workspace_id})
        # 409 = already exists, that's fine
        if resp.status_code not in (200, 201, 409):
            resp.raise_for_status()

    def _ensure_peer(self) -> None:
        resp = self._request(
            "POST",
            f"/workspaces/{self._workspace_id}/peers",
            json={"id": self._peer_id},
        )
        if resp.status_code not in (200, 201, 409):
            resp.raise_for_status()

    def _ensure_session(self) -> None:
        resp = self._request(
            "POST",
            f"/workspaces/{self._workspace_id}/sessions",
            json={"id": self._session_id},
        )
        if resp.status_code not in (200, 201, 409):
            resp.raise_for_status()

    def store(
        self,
        content: str,
        category: str = "factual",
        scope: str = "global",
        importance: float = 0.5,
    ) -> None:
        del category, scope, importance
        self._request(
            "POST",
            f"/workspaces/{self._workspace_id}/sessions/{self._session_id}/messages",
            json={"messages": [{"role": "user", "content": content, "peer_id": self._peer_id}]},
        )

    def recall(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> list[str]:
        del scope
        # Use session context endpoint which returns all messages,
        # then do client-side keyword matching (the /search endpoint
        # requires the deriver service to build conclusions first).
        resp = self._request(
            "GET",
            f"/workspaces/{self._workspace_id}/sessions/{self._session_id}/context",
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        messages = data.get("messages", [])
        # Score by word overlap with query (simple client-side ranking)
        query_words = set(query.lower().split())
        scored = []
        for msg in messages:
            content = msg.get("content", "")
            content_words = set(content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append((overlap / max(len(query_words), 1), content))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]

    def simulate_time(self, days: float) -> None:
        del days

    def simulate_access(self, content_substring: str) -> None:
        del content_substring

    def consolidate(self) -> None:
        pass

    def get_stats(self) -> dict[str, Any]:
        return {
            "backend": "honcho",
            "workspace_id": self._workspace_id,
            "session_id": self._session_id,
            "configured": True,
        }

    def reset(self) -> None:
        # Delete the session and create a fresh one
        self._request(
            "DELETE",
            f"/workspaces/{self._workspace_id}/sessions/{self._session_id}",
        )
        self._session_id = f"bench-{uuid.uuid4().hex[:8]}"
        self._ensure_session()


BACKEND_CLASS = HonchoBenchmarkAdapter
