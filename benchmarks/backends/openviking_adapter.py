"""Benchmark adapter for the OpenViking memory service.

OpenViking exposes a REST API for session-based memory storage and retrieval.
Memories are stored by posting messages to a session then committing that
session to trigger extraction.  Recall is a separate semantic search endpoint.

Important: memories are NOT searchable until the session is committed.  This
adapter therefore calls commit automatically after every store() call so that
benchmark queries can find freshly stored facts.  An explicit consolidate()
call is also available for callers that batch-store and then consolidate once.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Optional

import httpx

from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import BenchmarkableStore

logger = logging.getLogger(__name__)

BACKEND_NAME = "openviking"
BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,
    turn_sync=True,
)

_DEFAULT_ENDPOINT = "http://127.0.0.1:1933"
_DEFAULT_TIMEOUT = 30.0  # seconds


# ---------------------------------------------------------------------------
# Low-level HTTP client
# ---------------------------------------------------------------------------

class _VikingClient:
    """Thin synchronous wrapper around the OpenViking REST API.

    All methods raise httpx.HTTPStatusError on non-2xx responses so callers
    can handle errors at the adapter level.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Session message posting
    # ------------------------------------------------------------------

    def post_message(self, session_id: str, content: str) -> dict[str, Any]:
        """POST a user message to a session.

        Endpoint: POST /api/v1/sessions/{session_id}/messages
        Body: {"role": "user", "parts": [{"type": "text", "text": content}]}
        """
        url = f"/api/v1/sessions/{session_id}/messages"
        payload = {
            "role": "user",
            "parts": [{"type": "text", "text": content}],
        }
        response = self._client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def commit_session(self, session_id: str) -> dict[str, Any]:
        """POST a commit request so the service extracts memories from messages.

        Endpoint: POST /api/v1/sessions/{session_id}/commit
        """
        url = f"/api/v1/sessions/{session_id}/commit"
        response = self._client.post(url)
        response.raise_for_status()
        return response.json()

    def wait_for_task(self, task_id: str, timeout: float = 120.0,
                      poll_interval: float = 2.0) -> bool:
        """Poll a task until it completes or times out.

        Returns True if the task completed successfully, False on timeout/error.
        """
        import time
        start = time.time()
        while time.time() - start < timeout:
            try:
                url = f"/api/v1/tasks/{task_id}"
                response = self._client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("result", {}).get("status", "")
                    if status in ("completed", "done", "success"):
                        return True
                    if status in ("failed", "error"):
                        logger.warning("Task %s failed: %s", task_id, data)
                        return False
            except Exception:  # noqa: BLE001
                pass
            time.sleep(poll_interval)
        logger.warning("Task %s timed out after %.0fs", task_id, timeout)
        return False

    # ------------------------------------------------------------------
    # Combined store helper (message + commit + wait)
    # ------------------------------------------------------------------

    def store(self, session_id: str, content: str) -> None:
        """Post a message, commit, and wait for extraction to complete.

        OpenViking extraction is async — memories aren't searchable until the
        commit task finishes.  This method polls the task status to ensure
        facts are available before the next benchmark query.
        """
        self.post_message(session_id, content)
        result = self.commit_session(session_id)
        # Wait for extraction to complete
        task_id = result.get("result", {}).get("task_id")
        if task_id:
            self.wait_for_task(task_id, timeout=60.0, poll_interval=1.0)

    def store_deferred(self, session_id: str, content: str) -> str | None:
        """Post a message and commit WITHOUT waiting for extraction.

        Returns the task_id so the caller can batch-wait later.
        Used by the benchmark adapter to avoid per-fact extraction waits.
        """
        self.post_message(session_id, content)
        result = self.commit_session(session_id)
        return result.get("result", {}).get("task_id")

    def wait_all_tasks(self, task_ids: list[str],
                       timeout: float = 300.0) -> None:
        """Wait for all extraction tasks to complete (shared timeout)."""
        import time as _time
        deadline = _time.monotonic() + timeout
        for task_id in task_ids:
            if not task_id:
                continue
            remaining = max(0.1, deadline - _time.monotonic())
            self.wait_for_task(task_id, timeout=remaining, poll_interval=2.0)

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def search(self, session_id: str, query: str,
               top_k: int = 10) -> list[dict[str, Any]]:
        """POST a semantic search and return the list of memory dicts.

        Endpoint: POST /api/v1/search/find
        Body: {"query": query, "top_k": top_k, "mode": "fast"}
        Returns: result["result"]["memories"] — each item has an "abstract" key.
        """
        url = "/api/v1/search/find"
        payload = {
            "query": query,
            "top_k": top_k,
            "mode": "fast",
            "session_id": session_id,
        }
        response = self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        try:
            return data["result"]["memories"]
        except (KeyError, TypeError) as exc:
            logger.warning("Unexpected search response structure: %s (%s)", data, exc)
            return []

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def delete_session(self, session_id: str) -> None:
        """Attempt to delete a session; silently ignores 404."""
        url = f"/api/v1/sessions/{session_id}"
        try:
            response = self._client.delete(url)
            if response.status_code not in (200, 204, 404):
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.debug("Could not delete session %s: %s", session_id, exc)

    def close(self) -> None:
        """Release the underlying httpx connection pool."""
        self._client.close()


# ---------------------------------------------------------------------------
# BenchmarkableStore adapter
# ---------------------------------------------------------------------------

class OpenVikingBenchmarkAdapter(BenchmarkableStore):
    """Adapter exposing the OpenViking REST service through BenchmarkableStore.

    Environment variables
    ---------------------
    OPENVIKING_ENDPOINT : str, optional
        Base URL of the OpenViking service (default: http://127.0.0.1:1933).
        If not set the adapter falls back to the default localhost address.
    OPENVIKING_API_KEY : str, optional
        Bearer token for authenticated deployments.  Omit for local instances.
    """

    def __init__(self, **kwargs) -> None:
        endpoint = os.environ.get("OPENVIKING_ENDPOINT", _DEFAULT_ENDPOINT)
        api_key = os.environ.get("OPENVIKING_API_KEY")
        timeout = float(kwargs.get("timeout", _DEFAULT_TIMEOUT))

        self._endpoint = endpoint.rstrip("/")
        self._client = _VikingClient(
            base_url=self._endpoint,
            api_key=api_key,
            timeout=timeout,
        )
        self._session_id: str = self._new_session_id()
        self._pending_tasks: list[str] = []
        logger.debug(
            "OpenVikingBenchmarkAdapter initialised — endpoint=%s session=%s",
            self._endpoint,
            self._session_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _new_session_id() -> str:
        return f"bench-{uuid.uuid4().hex}"

    # ------------------------------------------------------------------
    # BenchmarkableStore interface
    # ------------------------------------------------------------------

    def store(self, content: str, category: str = "factual",
              scope: str = "global", importance: float = 0.5) -> None:
        """Store content via deferred commit (extraction runs in background).

        OpenViking's extraction pipeline runs two sequential LLM calls per
        commit (archive summary + memory extraction). Waiting per-store makes
        benchmarking infeasible. Instead we fire-and-forget the commit and
        collect task IDs. The pending tasks are flushed before the first
        recall() call.
        """
        del category, scope, importance
        try:
            task_id = self._client.store_deferred(self._session_id, content)
            if task_id:
                self._pending_tasks.append(task_id)
        except httpx.HTTPStatusError as exc:
            logger.error(
                "OpenViking store failed (session=%s): %s",
                self._session_id, exc,
            )
            raise
        except httpx.RequestError as exc:
            logger.error(
                "OpenViking connection error during store (session=%s): %s",
                self._session_id, exc,
            )
            raise

    def _flush_pending(self) -> None:
        """Wait for all deferred extraction tasks to complete."""
        if self._pending_tasks:
            logger.debug("Flushing %d pending extraction tasks", len(self._pending_tasks))
            self._client.wait_all_tasks(self._pending_tasks, timeout=300.0)
            self._pending_tasks.clear()

    def recall(self, query: str, top_k: int = 10,
               scope: Optional[str] = None) -> list[str]:
        """Recall memories matching query via semantic search.

        Flushes any pending extraction tasks first so stored facts are
        searchable. scope is accepted for interface compliance but ignored.
        Returns a list of abstract strings for the top-k results.
        """
        del scope
        self._flush_pending()
        try:
            memories = self._client.search(self._session_id, query, top_k=top_k)
        except httpx.HTTPStatusError as exc:
            logger.error(
                "OpenViking recall failed (session=%s): %s",
                self._session_id, exc,
            )
            return []
        except httpx.RequestError as exc:
            logger.error(
                "OpenViking connection error during recall (session=%s): %s",
                self._session_id, exc,
            )
            return []

        results: list[str] = []
        for mem in memories[:top_k]:
            abstract = mem.get("abstract")
            if abstract and isinstance(abstract, str):
                results.append(abstract)
            else:
                logger.debug("Memory entry missing 'abstract' field: %s", mem)
        return results

    def simulate_time(self, days: float) -> None:
        """No-op: OpenViking has no time-simulation API."""
        del days
        return None

    def simulate_access(self, content_substring: str) -> None:
        """No-op: OpenViking has no rehearsal API."""
        del content_substring
        return None

    def consolidate(self) -> None:
        """Commit the current session to flush any uncommitted messages."""
        try:
            self._client.commit_session(self._session_id)
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "OpenViking consolidate failed (session=%s): %s",
                self._session_id, exc,
            )
        except httpx.RequestError as exc:
            logger.warning(
                "OpenViking connection error during consolidate (session=%s): %s",
                self._session_id, exc,
            )

    def get_stats(self) -> dict[str, Any]:
        return {
            "backend": "openviking",
            "endpoint": self._endpoint,
            "session": self._session_id,
        }

    def reset(self) -> None:
        """Discard the current session and start a fresh one.

        Attempts to delete the old session from the server so it does not
        pollute future recall results.  Failure to delete is logged but does
        not raise so callers always get a clean adapter state.
        """
        self._pending_tasks.clear()
        old_session_id = self._session_id
        self._session_id = self._new_session_id()
        logger.debug(
            "OpenVikingBenchmarkAdapter reset — old=%s new=%s",
            old_session_id,
            self._session_id,
        )
        try:
            self._client.delete_session(old_session_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not delete old session %s: %s", old_session_id, exc)

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


BACKEND_CLASS = OpenVikingBenchmarkAdapter
