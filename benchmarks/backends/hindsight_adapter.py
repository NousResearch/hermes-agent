"""Benchmark adapter for the Hindsight memory backend.

Supports both cloud mode (HINDSIGHT_API_KEY) and local mode
(HINDSIGHT_BASE_URL pointing to a local hindsight-api server).

The hindsight_client SDK uses aiohttp internally and owns its own event
loop.  To avoid conflicts, all SDK calls are dispatched to a single
dedicated worker thread that holds the client instance.
"""

from __future__ import annotations

import os
import queue
import threading
import uuid
from typing import Any, Optional

from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import BenchmarkableStore

BACKEND_NAME = "hindsight"
BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,
)


class _HindsightWorker:
    """Single-threaded worker that owns the Hindsight client."""

    def __init__(self, client_kwargs: dict):
        self._client_kwargs = client_kwargs
        self._client = None
        self._cmd_q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        from hindsight_client import Hindsight
        self._client = Hindsight(**self._client_kwargs)
        while True:
            fn, result_q = self._cmd_q.get()
            if fn is None:
                break
            try:
                result_q.put(("ok", fn(self._client)))
            except Exception as exc:
                result_q.put(("err", exc))

    def call(self, fn, timeout: float = 120.0):
        result_q: queue.Queue = queue.Queue(maxsize=1)
        self._cmd_q.put((fn, result_q))
        kind, value = result_q.get(timeout=timeout)
        if kind == "err":
            raise value
        return value


class HindsightBenchmarkAdapter(BenchmarkableStore):
    """Adapter exposing Hindsight through BenchmarkableStore."""

    def __init__(self, **kwargs):
        api_key = os.environ.get("HINDSIGHT_API_KEY")
        base_url = os.environ.get("HINDSIGHT_BASE_URL", "")

        if not api_key and not base_url:
            raise RuntimeError(
                "Set HINDSIGHT_API_KEY (cloud) or HINDSIGHT_BASE_URL "
                "(local, e.g. http://localhost:8888) before running "
                "Hindsight benchmarks."
            )

        client_kwargs = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key

        self._client_kwargs = client_kwargs
        self._worker: Optional[_HindsightWorker] = None
        self._base_url: str = base_url
        self._bank_id: str = kwargs.get("bank_id", "benchmark-bank")
        self._budget: str = kwargs.get("budget", "mid")

    def _ensure_worker(self) -> _HindsightWorker:
        if self._worker is None:
            self._worker = _HindsightWorker(self._client_kwargs)
        return self._worker

    def store(
        self,
        content: str,
        category: str = "factual",
        scope: str = "global",
        importance: float = 0.5,
    ) -> None:
        del category, scope, importance
        bank_id = self._bank_id
        self._ensure_worker().call(
            lambda c: c.retain(bank_id=bank_id, content=content, context="benchmark")
        )

    def recall(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> list[str]:
        del scope
        bank_id = self._bank_id
        budget = self._budget
        resp = self._ensure_worker().call(
            lambda c: c.recall(bank_id=bank_id, query=query, budget=budget)
        )
        return [r.text for r in resp.results[:top_k]]

    def simulate_time(self, days: float) -> None:
        del days

    def simulate_access(self, content_substring: str) -> None:
        del content_substring

    def consolidate(self) -> None:
        pass

    def get_stats(self) -> dict[str, Any]:
        return {
            "backend": "hindsight",
            "bank_id": self._bank_id,
            "base_url": self._base_url,
            "configured": True,
        }

    def reset(self) -> None:
        """Rotate to a fresh bank so scenarios cannot see prior data.

        store() and recall() both use self._bank_id, so a no-op reset()
        would let facts from one scenario leak into the next.  We rotate
        to a unique bank name instead of deleting the old one (Hindsight
        may not expose a delete API in all deployments).
        """
        self._bank_id = f"benchmark-bank-{uuid.uuid4().hex[:8]}"
        # Drop the cached worker so the next call creates a fresh client
        # bound to the new bank.
        if self._worker is not None:
            self._worker = None


BACKEND_CLASS = HindsightBenchmarkAdapter
