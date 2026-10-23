"""Benchmark adapter for the Mem0 cloud memory service.

This adapter wraps the Mem0 MemoryClient so it can be evaluated by the
hermes-agent benchmark harness through the standard BenchmarkableStore
interface.

Mem0 is a managed, server-side memory service.  Because all state lives
in Mem0's cloud, several lifecycle operations that make sense only for
local stores (time simulation, access rehearsal, consolidation, and
blanket reset) are implemented as explicit no-ops.  The capability
declaration reflects this honestly so the benchmark runner skips the
corresponding test categories rather than counting them as failures.

Environment requirements
------------------------
MEM0_API_KEY : str
    A valid Mem0 API key.  The adapter raises RuntimeError at construction
    time if the variable is absent or empty so failures surface immediately
    rather than on the first API call.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import BenchmarkableStore

# ---------------------------------------------------------------------------
# Module-level exports consumed by the benchmark runner
# ---------------------------------------------------------------------------

BACKEND_NAME = "mem0"

BACKEND_CAPABILITIES = BackendCapabilities(
    universal_store_recall=True,
    time_simulation=False,
    access_rehearsal=False,
    consolidation=False,
    scopes=False,
    typed_facts=False,
    supersession=False,
    reward_learning=False,
    exploration=False,
    turn_sync=False,
    precompress_hook=False,
    session_end_hook=False,
    delegation_hook=False,
)

class Mem0BenchmarkAdapter(BenchmarkableStore):
    """Adapter that exposes Mem0 MemoryClient through BenchmarkableStore.

    All memories written during a benchmark run are scoped to the fixed
    user ID ``benchmark-user``.  This keeps benchmark data isolated from
    any production user IDs that might share the same API key.

    Note that ``reset()`` is intentionally a no-op: the Mem0 API does not
    expose a bulk-delete endpoint suitable for use in an automated benchmark
    loop.  Runners that require a clean slate between runs should use a
    dedicated API key for benchmarking.
    """

    def __init__(self, **kwargs) -> None:
        """Initialise the adapter and verify the Mem0 API key is available.

        Args:
            **kwargs: Accepted but ignored; present for interface compatibility
                with other adapters that take optional tuning parameters.

        Raises:
            RuntimeError: If the ``MEM0_API_KEY`` environment variable is not
                set or is empty.
            ImportError: If the ``mem0`` package is not installed in the
                current environment.
        """
        api_key = os.environ.get("MEM0_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "MEM0_API_KEY environment variable is not set. "
                "Export a valid Mem0 API key before running the benchmark."
            )

        try:
            from mem0 import MemoryClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'mem0' package is required for this adapter. "
                "Install it with: pip install mem0ai"
            ) from exc

        self._client = MemoryClient(api_key=api_key)
        self._user_id: str = f"bench-{__import__('uuid').uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # BenchmarkableStore implementation
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        category: str = "factual",
        scope: str = "global",
        importance: float = 0.5,
    ) -> None:
        """Store a memory in Mem0.

        Args:
            content: The text to remember.
            category: Semantic category (e.g. ``"factual"``, ``"user_pref"``).
                      Passed through for interface compatibility; Mem0 does not
                      natively filter by category.
            scope: Scope tag (``"global"``, ``"session"``, …).  Ignored because
                   Mem0 scoping is handled by its own user/agent/run hierarchy.
            importance: Salience weight in [0, 1].  Ignored; Mem0 manages
                        relevance internally.
        """
        del category, scope, importance  # unused by this backend
        self._client.add(
            [{"role": "user", "content": content}],
            user_id=self._user_id,
            infer=False,
        )

    def recall(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> List[str]:
        """Retrieve memories relevant to *query* from Mem0.

        Args:
            query: Natural-language search string.
            top_k: Maximum number of results to return.
            scope: Optional scope filter.  Ignored because Mem0 scoping is
                   managed by user/agent/run identifiers, not arbitrary tags.

        Returns:
            A list of memory text strings, ranked by Mem0's relevance score,
            truncated to at most *top_k* entries.
        """
        del scope  # unused by this backend
        response = self._client.search(
            query,
            user_id=self._user_id,
            top_k=top_k,
            filters={"user_id": self._user_id},
        )
        # The response may be a dict with "results" key or a list directly
        results = response.get("results", response) if isinstance(response, dict) else response
        return [r["memory"] for r in results[:top_k]]

    def simulate_time(self, days: float) -> None:
        """No-op — Mem0 is a server-side service with no local clock.

        Args:
            days: Number of days to advance (ignored).
        """
        del days
        return None

    def simulate_access(self, content_substring: str) -> None:
        """No-op — Mem0 has no dedicated rehearsal or access-boost API.

        Args:
            content_substring: Substring identifying the memory to rehearse
                               (ignored).
        """
        del content_substring
        return None

    def consolidate(self) -> None:
        """No-op — Mem0 consolidates memories automatically on the server."""
        return None

    def get_stats(self) -> dict[str, Any]:
        """Return basic adapter statistics.

        Returns:
            A dictionary with backend identification and configuration status.
            Extended runtime statistics (e.g. memory count) are not included
            because they would require an extra API round-trip that may not be
            available in all Mem0 plans.
        """
        return {
            "backend": BACKEND_NAME,
            "configured": True,
            "user_id": self._user_id,
        }

    def reset(self) -> None:
        """Rotate to a fresh user_id — guarantees clean state for next scenario.

        Rotating the user_id avoids cross-scenario contamination without
        requiring a bulk-delete round-trip (which is slow and may be async).
        """
        import uuid
        self._user_id = f"bench-{uuid.uuid4().hex[:12]}"


# Module-level alias consumed by the benchmark runner's dynamic loader.
BACKEND_CLASS = Mem0BenchmarkAdapter
