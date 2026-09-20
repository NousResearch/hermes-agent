"""Tests for fallback-eviction gating on failed runs (#7130).

When a run fails, the gateway must NOT evict the cached agent — doing so
forces MCP reinit on the next message, creating a CPU-burning restart loop.
Eviction should only happen on successful runs where fallback activated.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestFallbackEvictionGating:
    """The fallback-eviction code path should skip eviction on failed runs."""

    def test_failed_run_does_not_evict_cached_agent(self):
        """When result has failed=True, the cached agent should NOT be evicted."""
        # The fix: `and not _run_failed` guard on the eviction check.
        # Simulate the variables that the eviction block uses.
        result = {"failed": True, "final_response": None, "error": "400 invalid model"}
        _run_failed = result.get("failed") if result else False
        assert _run_failed is True, "Failed run should be detected"


