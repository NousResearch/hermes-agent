"""Performance regression tests for MemoryManager.

These tests ensure multi-provider operations stay within reasonable
time budgets to prevent regressions as the feature grows.
"""

import os
import time

import pytest
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class QuickProvider(MemoryProvider):
    """Minimal provider for performance testing."""

    def __init__(self, name, tool_count=10):
        self._name = name
        self._tools = [
            {"name": f"{name}_tool_{i}", "description": f"T{i}", "parameters": {}}
            for i in range(tool_count)
        ]

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def initialize(self, **kwargs):
        pass

    def system_prompt_block(self):
        return ""

    def prefetch(self, query, **kwargs):
        return ""

    def get_tool_schemas(self):
        return self._tools

    def handle_tool_call(self, tool_name, args, **kwargs):
        return "{}"

    def shutdown(self):
        pass


class TestMemoryManagerPerformance:
    """Performance regression tests."""

    def test_add_provider_single_under_1ms(self):
        """Adding a single provider should complete in under 1ms."""
        mgr = MemoryManager()
        p = QuickProvider("perf", tool_count=10)

        start = time.perf_counter()
        mgr.add_provider(p)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.001, f"add_provider took {elapsed*1000:.3f}ms, expected <1ms"

    @pytest.mark.skipif(os.getenv("CI"), reason="timing-sensitive, skipped in CI")
    def test_add_ten_providers_under_5ms(self):
        """Adding 10 providers (100 tools total) should complete in under 5ms."""
        mgr = MemoryManager()
        providers = [QuickProvider(f"perf{i}", tool_count=10) for i in range(10)]

        start = time.perf_counter()
        for p in providers:
            mgr.add_provider(p)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.005, f"10x add_provider took {elapsed*1000:.3f}ms, expected <5ms"

    def test_get_all_tool_schemas_under_1ms(self):
        """Collecting schemas from 10 providers should complete in under 1ms."""
        mgr = MemoryManager()
        for i in range(10):
            mgr.add_provider(QuickProvider(f"perf{i}", tool_count=10))

        start = time.perf_counter()
        schemas = mgr.get_all_tool_schemas()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.001, f"get_all_tool_schemas took {elapsed*1000:.3f}ms, expected <1ms"
        assert len(schemas) == 100

    def test_remove_provider_under_1ms(self):
        """Removing a provider with 50 tools should complete in under 1ms."""
        mgr = MemoryManager()
        p = QuickProvider("perf", tool_count=50)
        mgr.add_provider(p)

        start = time.perf_counter()
        mgr.remove_provider("perf")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.001, f"remove_provider took {elapsed*1000:.3f}ms, expected <1ms"

    def test_concurrent_add_still_fast(self):
        """Concurrent additions should not cause pathological slowdown."""
        import threading
        mgr = MemoryManager()
        providers = [QuickProvider(f"ct{i}", tool_count=5) for i in range(20)]

        def adder(p):
            mgr.add_provider(p)

        start = time.perf_counter()
        threads = [threading.Thread(target=adder, args=(p,)) for p in providers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        # 20 providers, should complete in under 50ms even with lock contention
        assert elapsed < 0.05, f"20 concurrent adds took {elapsed*1000:.3f}ms, expected <50ms"
        assert len(mgr.providers) == 20
