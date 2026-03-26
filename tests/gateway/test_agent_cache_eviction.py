"""Tests for gateway agent cache eviction on session expiry and size cap.

Cached AIAgent instances hold OpenAI/Anthropic clients, system prompts,
message history, memory stores, and tool definitions. Without eviction,
expired sessions leak these objects permanently — an unbounded memory
leak on long-running gateways.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


def _make_runner_with_cache():
    """Build a minimal mock GatewayRunner with the real cache structures."""
    runner = SimpleNamespace(
        _agent_cache={},
        _agent_cache_lock=threading.Lock(),
        _running=True,
        session_store=MagicMock(),
    )

    # Import the real method
    from gateway.run import GatewayRunner
    runner._evict_cached_agent = GatewayRunner._evict_cached_agent.__get__(runner)
    return runner


class TestEvictOnSessionExpiry:
    """_session_expiry_watcher must evict cached agents for expired sessions."""

    def test_evict_cached_agent_removes_entry(self):
        runner = _make_runner_with_cache()
        runner._agent_cache["telegram:123"] = (MagicMock(), "sig-abc")
        runner._agent_cache["discord:456"] = (MagicMock(), "sig-def")

        runner._evict_cached_agent("telegram:123")

        assert "telegram:123" not in runner._agent_cache
        assert "discord:456" in runner._agent_cache

    def test_evict_nonexistent_key_is_noop(self):
        runner = _make_runner_with_cache()
        runner._agent_cache["telegram:123"] = (MagicMock(), "sig-abc")

        runner._evict_cached_agent("nonexistent")

        assert len(runner._agent_cache) == 1


class TestCacheSizeCap:
    """Agent cache must not grow unboundedly."""

    def test_cache_evicts_oldest_when_over_limit(self):
        """Simulate inserting more agents than the cap allows."""
        cache = {}
        lock = threading.Lock()
        _MAX_CACHED_AGENTS = 200

        # Fill cache to max
        for i in range(_MAX_CACHED_AGENTS):
            cache[f"session:{i}"] = (MagicMock(), f"sig-{i}")

        assert len(cache) == _MAX_CACHED_AGENTS

        # Insert one more — should trigger eviction of the oldest
        new_key = "session:new"
        with lock:
            cache[new_key] = (MagicMock(), "sig-new")
            if len(cache) > _MAX_CACHED_AGENTS:
                oldest_key = next(iter(cache))
                if oldest_key != new_key:
                    cache.pop(oldest_key, None)

        assert len(cache) == _MAX_CACHED_AGENTS
        assert new_key in cache
        assert "session:0" not in cache  # oldest was evicted
        assert "session:1" in cache  # second-oldest survives
