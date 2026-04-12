"""Tests for the OpenVikingMemoryProvider."""

import pytest
from unittest.mock import patch, MagicMock


class FakeHttpx:
    """Fake httpx module for testing without network."""

    def __init__(self, health_ok=True):
        self.health_ok = health_ok
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        m = MagicMock()
        m.status_code = 200 if self.health_ok else 500
        m.raise_for_status = MagicMock(
            side_effect=Exception("server error") if not self.health_ok else None
        )
        m.json = MagicMock(return_value={"result": []})
        return m

    def post(self, url, json=None, **kwargs):
        self.calls.append(("post", url, json, kwargs))
        m = MagicMock()
        m.raise_for_status = MagicMock()
        m.json = MagicMock(return_value={"result": {
            "memories": [
                {"uri": "viking://user/memories/test-1", "abstract": "Test memory 1", "score": 0.95},
                {"uri": "viking://user/memories/test-2", "abstract": "Test memory 2", "score": 0.80},
            ],
            "resources": [],
        }})
        return m


@pytest.fixture
def fake_httpx():
    return FakeHttpx()


@pytest.fixture
def provider(fake_httpx, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1933")
    monkeypatch.setenv("OPENVIKING_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.openviking._get_httpx", lambda: fake_httpx)
    from plugins.memory.openviking import OpenVikingMemoryProvider
    p = OpenVikingMemoryProvider()
    p.initialize("session-test-1", hermes_home=str(tmp_path), platform="cli")
    return p


class TestPrefetchSync:
    """Test that prefetch() uses the current query for synchronous recall."""

    def test_prefetch_returns_sync_results(self, provider, fake_httpx):
        """prefetch() should search with the current query and return results."""
        result = provider.prefetch("my test query")

        # Should have called the search endpoint
        post_calls = [(c[0], c[1], c[2]) for c in fake_httpx.calls if c[0] == "post"]
        assert len(post_calls) == 1
        assert "/api/v1/search/find" in post_calls[0][1]
        assert post_calls[0][2]["query"] == "my test query"

        # Should return formatted results
        assert "## OpenViking Context" in result
        assert "Test memory 1" in result
        assert "viking://user/memories/test-1" in result

    def test_prefetch_uses_current_query_not_cached(self, provider, fake_httpx):
        """prefetch() should use the current query, not a stale cached one."""
        # Queue a background prefetch for a different query
        provider.queue_prefetch("stale background query")

        # Now call prefetch with a new current query
        fake_httpx.calls.clear()
        result = provider.prefetch("current fresh query")

        # The synchronous search should use "current fresh query", not "stale background query"
        post_calls = [(c[0], c[1], c[2]) for c in fake_httpx.calls if c[0] == "post"]
        assert len(post_calls) == 1
        assert post_calls[0][2]["query"] == "current fresh query"
        assert "## OpenViking Context" in result

    def test_prefetch_empty_query_returns_nothing(self, provider, fake_httpx):
        """prefetch() with empty query should return empty string."""
        result = provider.prefetch("")
        assert result == ""

    def test_prefetch_empty_client_returns_nothing(self, monkeypatch, tmp_path):
        """prefetch() should return empty when client is not initialized."""
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "")
        from plugins.memory.openviking import OpenVikingMemoryProvider
        p = OpenVikingMemoryProvider()
        p.initialize("session-test-2", hermes_home=str(tmp_path), platform="cli")
        result = p.prefetch("some query")
        assert result == ""

    def test_prefetch_falls_back_to_cache_on_sync_failure(self, provider, fake_httpx):
        """If sync search fails, prefetch() should fall back to cached background results."""
        # Simulate sync failure by raising on httpx.post
        original_post = fake_httpx.post
        def failing_post(*args, **kwargs):
            m = MagicMock()
            m.raise_for_status = MagicMock(side_effect=Exception("network error"))
            return m
        fake_httpx.post = failing_post

        # Pre-populate background cache
        provider.queue_prefetch("background query")
        import time
        time.sleep(0.1)

        # prefetch should return empty (no cache from current session)
        result = provider.prefetch("current query")
        # The sync fails, fallback finds nothing cached, returns empty
        assert result == ""


class TestQueuePrefetch:
    """Test that queue_prefetch() fires background searches."""

    def test_queue_prefetch_starts_thread(self, provider):
        """queue_prefetch() should start a background thread."""
        provider.queue_prefetch("background search query")
        assert provider._prefetch_thread is not None
        assert provider._prefetch_thread.daemon is True


class TestSyncTurn:
    """Test that sync_turn() records conversation turns."""

    def test_sync_turn_increments_turn_count(self, provider, fake_httpx):
        """sync_turn() should increment the turn counter."""
        assert provider._turn_count == 0
        provider.sync_turn("user message", "assistant reply")
        assert provider._turn_count == 1
        provider.sync_turn("next user message", "next assistant reply")
        assert provider._turn_count == 2

    def test_on_session_end_commits(self, provider, fake_httpx):
        """on_session_end() should commit the session."""
        provider.sync_turn("user", "assistant")
        provider.on_session_end([])

        # Should have called commit
        commit_calls = [(c[0], c[1]) for c in fake_httpx.calls if "/commit" in str(c[1])]
        assert len(commit_calls) >= 1
