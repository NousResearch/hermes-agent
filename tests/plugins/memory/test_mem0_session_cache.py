"""Tests for Mem0 session-level retrieval cache.

Covers #25971: prefetch() must reuse the first turn's search result for all
subsequent turns in the same session, so the injected context text stays
constant and Anthropic prefix cache entries are not invalidated.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(search_results=None, config_overrides=None):
    """Return an initialised Mem0MemoryProvider with a fake backend."""
    fake_backend = MagicMock()
    results_payload = search_results if search_results is not None else [
        {"memory": "user likes dark mode"}
    ]
    fake_backend.search.return_value = results_payload

    config = {
        "api_key": "test-key",
        "user_id": "test-user",
        "agent_id": "hermes",
        "rerank": False,
        "retrieve_per_session": True,
        "keyword_search": False,
    }
    if config_overrides:
        config.update(config_overrides)

    from plugins.memory.mem0 import Mem0MemoryProvider

    provider = Mem0MemoryProvider()
    with patch("plugins.memory.mem0._load_config", return_value=config), \
         patch.object(Mem0MemoryProvider, "_create_backend", return_value=fake_backend):
        provider.initialize("sess-A")

    return provider, fake_backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMem0SessionCache:

    def test_prefetch_caches_result_for_session(self):
        """client.search must be called only once when session_id is the same."""
        provider, fake_client = _make_provider()

        provider.prefetch("tell me about the user", session_id="sess-A")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=2)

        provider.prefetch("tell me about the user again", session_id="sess-A")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=2)

        assert fake_client.search.call_count == 1, (
            "Expected 1 search call, got " + str(fake_client.search.call_count)
        )

    def test_prefetch_returns_cached_on_second_call(self):
        """Both prefetch() calls must return identical text when the cache is active."""
        provider, fake_client = _make_provider()

        result1 = provider.prefetch("query one", session_id="sess-A")

        # Second turn, cache hit on the new query for the same session.
        result2 = provider.prefetch("query two", session_id="sess-A")

        assert result1 == result2
        assert "dark mode" in result1

    def test_session_cache_invalidated_on_session_switch(self):
        """After on_session_switch with a new id, search must be called again."""
        provider, fake_client = _make_provider()

        provider.prefetch("first query", session_id="sess-A")

        # Switch to a new session, evicting sess-A entry
        provider.on_session_switch("sess-B", parent_session_id="sess-A")

        # Second session must hit the API again.
        provider.prefetch("second query", session_id="sess-B")

        assert fake_client.search.call_count == 2, (
            "Expected 2 search calls after session switch, got " + str(fake_client.search.call_count)
        )

    def test_session_cache_disabled_via_config(self):
        """When retrieve_per_session=False, search must be called on every turn."""
        provider, fake_client = _make_provider(
            config_overrides={"retrieve_per_session": False}
        )
        provider._session_cache_enabled = False

        provider.prefetch("first", session_id="sess-A")
        provider.prefetch("second", session_id="sess-A")

        assert fake_client.search.call_count == 2, (
            "Expected 2 search calls when cache disabled, got " + str(fake_client.search.call_count)
        )

    def test_session_cache_cleared_on_reset(self):
        """on_session_switch with reset=True must wipe the entire cache."""
        provider, fake_client = _make_provider()

        # Populate the cache
        provider.prefetch("query", session_id="sess-A")
        assert "sess-A" in provider._session_cache

        # Hard reset
        provider.on_session_switch("sess-new", reset=True)

        assert provider._session_cache == {}
        assert provider._current_session_id == "sess-new"
