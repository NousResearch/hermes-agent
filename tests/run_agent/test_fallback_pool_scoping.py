"""Fallback attaches the endpoint-scoped credential pool, not the bare name.

``_rebind_fallback_credential_pool`` used to ``load_pool(fb_provider)``
directly: for ``provider="custom"`` that key has no rows (no rotation on
fallback), and named customs missed their durable ``providers.<slug>`` key.
It now resolves via ``resolve_runtime_pool_key`` and gates on
``credential_pool_matches_provider``, mirroring the restore path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import _rebind_fallback_credential_pool


def _make_pool(provider):
    pool = MagicMock()
    pool.provider = provider
    pool.has_credentials.return_value = True
    return pool


def _agent():
    agent = MagicMock()
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    return agent


class TestFallbackPoolScoping:
    def test_plain_provider_loads_scoped_key(self):
        agent = _agent()
        pool = _make_pool("openai")
        seen = {}

        def fake_load(key):
            seen["key"] = key
            return pool

        with patch("agent.credential_pool.load_pool", side_effect=fake_load):
            _rebind_fallback_credential_pool(
                agent, "openai", "gpt-5", "https://api.openai.com/v1")
        assert seen["key"] == "openai"
        assert agent._credential_pool is pool

    def test_mismatched_pool_is_not_attached(self):
        agent = _agent()
        pool = _make_pool("someone-else")
        with patch("agent.credential_pool.load_pool", return_value=pool):
            _rebind_fallback_credential_pool(
                agent, "openai", "gpt-5", "https://api.openai.com/v1")
        assert agent._credential_pool is None

    def test_empty_pool_is_not_attached(self):
        agent = _agent()
        pool = _make_pool("openai")
        pool.has_credentials.return_value = False
        with patch("agent.credential_pool.load_pool", return_value=pool):
            _rebind_fallback_credential_pool(
                agent, "openai", "gpt-5", "https://api.openai.com/v1")
        assert agent._credential_pool is None
