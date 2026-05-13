"""Tests for credential pool loading during /model switch.

Verifies that switch_model() loads the credential pool for the new
provider and includes it in the _primary_runtime snapshot so it
survives _restore_primary_runtime().

Issue: #25273
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(provider="zai", model="glm-5.1"):
    """Create a minimal AIAgent-like object for testing switch_model."""
    from run_agent import AIAgent
    # Use __new__ to skip __init__'s heavy setup
    agent = object.__new__(AIAgent)
    agent.model = model
    agent.provider = provider
    agent.base_url = "https://api.example.com/v1"
    agent.api_mode = "chat_completions"
    agent.api_key = "test-key"
    agent._credential_pool = None
    agent._client_kwargs = {}
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent._transport_cache = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._config_context_length = None
    agent._cached_system_prompt = None
    agent.verbose_logging = False
    agent.quiet_mode = True
    # Minimal context compressor mock — switch_model and _restore_primary_runtime
    # call update_model() unconditionally.
    _cc = MagicMock()
    _cc.model = model
    _cc.base_url = agent.base_url
    _cc.api_key = "test-key"
    _cc.provider = provider
    _cc.context_length = 128000
    _cc.threshold_tokens = 64000
    agent.context_compressor = _cc
    agent._primary_runtime = {
        "model": model,
        "provider": provider,
        "base_url": agent.base_url,
        "api_mode": "chat_completions",
        "api_key": "test-key",
        "client_kwargs": {},
        "use_prompt_caching": False,
        "use_native_cache_layout": False,
        "credential_pool": None,
        "compressor_model": model,
        "compressor_base_url": agent.base_url,
        "compressor_api_key": "test-key",
        "compressor_provider": provider,
        "compressor_context_length": 128000,
        "compressor_threshold_tokens": 64000,
    }
    return agent


def _switch_mocks():
    """Common mocks needed for switch_model to run without side effects."""
    return pytest.fixture(autouse=False)


class FakePool:
    """Minimal CredentialPool stub."""

    def __init__(self, has_creds=True, entries_count=3):
        self._has_creds = has_creds
        self._entries_count = entries_count

    def has_credentials(self):
        return self._has_creds

    def entries(self):
        return [MagicMock() for _ in range(self._entries_count)]


def _patch_switch_deps():
    """Return a context manager with all mocks needed for switch_model."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None))
    stack.enter_context(patch("hermes_cli.providers.determine_api_mode", return_value="codex_responses"))
    stack.enter_context(patch("agent.model_metadata.get_model_context_length", return_value=128000))
    return stack


# ---------------------------------------------------------------------------
# Tests: switch_model loads pool for new provider
# ---------------------------------------------------------------------------

class TestSwitchModelCredentialPool:

    def test_switch_loads_pool_for_new_provider(self):
        """switch_model should call load_pool for the new provider."""
        agent = _make_agent(provider="zai")
        assert agent._credential_pool is None

        fake_pool = FakePool(has_creds=True, entries_count=3)

        with _patch_switch_deps() as stack, \
             patch("agent.credential_pool.load_pool", return_value=fake_pool) as mock_load, \
             patch.object(agent, "_create_openai_client", return_value=MagicMock()):
            agent.switch_model(
                new_model="gpt-5.5",
                new_provider="openai-codex",
                api_key="new-key",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses",
            )

        mock_load.assert_called_with("openai-codex")
        assert agent._credential_pool is fake_pool

    def test_switch_sets_pool_none_when_no_credentials(self):
        """switch_model should set pool to None when new provider has no pool."""
        agent = _make_agent(provider="zai")
        fake_pool = FakePool(has_creds=False, entries_count=0)

        with _patch_switch_deps() as stack, \
             patch("agent.credential_pool.load_pool", return_value=fake_pool), \
             patch.object(agent, "_create_openai_client", return_value=MagicMock()):
            agent.switch_model(
                new_model="gpt-5.5",
                new_provider="openai-codex",
                api_key="key",
                base_url="https://api.openai.com/v1",
                api_mode="chat_completions",
            )

        assert agent._credential_pool is None

    def test_switch_handles_load_pool_exception(self):
        """switch_model should set pool to None if load_pool raises."""
        agent = _make_agent(provider="zai")

        with _patch_switch_deps() as stack, \
             patch("agent.credential_pool.load_pool", side_effect=RuntimeError("boom")), \
             patch.object(agent, "_create_openai_client", return_value=MagicMock()):
            agent.switch_model(
                new_model="gpt-5.5",
                new_provider="openai-codex",
                api_key="key",
                base_url="https://api.openai.com/v1",
                api_mode="chat_completions",
            )

        assert agent._credential_pool is None

    def test_switch_includes_pool_in_primary_runtime(self):
        """_primary_runtime snapshot should include credential_pool."""
        agent = _make_agent(provider="zai")
        fake_pool = FakePool(has_creds=True, entries_count=3)

        with _patch_switch_deps() as stack, \
             patch("agent.credential_pool.load_pool", return_value=fake_pool), \
             patch.object(agent, "_create_openai_client", return_value=MagicMock()):
            agent.switch_model(
                new_model="gpt-5.5",
                new_provider="openai-codex",
                api_key="new-key",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses",
            )

        assert "credential_pool" in agent._primary_runtime
        assert agent._primary_runtime["credential_pool"] is fake_pool

    def test_switch_same_provider_keeps_pool(self):
        """Switching models within the same provider keeps the pool."""
        fake_pool = FakePool(has_creds=True, entries_count=2)
        agent = _make_agent(provider="openai-codex")
        agent._credential_pool = fake_pool

        with _patch_switch_deps() as stack, \
             patch("agent.credential_pool.load_pool", return_value=fake_pool) as mock_load, \
             patch.object(agent, "_create_openai_client", return_value=MagicMock()):
            agent.switch_model(
                new_model="o3",
                new_provider="openai-codex",
                api_key="key",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses",
            )

        mock_load.assert_called_with("openai-codex")
        assert agent._credential_pool is fake_pool

    def test_restore_primary_preserves_pool_from_snapshot(self):
        """_restore_primary_runtime should restore pool from _primary_runtime."""
        agent = _make_agent(provider="openai-codex")
        fake_pool = FakePool(has_creds=True, entries_count=3)
        agent._credential_pool = fake_pool

        # Simulate: switch_model wrote the snapshot with the pool
        agent._primary_runtime = {
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "api_key": "some-key",
            "client_kwargs": {"api_key": "some-key", "base_url": "https://chatgpt.com/backend-api/codex"},
            "use_prompt_caching": False,
            "use_native_cache_layout": False,
            "credential_pool": fake_pool,
            "compressor_model": "gpt-5.5",
            "compressor_base_url": "https://chatgpt.com/backend-api/codex",
            "compressor_api_key": "some-key",
            "compressor_provider": "openai-codex",
            "compressor_context_length": 128000,
            "compressor_threshold_tokens": 64000,
        }
        # Simulate fallback was activated
        agent._fallback_activated = True

        with patch.object(agent, "_apply_client_headers_for_base_url"), \
             patch.object(agent, "_replace_primary_openai_client"):
            result = agent._restore_primary_runtime()

        assert result is True
        assert agent._credential_pool is fake_pool

    def test_restore_primary_keeps_pool_when_snapshot_has_none(self):
        """_restore_primary_runtime should not clear pool if snapshot has None pool."""
        agent = _make_agent(provider="openai-codex")
        fake_pool = FakePool(has_creds=True, entries_count=3)
        agent._credential_pool = fake_pool

        # Simulate: snapshot with credential_pool=None (provider had pool at
        # construction time, snapshot captured it as None)
        agent._primary_runtime = {
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "api_key": "some-key",
            "client_kwargs": {"api_key": "some-key", "base_url": "https://chatgpt.com/backend-api/codex"},
            "use_prompt_caching": False,
            "use_native_cache_layout": False,
            "credential_pool": None,
            "compressor_model": "gpt-5.5",
            "compressor_base_url": "https://chatgpt.com/backend-api/codex",
            "compressor_api_key": "some-key",
            "compressor_provider": "openai-codex",
            "compressor_context_length": 128000,
            "compressor_threshold_tokens": 64000,
        }
        agent._fallback_activated = True

        with patch.object(agent, "_apply_client_headers_for_base_url"), \
             patch.object(agent, "_replace_primary_openai_client"):
            result = agent._restore_primary_runtime()

        assert result is True
        # Pool should remain — the guard only restores when snapshot pool is not None
        assert agent._credential_pool is fake_pool

    def test_restore_primary_keeps_pool_when_snapshot_predates_field(self):
        """_restore_primary_runtime should not touch pool if snapshot lacks credential_pool."""
        agent = _make_agent(provider="openai-codex")
        fake_pool = FakePool(has_creds=True, entries_count=3)
        agent._credential_pool = fake_pool

        # Simulate: snapshot without credential_pool key (older format)
        agent._primary_runtime = {
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "api_key": "some-key",
            "client_kwargs": {"api_key": "some-key", "base_url": "https://chatgpt.com/backend-api/codex"},
            "use_prompt_caching": False,
            "use_native_cache_layout": False,
            # No credential_pool key — predates the fix
            "compressor_model": "gpt-5.5",
            "compressor_base_url": "https://chatgpt.com/backend-api/codex",
            "compressor_api_key": "some-key",
            "compressor_provider": "openai-codex",
            "compressor_context_length": 128000,
            "compressor_threshold_tokens": 64000,
        }
        agent._fallback_activated = True

        with patch.object(agent, "_apply_client_headers_for_base_url"), \
             patch.object(agent, "_replace_primary_openai_client"):
            result = agent._restore_primary_runtime()

        assert result is True
        # Pool should remain unchanged (construction-time pool)
        assert agent._credential_pool is fake_pool
