from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.agent_runtime_helpers import restore_primary_runtime, switch_model
from agent.chat_completion_helpers import try_activate_fallback


def _pool(provider: str, entry_id: str, api_key: str | None = None):
    pool = MagicMock()
    pool.provider = provider
    pool.has_credentials.return_value = True
    pool.has_available.return_value = False
    pool.current.return_value = SimpleNamespace(id=entry_id)
    pool.entry_id_for_api_key.side_effect = (
        lambda candidate: entry_id if candidate == api_key else None
    )
    return pool


def _switch_agent(pool):
    agent = MagicMock()
    agent.provider = "provider-a"
    agent.model = "model-a"
    agent.base_url = "https://a.example/v1"
    agent.api_key = "key-a"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent._client_kwargs = {
        "api_key": "key-a",
        "base_url": "https://a.example/v1",
    }
    agent._anthropic_client = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._is_anthropic_oauth = False
    agent._config_context_length = None
    agent._transport_cache = {}
    agent._cached_system_prompt = "cached"
    agent.context_compressor = None
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._rate_limited_until = 0
    agent._credential_pool = pool
    agent._credential_pool_entry_id = "entry-a"
    agent._anthropic_prompt_cache_policy.return_value = (False, False)
    return agent


def test_switch_model_rebinds_entry_id_to_new_runtime_key():
    agent = _switch_agent(_pool("provider-a", "entry-a", "key-a"))
    rebound_pool = _pool("provider-b", "entry-b", "key-b")

    with patch("agent.credential_pool.load_pool", return_value=rebound_pool):
        switch_model(
            agent,
            new_model="model-b",
            new_provider="provider-b",
            api_key="key-b",
            base_url="https://b.example/v1",
            api_mode="chat_completions",
        )

    assert agent._credential_pool is rebound_pool
    assert agent._credential_pool_entry_id == "entry-b"


def test_fallback_attach_rebinds_entry_id_to_fallback_runtime_key():
    agent = _switch_agent(_pool("provider-a", "entry-a", "key-a"))
    agent._fallback_chain = [{"provider": "provider-b", "model": "model-b"}]
    agent._buffer_status = MagicMock()
    agent._is_azure_openai_url.return_value = False
    agent._is_direct_openai_url.return_value = False
    agent._provider_model_requires_responses_api.return_value = False
    agent._replace_primary_openai_client = MagicMock()
    fallback_client = SimpleNamespace(
        api_key="key-b",
        base_url="https://b.example/v1",
        _custom_headers={},
    )
    fallback_pool = _pool("provider-b", "entry-b", "key-b")

    with (
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "model-b"),
        ),
        patch("agent.credential_pool.load_pool", return_value=fallback_pool),
    ):
        assert try_activate_fallback(agent) is True

    assert agent._credential_pool is fallback_pool
    assert agent._credential_pool_entry_id == "entry-b"


def test_restore_primary_clears_fallback_entry_id_when_pool_is_rebound():
    agent = _switch_agent(_pool("provider-b", "entry-b"))
    agent._credential_pool_entry_id = "entry-b"
    agent._fallback_activated = True
    agent._primary_runtime = {
        "model": "model-a",
        "provider": "provider-a",
        "base_url": "https://a.example/v1",
        "api_mode": "chat_completions",
        "api_key": "key-a",
        "client_kwargs": {
            "api_key": "key-a",
            "base_url": "https://a.example/v1",
        },
        "use_prompt_caching": False,
        "use_native_cache_layout": False,
        "compressor_model": "model-a",
        "compressor_context_length": 1000,
        "compressor_base_url": "https://a.example/v1",
        "compressor_api_key": "key-a",
        "compressor_provider": "provider-a",
        "compressor_api_mode": "chat_completions",
    }
    agent.context_compressor = MagicMock()
    agent._create_openai_client.return_value = MagicMock()
    primary_pool = _pool("provider-a", "entry-a")

    with patch("agent.credential_pool.load_pool", return_value=primary_pool):
        assert restore_primary_runtime(agent) is True

    assert agent._credential_pool is primary_pool
    assert agent._credential_pool_entry_id is None
