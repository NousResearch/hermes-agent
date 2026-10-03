"""#110831: a fallback destination that resolved without any API key on a paid, remote
provider with an empty credential pool must be skipped instead of committed — the committed
switch traded a "model unavailable" 400 for an endless 401 loop on a pool with nothing to
rotate. Virtual/external-process overlays, local endpoints, pool credentials, and
entry-declared or client-held keys still activate.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent

# Assembled at runtime so no credential-shaped literal sits in source (scanner guard).
_TEST_KEY = "test-" + "key-12345678"
_FALLBACK_KEY = "fallback-" + "key-1234"


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key=_TEST_KEY, base_url="https://my-llm.example.com/v1", provider="custom",
            quiet_mode=True, skip_context_files=True, skip_memory=True, fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _fb_client(base_url, client_key=_FALLBACK_KEY):
    fb_client = MagicMock()
    fb_client.api_key = client_key
    fb_client.base_url = base_url
    return fb_client


def _pool(has_credentials):
    pool = MagicMock()
    pool.has_credentials.return_value = has_credentials
    return pool


def test_unauthenticated_destination_is_skipped_instead_of_committed():
    agent = _make_agent(fallback_model={"provider": "opencode", "model": "deepseek-v4-flash"})
    agent.provider, agent.model = "zai", "glm-5.3"
    with (
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(_fb_client("https://opencode.ai/zen/v1", client_key=""), None)),
        patch("agent.credential_pool.load_pool", return_value=_pool(False)),
    ):
        assert agent._try_activate_fallback() is False
    assert (agent.provider, agent.model) == ("zai", "glm-5.3")
    assert agent._unavailable_fallback_keys == {("opencode", "deepseek-v4-flash", "")}


def test_destination_with_pool_credentials_still_activates():
    agent = _make_agent(fallback_model={"provider": "opencode", "model": "deepseek-v4-flash"})
    with (
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(_fb_client("https://opencode.ai/zen/v1", client_key=""), None)),
        patch("agent.credential_pool.load_pool", return_value=_pool(True)),
    ):
        assert agent._try_activate_fallback() is True
    assert (agent.provider, agent.model) == ("opencode", "deepseek-v4-flash")


def test_destination_with_a_usable_client_key_still_activates():
    agent = _make_agent(fallback_model={"provider": "opencode", "model": "deepseek-v4-flash"})
    with (
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(_fb_client("https://opencode.ai/zen/v1"), None)),
        patch("agent.credential_pool.load_pool", return_value=_pool(False)),
    ):
        assert agent._try_activate_fallback() is True
    assert (agent.provider, agent.model) == ("opencode", "deepseek-v4-flash")


def test_virtual_auth_overlay_destination_activates_without_credentials():
    agent = _make_agent(fallback_model={"provider": "moa", "model": "moa-1"})
    with (
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(_fb_client("https://moa.example.com/v1", client_key=""), None)),
        patch("agent.credential_pool.load_pool", return_value=_pool(False)),
    ):
        assert agent._try_activate_fallback() is True
    assert (agent.provider, agent.model) == ("moa", "moa-1")


def test_local_endpoint_destination_activates_without_credentials():
    agent = _make_agent(fallback_model={"provider": "ollama", "model": "llama4"})
    with (
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(_fb_client("http://127.0.0.1:11434/v1", client_key=""), None)),
        patch("agent.credential_pool.load_pool", return_value=_pool(False)),
    ):
        assert agent._try_activate_fallback() is True
    assert (agent.provider, agent.model) == ("ollama", "llama4")


def test_entry_declared_api_key_activates_without_pool():
    agent = _make_agent(fallback_model={
        "provider": "opencode", "model": "deepseek-v4-flash", "api_key": "inline-" + "key-5678"})
    with (
        patch("agent.auxiliary_client.resolve_provider_client",
              return_value=(_fb_client("https://opencode.ai/zen/v1", client_key=""), None)),
        patch("agent.credential_pool.load_pool", return_value=_pool(False)),
    ):
        assert agent._try_activate_fallback() is True
    assert (agent.provider, agent.model) == ("opencode", "deepseek-v4-flash")


def test_pool_read_failure_does_not_block_the_fallback():
    from agent.chat_completion_helpers import _fallback_destination_can_authenticate
    with patch("agent.credential_pool.load_pool", side_effect=RuntimeError("pool unreadable")):
        assert _fallback_destination_can_authenticate("zai", "https://api.z.ai/api/paas/v4/") is True
