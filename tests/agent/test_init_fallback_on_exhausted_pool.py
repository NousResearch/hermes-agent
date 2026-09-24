"""Regression test for #17929: AIAgent.__init__ should try fallback_model
when primary provider credentials are exhausted."""
import pytest
from unittest.mock import patch, MagicMock
from run_agent import AIAgent


def _make_tool_defs():
    return [{"type": "function", "function": {"name": "web_search",
             "description": "search", "parameters": {"type": "object", "properties": {}}}}]


def _mock_client(api_key="fb-key-1234567890", base_url="https://fb.example.com/v1"):
    c = MagicMock()
    c.api_key = api_key
    c.base_url = base_url
    c._default_headers = None
    return c


def test_init_tries_fallback_when_primary_returns_none():
    """When resolve_provider_client returns None for primary but succeeds for
    a fallback entry, __init__ should NOT raise RuntimeError."""
    fb = _mock_client()

    def fake_resolve(provider, model=None, raw_codex=False,
                     explicit_base_url=None, explicit_api_key=None):
        if provider == "tencent-token-plan":
            return fb, "kimi2.5"
        return None, None  # primary exhausted

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):

        agent = AIAgent(
            provider="alibaba-coding-plan",
            model="qwen3.6-plus",
            api_key=None,
            base_url=None,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{"provider": "tencent-token-plan", "model": "kimi2.5"}],
        )
        assert agent.provider == "tencent-token-plan"
        assert agent.model == "kimi2.5"
        assert agent._fallback_activated is True


def test_init_raises_when_no_fallback_configured():
    """When primary returns None and no fallback is set, should raise."""
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):

        with pytest.raises(RuntimeError, match="no API key was found"):
            AIAgent(
                provider="alibaba-coding-plan",
                model="qwen3.6-plus",
                api_key=None,
                base_url=None,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=None,
            )


def test_init_tries_fallback_when_openrouter_pool_exhausted():
    """Regression: an exhausted ``openrouter`` pool (single credential entry
    in 429-cooldown) must still reach fallback_providers at init.

    Before the fix, the init-time fallback block was nested inside the
    ``_explicit not in {auto, openrouter, custom}`` guard, so the default
    openrouter setup skipped the chain entirely and raised the misleading
    "No LLM provider configured" RuntimeError (2026-08-23 outage).
    """
    fb = _mock_client(api_key="local-key-1234567890",
                      base_url="http://127.0.0.1:11434/v1")

    def fake_resolve(provider, model=None, raw_codex=False,
                     explicit_base_url=None, explicit_api_key=None):
        if provider == "custom" and explicit_base_url:
            return fb, "qwen3.5:4b"
        return None, None  # openrouter pool exhausted

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):

        agent = AIAgent(
            provider="openrouter",
            model="z-ai/glm-5.2:free",
            api_key=None,
            base_url=None,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[
                {"provider": "openrouter", "model": "poolside/laguna-s-2.1:free"},
                {"provider": "custom", "model": "qwen3.5:4b",
                 "base_url": "http://127.0.0.1:11434/v1"},
            ],
        )
        assert agent.provider == "custom"
        assert agent.model == "qwen3.5:4b"
        assert agent._fallback_activated is True


def test_init_openrouter_exhausted_without_chain_keeps_generic_error():
    """openrouter exhausted + no usable fallback keeps the generic
    'No LLM provider configured' error (not the named-provider one)."""
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):

        with pytest.raises(RuntimeError, match="No LLM provider configured"):
            AIAgent(
                provider="openrouter",
                model="z-ai/glm-5.2:free",
                api_key=None,
                base_url=None,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=[{"provider": "openrouter",
                                 "model": "poolside/laguna-s-2.1:free"}],
            )


def test_init_fallback_moves_requested_provider_and_capabilities_to_the_fallback():
    """The entry now serving owns requested_provider and capabilities, as in a runtime fallback.
    Leaving the failed primary's there lets its declaration outlive the switch — and a child's
    model pin, which reads requested_provider, borrow it (review 6 of the OAuth-proxy work)."""
    fb = _mock_client()
    fb.capabilities = {"anthropic_oauth_proxy": False}

    def fake_resolve(provider, model=None, raw_codex=False, explicit_base_url=None, explicit_api_key=None):
        return (fb, "fb-model") if provider == "relay2" else (None, None)

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        agent = AIAgent(
            provider="relay", requested_provider="relay", capabilities={"anthropic_oauth_proxy": True},
            model="m", api_key=None, base_url=None, quiet_mode=True, skip_context_files=True,
            skip_memory=True, fallback_model=[{"provider": "relay2", "model": "fb-model"}],
        )
    assert agent._fallback_activated is True
    assert (agent.provider, agent.requested_provider) == ("relay2", "relay2")
    assert agent.capabilities == {"anthropic_oauth_proxy": False}


@pytest.mark.parametrize(
    "transport,declared",
    [
        ("anthropic_messages", {"anthropic_oauth_proxy": True, "vision": True}),
        ("anthropic_messages", {"anthropic_oauth_proxy": False, "vision": True}),
        # An OpenAI-wire relay: the routed client carries no ``capabilities`` of its own.
        ("chat_completions", {"anthropic_oauth_proxy": True, "vision": True}),
    ],
)
def test_init_fallback_takes_the_routes_declared_capabilities(tmp_path, monkeypatch, transport, declared):
    """Real config, real ``resolve_provider_client`` for the fallback: the session takes the map the
    entry declares (not only the OAuth bit the Anthropic wrapper happens to carry), and a
    query-bearing endpoint keeps its tenant query on the client the session goes on with."""
    import yaml

    import agent.auxiliary_client as aux

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_RELAY2_KEY", "relay2-key")
    for stale in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN"):
        monkeypatch.delenv(stale, raising=False)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"providers": {"relay2": {
        "api": "https://relay2.example.com/t?team=a", "key_env": "TEST_RELAY2_KEY",
        "transport": transport, "capabilities": declared,
    }}}), encoding="utf-8")
    real_resolve = aux.resolve_provider_client

    def primary_exhausted(provider, model=None, **kwargs):
        return (None, None) if provider == "alibaba-coding-plan" else real_resolve(provider, model, **kwargs)

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=primary_exhausted), \
         patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("model_tools.check_toolset_requirements", return_value={}):
        agent = AIAgent(
            provider="alibaba-coding-plan", model="qwen3.6-plus", api_key=None, base_url=None,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=[{"provider": "relay2", "model": "fb-model"}],
        )
    assert agent._fallback_activated is True
    assert agent.capabilities == declared
    if transport == "chat_completions":
        assert agent._client_kwargs.get("default_query") == {"team": "a"}
