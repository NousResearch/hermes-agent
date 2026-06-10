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
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=MagicMock()):

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
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=MagicMock()):

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


def test_init_raises_graceful_error_when_api_key_empty_chat_completions():
    """When resolve_provider_client returns a client with an empty api_key,
    the chat_completions guard must raise RuntimeError with the standardized
    message before the OpenAI SDK is instantiated."""
    def fake_resolve(*args, **kwargs):
        c = MagicMock()
        c.api_key = ""
        c.base_url = "https://openrouter.ai/api/v1"
        c._custom_headers = None
        return c, "gpt-4o"

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve), \
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=MagicMock()):

        with pytest.raises(RuntimeError, match=r"no API key \(param empty, env vars unset\)"):
            AIAgent(
                provider="openrouter",
                model="gpt-4o",
                api_key="",
                base_url=None,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )


def test_init_raises_graceful_error_when_api_key_empty_anthropic():
    """When api_key is explicitly empty for the anthropic provider, the
    anthropic_messages guard must raise RuntimeError with the standardized
    message before build_anthropic_client is called."""
    with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value=None), \
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()):

        with pytest.raises(RuntimeError, match=r"no API key \(param empty, env vars unset\)"):
            AIAgent(
                provider="anthropic",
                model="claude-opus-4",
                api_key="",
                base_url="https://api.anthropic.com",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
