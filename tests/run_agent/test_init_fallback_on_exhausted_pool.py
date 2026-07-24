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


def test_init_fallback_key_env_uses_active_profile_secret_scope(monkeypatch):
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    fb = _mock_client(api_key="profile-fallback-key")
    fallback_explicit_keys = []

    def fake_resolve(provider, model=None, raw_codex=False,
                     explicit_base_url=None, explicit_api_key=None):
        if provider == "tencent-token-plan":
            fallback_explicit_keys.append(explicit_api_key)
            return fb, "kimi2.5"
        return None, None

    monkeypatch.setenv("FB_KEY", "global-default-profile-key")
    token = set_secret_scope({"FB_KEY": "profile-fallback-key"})
    try:
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=fake_resolve,
        ), patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs()
        ), patch(
            "run_agent.check_toolset_requirements", return_value={}
        ), patch("run_agent.OpenAI", return_value=MagicMock()):
            agent = AIAgent(
                provider="alibaba-coding-plan",
                model="qwen3.6-plus",
                api_key=None,
                base_url=None,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=[
                    {
                        "provider": "tencent-token-plan",
                        "model": "kimi2.5",
                        "key_env": "FB_KEY",
                    }
                ],
            )
    finally:
        reset_secret_scope(token)

    assert agent._fallback_activated is True
    assert fallback_explicit_keys == ["profile-fallback-key"]


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


def test_init_does_not_use_real_dotenv_fallback_under_empty_scope(
    tmp_path, monkeypatch
):
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text(
        "OPENROUTER_API_KEY=profile-file-key\n", encoding="utf-8"
    )
    token = set_secret_scope({})
    try:
        with patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs()
        ), patch(
            "run_agent.check_toolset_requirements", return_value={}
        ), patch("run_agent.OpenAI", return_value=MagicMock()):
            with pytest.raises(RuntimeError, match="no API key was found"):
                AIAgent(
                    provider="alibaba-coding-plan",
                    model="primary-model",
                    quiet_mode=True,
                    skip_context_files=True,
                    skip_memory=True,
                    fallback_model=[
                        {"provider": "openrouter", "model": "fallback-model"}
                    ],
                )
    finally:
        reset_secret_scope(token)
