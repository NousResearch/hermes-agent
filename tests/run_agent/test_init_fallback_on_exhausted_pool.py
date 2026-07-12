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


def test_init_policy_off_never_resolves_configured_fallback():
    """Init/auth fallback must obey ``off`` before resolving any backup."""
    calls = []

    def fake_resolve(provider, model=None, raw_codex=False, **kwargs):
        calls.append((provider, model))
        return None, None

    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"fallback_policy": "off"},
    ), patch(
        "agent.auxiliary_client.resolve_provider_client",
        side_effect=fake_resolve,
    ), patch(
        "run_agent.get_tool_definitions", return_value=_make_tool_defs()
    ), patch(
        "run_agent.check_toolset_requirements", return_value={}
    ), patch("run_agent.OpenAI", return_value=MagicMock()):
        with pytest.raises(
            RuntimeError,
            match="Fallback policy is off, so no backup provider was attempted",
        ):
            AIAgent(
                provider="alibaba-coding-plan",
                model="qwen3.6-plus",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=[
                    {"provider": "openrouter", "model": "remote-backup"}
                ],
            )

    assert calls == [("alibaba-coding-plan", "qwen3.6-plus")]


def test_init_local_only_skips_remote_and_queues_local_decision():
    """Init fallback filters the captured chain and explains the local switch."""
    local = {
        "provider": "custom",
        "model": "local-backup",
        "base_url": "http://127.0.0.1:8000/v1",
    }
    config = {
        "fallback_policy": "local-only",
        "fallback_providers": [
            {"provider": "openrouter", "model": "remote-backup"},
            local,
        ],
    }
    calls = []
    local_client = _mock_client(base_url=local["base_url"])

    def fake_resolve(provider, model=None, raw_codex=False, **kwargs):
        calls.append((provider, model))
        if provider == "custom":
            return local_client, model
        return None, None

    with patch(
        "hermes_cli.config.load_config_readonly", return_value=config
    ), patch(
        "agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve
    ), patch(
        "run_agent.get_tool_definitions", return_value=_make_tool_defs()
    ), patch(
        "run_agent.check_toolset_requirements", return_value={}
    ), patch("run_agent.OpenAI", return_value=MagicMock()):
        agent = AIAgent(
            provider="alibaba-coding-plan",
            model="qwen3.6-plus",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=config["fallback_providers"],
        )

    assert calls == [
        ("alibaba-coding-plan", "qwen3.6-plus"),
        ("custom", "local-backup"),
    ]
    assert agent.provider == "custom"
    assert agent.model == "local-backup"
    assert agent._fallback_policy == "local-only"
    assert agent._fallback_chain == [local]
    assert "policy local-only" in agent._pending_fallback_notice
    assert "credentials unavailable" in agent._pending_fallback_notice


def test_init_fallback_is_revoked_before_first_request_when_policy_changes():
    """A cached init route cannot survive a live switch to ``off``."""
    fallback = {"provider": "openrouter", "model": "remote-backup"}
    config = {
        "fallback_policy": "any",
        "fallback_providers": [fallback],
    }
    fallback_client = _mock_client()

    def fake_resolve(provider, model=None, raw_codex=False, **kwargs):
        if provider == "openrouter":
            return fallback_client, model
        return None, None

    with patch(
        "hermes_cli.config.load_config_readonly", return_value=config
    ), patch(
        "agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve
    ), patch(
        "run_agent.get_tool_definitions", return_value=_make_tool_defs()
    ), patch(
        "run_agent.check_toolset_requirements", return_value={}
    ), patch("run_agent.OpenAI", return_value=MagicMock()):
        agent = AIAgent(
            provider="alibaba-coding-plan",
            model="qwen3.6-plus",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[fallback],
        )

    statuses = []
    agent.status_callback = lambda kind, text: statuses.append((kind, text))
    fallback_client.chat.completions.create.reset_mock()

    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"fallback_policy": "off", "fallback_providers": []},
    ):
        with pytest.raises(RuntimeError, match="no model request was sent"):
            agent.run_conversation("must not reach the fallback")

    fallback_client.chat.completions.create.assert_not_called()
    assert agent._pending_fallback_notice is None
    assert len(statuses) == 1
    assert statuses[0][0] == "fallback"
    assert "policy off" in statuses[0][1]
