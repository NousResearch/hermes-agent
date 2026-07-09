"""Fallback activation must honor fixed model policy before switching."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import try_activate_fallback


_FIXED_POLICY = {
    "model_policy": {
        "fixed_model": "gpt-5.5",
        "forbid_lower_fallback": True,
    }
}


def _recursive_agent(fallback_model: str):
    statuses: list[str] = []
    agent = SimpleNamespace(
        _fallback_chain=[{"provider": "openrouter", "model": fallback_model}],
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _primary_runtime={"provider": "openai-codex"},
        provider="openai-codex",
        model="gpt-5.5",
        base_url="https://chatgpt.com/backend-api/codex",
        _buffer_status=statuses.append,
    )
    agent._try_activate_fallback = lambda reason=None: try_activate_fallback(agent, reason)
    return agent, statuses


def test_disallowed_fallback_is_blocked_with_status_before_provider_resolution(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _FIXED_POLICY)
    agent, statuses = _recursive_agent("openai/gpt-5.4-mini")

    with patch("agent.auxiliary_client.resolve_provider_client") as resolve_client:
        result = try_activate_fallback(agent)

    assert result is False
    resolve_client.assert_not_called()
    assert agent.model == "gpt-5.5"
    assert agent.provider == "openai-codex"
    assert any("model policy" in msg.lower() and "gpt-5.5" in msg for msg in statuses)


def _activating_agent(fallback_model: str):
    statuses: list[str] = []
    agent = SimpleNamespace(
        _fallback_chain=[{"provider": "openrouter", "model": fallback_model}],
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _primary_runtime={"provider": "openai-codex"},
        provider="openai-codex",
        model="gpt-5.5",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        api_key="primary-key",
        _config_context_length=None,
        _transport_cache={},
        _credential_pool=None,
        context_compressor=None,
        _cached_system_prompt="Model: gpt-5.5\nProvider: openai-codex",
        _buffer_status=statuses.append,
        _is_azure_openai_url=MagicMock(return_value=False),
        _is_direct_openai_url=MagicMock(return_value=False),
        _provider_model_requires_responses_api=MagicMock(return_value=False),
        _anthropic_prompt_cache_policy=MagicMock(return_value=(False, False)),
        _ensure_lmstudio_runtime_loaded=MagicMock(),
        _replace_primary_openai_client=MagicMock(return_value=True),
    )
    agent._try_activate_fallback = lambda reason=None: try_activate_fallback(agent, reason)
    return agent, statuses


def test_provider_prefixed_gpt55_fallback_is_allowed_under_fixed_policy(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _FIXED_POLICY)
    agent, statuses = _activating_agent("openai/gpt-5.5")
    fake_client = SimpleNamespace(
        api_key="fallback-key",
        base_url="https://openrouter.ai/api/v1",
    )

    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(fake_client, "gpt-5.5")),
        patch("hermes_cli.model_normalize.normalize_model_for_provider", return_value="gpt-5.5"),
        patch("agent.chat_completion_helpers.get_provider_request_timeout", return_value=None),
    ):
        result = try_activate_fallback(agent)

    assert result is True
    assert agent.model == "gpt-5.5"
    assert agent.provider == "openrouter"
    assert agent._fallback_activated is True
    assert any("switching to fallback" in msg.lower() for msg in statuses)
