"""An explicitly bound fallback key must survive the pre-request Anthropic credential refresh."""

from unittest.mock import MagicMock, patch

from agent.route_binding import reinstall_primary_runtime
from run_agent import AIAgent

_API_KEY = "sk-ant-api03-explicit-fallback-key"
_OAUTH = "sk-ant-oat01-claude-code-subscription-token"


def _agent(key=_API_KEY, explicit=_API_KEY):
    agent = AIAgent.__new__(AIAgent)
    agent.provider = "anthropic"
    agent.model = "claude-opus-4-1"
    agent.api_mode = "anthropic_messages"
    agent.api_key = key
    agent._anthropic_api_key = key
    agent._anthropic_base_url = "https://api.anthropic.com"
    agent._anthropic_client = MagicMock()
    agent._is_anthropic_oauth = False
    agent._explicit_route_api_key = explicit
    return agent


def _refresh(agent, resolved):
    with patch("agent.anthropic_credentials.resolve_anthropic_token", return_value=resolved), \
            patch.object(agent, "_build_direct_anthropic_client", return_value=MagicMock()) as build:
        return agent._try_refresh_anthropic_client_credentials(), build


def test_explicit_api_key_not_replaced_by_global_oauth_token():
    agent = _agent()
    changed, build = _refresh(agent, _OAUTH)
    assert changed is False
    build.assert_not_called()
    assert agent._anthropic_api_key == _API_KEY
    assert agent.api_key == _API_KEY


def test_without_explicit_key_global_refresh_still_applies():
    agent = _agent(key="sk-ant-api03-old", explicit=None)
    changed, _ = _refresh(agent, _OAUTH)
    assert changed is True
    assert agent._anthropic_api_key == _OAUTH


def test_explicit_oauth_token_still_rotates():
    old = "sk-ant-oat01-old-explicit"
    agent = _agent(key=old, explicit=old)
    agent._is_anthropic_oauth = True
    changed, _ = _refresh(agent, _OAUTH)
    assert changed is True
    assert agent._anthropic_api_key == _OAUTH


def test_reinstall_primary_runtime_clears_explicit_key():
    agent = _agent()
    with patch("agent.route_binding.reinstall_runtime_snapshot"), \
            patch("agent.agent_runtime_helpers._rebind_primary_credential_pool"):
        reinstall_primary_runtime(agent, {}, "p", "m", None, None)
    assert agent._explicit_route_api_key is None
    assert agent._fallback_activated is False


def test_bind_route_entry_records_entry_key(monkeypatch):
    from agent import route_binding

    monkeypatch.setenv("ANTHROPIC_FALLBACK_API_KEY", _API_KEY)
    agent = _agent(explicit=None)
    agent.provider, agent.base_url = "claude-sub", "https://example.invalid"
    client = MagicMock()
    client.base_url = "https://api.anthropic.com"
    entry = {"provider": "anthropic", "model": "claude-opus-4-1", "key_env": "ANTHROPIC_FALLBACK_API_KEY"}
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "claude-opus-4-1")) as rpc, \
            patch("agent.chat_completion_helpers._rebind_fallback_credential_pool"), \
            patch("agent.chat_completion_helpers._update_fallback_context_compressor"), \
            patch("agent.chat_completion_helpers._reresolve_fallback_reasoning_config"), \
            patch("agent.chat_completion_helpers._rescope_fallback_extra_body"), \
            patch("agent.client_lifecycle._swap_fallback_clients"), \
            patch("agent.agent_runtime_helpers.sync_credential_pool_entry_id"), \
            patch.object(agent, "_anthropic_prompt_cache_policy", return_value=(False, False)), \
            patch.object(agent, "_ensure_lmstudio_runtime_loaded"):
        route_binding.bind_route_entry(agent, entry, "anthropic", "claude-opus-4-1")
    assert rpc.call_args.kwargs["explicit_api_key"] == _API_KEY
    assert agent._explicit_route_api_key == _API_KEY
