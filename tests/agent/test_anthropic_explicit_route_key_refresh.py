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


_PRIMARY_OAUTH = "sk-ant-oat01-test-primary"


def test_stale_explicit_key_marker_does_not_block_oauth_rotation():
    # Back on the OAuth primary while the marker still holds a key from an earlier route.
    agent = _agent(key=_PRIMARY_OAUTH, explicit=_API_KEY)
    agent._is_anthropic_oauth = True
    changed, _ = _refresh(agent, _OAUTH)
    assert changed is True
    assert agent._anthropic_api_key == _OAUTH


def test_explicit_key_in_use_still_skips_refresh():
    agent = _agent(key="sk-ant-api03-test-in-use", explicit="sk-ant-api03-test-in-use")
    changed, build = _refresh(agent, _OAUTH)
    assert changed is False
    build.assert_not_called()
    assert agent._anthropic_api_key == "sk-ant-api03-test-in-use"


def test_voice_turn_end_restore_reenables_oauth_refresh(monkeypatch):
    """begin_voice_turn_route binds an aux entry with an explicit key; end_voice_turn_route restores the
    primary via reinstall_runtime_snapshot (which never clears the marker)."""
    from agent import route_binding, voice_turn_route

    voice_key = "sk-ant-api03-test-voice"
    monkeypatch.setenv("VOICE_TEST_KEY", voice_key)
    agent = _agent(key=_PRIMARY_OAUTH, explicit=None)
    agent._is_anthropic_oauth = True
    agent._voice_turn_pending = True
    agent._cached_system_prompt = "sys"
    agent.requested_provider, agent.base_url = "anthropic", "https://api.anthropic.com"
    cfg = {"provider": "anthropic", "model": "claude-haiku-4-5", "key_env": "VOICE_TEST_KEY"}
    client = MagicMock()
    client.base_url = "https://api.anthropic.com"

    def swap(agent_, *a, **kw):
        agent_._anthropic_api_key = voice_key

    def restore(agent_, snapshot):
        agent_._anthropic_api_key = _PRIMARY_OAUTH

    with patch("agent.auxiliary_task_config._get_auxiliary_task_config", return_value=cfg), \
            patch.object(voice_turn_route, "_capture", return_value={"snapshot": {}, **{
                n: None for n in voice_turn_route._EXTRA_FIELDS}}), \
            patch.object(voice_turn_route, "_fits", return_value=True), \
            patch("agent.chat_completion_helpers.rewrite_prompt_model_identity"), \
            patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "claude-haiku-4-5")), \
            patch("agent.chat_completion_helpers._rebind_fallback_credential_pool"), \
            patch("agent.chat_completion_helpers._update_fallback_context_compressor"), \
            patch("agent.chat_completion_helpers._reresolve_fallback_reasoning_config"), \
            patch("agent.chat_completion_helpers._rescope_fallback_extra_body"), \
            patch("agent.client_lifecycle._swap_fallback_clients", side_effect=swap), \
            patch("agent.agent_runtime_helpers.sync_credential_pool_entry_id"), \
            patch.object(agent, "_anthropic_prompt_cache_policy", return_value=(False, False)), \
            patch.object(agent, "_ensure_lmstudio_runtime_loaded"), \
            patch.object(route_binding, "reinstall_runtime_snapshot", side_effect=restore):
        voice_turn_route.begin_voice_turn_route(agent, [], "sys")
        assert agent._explicit_route_api_key == voice_key
        # While the voice key is in use the refresh is skipped.
        assert _refresh(agent, _OAUTH)[0] is False
        voice_turn_route.end_voice_turn_route(agent)

    assert agent._anthropic_api_key == _PRIMARY_OAUTH
    assert agent._explicit_route_api_key == voice_key  # marker is stale by design; the comparison handles it
    changed, _ = _refresh(agent, _OAUTH)
    assert changed is True
    assert agent._anthropic_api_key == _OAUTH
