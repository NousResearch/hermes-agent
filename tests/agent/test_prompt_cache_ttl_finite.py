"""Finite admissions and background routes keep the short auto cache tier."""
from types import SimpleNamespace

import pytest

from agent.agent_init import _init_prompt_cache_config
from gateway.session_finite import finite_turn_scope


@pytest.fixture
def init_cache(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", dict)
    for key in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(key, raising=False)

    def build(**kwargs):
        agent = SimpleNamespace(_anthropic_prompt_cache_policy=lambda: (True, True), **kwargs)
        _init_prompt_cache_config(agent)
        return agent

    return build


@pytest.mark.parametrize("unattended", [False, True], ids=["chat-q", "direct-z"])
def test_gateway_finite_admission_keeps_5m_without_launch_env(init_cache, unattended):
    # Managed workers construct the agent on the admission's finite-turn context;
    # the gateway daemon has neither CLI launch environment marker.
    with finite_turn_scope(True, unattended=unattended):
        assert init_cache(platform="cli")._cache_ttl == "5m"
    with finite_turn_scope(False):
        assert init_cache(platform="cli")._cache_ttl == "1h"


def test_direct_z_launch_marker_keeps_5m(init_cache, monkeypatch):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    assert init_cache(platform="cli")._cache_ttl == "5m"


@pytest.mark.parametrize("platform", ["a2a", "msgraph_webhook"])
def test_automation_sources_keep_5m(init_cache, platform):
    assert init_cache(platform=platform)._cache_ttl == "5m"


@pytest.mark.parametrize("ttl, expected", [("auto", "5m"), ("1h", "1h")])
def test_bedrock_auto_is_short_but_explicit_pin_survives(init_cache, monkeypatch, ttl, expected):
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", lambda: {"prompt_caching": {"cache_ttl": ttl}})
    assert init_cache(platform="cli", provider="bedrock")._cache_ttl == expected


def test_missing_auxiliary_ttl_uses_auto_without_changing_short_tier(init_cache):
    from agent.agent_runtime_helpers import (
        _raw_cache_ttl_from_config, configured_cache_ttl, prompt_caching_disabled_from_config,
    )
    from agent.prompt_caching import effective_cache_ttl

    assert _raw_cache_ttl_from_config(None) == "auto"
    assert configured_cache_ttl() is None
    assert effective_cache_ttl(configured_cache_ttl()) == "5m"
    assert prompt_caching_disabled_from_config() is False


@pytest.mark.parametrize("ttl, expected", [("auto", "5m"), ("1h", "1h")])
@pytest.mark.parametrize("request_path", ["assembly", "failover"])
def test_bedrock_switch_at_request_time_preserves_only_explicit_pin(monkeypatch, ttl, expected, request_path):
    import copy
    import logging
    import hermes_cli.config
    from agent.conversation_loop import _redecorate_prompt_cache_for_provider
    from agent.turn_request_assembly import assemble_api_request
    from run_agent import AIAgent

    cfg = {"prompt_caching": {"cache_ttl": ttl}}
    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", lambda: cfg)
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: cfg)
    for key in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(key, raising=False)
    agent = AIAgent(
        platform="cli", provider="openrouter", api_key="fixture-key",
        base_url="https://openrouter.ai/api/v1", model="anthropic/claude-sonnet-4.6",
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, enabled_toolsets=["file"],
    )
    history = [{"role": "system", "content": "Stable prompt"}, {"role": "user", "content": "Hello"}]
    before = copy.deepcopy(history)
    try:
        assert agent._cache_ttl == "1h"
        agent.provider = "bedrock"
        agent._current_turn_timestamp = 1.0
        if request_path == "failover":
            messages, _, _ = _redecorate_prompt_cache_for_provider(agent, history)
        else:
            result = assemble_api_request(
                agent, messages=history, current_turn_user_idx=1, _ext_prefetch_cache="",
                _plugin_user_context="", moa_config=None, active_system_prompt="Stable prompt",
                original_user_message="Hello", pending_moa_prepared_request=None,
                request_logger=logging.getLogger(__name__),
            )
            messages = result.api_messages
        markers = [part["cache_control"] for message in messages
                   for part in message["content"] if isinstance(part, dict) and "cache_control" in part]
        assert markers
        assert all(marker == ({"type": "ephemeral"} if expected == "5m"
                              else {"type": "ephemeral", "ttl": "1h"}) for marker in markers)
        assert history == before
    finally:
        agent.close()
