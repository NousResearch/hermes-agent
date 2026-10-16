"""Go Gemini reservations follow the effective route across caller boundaries (#139526)."""

import socket

import httpx
import pytest

from agent.context_compressor import ContextCompressor
from agent.gemini_native_adapter import GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
from agent.route_binding import bind_route_entry, reinstall_runtime_snapshot


GO_BASE = "https://opencode.ai/zen/go/v1"
GO_MODEL = "gemini-4-argon-contributor"
OTHER_BASE = "https://example.test/v1"
OTHER_MODEL = "gpt-4o"


@pytest.fixture
def make_agent(monkeypatch, tmp_path):
    # Real configuration/imports/clients, with all HTTP forbidden at the I/O boundary.
    (tmp_path / "config.yaml").write_text(
        "model:\n  context_length: 131072\ncompression:\n  threshold: 0.5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def no_http(*args, **kwargs):
        raise AssertionError("This route-budget regression must not access the network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_http)
    monkeypatch.setattr(socket.socket, "connect", no_http)
    from run_agent import AIAgent
    agents = []

    def build(provider="opencode-go", requested_provider="opencode-go", max_tokens=None):
        agent = AIAgent(
            provider=provider, requested_provider=requested_provider,
            model=GO_MODEL, base_url=GO_BASE, api_key="fixture-only",
            max_tokens=max_tokens, skip_memory=True, skip_context_files=True,
            quiet_mode=True, enabled_toolsets=[], save_trajectories=False,
        )
        agents.append(agent)
        assert isinstance(agent.context_compressor, ContextCompressor)
        return agent

    yield build
    for agent in agents:
        agent._close_cached_request_openai_client(reason="test_cleanup")
        agent.client.close()


def leave_go(agent, boundary):
    if boundary == "switch":
        agent.switch_model(
            OTHER_MODEL, "custom", api_key="fixture-only", base_url=OTHER_BASE,
            api_mode="chat_completions",
        )
    else:
        assert bind_route_entry(
            agent, {"base_url": OTHER_BASE, "api_key": "fixture-only"},
            "custom", OTHER_MODEL,
        ) is not None


@pytest.mark.parametrize("provider", ["opencode-go", "custom"])
@pytest.mark.parametrize("boundary", ["switch", "fallback"])
@pytest.mark.parametrize("limit", [None, 8192])
def test_go_exit_and_restore_recompute_real_compressor(make_agent, provider, boundary, limit):
    agent = make_agent(provider=provider, max_tokens=limit)
    cc = agent.context_compressor
    go_reservation = limit or GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    assert cc.max_tokens == go_reservation
    initial_threshold = cc.threshold_tokens
    snapshot = dict(agent._primary_runtime)

    leave_go(agent, boundary)
    assert cc.max_tokens == limit
    # Compare the actual trigger to a fresh destination compressor, not just metadata.
    expected = ContextCompressor(
        model=OTHER_MODEL, provider="custom", base_url=OTHER_BASE,
        config_context_length=cc.context_length, max_tokens=limit,
    )
    assert cc.threshold_tokens == expected.threshold_tokens
    assert agent.max_tokens == limit

    reinstall_runtime_snapshot(agent, snapshot)
    assert cc.max_tokens == go_reservation
    assert cc.threshold_tokens == initial_threshold
    assert agent.provider == provider
    # Restoration must also preserve provenance for the NEXT departure.
    leave_go(agent, boundary)
    assert cc.max_tokens == limit


@pytest.mark.parametrize("boundary", ["switch", "fallback"])
def test_alias_entry_retains_provenance_for_later_exit(make_agent, boundary):
    from agent.agent_runtime_helpers import _update_switch_compressor, _build_primary_runtime_snapshot
    from agent.chat_completion_helpers import _update_fallback_context_compressor

    agent = make_agent()
    leave_go(agent, "switch")
    # Exercise the compressor callers with the supported internal alias identity.
    agent.model, agent.provider, agent.requested_provider = GO_MODEL, "custom", "opencode-go"
    agent.base_url = GO_BASE
    if boundary == "switch":
        _update_switch_compressor(agent, [], 131072, {})
    else:
        agent._config_context_length = 131072
        _update_fallback_context_compressor(agent)
    assert agent.context_compressor.max_tokens == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    snapshot = _build_primary_runtime_snapshot(agent, agent.api_mode)
    leave_go(agent, boundary)
    assert agent.context_compressor.max_tokens is None
    reinstall_runtime_snapshot(agent, snapshot)
    assert agent.context_compressor.max_tokens == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    leave_go(agent, boundary)
    assert agent.context_compressor.max_tokens is None


@pytest.mark.parametrize("boundary", ["switch", "fallback"])
@pytest.mark.parametrize("requested", ["custom", "custom:go"])
def test_arbitrary_custom_route_keeps_existing_reservation(make_agent, boundary, requested):
    agent = make_agent(provider="custom", requested_provider=requested)
    cc = agent.context_compressor
    assert cc.provider == "custom"
    assert cc.max_tokens is None
    cc.update_model(cc.model, cc.context_length, base_url=cc.base_url,
                    provider=cc.provider, max_tokens=4096)
    leave_go(agent, boundary)
    assert cc.max_tokens == 4096
    assert cc.provider == "custom"


@pytest.mark.parametrize("boundary", ["switch", "fallback"])
def test_plugin_update_contract_keeps_runtime_provider(make_agent, boundary):
    from agent.agent_runtime_helpers import _update_switch_compressor
    from agent.chat_completion_helpers import _update_fallback_context_compressor

    class PluginEngine:
        context_length = 131072
        threshold_tokens = 65536

        # Deliberately the existing plugin ABI: no max_tokens or **kwargs.
        def update_model(self, model, context_length, base_url="", api_key="",
                         provider="", api_mode=""):
            self.provider = provider

    agent = make_agent(provider="custom")
    engine = agent.context_compressor = PluginEngine()
    if boundary == "switch":
        _update_switch_compressor(agent, [], 131072, {})
    else:
        _update_fallback_context_compressor(agent)
    assert engine.provider == "custom"


def test_failed_compressor_update_keeps_go_provenance(make_agent, monkeypatch):
    agent = make_agent(provider="custom")
    cc = agent.context_compressor
    original = (cc.provider, cc.model, cc.base_url, cc.max_tokens, cc.threshold_tokens)

    def fail_update(*args, **kwargs):
        raise RuntimeError("compressor update refused")

    with monkeypatch.context() as patch:
        patch.setattr(cc, "update_model", fail_update)
        with pytest.raises(RuntimeError, match="compressor update refused"):
            leave_go(agent, "switch")
    assert (cc.provider, cc.model, cc.base_url, cc.max_tokens, cc.threshold_tokens) == original
    assert agent.provider == "custom"
    assert agent.requested_provider == "opencode-go"
    leave_go(agent, "switch")
    assert cc.max_tokens is None
