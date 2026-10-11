"""delegation.hot_reload_model — per-request model/provider rebind for running subagents.

Off (default) = the child's spawn-time provider:model is frozen and reused. On = the next
provider API request re-reads delegation.provider/model and rebinds the live route in place.
"""

from __future__ import annotations

import logging

import pytest

from agent import subagent_hot_reload as shr
from agent.turn_api_request import build_api_request


class _Agent:
    def __init__(self, *, model="spawn-model", provider="openrouter", hot_reload=False):
        self.model = model
        self.provider = provider
        self.api_key = "k"
        self._delegation_hot_reload_model = hot_reload


def _bundle(model="new-model", provider="openrouter"):
    return {
        "model": model, "provider": provider, "api_key": "k",
        "base_url": "https://example.invalid/v1", "api_mode": "chat_completions",
    }


def test_flag_frozen_at_spawn_defaults_off():
    assert shr.hot_reload_enabled(object()) is False
    assert shr.hot_reload_enabled(_Agent(hot_reload=False)) is False
    assert shr.hot_reload_enabled(_Agent(hot_reload=True)) is True


def test_default_config_key_is_false():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["delegation"]["hot_reload_model"] is False


class TestResolveRoute:
    def test_pure_inherit_child_has_no_route_to_reread(self, monkeypatch):
        monkeypatch.setattr(
            "tools.delegate_tool_config._load_config", lambda: {"provider": "", "model": ""}
        )
        assert shr.resolve_hot_reload_route(object()) is None

    def test_pinned_child_resolves_the_configured_bundle(self, monkeypatch):
        monkeypatch.setattr(
            "tools.delegate_tool_config._load_config",
            lambda: {"provider": "nous", "model": "hermes-4"},
        )
        monkeypatch.setattr(
            "tools.delegate_tool_config._resolve_delegation_credentials",
            lambda cfg, agent: {"model": "hermes-4", "provider": "nous"},
        )
        assert shr.resolve_hot_reload_route(object()) == {"model": "hermes-4", "provider": "nous"}


class TestRefreshBranches:
    def test_off_reuses_spawn_time_binding(self, monkeypatch):
        agent = _Agent(model="spawn-model", hot_reload=False)
        calls = []
        monkeypatch.setattr(shr, "resolve_hot_reload_route", lambda a: _bundle())
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", lambda *a, **k: calls.append(a))

        shr.refresh_subagent_model(agent)

        assert agent.model == "spawn-model"  # frozen exactly as at spawn
        assert calls == []

    def test_on_rebinds_to_the_new_value_on_the_next_request(self, monkeypatch):
        agent = _Agent(model="spawn-model", hot_reload=True)
        seen = []

        def fake_switch(a, new_model, new_provider, api_key="", base_url="", api_mode=""):
            a.model, a.provider = new_model, new_provider
            seen.append((new_model, new_provider, base_url, api_mode))

        monkeypatch.setattr(shr, "resolve_hot_reload_route", lambda a: _bundle())
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", fake_switch)

        shr.refresh_subagent_model(agent)

        assert agent.model == "new-model"
        assert agent.provider == "openrouter"
        assert seen == [("new-model", "openrouter", "https://example.invalid/v1", "chat_completions")]

    def test_unchanged_route_does_not_rebuild_the_client(self, monkeypatch):
        agent = _Agent(model="new-model", provider="openrouter", hot_reload=True)
        calls = []
        monkeypatch.setattr(shr, "resolve_hot_reload_route", lambda a: _bundle(model="new-model"))
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", lambda *a, **k: calls.append(a))

        shr.refresh_subagent_model(agent)

        assert calls == []

    def test_unpinned_child_keeps_binding(self, monkeypatch):
        agent = _Agent(hot_reload=True)
        calls = []
        monkeypatch.setattr(shr, "resolve_hot_reload_route", lambda a: None)
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", lambda *a, **k: calls.append(a))

        shr.refresh_subagent_model(agent)

        assert agent.model == "spawn-model"
        assert calls == []

    def test_resolution_failure_keeps_binding_and_never_raises(self, monkeypatch, caplog):
        agent = _Agent(hot_reload=True)

        def boom(_a):
            raise ValueError("bad provider pin")

        monkeypatch.setattr(shr, "resolve_hot_reload_route", boom)

        with caplog.at_level(logging.WARNING):
            shr.refresh_subagent_model(agent)  # must not raise

        assert agent.model == "spawn-model"
        assert any("hot reload" in r.message for r in caplog.records)


def test_build_api_request_consults_hot_reload_before_building_the_payload(monkeypatch):
    """The rebind runs at the top of the per-attempt request builder."""
    import agent.turn_api_request as tar

    class _Consulted(RuntimeError):
        pass

    def consult(_agent):
        raise _Consulted

    monkeypatch.setattr(tar, "refresh_subagent_model", consult)

    with pytest.raises(_Consulted):
        build_api_request(
            object(), api_messages=[], _moa_prepared_request=None, tools_for_api=None,
            system_message=None, messages=[], original_user_message="", approx_tokens=0,
            total_chars=0, retry_count=0, api_call_count=0, api_request_id="r1",
            api_start_time=0.0, effective_task_id="t1", turn_id="turn1",
        )
