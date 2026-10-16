"""A resumed ACP session must re-resolve its named provider, not fall back (#139908).

Three coupled defects sent a resumed turn's credentials to an endpoint the session never
used: ``_persist`` stored the resolved KIND (``custom``) instead of the named route
(``custom:<name>``); ``_make_agent`` swallowed the unresolvable-bare-kind AuthError and
rebuilt on the config-default endpoint; and ``_switch_model(keep_endpoint=True)`` treated
an empty provider as "unchanged", pinning that wrong base_url under the new provider's key.
"""

import json
from types import SimpleNamespace

import pytest

from acp_adapter.session import SessionManager
from hermes_state import SessionDB


def _agent(**attrs):
    fields = {"model": "some-model", "provider": "custom",
              "base_url": "http://127.0.0.1:9001/v1", "api_mode": "chat"}
    fields.update(attrs)
    return SimpleNamespace(**fields)


def test_persist_stores_the_named_route_not_the_bare_kind(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    manager = SessionManager(db=db, agent_factory=lambda: _agent(provider_route="custom:gw"))
    state = manager.create_session(cwd=str(tmp_path))
    state.history.append({"role": "user", "content": "turn 1 on endpoint A"})
    manager.save_session(state.session_id)
    meta = json.loads(db.get_session(state.session_id)["model_config"])
    assert meta["provider"] == "custom:gw"
    db.close()


def test_persist_falls_back_to_the_kind_without_a_route(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    manager = SessionManager(db=db, agent_factory=lambda: _agent())
    state = manager.create_session(cwd=str(tmp_path))
    state.history.append({"role": "user", "content": "compat"})
    manager.save_session(state.session_id)
    assert json.loads(db.get_session(state.session_id)["model_config"])["provider"] == "custom"
    db.close()


def _patch_resolution(monkeypatch, *, runtime=None, error=None):
    monkeypatch.setattr("hermes_cli.config.load_config",
                        lambda: {"model": {"default": "some-model", "provider": "custom:home"}, "agent": {}})
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda cfg, platform: set())
    monkeypatch.setattr("hermes_cli.tools_config.enabled_mcp_server_names", lambda cfg: set())
    monkeypatch.setattr("hermes_constants.resolve_reasoning_config", lambda cfg, model: None)
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
                        lambda **_kw: None)
    built: list[dict] = []

    def _fake_agent(**kwargs):
        built.append(kwargs)
        return _agent()

    monkeypatch.setattr("run_agent.AIAgent", _fake_agent)

    def _resolve(*, requested=None, **_kw):
        if error is not None:
            raise error
        return runtime or {"provider": "custom", "api_mode": "chat", "base_url": "http://a/v1",
                           "api_key": "k", "credential_pool": None, "command": None, "args": []}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _resolve)
    return built


def test_make_agent_stamps_the_resolved_named_route(monkeypatch, tmp_path):
    built = _patch_resolution(monkeypatch)
    db = SessionDB(tmp_path / "state.db")
    manager = SessionManager(db=db, agent_factory=None)
    agent = manager._make_agent(session_id="s-route", cwd=str(tmp_path), requested_provider="custom:gw")
    assert agent.provider_route == "custom:gw"
    assert built[0]["provider"] == "custom"  # AIAgent still gets the bare kind
    db.close()


def test_make_agent_fails_closed_when_a_named_request_cannot_resolve(monkeypatch, tmp_path):
    from hermes_cli.auth import AuthError

    _patch_resolution(monkeypatch, error=AuthError(
        "provider 'custom' resolved without credentials (no endpoint or API key configured). "
        "If this is a named custom provider, use its real name (see providers: in config.yaml).",
        provider="custom", code="missing_api_key"))
    db = SessionDB(tmp_path / "state.db")
    manager = SessionManager(db=db, agent_factory=None)
    # The explicit request surfaces its error instead of silently rebuilding on the
    # config-default endpoint (a different endpoint, later sent the target provider's key).
    with pytest.raises(AuthError, match="use its real name"):
        manager._make_agent(session_id="s-closed", cwd=str(tmp_path), requested_provider="custom")
    db.close()


def test_restore_of_a_legacy_bare_kind_row_fails_closed(monkeypatch, tmp_path):
    from hermes_cli.auth import AuthError

    _patch_resolution(monkeypatch, error=AuthError(
        "provider 'custom' resolved without credentials", provider="custom", code="missing_api_key"))
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="legacy", source="acp", model="some-model",
                      model_config={"cwd": str(tmp_path), "provider": "custom",
                                    "base_url": "http://127.0.0.1:9001/v1", "api_mode": "chat"},
                      cwd=str(tmp_path))
    db.append_message("legacy", "user", "turn 1 on endpoint A")
    manager = SessionManager(db=db, agent_factory=None)
    assert manager._restore("legacy") is None  # resume refuses rather than mispair key/endpoint
    db.close()


def test_switch_model_never_pins_the_endpoint_for_an_unknown_provider(monkeypatch):
    from acp_adapter.server import HermesACPAgent
    from hermes_cli.model_switch import ModelSwitchResult

    monkeypatch.setattr("hermes_cli.config.load_config", dict)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kw: ModelSwitchResult(
        success=True, new_model="some-model", target_provider="custom:gw"))
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_events.record_model_switch",
                        lambda **_kw: None)
    made: dict = {}

    class _SM:
        def _make_agent(self, **kw):
            made.update(kw)
            return _agent()

        def save_session(self, sid):
            pass

    agent = HermesACPAgent(session_manager=_SM())
    # A resume that fell back leaves provider empty; that must not keep the stale base_url
    # while the switch hands the new provider's credentials to _make_agent.
    state = SimpleNamespace(session_id="s1", cwd=".", model="some-model",
                            agent=_agent(provider="", provider_route=None, base_url="http://wrong/v1"))
    agent._switch_model(state, "custom:gw:some-model", keep_endpoint=True)
    assert "base_url" not in made  # endpoint re-resolved from the named provider, not pinned

    # Same named provider on both sides still carries the endpoint over.
    made.clear()
    state.agent = _agent(provider_route="custom:gw", base_url="http://right/v1")
    agent._switch_model(state, "custom:gw:some-model", keep_endpoint=True)
    assert made["base_url"] == "http://right/v1"
