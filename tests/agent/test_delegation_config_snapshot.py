"""Same-name route edits must not change a live or resumed session's child model."""

import hermes_yaml as yaml
import pytest

from tests.agent import test_delegation_switch_next_session as switch_tests

_agent, _turn = switch_tests._agent, switch_tests._turn


@pytest.fixture
def options_backend(monkeypatch):
    yield from switch_tests.options_backend.__wrapped__(monkeypatch)


def _model(agent):
    from tools.delegate_tool_config import _load_config
    from tools.delegate_tool_routes import resolve_task_routes
    routes, error = resolve_task_routes([{"goal": "work", "option": "fast"}], _load_config(), agent)
    assert error is None
    assert routes[0].error is None
    return routes[0].creds["model"]


@pytest.mark.parametrize("surface", ["startup", "refresh", "acp"])
def test_config_edit_between_capture_and_schema_keeps_matched_routes(options_backend, monkeypatch, surface, caplog):
    import model_tools
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    from tools.delegate_tool_config import _load_config
    from tools.delegate_tool_routes import resolve_task_routes

    srv, home = options_backend
    agent = _agent(srv, home, "schema-race") if surface != "startup" else None
    real = model_tools.get_tool_definitions
    builds = []

    def edit_then_build(*args, **kwargs):
        config = yaml.safe_load((home / "config.yaml").read_text())
        config["delegation"]["backends"]["probe-options"]["options"] = [{
            "name": "renamed", "description": "edited during build", "model": "new-model",
            "base_url": "http://127.0.0.1:9/v1"}]
        (home / "config.yaml").write_text(yaml.safe_dump(config))
        model_tools._clear_tool_defs_cache()
        result = real(*args, **kwargs)
        builds.append(result)
        return result

    monkeypatch.setattr(model_tools, "get_tool_definitions", edit_then_build)
    if surface == "startup":
        agent = _agent(srv, home, "schema-race")
    elif surface == "acp":
        import asyncio
        from types import SimpleNamespace
        from acp.schema import McpServerStdio
        from acp_adapter.server import HermesACPAgent
        monkeypatch.setattr("tools.mcp_tool_discovery.register_mcp_servers", lambda _: None)
        state = SimpleNamespace(agent=agent, cwd=str(home), session_id="schema-race")
        asyncio.run(HermesACPAgent._register_session_mcp_servers(
            SimpleNamespace(), state, [McpServerStdio(name="race", command="unused", args=[], env=[])]))
    else:
        refresh_agent_mcp_tools(agent)
    assert agent is not None
    try:
        assert len(builds) == 1
        assert "failed to refresh tool surface" not in caplog.text
        assert "failed to register ACP MCP servers" not in caplog.text
        tool = next(t["function"] for t in agent.tools if t["function"]["name"] == "delegate_task")
        choices = tool["parameters"]["properties"]["tasks"]["items"]["properties"]["option"]["enum"]
        assert choices[0] == "fast"  # captured A, not the subsequently saved B
        routes, error = resolve_task_routes([{"goal": "work", "option": choices[0]}], _load_config(), agent)
        assert error is None and routes[0].error is None
        assert routes[0].creds["model"] == "fast-model"
    finally:
        agent.close()


def test_newer_publish_cannot_be_overwritten_by_old_refresh_config(options_backend, monkeypatch):
    from tools import mcp_tool_agent as mcp
    from agent import delegation_backends as backends

    srv, home = options_backend
    agent = _agent(srv, home, "publish-race")
    publish = mcp._publish_tool_snapshot
    adopt = backends.adopt_built_backend
    interleaved = False

    def assert_locked(*args, **kwargs):
        assert mcp._agent_tools_lock.locked()
        return adopt(*args, **kwargs)

    def publish_then_interleave(*args, **kwargs):
        nonlocal interleaved
        result = publish(*args, **kwargs)
        if not interleaved:
            interleaved = True
            config = yaml.safe_load((home / "config.yaml").read_text())
            config["delegation"]["backends"]["probe-options"]["options"][0]["model"] = "newer-model"
            (home / "config.yaml").write_text(yaml.safe_dump(config))
            mcp.refresh_agent_mcp_tools(agent)
        return result

    monkeypatch.setattr(backends, "adopt_built_backend", assert_locked)
    monkeypatch.setattr(mcp, "_publish_tool_snapshot", publish_then_interleave)
    try:
        mcp.refresh_agent_mcp_tools(agent)
        assert _model(agent) == "newer-model"
        import json
        pin = json.loads(agent._session_db.get_session("publish-race")["tool_names"])
        assert pin["delegation_backend_config"]["backends"]["probe-options"]["options"][0]["model"] == "newer-model"
    finally:
        agent.close()


def test_snapshot_is_immutable_and_schema_scope_is_detached(options_backend):
    from dataclasses import FrozenInstanceError
    from agent.delegation_backends import saved_backend_snapshot, delegation_schema_config
    from tools.delegate_tool_config import _load_config

    snapshot = saved_backend_snapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot.serialized = "changed"
    with delegation_schema_config(snapshot):
        _load_config()["backends"]["probe-options"]["options"][0]["model"] = "mutated"
        assert _load_config()["backends"]["probe-options"]["options"][0]["model"] == "fast-model"
    assert "fast-model" not in repr(snapshot)


def test_no_delegate_tool_does_not_persist_runtime_credentials(options_backend):
    import json
    from types import SimpleNamespace
    from tools.mcp_tool_agent import persist_agent_tool_names
    from hermes_state import SessionDB

    _, home = options_backend
    db = SessionDB(home / "no-delegate.db")
    db.ensure_session("no-delegate", source="cli")
    agent = SimpleNamespace(tools=[], _session_db=db, session_id="no-delegate",
                            _delegation_backend="probe-options",
                            _delegation_backend_config={"api_key": "synthetic-explicit-key"})
    try:
        persist_agent_tool_names(agent)
        pin = json.loads(db.get_session("no-delegate")["tool_names"])
        assert "delegation_backend_config" not in pin
        assert "delegation_backend" not in pin
    finally:
        db.close()


def test_snapshot_retains_only_active_runtime_settings(options_backend, monkeypatch):
    from agent.delegation_backends import saved_backend_config
    from agent.delegation_registry import get_active_provider
    from tools.delegate_tool_config import _load_config

    _, home = options_backend
    config = yaml.safe_load((home / "config.yaml").read_text())
    section = config["delegation"]
    section["backends"]["unrelated"] = {"api_key": "not-for-this-session"}
    section["backends"]["probe-options"]["library"] = {"secret": "not-runtime"}
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    provider = get_active_provider(_load_config())
    monkeypatch.setattr(provider, "snapshot_settings", lambda cfg: {"options": provider.settings(cfg)["options"]})
    snapshot = saved_backend_config()
    assert set(snapshot["backends"]) == {"probe-options"}
    assert set(snapshot["backends"]["probe-options"]) == {"options"}
    snapshot["backends"]["probe-options"]["options"][0]["model"] = "mutated-copy"
    assert saved_backend_config()["backends"]["probe-options"]["options"][0]["model"] == "fast-model"


def test_now_request_is_not_consumed_by_another_profile_with_the_same_session_key(options_backend, tmp_path):
    from types import SimpleNamespace
    from agent.delegation_backends import apply_requested_tool_rebuild, set_global_backend
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    srv, home = options_backend
    agent = _agent(srv, home, "shared-key")
    history = _turn(agent, "hello", [])
    set_global_backend("default", session_key="shared-key", apply_now=True)
    other = tmp_path / "other-profile"
    other.mkdir()
    token = set_hermes_home_override(str(other))
    try:
        assert not apply_requested_tool_rebuild(SimpleNamespace(session_id="shared-key"))
    finally:
        reset_hermes_home_override(token)
    _turn(agent, "apply in owner", history)
    assert agent._delegation_backend == "default"
    agent.close()


def test_same_backend_settings_stay_pinned_until_now_or_new_session(options_backend):
    from agent.delegation_backends import set_global_backend

    srv, home = options_backend
    agent = _agent(srv, home, "config-pin")
    history = _turn(agent, "hello", [])
    before = srv.main_requests()[-1]["tools"]
    assert _model(agent) == "fast-model"
    config = yaml.safe_load((home / "config.yaml").read_text())
    config["delegation"]["backends"]["probe-options"]["options"][0]["model"] = "changed-model"
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    assert _model(agent) == "fast-model"
    agent.close()

    resumed = _agent(srv, home, "config-pin")
    history = _turn(resumed, "still here", history)
    assert srv.main_requests()[-1]["tools"] == before
    assert _model(resumed) == "fast-model"
    other = _agent(srv, home, "config-other")
    other_history = _turn(other, "new chat", [])
    assert _model(other) == "changed-model"

    # Same name, identical schema: --now must refresh and persist the route anyway.
    config["delegation"]["backends"]["probe-options"]["options"][0]["model"] = "now-model"
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    set_global_backend("probe-options", session_key="config-pin", apply_now=True)
    history = _turn(resumed, "apply", history)
    assert _model(resumed) == "now-model"
    _turn(other, "unchanged", other_history)
    assert _model(other) == "changed-model"
    other.close()
    resumed.close()
    final = _agent(srv, home, "config-pin")
    _turn(final, "resume now", history)
    assert _model(final) == "now-model"
    final.close()
