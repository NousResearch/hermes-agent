"""Architectural witnesses for desktop.plugin_routed_session API 1."""

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _tree(relative: str) -> ast.AST:
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def _declares_method(tree: ast.AST, method_name: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "method"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == method_name
        for node in ast.walk(tree)
    )


def test_routed_session_contract_binds_overrides_before_agent_construction():
    contracts = _tree("tui_gateway/contracts/sessions.py")
    create_params = next(
        node for node in contracts.body if isinstance(node, ast.ClassDef) and node.name == "SessionCreateParams"
    )
    fields = {
        node.target.id
        for node in create_params.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert {"model", "provider", "reasoning_effort"} <= fields
    assert _declares_method(contracts, "session.create")
    assert _declares_method(_tree("tui_gateway/contracts/prompt_voice.py"), "prompt.submit")
    config_models = _tree("tui_gateway/contracts/config_free_tier_control.py")
    assert _declares_method(config_models, "model.options")
    assert _declares_method(config_models, "config.get")

    methods = _tree("tui_gateway/methods_session.py")
    create = next(node for node in ast.walk(methods) if isinstance(node, ast.FunctionDef) and node.name == "_create_session")
    override_line = min(
        node.lineno
        for node in ast.walk(create)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_create_overrides"
    )
    schedule_line = min(
        node.lineno
        for node in ast.walk(create)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_schedule_agent_build"
    )
    assert override_line < schedule_line


def test_routed_session_contract_exposes_profile_scoped_plugin_doors():
    sdk = (ROOT / "apps/desktop/src/sdk/index.ts").read_text(encoding="utf-8")
    for symbol in ("profileRoutes", "requestProfile", "retainProfile", "openSession"):
        assert symbol in sdk
    context = (ROOT / "apps/desktop/src/contrib/plugin.ts").read_text(encoding="utf-8")
    assert "rest:" in context
    dashboard = _tree("hermes_cli/web_server_dashboard.py")
    functions = {
        node.name for node in ast.walk(dashboard) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"_plugin_route_secret_scope", "_mount_plugin_api_routes"} <= functions
    secrets = _tree("agent/secret_scope.py")
    config = _tree("hermes_cli/config.py")
    assert any(isinstance(node, ast.FunctionDef) and node.name == "get_secret" for node in ast.walk(secrets))
    assert any(isinstance(node, ast.FunctionDef) and node.name == "load_config_readonly" for node in ast.walk(config))


def test_routed_session_contract_stages_requested_binding_before_build(monkeypatch, tmp_path):
    """The published marker is backed by behavior, not only symbol presence."""
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server

    (tmp_path / "config.yaml").write_text(
        "model:\n  default: claude-opus-5\n  provider: anthropic\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_profile_home", lambda *_a: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    scheduled = []
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: scheduled.append(sid))
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_a: None)
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda *_a: None)

    response = server._methods["session.create"]("contract", {
        "cols": 80,
        "source": "desktop",
        "model": "claude-sonnet-4.6",
        "provider": "anthropic",
        "reasoning_effort": "high",
    })

    assert "error" not in response, response
    sid = response["result"]["session_id"]
    session = server._sessions[sid]
    assert session["model_override"] == {"model": "claude-sonnet-4.6", "provider": "anthropic"}
    assert session["create_reasoning_override"] == {"enabled": True, "effort": "high"}
    assert scheduled == [sid]
