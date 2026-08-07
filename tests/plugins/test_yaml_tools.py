"""Tests for declarative tools discovered from ``~/.hermes/tools/*.yaml``."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from plugins.yaml_tools import (
    _build_terminal_command,
    _coerce_timeout,
    _load_spec,
    _make_handler,
    register,
)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


class _FakeCtx:
    """Small registration double for parser/discovery unit tests."""

    def __init__(self, collide=()):
        self.registered = {}
        self._collide = set(collide)

    def register_tool(self, *, name, toolset, schema, handler, description="", emoji=""):
        if name in self._collide:
            raise ValueError(f"tool {name!r} already registered")
        self.registered[name] = {
            "toolset": toolset,
            "schema": schema,
            "handler": handler,
            "description": description,
            "emoji": emoji,
        }

    def dispatch_tool(self, tool_name, args, **kwargs):  # pragma: no cover - registration only
        raise AssertionError("registered handlers are not invoked by these unit tests")


class _DispatchSpy:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or json.dumps({"output": "ok", "exit_code": 0})

    def __call__(self, tool_name, args, **kwargs):
        self.calls.append((tool_name, args, kwargs))
        return self.response


def _write_tool(home, filename, text):
    tools = home / "tools"
    tools.mkdir(exist_ok=True)
    (tools / filename).write_text(text, encoding="utf-8")


@pytest.fixture()
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


@pytest.fixture()
def isolated_registry(monkeypatch):
    """Give the real PluginManager an isolated copy of registry state."""
    import tools.terminal_tool as terminal_mod
    from tools.registry import registry

    for attr in (
        "_tools",
        "_toolset_checks",
        "_toolset_aliases",
        "_plugin_override_policy",
    ):
        monkeypatch.setattr(registry, attr, getattr(registry, attr).copy())
    monkeypatch.setattr(registry, "_generation", registry._generation)
    return registry, terminal_mod


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------

def test_load_spec_builds_function_schema(home):
    _write_tool(home, "greet.yaml", (
        "name: greet\n"
        "description: Say hi\n"
        "command: 'echo hi $HERMES_TOOL_ARG_WHO'\n"
        "parameters:\n"
        "  who:\n"
        "    type: string\n"
        "    description: whom to greet\n"
        "    required: true\n"
        "  loud:\n"
        "    type: boolean\n"
    ))
    name, schema, command, timeout = _load_spec(home / "tools" / "greet.yaml")
    assert name == "greet"
    assert command == "echo hi $HERMES_TOOL_ARG_WHO"
    assert timeout == 60
    assert schema["name"] == "greet"
    assert schema["parameters"]["type"] == "object"
    assert schema["parameters"]["properties"]["who"] == {
        "type": "string", "description": "whom to greet",
    }
    assert schema["parameters"]["properties"]["loud"] == {"type": "boolean"}
    assert schema["parameters"]["required"] == ["who"]


def test_load_spec_enum_and_timeout_cap(home):
    _write_tool(home, "pick.yaml", (
        "name: pick\n"
        "command: 'echo $HERMES_TOOL_ARG_MODE'\n"
        "timeout: 99999\n"
        "parameters:\n"
        "  mode:\n"
        "    type: string\n"
        "    enum: [a, b, c]\n"
    ))
    _, schema, _, timeout = _load_spec(home / "tools" / "pick.yaml")
    assert schema["parameters"]["properties"]["mode"]["enum"] == ["a", "b", "c"]
    assert timeout == 600


@pytest.mark.parametrize("body", [
    "description: no name\ncommand: 'echo x'\n",
    "name: no_cmd\n",
    "name: 'bad name'\ncommand: 'echo x'\n",
    "name: ok\ncommand: 'echo x'\nparameters:\n  q:\n    type: date\n",
    "name: ok\ncommand: 'echo x'\ntimeout: -5\n",
    "- just\n- a\n- list\n",
])
def test_load_spec_rejects_malformed(home, body):
    _write_tool(home, "bad.yaml", body)
    with pytest.raises(ValueError):
        _load_spec(home / "tools" / "bad.yaml")


def test_load_spec_rejects_case_folded_parameter_collision(home):
    _write_tool(home, "collision.yaml", (
        "name: collision\n"
        "command: 'echo x'\n"
        "parameters:\n"
        "  query: {type: string}\n"
        "  QUERY: {type: string}\n"
    ))
    with pytest.raises(ValueError, match="same environment variable"):
        _load_spec(home / "tools" / "collision.yaml")


def test_coerce_timeout_defaults_and_bounds():
    assert _coerce_timeout(None) == 60
    assert _coerce_timeout(30) == 30
    assert _coerce_timeout(10_000) == 600
    with pytest.raises(ValueError):
        _coerce_timeout(0)
    with pytest.raises(ValueError):
        _coerce_timeout("soon")


# ---------------------------------------------------------------------------
# Discovery + registration
# ---------------------------------------------------------------------------

def test_register_discovers_and_skips_malformed(home):
    _write_tool(home, "good.yaml", "name: good\ncommand: 'echo ok'\n")
    _write_tool(home, "broken.yaml", "name: 'has space'\ncommand: 'echo no'\n")
    _write_tool(home, "notyaml.txt", "name: ignored\ncommand: 'echo no'\n")
    ctx = _FakeCtx()
    register(ctx)
    assert set(ctx.registered) == {"good"}
    assert ctx.registered["good"]["toolset"] == "custom"


def test_register_survives_context_rejection(home):
    _write_tool(home, "dup.yaml", "name: context_rejected\ncommand: 'echo no'\n")
    _write_tool(home, "fine.yaml", "name: fine\ncommand: 'echo yes'\n")
    ctx = _FakeCtx(collide={"context_rejected"})
    register(ctx)
    assert set(ctx.registered) == {"fine"}


def test_register_no_tools_dir_is_noop(home):
    ctx = _FakeCtx()
    register(ctx)
    assert ctx.registered == {}


# ---------------------------------------------------------------------------
# Execution contract
# ---------------------------------------------------------------------------

def _handler_for(command, params, timeout=60, response=None):
    dispatch = _DispatchSpy(response)
    return _make_handler(dispatch, command, params, timeout), dispatch


def _run_bash(command, *, env=None):
    return subprocess.run(
        [shutil.which("bash"), "-c", command],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_handler_routes_namespaced_params_through_terminal():
    handler, dispatch = _handler_for(
        'printf "%s %s" "$HERMES_TOOL_ARG_GREETING" "$HERMES_TOOL_ARG_NAME"',
        ["greeting", "name"],
        timeout=37,
    )
    result = handler(
        {"greeting": "hello", "name": "ada"},
        task_id="task-unit",
        session_id="session-unit",
    )
    assert json.loads(result) == {"output": "ok", "exit_code": 0}
    assert len(dispatch.calls) == 1
    tool_name, terminal_args, kwargs = dispatch.calls[0]
    assert tool_name == "terminal"
    assert terminal_args["timeout"] == 37
    assert kwargs == {"task_id": "task-unit", "session_id": "session-unit"}
    completed = _run_bash(terminal_args["command"])
    assert completed.returncode == 0
    assert completed.stdout == "hello ada"


def test_handler_boolean_stringified():
    handler, dispatch = _handler_for(
        'printf %s "$HERMES_TOOL_ARG_FLAG"', ["flag"]
    )
    handler({"flag": True})
    completed = _run_bash(dispatch.calls[0][1]["command"])
    assert completed.stdout == "true"


def test_handler_returns_terminal_pending_approval_unchanged():
    pending = {
        "output": "",
        "exit_code": -1,
        "error": "",
        "status": "pending_approval",
        "approval_pending": True,
        "pattern_key": "dangerous-test",
    }
    response = json.dumps(pending)
    handler, dispatch = _handler_for("rm -rf /", [], response=response)
    assert handler({}) == response
    assert dispatch.calls[0][0] == "terminal"


def test_handler_rejects_missing_required_parameter_without_dispatching():
    dispatch = _DispatchSpy()
    handler = _make_handler(
        dispatch,
        'printf %s "$HERMES_TOOL_ARG_QUERY"',
        ["query"],
        60,
        required_names=["query"],
    )
    result = json.loads(handler({}))
    assert "Missing required parameter" in result["error"]
    assert result["success"] is False
    assert dispatch.calls == []


def test_missing_optional_parameter_shadows_inherited_value():
    command = _build_terminal_command(
        'printf %s "$HERMES_TOOL_ARG_QUERY"', ["query"], {}
    )
    run_env = os.environ.copy()
    run_env["HERMES_TOOL_ARG_QUERY"] = "stale-host-value"
    completed = _run_bash(command, env=run_env)
    assert completed.returncode == 0
    assert completed.stdout == ""


def test_template_trailing_comment_does_not_consume_subshell_close():
    command = _build_terminal_command("printf ok # trailing comment", [], {})
    completed = _run_bash(command)
    assert completed.returncode == 0
    assert completed.stdout == "ok"


def test_model_params_do_not_clobber_inherited_environment():
    command = _build_terminal_command(
        'printf "%s|%s|%s" "$HERMES_TOOL_ARG_PATH" "$PATH" "$TEST_API_KEY"',
        ["path"],
        {"path": "/model/value"},
    )
    run_env = os.environ.copy()
    run_env["PATH"] = "/ambient/path"
    run_env["TEST_API_KEY"] = "ambient-secret"
    completed = _run_bash(command, env=run_env)
    assert completed.returncode == 0
    assert completed.stdout == "/model/value|/ambient/path|ambient-secret"


def test_handler_rejects_nul_without_dispatching():
    handler, dispatch = _handler_for("printf ignored", ["query"])
    result = json.loads(handler({"query": "before\x00after"}))
    assert "NUL" in result["error"]
    assert result["success"] is False
    assert dispatch.calls == []


def test_shell_injection_is_neutralized(tmp_path):
    sentinel = tmp_path / "PWNED"
    payload = f"$(touch {sentinel}); printf INJECTED; 'quoted'"
    command = _build_terminal_command(
        'printf "value=%s" "$HERMES_TOOL_ARG_ARG"',
        ["arg"],
        {"arg": payload},
    )
    completed = _run_bash(command)
    assert completed.returncode == 0
    assert completed.stdout == f"value={payload}"
    assert not sentinel.exists()


def test_terminal_environment_eval_layer_preserves_quoting(tmp_path):
    """Exercise the extra eval layer used by real terminal environments."""
    from tools.environments.local import LocalEnvironment

    sentinel = tmp_path / "PWNED_THROUGH_EVAL"
    payload = f"line one\n$(touch {sentinel}); 'single' \\ backslash"
    command = _build_terminal_command(
        'printf %s "$HERMES_TOOL_ARG_ARG"', ["arg"], {"arg": payload}
    )
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    try:
        result = environment.execute(command, timeout=10, bounded_capture=True)
    finally:
        environment.cleanup()
    assert result["returncode"] == 0
    assert result["output"] == payload
    assert not sentinel.exists()


def test_plugin_manager_registry_e2e(
    home, isolated_registry, monkeypatch, tmp_path,
):
    """Exercise YAML discovery -> PluginManager -> registry -> terminal."""
    registry, terminal_mod = isolated_registry
    tool_name = "yaml_tools_e2e_probe"
    sentinel = tmp_path / "PWNED_E2E"
    payload = f"$(touch {sentinel}); printf INJECTED; 'quoted'"
    _write_tool(home, "e2e.yaml", (
        f"name: {tool_name}\n"
        "description: E2E custom tool\n"
        "command: >-\n"
        "  printf '%s|%s|%s|%s'\n"
        "  \"$HERMES_TOOL_ARG_QUERY\" \"$PATH\" \"$QUERY\" \"$TEST_API_KEY\"\n"
        "timeout: 37\n"
        "parameters:\n"
        "  query:\n"
        "    type: string\n"
        "    required: true\n"
    ))
    _write_tool(home, "terminal-collision.yaml", (
        "name: terminal\n"
        "description: Must not replace or claim the built-in\n"
        "command: 'printf collision'\n"
    ))

    pending = {
        "output": "",
        "exit_code": -1,
        "error": "",
        "status": "pending_approval",
        "approval_pending": True,
        "pattern_key": "e2e-approval",
    }
    terminal_calls = []

    def fake_terminal_tool(**kwargs):
        terminal_calls.append(kwargs)
        return json.dumps(pending)

    monkeypatch.setattr(terminal_mod, "terminal_tool", fake_terminal_tool)

    from hermes_cli.plugins import PluginManager
    from toolsets import resolve_toolset

    plugin_dir = Path(__file__).resolve().parents[2] / "plugins" / "yaml_tools"
    manager = PluginManager()
    manifest = manager._parse_manifest(
        plugin_dir / "plugin.yaml", plugin_dir, "bundled", ""
    )
    assert manifest is not None

    parent_module_name = "hermes_plugins"
    module_name = "hermes_plugins.yaml_tools"
    previous_parent_module = sys.modules.get(parent_module_name)
    previous_module = sys.modules.get(module_name)
    try:
        original_terminal_entry = registry.get_entry("terminal")
        assert original_terminal_entry is not None
        manager._load_plugin(manifest)

        loaded = manager._plugins[manifest.key or manifest.name]
        assert loaded.error is None
        assert set(loaded.tools_registered) == {tool_name}
        assert manager._plugin_tool_names == {tool_name}
        assert registry.get_entry(tool_name) is not None
        assert registry.get_entry("terminal") is original_terminal_entry
        assert tool_name in resolve_toolset("custom")
        definitions = registry.get_definitions({tool_name}, quiet=True)
        assert definitions[0]["function"]["name"] == tool_name
        assert definitions[0]["function"]["parameters"]["required"] == ["query"]
        assert registry.get_entry("terminal").handler is terminal_mod._handle_terminal

        dispatch_trace = []
        real_dispatch = registry.dispatch

        def traced_dispatch(name, args, **kwargs):
            dispatch_trace.append(name)
            return real_dispatch(name, args, **kwargs)

        monkeypatch.setattr(registry, "dispatch", traced_dispatch)
        result = registry.dispatch(
            tool_name,
            {"query": payload},
            task_id="task-e2e",
            session_id="session-e2e",
        )
        assert json.loads(result) == pending
        assert dispatch_trace == [tool_name, "terminal"]
        assert len(terminal_calls) == 1
        call = terminal_calls[0]
        assert call["timeout"] == 37
        assert call["task_id"] == "task-e2e"
        assert call["session_id"] == "session-e2e"

        runtime_command = call["command"]
        assert "export HERMES_TOOL_ARG_QUERY=" in runtime_command
        assert "export QUERY=" not in runtime_command

        run_env = os.environ.copy()
        ambient_path = run_env.get("PATH", "")
        run_env.update({
            "PATH": ambient_path,
            "QUERY": "ambient-query",
            "TEST_API_KEY": "ambient-secret",
        })
        completed = _run_bash(runtime_command, env=run_env)
        assert completed.returncode == 0
        assert completed.stdout == (
            f"{payload}|{ambient_path}|ambient-query|ambient-secret"
        )
        assert not sentinel.exists()

        # PluginManager's force path clears its own attribution state but
        # intentionally keeps registry entries. A reload must replace this
        # plugin's prior custom handler instead of treating it as a collision.
        previous_handler = registry.get_entry(tool_name).handler
        _write_tool(home, "e2e.yaml", (
            f"name: {tool_name}\n"
            "description: Reloaded E2E custom tool\n"
            "command: 'printf \"reloaded:%s\" \"$HERMES_TOOL_ARG_QUERY\"'\n"
            "parameters:\n"
            "  query:\n"
            "    type: string\n"
            "    required: true\n"
        ))
        manager._plugins.clear()
        manager._plugin_tool_names.clear()
        dispatch_trace.clear()
        terminal_calls.clear()

        manager._load_plugin(manifest)
        reloaded = manager._plugins[manifest.key or manifest.name]
        assert set(reloaded.tools_registered) == {tool_name}
        assert registry.get_entry(tool_name).handler is not previous_handler
        result = registry.dispatch(tool_name, {"query": "fresh"})
        assert json.loads(result) == pending
        assert dispatch_trace == [tool_name, "terminal"]
        assert len(terminal_calls) == 1
        completed = _run_bash(terminal_calls[0]["command"])
        assert completed.returncode == 0
        assert completed.stdout == "reloaded:fresh"
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        if previous_parent_module is None:
            sys.modules.pop(parent_module_name, None)
        else:
            sys.modules[parent_module_name] = previous_parent_module


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
