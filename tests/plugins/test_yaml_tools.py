"""Tests for the yaml_tools plugin (user-defined tools via ~/.hermes/tools/*.yaml).

Covers:
  * schema construction from a YAML spec (types, required, enum, description)
  * validation / rejection of malformed specs
  * discovery + registration through a fake plugin context
  * malformed files and name collisions are skipped, never fatal
  * command execution: parameter passing (name + UPPER form), non-zero exit,
    timeout, output capture
  * SECURITY: model-supplied argument values cannot inject shell
"""

from __future__ import annotations

import json
import shutil
import subprocess
from unittest.mock import patch

import pytest

from plugins.yaml_tools import (
    _coerce_timeout,
    _load_spec,
    _make_handler,
    register,
)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


# ---------------------------------------------------------------------------
# Fake plugin context — collects register_tool() calls like the real loader.
# ---------------------------------------------------------------------------

class _FakeCtx:
    def __init__(self, collide=()):
        self.registered = {}
        self._collide = set(collide)

    def register_tool(self, *, name, toolset, schema, handler, description="", emoji=""):
        if name in self._collide:
            raise ValueError(f"tool '{name}' already registered by another toolset")
        self.registered[name] = {
            "toolset": toolset, "schema": schema, "handler": handler,
            "description": description, "emoji": emoji,
        }


def _write_tool(home, filename, text):
    tools = home / "tools"
    tools.mkdir(exist_ok=True)
    (tools / filename).write_text(text, encoding="utf-8")


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------

def test_load_spec_builds_function_schema(home):
    _write_tool(home, "greet.yaml", (
        "name: greet\n"
        "description: Say hi\n"
        "command: 'echo hi $who'\n"
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
    assert command == "echo hi $who"
    assert timeout == 60  # default
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
        "command: 'echo $mode'\n"
        "timeout: 99999\n"
        "parameters:\n"
        "  mode:\n"
        "    type: string\n"
        "    enum: [a, b, c]\n"
    ))
    _, schema, _, timeout = _load_spec(home / "tools" / "pick.yaml")
    assert schema["parameters"]["properties"]["mode"]["enum"] == ["a", "b", "c"]
    assert timeout == 600  # capped at _MAX_TIMEOUT


@pytest.mark.parametrize("body,reason", [
    ("description: no name\ncommand: 'echo x'\n", "missing name"),
    ("name: no_cmd\n", "missing command"),
    ("name: 'bad name'\ncommand: 'echo x'\n", "invalid tool name"),
    ("name: ok\ncommand: 'echo x'\nparameters:\n  q:\n    type: date\n", "bad type"),
    ("name: ok\ncommand: 'echo x'\ntimeout: -5\n", "bad timeout"),
    ("- just\n- a\n- list\n", "not a mapping"),
])
def test_load_spec_rejects_malformed(home, body, reason):
    _write_tool(home, "bad.yaml", body)
    with pytest.raises(ValueError):
        _load_spec(home / "tools" / "bad.yaml")


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


def test_register_survives_name_collision(home):
    # A YAML tool whose name collides with a built-in: the fake ctx raises
    # (like the real registry); register() must skip it, not crash.
    _write_tool(home, "dup.yaml", "name: read_file\ncommand: 'echo no'\n")
    _write_tool(home, "fine.yaml", "name: fine\ncommand: 'echo yes'\n")
    ctx = _FakeCtx(collide={"read_file"})
    register(ctx)  # must not raise
    assert set(ctx.registered) == {"fine"}


def test_register_no_tools_dir_is_noop(home):
    ctx = _FakeCtx()
    register(ctx)  # ~/.hermes/tools/ does not exist
    assert ctx.registered == {}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _handler_for(command, params, timeout=60):
    return _make_handler(command, params, timeout)


def test_handler_passes_params_as_env_name_and_upper():
    # Template references both $greeting (exact) and $NAME (upper-cased).
    h = _handler_for('echo "$greeting $NAME"', ["greeting", "name"])
    out = json.loads(h({"greeting": "hello", "name": "ada"}))
    assert out["exit_code"] == 0
    assert out["output"].strip() == "hello ada"


def test_handler_boolean_stringified():
    h = _handler_for('echo "$flag"', ["flag"])
    assert json.loads(h({"flag": True}))["output"].strip() == "true"


def test_handler_nonzero_exit_is_error():
    h = _handler_for("exit 3", [])
    out = json.loads(h({}))
    assert out["error"].startswith("Command exited with status 3")
    assert out["success"] is False


def test_handler_timeout():
    # Patch subprocess.run to raise TimeoutExpired so we exercise the handler's
    # timeout branch without spawning/killing a real long-running process
    # (the test harness's live-system guard blocks the os.kill cleanup that a
    # real timeout would trigger).
    h = _handler_for("sleep 5", [], timeout=1)
    with patch("plugins.yaml_tools.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="sleep 5", timeout=1)):
        out = json.loads(h({}))
    assert "timed out after 1s" in out["error"]
    assert out["success"] is False


def test_handler_shell_injection_is_neutralized(tmp_path):
    # The classic attack: a parameter value carrying shell metacharacters.
    # With env-var passing the value is inert — no command substitution, no
    # extra command runs, the sentinel file is never created.
    sentinel = tmp_path / "PWNED"
    h = _handler_for('echo "value=$arg"', ["arg"])
    payload = f'$(touch {sentinel}); echo INJECTED'
    out = json.loads(h({"arg": payload}))
    assert not sentinel.exists()                 # no command executed
    assert out["output"].strip() == f"value={payload}"  # value stayed literal


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
