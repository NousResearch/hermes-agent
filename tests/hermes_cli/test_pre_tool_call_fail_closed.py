"""``pre_tool_call`` fails closed on every dispatch seam (regression for #160512).

Three call sites resolve the plugin ``pre_tool_call`` verdict: the agent-loop executor
(``agent.tool_executor._pre_tool_block``), ``invoke_tool``'s helper
(``agent.agent_runtime_helpers._pre_tool_block_message``) and ``model_tools.handle_function_call``
(execute_code RPC, connectors, MCP serve, tool_search re-dispatch). Each one used to swallow a
dispatch/discovery error and run the tool, and none of them noticed that a guard plugin the operator
marked ``fail_closed`` had failed to load or was missing, so the tool ran unchecked.

Plugins here load through real discovery from a temp ``HERMES_HOME``.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import plugins as plugins_mod

_ARGS = {"command": "echo hi"}


def _write_plugin(home, name, init_body, manifest_extra=""):
    pdir = home / "plugins" / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.yaml").write_text(
        f"name: {name}\nversion: '0.0.1'\ndescription: test guard\nprovides_hooks: [pre_tool_call]\n{manifest_extra}",
        encoding="utf-8")
    (pdir / "__init__.py").write_text(init_body, encoding="utf-8")


_BLOCKING_GUARD = (
    "def _pre(tool_name='', **kw):\n"
    "    if tool_name == 'terminal':\n"
    "        return {'action': 'block', 'message': 'guard says no'}\n"
    "def register(ctx):\n"
    "    ctx.register_hook('pre_tool_call', _pre)\n"
)
# The reported shape: a guard that needs an external binary and cannot start without it.
_MISSING_BINARY_GUARD = (
    "import shutil\n"
    "def register(ctx):\n"
    "    if shutil.which('tirith-160512-not-installed') is None:\n"
    "        raise RuntimeError('tirith-160512-not-installed not found on PATH')\n"
    "    ctx.register_hook('pre_tool_call', lambda **kw: None)\n"
)
_BROKEN_IMPORT_GUARD = "import module_that_does_not_exist_160512\ndef register(ctx):\n    pass\n"


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    yield home
    plugins_mod._reset_plugin_managers_for_tests()


def _configure(home, plugins_cfg):
    # JSON is YAML: keeps the config literal exact.
    (home / "config.yaml").write_text(json.dumps({"plugins": plugins_cfg}), encoding="utf-8")
    plugins_mod._reset_plugin_managers_for_tests()


def _executor_verdict():
    from agent.tool_executor import _pre_tool_block, _ToolCallRef

    agent = SimpleNamespace(session_id="s1", _current_turn_id="t1", _current_api_request_id="")
    ref = _ToolCallRef(name="terminal", args=dict(_ARGS), task_id="task", call_id="c1", trace=[])
    return _pre_tool_block(agent, ref)[0]


def _invoke_tool_verdict():
    from agent.agent_runtime_helpers import _pre_tool_block_message

    agent = SimpleNamespace(session_id="s1", _current_turn_id="t1", _current_api_request_id="")
    return _pre_tool_block_message(agent, "terminal", dict(_ARGS), "task", "c1", [])[0]


def _handle_function_call_verdict():
    """``None`` when the registry ran the tool, else the error the caller got back."""
    import model_tools

    ran = []
    with patch.object(model_tools.registry, "dispatch",
                      side_effect=lambda name, args, **kw: ran.append(name) or json.dumps({"output": "ran"})):
        result = model_tools.handle_function_call("terminal", dict(_ARGS), task_id="task", tool_call_id="c1")
    return None if ran else json.loads(result).get("error")


SEAMS = {"agent_executor": _executor_verdict, "invoke_tool": _invoke_tool_verdict,
         "handle_function_call": _handle_function_call_verdict}


@pytest.mark.parametrize("seam", sorted(SEAMS))
@pytest.mark.parametrize("failure", ["dispatch", "discovery"])
def test_dispatch_error_blocks_and_warns(home, seam, failure, caplog):
    _configure(home, {"enabled": []})
    boom = RuntimeError("policy engine exploded")
    target = (patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=boom) if failure == "dispatch"
              else patch.object(plugins_mod.PluginManager, "_discover_and_load_inner", side_effect=boom))
    with caplog.at_level(logging.WARNING), target:
        verdict = SEAMS[seam]()
    assert verdict is not None, f"{seam} ran terminal after the pre_tool_call check raised"
    assert "terminal" in verdict and "RuntimeError" in verdict and "policy engine exploded" in verdict
    assert any(r.levelno == logging.WARNING and "terminal" in r.getMessage() for r in caplog.records)


# How an operator marks a guard fail-closed: in its manifest, in config, or the catalog-guard shape
# (``config_schema.fail_closed.default: true`` on a pre_tool_call plugin).
MARKINGS = {
    "manifest": ("fail_closed: true\n", {}),
    "entry": ("", {"fail_closed": True}),
    "entry_settings": ("", {"settings": {"fail_closed": True}}),
    "config_schema_default": ("config_schema:\n  fail_closed:\n    type: bool\n    default: true\n", {}),
}
BROKEN = {"register_raises": _MISSING_BINARY_GUARD, "import_fails": _BROKEN_IMPORT_GUARD}


@pytest.mark.parametrize("seam", sorted(SEAMS))
@pytest.mark.parametrize("marking", sorted(MARKINGS))
@pytest.mark.parametrize("broken", sorted(BROKEN))
def test_enabled_fail_closed_plugin_that_failed_to_load_blocks(home, seam, marking, broken):
    manifest_extra, entry = MARKINGS[marking]
    _write_plugin(home, "guard", BROKEN[broken], manifest_extra)
    _configure(home, {"enabled": ["guard"], "entries": {"guard": entry} if entry else {}})
    verdict = SEAMS[seam]()
    assert verdict is not None, f"{seam} ran terminal while fail-closed plugin 'guard' was not loaded"
    assert "guard" in verdict


@pytest.mark.parametrize("seam", sorted(SEAMS))
def test_requires_hermes_failure_of_fail_closed_plugin_blocks(home, seam, monkeypatch):
    # The suite seeds an "unknown" version, which every requirement accepts.
    monkeypatch.setattr("hermes_cli.plugins_manifest.running_hermes_version", lambda: "1.0.0")
    _write_plugin(home, "guard", _BLOCKING_GUARD, "fail_closed: true\nrequires_hermes: '>=999.0'\n")
    _configure(home, {"enabled": ["guard"]})
    verdict = SEAMS[seam]()
    assert verdict is not None and "guard" in verdict and "999" in verdict


@pytest.mark.parametrize("seam", sorted(SEAMS))
@pytest.mark.parametrize("case", ["files_missing", "enabled_is_a_string"])
def test_configured_fail_closed_plugin_that_is_not_running_blocks(home, seam, case):
    if case == "files_missing":
        cfg = {"enabled": ["guard"], "entries": {"guard": {"fail_closed": True}}}
    else:
        _write_plugin(home, "guard", _BLOCKING_GUARD)
        cfg = {"enabled": "guard", "entries": {"guard": {"fail_closed": True}}}
    _configure(home, cfg)
    verdict = SEAMS[seam]()
    assert verdict is not None, f"{seam} ran terminal while fail-closed plugin 'guard' was not running"
    assert "guard" in verdict


@pytest.mark.parametrize("seam", sorted(SEAMS))
@pytest.mark.parametrize("opt_out", ["disabled", "entry_false"])
def test_operator_opt_out_lets_tools_run(home, seam, opt_out):
    _write_plugin(home, "guard", _MISSING_BINARY_GUARD, "fail_closed: true\n")
    cfg = ({"enabled": [], "disabled": ["guard"], "entries": {"guard": {"fail_closed": True}}}
           if opt_out == "disabled" else
           {"enabled": ["guard"], "entries": {"guard": {"fail_closed": False}}})
    _configure(home, cfg)
    assert SEAMS[seam]() is None


@pytest.mark.parametrize("seam", sorted(SEAMS))
def test_ordinary_plugin_load_failure_still_fails_open(home, seam):
    """A plugin nobody marked fail-closed that fails to load keeps today's behavior: logged, skipped."""
    _write_plugin(home, "helper", _MISSING_BINARY_GUARD)
    _write_plugin(home, "other", _BROKEN_IMPORT_GUARD)
    _configure(home, {"enabled": ["helper", "other"]})
    assert SEAMS[seam]() is None
    manager = plugins_mod.get_plugin_manager()
    assert manager._plugins["helper"].error and manager._plugins["other"].error


@pytest.mark.parametrize("seam", sorted(SEAMS))
def test_loaded_fail_closed_guard_still_decides(home, seam):
    """A healthy fail-closed guard is not blocked by the new check: its own verdict stands."""
    _write_plugin(home, "guard", _BLOCKING_GUARD, "fail_closed: true\n")
    _configure(home, {"enabled": ["guard"]})
    assert SEAMS[seam]() == "guard says no"


_PARTIAL_GUARD = (
    "import functools\n"
    "def _pre(ctx, *, tool_name='', **kw):\n"
    "    {body}\n"
    "def register(ctx):\n"
    "    ctx.register_hook('pre_tool_call', functools.partial(_pre, ctx))\n"
)


@pytest.mark.parametrize("seam", sorted(SEAMS))
@pytest.mark.parametrize("mode", ["blocks", "raises", "allows"])
def test_partial_callback_guards_work_on_every_seam(home, seam, mode):
    body = {"blocks": "return {'action': 'block', 'message': 'partial guard says no'}",
            "raises": "raise RuntimeError('guard crashed')", "allows": "return None"}[mode]
    _write_plugin(home, "guard", _PARTIAL_GUARD.format(body=body))
    _configure(home, {"enabled": ["guard"]})
    verdict = SEAMS[seam]()
    if mode == "allows":
        assert verdict is None
    elif mode == "blocks":
        assert verdict == "partial guard says no"
    else:
        assert verdict is not None and "RuntimeError" in verdict


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("seam", sorted(SEAMS))
def test_fail_closed_shell_hook_with_missing_binary_blocks(home, seam, tmp_path):
    from agent import shell_hooks

    _configure(home, {"enabled": []})
    shell_hooks.reset_for_tests()
    try:
        cfg = {"hooks": {"pre_tool_call": [{"matcher": "terminal", "fail_closed": True,
                                             "command": str(tmp_path / "not-installed-guard")}]}}
        assert len(shell_hooks.register_from_config(cfg, accept_hooks=True)) == 1
        verdict = SEAMS[seam]()
    finally:
        shell_hooks.reset_for_tests()
    assert verdict is not None and "failed closed" in verdict
