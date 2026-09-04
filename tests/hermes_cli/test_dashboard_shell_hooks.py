"""Dashboard/serve startup registers configured shell hooks before MCP."""

from __future__ import annotations

import shlex
import sys
import types
from argparse import Namespace

import pytest
import yaml


def _args() -> Namespace:
    return Namespace(
        headless_backend=True,
        host="127.0.0.1",
        ignore_user_config=False,
        insecure=False,
        isolated=False,
        no_open=True,
        open_profile="",
        port=0,
        skip_build=False,
        ssh_owner_nonce=None,
        ssh_session_token_file=None,
        status=False,
        stop=False,
    )


def _wire_dashboard_boundaries(monkeypatch, main_mod, order):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "default"
    )
    monkeypatch.setattr(
        "hermes_cli.resource_limits.apply_nofile_soft_limit", lambda: None
    )
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_quietly", lambda: None)
    monkeypatch.setattr(
        main_mod, "_maybe_setup_dashboard_auth_interactively", lambda _args: None
    )
    monkeypatch.setitem(sys.modules, "fastapi", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "hermes_logging",
        types.SimpleNamespace(setup_logging=lambda **_kwargs: None),
    )
    monkeypatch.setattr("hermes_cli.config.apply_terminal_config_to_env", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        lambda **_kwargs: order.append("mcp"),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.web_server",
        types.SimpleNamespace(start_server=lambda **_kwargs: order.append("server")),
    )


@pytest.fixture()
def dashboard_runtime(monkeypatch, tmp_path):
    from agent import shell_hooks
    from hermes_cli import main as main_mod
    from hermes_cli import plugins

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)

    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {})
    shell_hooks.reset_for_tests()
    yield main_mod, plugins, manager, shell_hooks, tmp_path
    shell_hooks.reset_for_tests()


def test_dashboard_registers_allowlisted_hook_before_mcp_and_server(
    dashboard_runtime, monkeypatch
):
    main_mod, plugins, manager, shell_hooks, home = dashboard_runtime
    hook_script = home / "block_hook.py"
    hook_script.write_text(
        "import json\nprint(json.dumps({'decision': 'block', "
        "'reason': 'blocked by dashboard hook'}))\n",
        encoding="utf-8",
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(hook_script))}"
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "hooks": {"pre_tool_call": [{"matcher": "terminal", "command": command}]}
        }),
        encoding="utf-8",
    )
    shell_hooks._record_approval("pre_tool_call", command)

    order = []
    _wire_dashboard_boundaries(monkeypatch, main_mod, order)
    monkeypatch.setattr(
        manager,
        "_discover_and_load_inner",
        lambda: order.append("plugins"),
    )
    real_register = shell_hooks.register_from_config

    def recording_register(config, *, accept_hooks=False):
        order.append(("shell-hooks", accept_hooks))
        return real_register(config, accept_hooks=accept_hooks)

    monkeypatch.setattr(shell_hooks, "register_from_config", recording_register)

    main_mod.cmd_dashboard(_args())

    assert order == ["plugins", ("shell-hooks", False), "mcp", "server"]
    assert plugins.get_plugin_manager() is manager
    block_message = plugins.get_pre_tool_call_block_message(
        "terminal", {"command": "echo allowed-to-reach-hook"}
    )
    assert block_message == "blocked by dashboard hook"


def test_dashboard_shell_hook_registration_failure_stops_startup(
    dashboard_runtime, monkeypatch, capsys
):
    main_mod, _plugins, manager, shell_hooks, _home = dashboard_runtime
    order = []
    _wire_dashboard_boundaries(monkeypatch, main_mod, order)
    monkeypatch.setattr(
        manager,
        "_discover_and_load_inner",
        lambda: order.append("plugins"),
    )

    def fail_registration(_config, *, accept_hooks=False):
        assert accept_hooks is False
        raise RuntimeError("registration exploded")

    monkeypatch.setattr(shell_hooks, "register_from_config", fail_registration)

    with pytest.raises(SystemExit) as exc_info:
        main_mod.cmd_dashboard(_args())

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Error:" in stderr
    assert "shell-hook registration" in stderr
    assert order == ["plugins"]
