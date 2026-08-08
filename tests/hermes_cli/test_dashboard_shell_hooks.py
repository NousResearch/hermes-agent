"""Desktop shell-hook registration regression tests."""

from __future__ import annotations

import os
import shlex
import sys
import types

import pytest
import yaml

from agent import shell_hooks
from hermes_cli import plugins
from hermes_cli.plugins import get_pre_tool_call_block_message


def _args(**overrides):
    defaults = {
        "headless_backend": True,
        "host": "127.0.0.1",
        "insecure": False,
        "isolated": True,
        "no_open": True,
        "open_profile": "",
        "port": 0,
        "skip_build": False,
        "ssh_owner_nonce": None,
        "ssh_session_token_file": None,
        "status": False,
        "stop": False,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def _fresh_hook_registry():
    previous_manager = plugins._plugin_manager
    plugins._plugin_manager = plugins.PluginManager()
    shell_hooks.reset_for_tests()
    yield
    shell_hooks.reset_for_tests()
    plugins._plugin_manager = previous_manager
    os.environ.pop("HERMES_PROFILE_SCOPED_UI", None)


def _configure_allowlisted_block_hook(tmp_path, monkeypatch) -> None:
    home = tmp_path / "hermes-home"
    script = tmp_path / "block.py"
    script.write_text(
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        'print(\'{"action": "block", "message": "desktop-hook-fired"}\')\n',
        encoding="utf-8",
    )
    command = shlex.join([sys.executable, str(script)])
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "hooks": {
                "pre_tool_call": [
                    {"command": command, "matcher": "terminal"},
                ],
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    shell_hooks._record_approval("pre_tool_call", command)


def _stub_dashboard_runtime(main_mod, monkeypatch, events) -> None:
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "default"
    )
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_quietly", lambda: None)
    monkeypatch.setattr("hermes_cli.config.apply_terminal_config_to_env", lambda: None)
    monkeypatch.setattr(plugins, "discover_plugins", lambda: events.append("plugins"))
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        lambda **_kwargs: events.append("mcp"),
    )
    monkeypatch.setattr(
        main_mod,
        "_maybe_setup_dashboard_auth_interactively",
        lambda _args: events.append("auth"),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_logging",
        types.SimpleNamespace(setup_logging=lambda **_kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "fastapi", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.web_server",
        types.SimpleNamespace(start_server=lambda **_kwargs: events.append("server")),
    )


def test_desktop_registers_allowlisted_hook_after_auth(tmp_path, monkeypatch):
    import hermes_cli.main as main_mod

    _configure_allowlisted_block_hook(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    events = []
    _stub_dashboard_runtime(main_mod, monkeypatch, events)

    original_register = shell_hooks.register_from_config

    def register(cfg, *, accept_hooks):
        events.append(("hooks", accept_hooks))
        return original_register(cfg, accept_hooks=accept_hooks)

    monkeypatch.setattr(shell_hooks, "register_from_config", register)

    main_mod.cmd_dashboard(_args())

    assert events == ["plugins", "mcp", "auth", ("hooks", False), "server"]
    assert os.environ["HERMES_PROFILE_SCOPED_UI"] == "1"
    assert (
        get_pre_tool_call_block_message("terminal", {"command": "echo should-not-run"})
        == "desktop-hook-fired"
    )


def test_browser_dashboard_does_not_register_profile_hooks(tmp_path, monkeypatch):
    import hermes_cli.main as main_mod

    _configure_allowlisted_block_hook(tmp_path, monkeypatch)
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    events = []
    _stub_dashboard_runtime(main_mod, monkeypatch, events)

    main_mod.cmd_dashboard(_args())

    assert events == ["plugins", "mcp", "auth", "server"]
    assert os.environ.get("HERMES_PROFILE_SCOPED_UI") is None
    assert (
        get_pre_tool_call_block_message("terminal", {"command": "echo allowed"}) is None
    )


def test_desktop_registration_failure_does_not_block_server(monkeypatch):
    import hermes_cli.main as main_mod

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    events = []
    _stub_dashboard_runtime(main_mod, monkeypatch, events)

    def fail_registration(_cfg, *, accept_hooks):
        events.append(("hooks", accept_hooks))
        raise RuntimeError("test registration failure")

    monkeypatch.setattr(shell_hooks, "register_from_config", fail_registration)

    main_mod.cmd_dashboard(_args())

    assert events == ["plugins", "mcp", "auth", ("hooks", False), "server"]


@pytest.mark.parametrize("flag", ["status", "stop"])
def test_desktop_lifecycle_commands_do_not_register_hooks(monkeypatch, flag):
    import hermes_cli.main as main_mod

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setattr(main_mod, "_report_dashboard_status", lambda: 0)
    monkeypatch.setattr(main_mod, "_find_stale_dashboard_pids", lambda: [])

    calls = []
    monkeypatch.setattr(
        shell_hooks,
        "register_from_config",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(SystemExit) as exc_info:
        main_mod.cmd_dashboard(_args(**{flag: True}))

    assert exc_info.value.code == 0
    assert calls == []
