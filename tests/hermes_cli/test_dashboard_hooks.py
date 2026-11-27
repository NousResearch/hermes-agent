"""Tests for dashboard/serve runtime config shell hook registration (#102504)."""

import sys
from argparse import Namespace
import pytest
import yaml

from hermes_cli import main as main_mod
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from agent import shell_hooks as shell_hooks_mod
from hermes_cli import plugins as plugins_mod


def _serve_args(**overrides) -> Namespace:
    base = {
        "accept_hooks": True,
        "command": "serve",
        "cron_command": None,
        "gateway_command": None,
        "headless_backend": True,
        "host": "127.0.0.1",
        "ignore_user_config": False,
        "insecure": False,
        "mcp_action": None,
        "no_open": True,
        "open_profile": "",
        "port": 0,
        "skip_build": True,
        "ssh_owner_nonce": None,
        "ssh_session_token_file": None,
        "status": False,
        "stop": False,
        "tui": False,
        "yolo": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_cmd_dashboard_status_exits_without_hook_or_plugin_registration(monkeypatch):
    """Management command --status must exit early without triggering plugin discovery or hook consent."""
    plugin_calls = []
    hook_calls = []

    monkeypatch.setattr(main_mod, "_report_dashboard_status", lambda: 0)
    monkeypatch.setattr(
        "hermes_cli.plugins.discover_plugins",
        lambda: plugin_calls.append("discover_plugins"),
    )
    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        lambda *a, **k: hook_calls.append("register_from_config"),
    )

    args = _serve_args(status=True)
    with pytest.raises(SystemExit) as exc_info:
        main_mod.cmd_dashboard(args)

    assert exc_info.value.code == 0
    assert len(plugin_calls) == 0
    assert len(hook_calls) == 0


def test_cmd_dashboard_stop_exits_without_hook_or_plugin_registration(monkeypatch):
    """Management command --stop must exit early without triggering plugin discovery or hook consent."""
    plugin_calls = []
    hook_calls = []

    monkeypatch.setattr(main_mod, "_find_stale_dashboard_pids", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.plugins.discover_plugins",
        lambda: plugin_calls.append("discover_plugins"),
    )
    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        lambda *a, **k: hook_calls.append("register_from_config"),
    )

    args = _serve_args(stop=True)
    with pytest.raises(SystemExit) as exc_info:
        main_mod.cmd_dashboard(args)

    assert exc_info.value.code == 0
    assert len(plugin_calls) == 0
    assert len(hook_calls) == 0


def test_cmd_dashboard_registers_hooks_after_plugins_real_path(tmp_path, monkeypatch):
    """On live serve/dashboard startup, plugins are discovered first, followed by shell hook and outbound webhook registration."""
    token = set_hermes_home_override(tmp_path)
    try:
        config_file = tmp_path / "config.yaml"
        config_data = {
            "hooks": {
                "pre_tool_call": [
                    {
                        "command": "echo 'test hook'",
                        "matcher": "terminal",
                    }
                ],
                "outbound_webhooks": [
                    {
                        "url": "http://127.0.0.1:9999/hook",
                        "events": ["on_session_start"],
                    }
                ],
            }
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f)

        events_order = []

        def _mock_discover_plugins():
            events_order.append("discover_plugins")

        registered_shell_hooks = []
        registered_webhooks = []

        def _mock_register_shell_hooks(cfg, accept_hooks=False):
            events_order.append("register_shell_hooks")
            registered_shell_hooks.append((cfg, accept_hooks))

        def _mock_register_webhooks(cfg):
            events_order.append("register_webhooks")
            registered_webhooks.append(cfg)

        monkeypatch.setattr("hermes_cli.plugins.discover_plugins", _mock_discover_plugins)
        monkeypatch.setattr("agent.shell_hooks.register_from_config", _mock_register_shell_hooks)
        monkeypatch.setattr("agent.outbound_webhooks.register_from_config", _mock_register_webhooks)
        monkeypatch.setattr("hermes_cli.main._sync_bundled_skills_quietly", lambda: None)
        monkeypatch.setattr("hermes_cli.resource_limits.apply_nofile_soft_limit", lambda: None)
        monkeypatch.setattr("hermes_cli.web_server.start_server", lambda **kw: events_order.append("start_server"))

        args = _serve_args(accept_hooks=True)
        main_mod.cmd_dashboard(args)

        # Assert ordering: plugin discovery MUST precede hook registration, and both must precede start_server
        assert events_order == [
            "discover_plugins",
            "register_shell_hooks",
            "register_webhooks",
            "start_server",
        ]

        assert len(registered_shell_hooks) == 1
        cfg, accept_hooks = registered_shell_hooks[0]
        assert accept_hooks is True
        assert cfg.get("hooks", {}).get("pre_tool_call") == config_data["hooks"]["pre_tool_call"]

        assert len(registered_webhooks) == 1
        assert registered_webhooks[0].get("hooks", {}).get("outbound_webhooks") == config_data["hooks"]["outbound_webhooks"]
    finally:
        reset_hermes_home_override(token)


def test_cmd_dashboard_preserves_plugin_precedence_in_hook_registry(tmp_path, monkeypatch):
    """Verify that plugin callbacks registered during discover_plugins precede config shell hooks in the hook list."""
    manager = plugins_mod.get_plugin_manager()
    manager._hooks.clear()
    shell_hooks_mod.reset_for_tests()

    def plugin_callback(**kwargs):
        return {"decision": "allow"}

    def fake_discover_plugins():
        # Plugin registers callback during discover_plugins
        manager._hooks.setdefault("pre_tool_call", []).append(plugin_callback)

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", fake_discover_plugins)
    monkeypatch.setattr("hermes_cli.main._sync_bundled_skills_quietly", lambda: None)
    monkeypatch.setattr("hermes_cli.resource_limits.apply_nofile_soft_limit", lambda: None)
    monkeypatch.setattr("hermes_cli.web_server.start_server", lambda **kw: None)

    token = set_hermes_home_override(tmp_path)
    try:
        config_file = tmp_path / "config.yaml"
        config_data = {
            "hooks": {
                "pre_tool_call": [
                    {
                        "command": "echo 'shell hook'",
                        "matcher": "terminal",
                    }
                ]
            }
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f)

        args = _serve_args(accept_hooks=True)
        main_mod.cmd_dashboard(args)

        callbacks = manager._hooks.get("pre_tool_call", [])
        assert len(callbacks) == 2
        # First callback is the plugin's callback
        assert callbacks[0] is plugin_callback
        # Second callback is the shell hook callback
        assert callable(callbacks[1])
    finally:
        reset_hermes_home_override(token)
        manager._hooks.clear()
        shell_hooks_mod.reset_for_tests()


def test_try_fast_serve_launch_dispatches_cleanly(monkeypatch):
    """_try_fast_serve_launch dispatches directly to cmd_dashboard without invoking _prepare_agent_startup."""
    call_order = []

    monkeypatch.setattr(sys, "argv", ["hermes", "serve", "--host", "127.0.0.1", "--port", "0"])
    monkeypatch.setenv("HERMES_DISABLE_FAST_SERVE_LAUNCH", "0")

    def _mock_prepare(args):
        call_order.append("prepare")

    def _mock_dashboard(args):
        call_order.append("dashboard")

    monkeypatch.setattr(main_mod, "_prepare_agent_startup", _mock_prepare)
    monkeypatch.setattr(main_mod, "cmd_dashboard", _mock_dashboard)

    res = main_mod._try_fast_serve_launch()
    assert res is True
    assert call_order == ["dashboard"]
