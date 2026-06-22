"""Tests for cube_sandbox plugin (high-risk tool overrides)."""

from __future__ import annotations

import os

import yaml

import pytest


def _seed_terminal_tool():
    from tools.registry import registry

    if registry.get_entry("terminal") is not None:
        return
    registry.register(
        name="terminal",
        toolset="terminal",
        schema={
            "name": "terminal",
            "description": "test stub",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
        handler=lambda args, **kw: "built-in",
    )


@pytest.fixture
def cube_plugin_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "SANDBOX_TYPE": "cube",
                "plugins": {"enabled": ["cube_sandbox"]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SANDBOX_TYPE", "cube")
    monkeypatch.setenv("CUBE_API_URL", "http://cube.test:3000")
    monkeypatch.setenv("CUBE_TEMPLATE_ID", "tpl-test")
    monkeypatch.setenv("CUBE_API_KEY", "test-key")
    return hermes_home


def test_plugin_overrides_terminal_handler(cube_plugin_env, monkeypatch):
    _seed_terminal_tool()
    from hermes_cli.plugins import PluginManager
    from tools.registry import registry

    builtin_handler = registry.get_entry("terminal").handler
    calls = {"cube_env": False}

    def _fake_terminal(**kwargs):
        import os

        calls["cube_env"] = os.getenv("TERMINAL_ENV") == "cube_sandbox"
        return '{"output":"ok","exit_code":0}'

    monkeypatch.setattr(
        "tools.terminal_tool.terminal_tool",
        _fake_terminal,
        raising=False,
    )

    mgr = PluginManager()
    mgr.discover_and_load(force=True)

    entry = registry.get_entry("terminal")
    assert entry is not None
    assert entry.handler is not builtin_handler

    entry.handler({"command": "echo hi"}, task_id="task-1")
    assert calls["cube_env"] is True


def test_plugin_skipped_without_sandbox_type(tmp_path, monkeypatch):
    from hermes_cli.plugins import PluginManager
    from tools.registry import registry

    _seed_terminal_tool()

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["cube_sandbox"]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SANDBOX_TYPE", raising=False)

    before = registry.get_entry("terminal").handler
    PluginManager().discover_and_load(force=True)
    after = registry.get_entry("terminal").handler

    assert after is before


def test_token_client_static_fallback(cube_plugin_env, monkeypatch):
    from hermes_cli.plugins import PluginManager

    monkeypatch.delenv("SANDBOX_TOKEN_API_URL", raising=False)
    monkeypatch.setenv("CUBE_API_KEY", "static-key")

    PluginManager().discover_and_load(force=True)
    from hermes_plugins.cube_sandbox.token_client import acquire_api_key

    assert acquire_api_key("task-1", "task") == "static-key"
