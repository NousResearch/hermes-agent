"""Tests for runtime refresh/restart advisories after config/tool changes."""

from argparse import Namespace
from unittest.mock import patch

from hermes_cli.change_impact import ImpactScope, advisory_for_config_key, format_advisory
from hermes_cli.tools_config import tools_disable_enable_command
from hermes_cli.config import set_config_value


def test_toolset_config_requires_new_session_advisory():
    advisory = advisory_for_config_key("platform_toolsets.cli")

    assert advisory is not None
    assert advisory.scope is ImpactScope.NEW_SESSION
    assert advisory.action == "/new"
    assert "tool schema" in advisory.reason.lower()
    assert "current session" in format_advisory(advisory).lower()


def test_gateway_platform_config_requires_gateway_restart_advisory():
    advisory = advisory_for_config_key("gateway.platforms.discord.enabled")

    assert advisory is not None
    assert advisory.scope is ImpactScope.GATEWAY_RESTART
    assert advisory.action == "hermes gateway restart"
    assert "gateway" in advisory.reason.lower()


def test_model_config_is_next_session_or_process_advisory():
    advisory = advisory_for_config_key("model.default")

    assert advisory is not None
    assert advisory.scope is ImpactScope.NEW_SESSION
    assert advisory.action == "/new"
    assert "new sessions" in advisory.reason.lower()


def test_env_secret_config_requires_gateway_restart_advisory(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").touch()

    set_config_value("DISCORD_BOT_TOKEN", "test-token")

    out = capsys.readouterr().out
    assert "Runtime note" in out
    assert "gateway restart" in out.lower()


def test_tools_disable_enable_prints_new_session_advisory(capsys):
    config = {"platform_toolsets": {"cli": ["web", "memory"]}}
    with patch("hermes_cli.tools_config.load_config", return_value=config), \
         patch("hermes_cli.tools_config.save_config"):
        tools_disable_enable_command(
            Namespace(tools_action="disable", names=["web"], platform="cli")
        )

    out = capsys.readouterr().out
    assert "Runtime note" in out
    assert "/new" in out
    assert "tool schema" in out.lower()
