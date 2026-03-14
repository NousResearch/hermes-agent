"""Tests for hermes_cli.tools_config platform tool persistence."""

from hermes_cli.tools_config import _get_platform_tools, _platform_toolset_summary


def test_get_platform_tools_uses_default_when_platform_not_configured():
    config = {}

    enabled = _get_platform_tools(config, "cli")

    assert enabled


def test_get_platform_tools_preserves_explicit_empty_selection():
    config = {"platform_toolsets": {"cli": []}}

    enabled = _get_platform_tools(config, "cli")

    assert enabled == set()


def test_platform_toolset_summary_uses_explicit_platform_list():
    config = {}

    summary = _platform_toolset_summary(config, platforms=["cli"])

    assert set(summary.keys()) == {"cli"}
    assert summary["cli"] == _get_platform_tools(config, "cli")


def test_save_platform_tools_preserves_mcp_server_names():
    """Ensure MCP server names are preserved when saving platform tools.

    Regression test for https://github.com/NousResearch/hermes-agent/issues/1247
    """
    from unittest.mock import patch
    from hermes_cli.tools_config import _save_platform_tools

    config = {
        "platform_toolsets": {
            "cli": ["web", "terminal", "time", "github", "custom-mcp-server"]
        }
    }

    new_selection = {"web", "browser"}

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", new_selection)

    saved_toolsets = config["platform_toolsets"]["cli"]

    assert "time" in saved_toolsets
    assert "github" in saved_toolsets
    assert "custom-mcp-server" in saved_toolsets
    assert "web" in saved_toolsets
    assert "browser" in saved_toolsets
    assert "terminal" not in saved_toolsets


def test_save_platform_tools_handles_empty_existing_config():
    """Saving platform tools works when no existing config exists."""
    from unittest.mock import patch
    from hermes_cli.tools_config import _save_platform_tools

    config = {}

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "telegram", {"web", "terminal"})

    saved_toolsets = config["platform_toolsets"]["telegram"]
    assert "web" in saved_toolsets
    assert "terminal" in saved_toolsets


def test_save_platform_tools_handles_invalid_existing_config():
    """Saving platform tools works when existing config is not a list."""
    from unittest.mock import patch
    from hermes_cli.tools_config import _save_platform_tools

    config = {
        "platform_toolsets": {
            "cli": "invalid-string-value"
        }
    }

    with patch("hermes_cli.tools_config.save_config"):
        _save_platform_tools(config, "cli", {"web"})

    saved_toolsets = config["platform_toolsets"]["cli"]
    assert "web" in saved_toolsets
