"""Saved platform MCP selections must not expand to unrelated servers."""

import pytest

from hermes_cli.tools_config import _get_platform_tools


@pytest.mark.parametrize("selection", ["linear", "mcp-linear"])
@pytest.mark.parametrize("include_defaults", [True, False])
def test_platform_mcp_selection_is_an_allowlist(selection, include_defaults):
    config = {
        "platform_toolsets": {"cli": ["hermes-cli", selection]},
        "mcp_servers": {
            "linear": {"command": "fixture-linear"},
            "github": {"command": "fixture-github"},
            "notion": {"command": "fixture-notion"},
        },
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=include_defaults)
    assert selection in enabled
    assert not ({"github", "mcp-github", "notion", "mcp-notion"} & enabled)


@pytest.mark.parametrize("selection", ["linear", "mcp-linear"])
def test_no_mcp_overrides_either_server_spelling(selection):
    config = {
        "platform_toolsets": {"cli": ["hermes-cli", selection, "no_mcp", "custom-fixture"]},
        "mcp_servers": {"linear": {"command": "fixture-linear"}},
    }
    enabled = _get_platform_tools(config, "cli")
    assert not ({"linear", "mcp-linear", "no_mcp"} & enabled)
    assert "custom-fixture" in enabled


@pytest.mark.parametrize("include_defaults", [True, False])
def test_unnamed_servers_keep_caller_selected_default_policy(include_defaults):
    config = {
        "platform_toolsets": {"cli": ["hermes-cli", "custom-fixture"]},
        "mcp_servers": {"linear": {"command": "fixture-linear"}},
    }
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=include_defaults)
    assert ("linear" in enabled) is include_defaults
    assert "custom-fixture" in enabled


@pytest.mark.parametrize("portable", [set(), {"portable-fixture"}, None])
def test_portable_discovery_empty_or_failure_keeps_native_selection(monkeypatch, portable):
    def discover():
        if portable is None:
            raise RuntimeError("fixture discovery failure")
        return portable

    monkeypatch.setattr("hermes_cli.plugins.get_portable_mcp_server_names_nowait", discover)
    config = {
        "platform_toolsets": {"cli": ["hermes-cli", "mcp-linear"]},
        "mcp_servers": {"linear": {"command": "fixture-linear"}},
    }
    enabled = _get_platform_tools(config, "cli")
    assert "mcp-linear" in enabled
    assert "portable-fixture" not in enabled


def test_portable_canonical_selection_does_not_enable_native_server(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.plugins.get_portable_mcp_server_names_nowait", lambda: {"portable-fixture"}
    )
    config = {
        "platform_toolsets": {"cli": ["hermes-cli", "mcp-portable-fixture"]},
        "mcp_servers": {"linear": {"command": "fixture-linear"}},
    }
    enabled = _get_platform_tools(config, "cli")
    assert "mcp-portable-fixture" in enabled
    assert "linear" not in enabled
