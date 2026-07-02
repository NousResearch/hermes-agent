from hermes_cli.platforms import PLATFORMS
from hermes_cli.tools_config import _get_platform_tools


def test_desktop_platform_uses_cli_default_toolset():
    assert PLATFORMS["desktop"].default_toolset == "hermes-cli"

    enabled = _get_platform_tools({}, "desktop", include_default_mcp_servers=False)

    assert "web" in enabled
    assert "terminal" in enabled
    assert "file" in enabled
    assert "hermes-desktop" not in enabled


def test_desktop_platform_inherits_saved_cli_toolsets():
    enabled = _get_platform_tools(
        {"platform_toolsets": {"cli": ["web"]}},
        "desktop",
        include_default_mcp_servers=False,
    )

    assert "web" in enabled
    assert "terminal" not in enabled


def test_desktop_platform_override_wins_when_present():
    enabled = _get_platform_tools(
        {"platform_toolsets": {"cli": ["terminal"], "desktop": ["web"]}},
        "desktop",
        include_default_mcp_servers=False,
    )

    assert "web" in enabled
    assert "terminal" not in enabled
