from hermes_cli.tools_config import _get_platform_tools


def test_whoop_plugin_toolset_defaults_off_for_cli_and_telegram():
    assert "whoop" not in _get_platform_tools({}, "cli", include_default_mcp_servers=False)
    assert "whoop" not in _get_platform_tools({}, "telegram", include_default_mcp_servers=False)


def test_whoop_plugin_toolset_can_be_enabled_explicitly_for_cli_and_telegram():
    config = {
        "platform_toolsets": {
            "cli": ["hermes-cli", "whoop"],
            "telegram": ["hermes-telegram", "whoop"],
        }
    }

    assert "whoop" in _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    assert "whoop" in _get_platform_tools(config, "telegram", include_default_mcp_servers=False)
