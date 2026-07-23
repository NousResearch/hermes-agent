"""Behavioral coverage for the default-off outbound messaging toolset."""

from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS, _get_platform_tools


def test_configurator_labels_messaging_as_outbound_side_effects():
    messaging = next(entry for entry in CONFIGURABLE_TOOLSETS if entry[0] == "messaging")

    assert "outbound" in messaging[2].lower()
    assert "side effect" in messaging[2].lower()


def test_default_telegram_excludes_messaging():
    enabled = _get_platform_tools({}, "telegram", include_default_mcp_servers=False)

    assert "messaging" not in enabled


def test_explicit_telegram_messaging_opt_in_includes_messaging():
    enabled = _get_platform_tools(
        {"platform_toolsets": {"telegram": ["hermes-telegram", "messaging"]}},
        "telegram",
        include_default_mcp_servers=False,
    )

    assert "messaging" in enabled


def test_removing_telegram_messaging_opt_in_excludes_messaging():
    enabled = _get_platform_tools(
        {"platform_toolsets": {"telegram": ["hermes-telegram"]}},
        "telegram",
        include_default_mcp_servers=False,
    )

    assert "messaging" not in enabled
