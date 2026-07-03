"""Non-runtime parity tests for gateway display config resolution.

These tests exercise only in-memory config dictionaries and pure display
helpers. They do not launch the gateway, platform adapters, agents, providers,
cron, or read the real profile config/credentials/runtime state.
"""

from __future__ import annotations

import pytest

from gateway.display_config import OVERRIDEABLE_KEYS, resolve_display_setting


@pytest.mark.parametrize(
    ("setting", "global_value", "platform_value", "expected"),
    [
        ("tool_progress", "off", "all", "all"),
        ("show_reasoning", False, "true", True),
        ("tool_progress_grouping", "accumulate", "separate", "separate"),
        ("tool_preview_length", 10, "25", 25),
        ("interim_assistant_messages", False, "on", True),
        ("long_running_notifications", False, "yes", True),
        ("busy_ack_detail", True, "0", False),
        ("cleanup_progress", True, "false", False),
    ],
)
def test_platform_display_overrides_win_over_global_display_values(
    setting: str,
    global_value: object,
    platform_value: object,
    expected: object,
) -> None:
    config = {
        "display": {
            setting: global_value,
            "platforms": {"telegram": {setting: platform_value}},
        }
    }

    assert resolve_display_setting(config, "telegram", setting) == expected


@pytest.mark.parametrize(
    ("platform_key", "setting", "global_value", "expected"),
    [
        # Global user config intentionally overrides built-in platform tiers.
        ("telegram", "tool_progress", "new", "new"),
        ("slack", "tool_progress", "all", "all"),
        ("webhook", "interim_assistant_messages", True, True),
        ("signal", "busy_ack_detail", True, True),
        ("discord", "tool_preview_length", "7", 7),
    ],
)
def test_global_display_values_win_when_platform_override_is_absent(
    platform_key: str,
    setting: str,
    global_value: object,
    expected: object,
) -> None:
    config = {"display": {setting: global_value, "platforms": {platform_key: {}}}}

    assert resolve_display_setting(config, platform_key, setting) == expected


def test_builtin_platform_defaults_are_quiet_for_mobile_and_batch_surfaces() -> None:
    assert resolve_display_setting({}, "telegram", "tool_progress") == "off"
    assert resolve_display_setting({}, "slack", "tool_progress") == "off"
    assert resolve_display_setting({}, "signal", "tool_progress") == "off"
    assert resolve_display_setting({}, "webhook", "tool_progress") == "off"
    assert resolve_display_setting({}, "webhook", "tool_preview_length") == 0
    assert resolve_display_setting({}, "discord", "tool_progress") == "all"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (False, "off"),
        (True, "all"),
        ("OFF", "off"),
        ("New", "new"),
        ("VERBOSE", "verbose"),
    ],
)
def test_tool_progress_normalises_yaml_boolean_and_string_forms(
    raw_value: object,
    expected: str,
) -> None:
    config = {"display": {"tool_progress": raw_value}}

    assert resolve_display_setting(config, "discord", "tool_progress") == expected


@pytest.mark.parametrize(
    ("setting", "raw_value", "expected"),
    [
        ("show_reasoning", "on", True),
        ("show_reasoning", "off", False),
        ("interim_assistant_messages", "yes", True),
        ("long_running_notifications", "no", False),
        ("busy_ack_detail", "1", True),
        ("cleanup_progress", "0", False),
    ],
)
def test_boolean_display_knobs_normalise_common_yaml_string_forms(
    setting: str,
    raw_value: str,
    expected: bool,
) -> None:
    config = {"display": {setting: raw_value}}

    assert resolve_display_setting(config, "discord", setting) is expected


def test_display_streaming_global_is_cli_only_but_platform_override_is_gateway_visible() -> None:
    config = {
        "streaming": {"enabled": False},
        "display": {
            "streaming": True,
            "platforms": {"telegram": {"streaming": "false"}},
        },
    }

    assert resolve_display_setting({"display": {"streaming": True}}, "telegram", "streaming", "fallback") == "fallback"
    assert resolve_display_setting(config, "telegram", "streaming", "fallback") is False


def test_legacy_tool_progress_overrides_remain_lower_precedence_than_platforms() -> None:
    config = {
        "display": {
            "tool_progress": "all",
            "tool_progress_overrides": {"telegram": "off", "slack": "new"},
            "platforms": {"telegram": {"tool_progress": "verbose"}},
        }
    }

    assert resolve_display_setting(config, "telegram", "tool_progress") == "verbose"
    assert resolve_display_setting(config, "slack", "tool_progress") == "new"


def test_invalid_grouping_and_preview_values_fall_back_safely() -> None:
    config = {
        "display": {
            "tool_progress_grouping": "one-message-per-cloud",
            "tool_preview_length": "not-an-int",
        }
    }

    assert resolve_display_setting(config, "discord", "tool_progress_grouping") == "accumulate"
    assert resolve_display_setting(config, "discord", "tool_preview_length") == 0


def test_unknown_setting_uses_explicit_fallback_without_reading_external_config() -> None:
    assert resolve_display_setting({}, "unknown-platform", "not_a_display_key", "sentinel") == "sentinel"


def test_overrideable_keys_cover_gateway_user_facing_display_knobs() -> None:
    assert {
        "tool_progress",
        "tool_progress_grouping",
        "show_reasoning",
        "tool_preview_length",
        "streaming",
        "interim_assistant_messages",
        "long_running_notifications",
        "busy_ack_detail",
        "cleanup_progress",
    }.issubset(OVERRIDEABLE_KEYS)
