"""Chat-type scoped display policy tests."""

from types import SimpleNamespace

from gateway.config import Platform
from gateway.display_config import resolve_source_display_setting, resolve_source_tool_progress
from gateway.warning_notifications import warning_notifications_enabled


def test_platform_chat_type_override_wins_over_every_broader_scope():
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_type="group")
    config = {
        "display": {
            "tool_progress": "all",
            "chat_types": {"group": {"tool_progress": "new"}},
            "platforms": {
                "telegram": {
                    "tool_progress": "verbose",
                    "chat_types": {"group": {"tool_progress": "off"}},
                }
            },
        }
    }

    assert resolve_source_display_setting(config, "telegram", source, "tool_progress") == "off"


def test_global_chat_type_override_wins_over_platform_and_global_values():
    source = SimpleNamespace(platform=Platform.DISCORD, chat_type="group")
    config = {
        "display": {
            "interim_assistant_messages": True,
            "chat_types": {"group": {"interim_assistant_messages": False}},
            "platforms": {"discord": {"interim_assistant_messages": True}},
        }
    }

    assert resolve_source_display_setting(
        config, "discord", source, "interim_assistant_messages",
    ) is False


def test_telegram_dm_uses_configured_display_values():
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_type="dm")
    user_config = {
        "display": {
            "tool_progress": "all",
            "interim_assistant_messages": True,
            "long_running_notifications": True,
            "platforms": {
                "telegram": {
                    "streaming": True,
                    "chat_types": {"group": {"streaming": False}},
                }
            },
        }
    }

    assert resolve_source_display_setting(user_config, "telegram", source, "tool_progress") == "all"
    assert resolve_source_display_setting(user_config, "telegram", source, "streaming") is True
    assert resolve_source_display_setting(user_config, "telegram", source, "interim_assistant_messages") is True
    assert resolve_source_display_setting(user_config, "telegram", source, "long_running_notifications") is True


def test_telegram_group_can_override_each_quiet_display_flag():
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_type="group")
    quiet = {
        "tool_progress": "off",
        "streaming": False,
        "interim_assistant_messages": False,
        "long_running_notifications": False,
        "busy_ack_detail": False,
        "busy_ack_enabled": False,
        "thinking_progress": False,
        "show_reasoning": False,
    }
    user_config = {
        "display": {
            "tool_progress": "all",
            "interim_assistant_messages": True,
            "long_running_notifications": True,
            "platforms": {"telegram": {"streaming": True, "chat_types": {"group": quiet}}},
            "busy_ack_detail": True,
            "busy_ack_enabled": True,
            "thinking_progress": True,
            "show_reasoning": True,
        }
    }

    for setting, expected in quiet.items():
        assert resolve_source_display_setting(user_config, "telegram", source, setting) == expected


def test_null_chat_type_value_inherits_from_platform():
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_type="forum")
    config = {
        "display": {
            "platforms": {
                "telegram": {
                    "streaming": True,
                    "chat_types": {"forum": {"streaming": None}},
                }
            }
        }
    }

    assert resolve_source_display_setting(config, "telegram", source, "streaming") is True


def test_chat_type_tool_progress_beats_platform_config_and_environment():
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_type="group")
    user_config = {
        "display": {
            "platforms": {
                "telegram": {
                    "tool_progress": "verbose",
                    "chat_types": {"group": {"tool_progress": "off"}},
                }
            }
        }
    }

    assert resolve_source_tool_progress(user_config, "telegram", source, "all") == ("off", True)


def test_chat_type_values_use_normal_display_normalisation():
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_type="channel")
    config = {
        "display": {
            "chat_types": {
                "channel": {"tool_progress": False, "busy_ack_enabled": "false"},
            }
        }
    }

    assert resolve_source_display_setting(config, "telegram", source, "tool_progress") == "off"
    assert resolve_source_display_setting(config, "telegram", source, "busy_ack_enabled") is False


def test_warning_notification_policy_uses_chat_type_override():
    source = SimpleNamespace(platform=Platform.TELEGRAM, chat_type="group")
    config = {
        "display": {
            "platforms": {
                "telegram": {
                    "suppress_warning_notifications": False,
                    "chat_types": {"group": {"suppress_warning_notifications": True}},
                }
            }
        }
    }

    assert warning_notifications_enabled(Platform.TELEGRAM, config, source) is False
