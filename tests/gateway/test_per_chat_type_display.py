"""Tests for per-chat-type display override resolution.

Covers ``resolve_display_setting``'s optional ``chat_type`` layer
(``display.platforms.<platform>.<chat_type>.<key>``).  The central invariant:
when ``chat_type`` is omitted, or no matching subdict exists, resolution is
byte-for-byte identical to the prior per-platform behavior.
"""

import pytest

from gateway.display_config import resolve_display_setting


# ---------------------------------------------------------------------------
# Backward compatibility: no chat_type => identical to per-platform behavior
# ---------------------------------------------------------------------------

def test_no_chat_type_uses_platform_override():
    cfg = {"display": {"platforms": {"slack": {"tool_progress": "all"}}}}
    # Without chat_type, the per-platform override wins (legacy behavior).
    assert resolve_display_setting(cfg, "slack", "tool_progress") == "all"


def test_no_chat_type_falls_through_to_global():
    cfg = {"display": {"tool_progress": "new"}}
    assert resolve_display_setting(cfg, "slack", "tool_progress") == "new"


def test_empty_config_returns_builtin_default():
    # slack's built-in default for tool_progress is "off".
    assert resolve_display_setting({}, "slack", "tool_progress") == "off"


def test_unknown_setting_returns_fallback():
    assert resolve_display_setting({}, "slack", "nonexistent", "FB") == "FB"


# ---------------------------------------------------------------------------
# The new layer: per-chat-type override sits ABOVE per-platform
# ---------------------------------------------------------------------------

def _split_cfg():
    """Verbose dm, quiet channel, on the same platform."""
    return {
        "display": {
            "platforms": {
                "teams": {
                    # plain per-platform baseline
                    "tool_progress": "all",
                    "interim_assistant_messages": True,
                    # per-chat-type scopes keyed on literal chat_type values
                    "dm": {
                        "tool_progress": "all",
                        "interim_assistant_messages": True,
                    },
                    "channel": {
                        "tool_progress": "off",
                        "interim_assistant_messages": False,
                    },
                }
            }
        }
    }


def test_channel_scope_overrides_platform():
    cfg = _split_cfg()
    assert resolve_display_setting(cfg, "teams", "tool_progress", chat_type="channel") == "off"


def test_dm_scope_keeps_verbose():
    cfg = _split_cfg()
    assert resolve_display_setting(cfg, "teams", "tool_progress", chat_type="dm") == "all"


def test_chat_type_case_insensitive():
    cfg = _split_cfg()
    # gateway chat_type values are lowercase, but normalise defensively.
    assert resolve_display_setting(cfg, "teams", "tool_progress", chat_type="CHANNEL") == "off"


def test_unconfigured_chat_type_falls_through_to_platform():
    # "group" has no subdict here; falls through to the platform baseline.
    cfg = _split_cfg()
    assert resolve_display_setting(cfg, "teams", "tool_progress", chat_type="group") == "all"


def test_scope_bool_normalisation():
    cfg = _split_cfg()
    assert resolve_display_setting(cfg, "teams", "interim_assistant_messages", chat_type="channel") is False
    assert resolve_display_setting(cfg, "teams", "interim_assistant_messages", chat_type="dm") is True


def test_scope_absent_setting_falls_through_to_platform():
    # channel scope sets tool_progress but NOT show_reasoning; show_reasoning
    # should fall through to the platform layer.
    cfg = {
        "display": {
            "platforms": {
                "teams": {
                    "show_reasoning": True,
                    "channel": {"tool_progress": "off"},
                }
            }
        }
    }
    assert resolve_display_setting(cfg, "teams", "show_reasoning", chat_type="channel") is True


def test_scope_only_no_platform_baseline():
    # Only a scope subdict, no sibling per-platform key for this setting.
    cfg = {
        "display": {
            "platforms": {
                "teams": {
                    "channel": {"tool_progress": "off"},
                }
            }
        }
    }
    assert resolve_display_setting(cfg, "teams", "tool_progress", chat_type="channel") == "off"
    # dm has no scope and no platform key => built-in default path.
    # (teams has no built-in entry => global default "all")
    assert resolve_display_setting(cfg, "teams", "tool_progress", chat_type="dm") == "all"


def test_chat_type_with_no_platform_dict_is_safe():
    cfg = {"display": {"tool_progress": "new"}}
    # No platforms dict at all — chat_type must not raise, falls to global.
    assert resolve_display_setting(cfg, "teams", "tool_progress", chat_type="channel") == "new"


def test_malformed_scope_value_ignored():
    # scope key present but not a dict — must be ignored, not crash.
    cfg = {
        "display": {
            "platforms": {
                "teams": {
                    "tool_progress": "all",
                    "channel": "not-a-dict",
                }
            }
        }
    }
    assert resolve_display_setting(cfg, "teams", "tool_progress", chat_type="channel") == "all"


def test_distinct_chat_types_independent():
    # group and channel configured separately resolve independently.
    cfg = {
        "display": {
            "platforms": {
                "discord": {
                    "tool_progress": "all",
                    "channel": {"tool_progress": "off"},
                    "thread": {"tool_progress": "new"},
                }
            }
        }
    }
    assert resolve_display_setting(cfg, "discord", "tool_progress", chat_type="channel") == "off"
    assert resolve_display_setting(cfg, "discord", "tool_progress", chat_type="thread") == "new"
    # group is unconfigured -> platform baseline
    assert resolve_display_setting(cfg, "discord", "tool_progress", chat_type="group") == "all"


def test_streaming_scope_override():
    # streaming has special handling (skips global display.<key>), but the
    # per-platform and per-chat-type layers still apply.
    cfg = {
        "display": {
            "platforms": {
                "teams": {
                    "streaming": True,
                    "channel": {"streaming": False},
                }
            }
        }
    }
    assert resolve_display_setting(cfg, "teams", "streaming", chat_type="dm") is True
    assert resolve_display_setting(cfg, "teams", "streaming", chat_type="channel") is False
