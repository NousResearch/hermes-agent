"""Tests for gateway/platform_routing.py — the platform routing helper cluster
extracted from gateway/run.py (see issue #54962 / #55138).

The helpers are pure policy functions, so these tests exercise the real
imports and real call paths (no mocks). A second block asserts that
gateway/run.py still re-exports every name, because external importers
(gateway/slash_commands.py, the /sethome tests, multiplex tests) import these
helpers from ``gateway.run`` and must keep working unchanged.
"""

from gateway.config import Platform, PlatformConfig
from gateway import platform_routing as pr


# --- _gateway_platform_value ----------------------------------------------

def test_platform_value_normalizes_enum():
    assert pr._gateway_platform_value(Platform.TELEGRAM) == "telegram"
    assert pr._gateway_platform_value(Platform.WHATSAPP_CLOUD) == "whatsapp_cloud"


def test_platform_value_normalizes_raw_strings():
    assert pr._gateway_platform_value("Telegram") == "telegram"
    assert pr._gateway_platform_value("  DISCORD  ") == "discord"


def test_platform_value_fail_closed_on_empty():
    # Unknown/empty platform must not match any surface.
    assert pr._gateway_platform_value(None) == ""
    assert pr._gateway_platform_value("") == ""


# --- _gateway_surface_passes_raw_text -------------------------------------

def test_raw_text_surfaces_pass_through():
    for surface in ("local", "api_server", "webhook", "msgraph_webhook"):
        assert pr._gateway_surface_passes_raw_text(surface)


def test_chat_surfaces_do_not_pass_raw_text():
    for chat in ("telegram", "discord", "slack", "whatsapp"):
        assert not pr._gateway_surface_passes_raw_text(chat)


def test_raw_text_check_fail_closed_on_unknown():
    assert not pr._gateway_surface_passes_raw_text(None)
    assert not pr._gateway_surface_passes_raw_text("")


# --- _non_conversational_metadata -----------------------------------------

def test_non_conversational_marker_applies_only_to_discord():
    metadata = {"thread_id": "t1"}
    marked = pr._non_conversational_metadata(metadata, platform=Platform.DISCORD)
    assert marked["non_conversational"] is True
    assert marked["thread_id"] == "t1"
    # Original dict is not mutated.
    assert "non_conversational" not in metadata


def test_non_conversational_marker_skips_other_platforms():
    metadata = {"thread_id": "t1"}
    for platform in (Platform.TELEGRAM, "slack", None):
        assert pr._non_conversational_metadata(metadata, platform=platform) is metadata


def test_non_conversational_marker_creates_dict_when_absent():
    marked = pr._non_conversational_metadata(None, platform="discord")
    assert marked == {"non_conversational": True}


# --- _home_target_env_var / _home_thread_env_var --------------------------

def test_home_target_env_var_known_conventions():
    # Matrix and Email deviate from the {PLATFORM}_HOME_CHANNEL convention
    # (see tests/gateway/test_home_target_env_var.py and PR #12698).
    assert pr._home_target_env_var("matrix") == "MATRIX_HOME_ROOM"
    assert pr._home_target_env_var("email") == "EMAIL_HOME_ADDRESS"


def test_home_target_env_var_fallback_suffix():
    assert pr._home_target_env_var("signal") == "SIGNAL_HOME_CHANNEL"


def test_home_thread_env_var_suffix():
    assert pr._home_thread_env_var("matrix") == "MATRIX_HOME_ROOM_THREAD_ID"
    assert pr._home_thread_env_var("signal") == "SIGNAL_HOME_CHANNEL_THREAD_ID"


# --- _platform_has_bot_credential -----------------------------------------

def test_token_platform_without_credential_fails():
    assert not pr._platform_has_bot_credential(Platform.TELEGRAM, PlatformConfig())


def test_token_platform_with_token_or_api_key_passes():
    assert pr._platform_has_bot_credential(
        Platform.TELEGRAM, PlatformConfig(token="123:ABC")
    )
    assert pr._platform_has_bot_credential(
        Platform.DISCORD, PlatformConfig(api_key="abc")
    )


def test_non_token_platform_always_passes():
    # Signal session paths / port-binding HTTP adapters don't use token.
    assert pr._platform_has_bot_credential(Platform.SIGNAL, PlatformConfig())


# --- _platform_config_key -------------------------------------------------

def test_platform_config_key_maps_local_to_cli():
    assert pr._platform_config_key(Platform.LOCAL) == "cli"


def test_platform_config_key_uses_enum_value_otherwise():
    assert pr._platform_config_key(Platform.TELEGRAM) == "telegram"
    assert pr._platform_config_key(Platform.API_SERVER) == "api_server"


# --- gateway/run.py re-exports (backward compatibility) -------------------

def test_run_py_reexports_routing_helpers():
    from gateway.run import (
        _GATEWAY_RAW_TEXT_PLATFORMS,
        _gateway_platform_value,
        _gateway_surface_passes_raw_text,
        _home_target_env_var,
        _home_thread_env_var,
        _non_conversational_metadata,
        _platform_config_key,
        _platform_has_bot_credential,
    )

    assert _GATEWAY_RAW_TEXT_PLATFORMS == pr._GATEWAY_RAW_TEXT_PLATFORMS
    assert _gateway_platform_value("Telegram") == "telegram"
    assert _gateway_surface_passes_raw_text("local") is True
    assert _home_target_env_var("matrix") == "MATRIX_HOME_ROOM"
    assert _home_thread_env_var("signal") == "SIGNAL_HOME_CHANNEL_THREAD_ID"
    assert _non_conversational_metadata(None, platform="discord") == {
        "non_conversational": True
    }
    assert _platform_config_key(Platform.LOCAL) == "cli"
    assert _platform_has_bot_credential(Platform.TELEGRAM, PlatformConfig()) is False
