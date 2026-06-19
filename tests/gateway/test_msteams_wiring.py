"""Smoke tests for Microsoft Teams adapter wiring (C1).

These tests verify the *integration points* are in place — enum value,
factory routing, env-var overrides, toolset registration, cron delivery
map, send_message platform map, target-ref parser.  They do NOT exercise
the Bot Framework / Graph protocol; that lives in separate tests once
the real adapter lands in C3+.
"""

from __future__ import annotations

import os

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.msteams import MsTeamsAdapter, check_msteams_requirements


# ---------------------------------------------------------------------------
# Platform enum + PlatformConfig
# ---------------------------------------------------------------------------

def test_platform_enum_has_msteams():
    assert Platform("msteams") is Platform.MSTEAMS
    assert Platform.MSTEAMS.value == "msteams"


def test_stub_adapter_instantiates_with_defaults():
    config = PlatformConfig(enabled=True, extra={"app_id": "test-app-id"})
    adapter = MsTeamsAdapter(config)
    assert adapter.platform is Platform.MSTEAMS
    assert adapter.name == "msteams"
    assert adapter._port == 3978
    assert adapter._path == "/api/messages"
    assert adapter._reply_style == "thread"
    assert adapter._require_mention is True
    assert adapter._dm_policy == "pairing"


def test_stub_adapter_connect_returns_false_without_protocol():
    """C1 stub must decline to connect — the protocol lands in C3."""
    config = PlatformConfig(enabled=True, extra={"app_id": "x"})
    adapter = MsTeamsAdapter(config)
    import asyncio
    assert asyncio.run(adapter.connect()) is False


# ---------------------------------------------------------------------------
# Env var overrides
# ---------------------------------------------------------------------------

def _clear_msteams_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MSTEAMS_"):
            monkeypatch.delenv(key, raising=False)


def test_env_overrides_populate_platform_config(monkeypatch):
    from gateway.config import _apply_env_overrides
    _clear_msteams_env(monkeypatch)
    monkeypatch.setenv("MSTEAMS_APP_ID", "app-123")
    monkeypatch.setenv("MSTEAMS_TENANT_ID", "tenant-xyz")
    monkeypatch.setenv("MSTEAMS_APP_PASSWORD", "hunter2")
    monkeypatch.setenv("MSTEAMS_AUTH_TYPE", "FEDERATED")  # case-insensitive
    monkeypatch.setenv("MSTEAMS_PORT", "4000")
    monkeypatch.setenv("MSTEAMS_PATH", "/bots/teams")
    monkeypatch.setenv("MSTEAMS_ALLOW_FROM", "aad-a , aad-b,, aad-c ")
    monkeypatch.setenv("MSTEAMS_GROUP_ALLOW_FROM", "aad-x,aad-y")
    monkeypatch.setenv("MSTEAMS_REQUIRE_MENTION", "false")
    monkeypatch.setenv("MSTEAMS_REPLY_STYLE", "top-level")
    monkeypatch.setenv("MSTEAMS_HISTORY_LIMIT", "25")
    monkeypatch.setenv("MSTEAMS_SHAREPOINT_SITE_ID", "site-abc")
    monkeypatch.setenv("MSTEAMS_USE_MANAGED_IDENTITY", "true")
    monkeypatch.setenv("MSTEAMS_MANAGED_IDENTITY_CLIENT_ID", "mi-123")

    config = GatewayConfig()
    _apply_env_overrides(config)

    pc = config.platforms[Platform.MSTEAMS]
    assert pc.enabled is True
    assert pc.token == "app-123"
    assert pc.extra["app_id"] == "app-123"
    assert pc.extra["tenant_id"] == "tenant-xyz"
    assert pc.extra["app_password"] == "hunter2"
    assert pc.extra["auth_type"] == "federated"
    assert pc.extra["port"] == 4000
    assert pc.extra["path"] == "/bots/teams"
    assert pc.extra["allow_from"] == ["aad-a", "aad-b", "aad-c"]
    assert pc.extra["group_allow_from"] == ["aad-x", "aad-y"]
    assert pc.extra["require_mention"] is False
    assert pc.extra["reply_style"] == "top-level"
    assert pc.extra["history_limit"] == 25
    assert pc.extra["sharepoint_site_id"] == "site-abc"
    assert pc.extra["use_managed_identity"] is True
    assert pc.extra["managed_identity_client_id"] == "mi-123"


def test_env_overrides_default_port_and_path(monkeypatch):
    from gateway.config import _apply_env_overrides
    _clear_msteams_env(monkeypatch)
    monkeypatch.setenv("MSTEAMS_APP_ID", "app-default")
    config = GatewayConfig()
    _apply_env_overrides(config)
    pc = config.platforms[Platform.MSTEAMS]
    assert pc.extra["port"] == 3978
    assert pc.extra["path"] == "/api/messages"
    assert pc.extra["auth_type"] == "secret"
    assert pc.extra["require_mention"] is True
    assert pc.extra["reply_style"] == "thread"


def test_env_overrides_invalid_port_falls_back_to_default(monkeypatch):
    from gateway.config import _apply_env_overrides
    _clear_msteams_env(monkeypatch)
    monkeypatch.setenv("MSTEAMS_APP_ID", "app-default")
    monkeypatch.setenv("MSTEAMS_PORT", "not-a-number")
    config = GatewayConfig()
    _apply_env_overrides(config)
    pc = config.platforms[Platform.MSTEAMS]
    assert pc.extra["port"] == 3978


def test_env_overrides_skipped_when_app_id_missing(monkeypatch):
    from gateway.config import _apply_env_overrides
    _clear_msteams_env(monkeypatch)
    # No MSTEAMS_APP_ID set
    monkeypatch.setenv("MSTEAMS_APP_PASSWORD", "orphan-secret")
    config = GatewayConfig()
    _apply_env_overrides(config)
    assert Platform.MSTEAMS not in config.platforms


# ---------------------------------------------------------------------------
# get_connected_platforms
# ---------------------------------------------------------------------------

def test_get_connected_platforms_requires_app_id():
    config = GatewayConfig()
    config.platforms[Platform.MSTEAMS] = PlatformConfig(enabled=True, extra={})
    assert Platform.MSTEAMS not in config.get_connected_platforms()

    config.platforms[Platform.MSTEAMS].extra["app_id"] = "abc"
    assert Platform.MSTEAMS in config.get_connected_platforms()


# ---------------------------------------------------------------------------
# Authorization maps (run.py)
# ---------------------------------------------------------------------------

def test_authorization_maps_include_msteams():
    import inspect
    from gateway.run import GatewayRunner
    src = inspect.getsource(GatewayRunner._is_user_authorized)
    assert "Platform.MSTEAMS" in src
    assert "MSTEAMS_ALLOWED_USERS" in src
    assert "MSTEAMS_ALLOW_ALL_USERS" in src

    src_dm = inspect.getsource(GatewayRunner._get_unauthorized_dm_behavior)
    assert "Platform.MSTEAMS" in src_dm


# ---------------------------------------------------------------------------
# Toolset
# ---------------------------------------------------------------------------

def test_toolsets_include_hermes_msteams():
    import toolsets
    assert "hermes-msteams" in toolsets.TOOLSETS
    assert "hermes-msteams" in toolsets.TOOLSETS["hermes-gateway"]["includes"]


# ---------------------------------------------------------------------------
# Cron delivery + send_message routing
# ---------------------------------------------------------------------------

def test_cron_known_delivery_platforms_includes_msteams():
    from cron import scheduler
    assert "msteams" in scheduler._KNOWN_DELIVERY_PLATFORMS
    assert scheduler._HOME_TARGET_ENV_VARS["msteams"] == "MSTEAMS_HOME_CHANNEL"


def test_send_message_parses_teams_conversation_id():
    from tools.send_message_tool import _parse_target_ref
    chat_id, thread_id, is_explicit = _parse_target_ref(
        "msteams", "19:abcdef@thread.tacv2",
    )
    assert chat_id == "19:abcdef@thread.tacv2"
    assert thread_id is None
    assert is_explicit is True


def test_send_message_platform_map_includes_msteams():
    # _handle_send resolves the platform name through the Platform enum
    # (``Platform(platform_name)``) rather than a hand-maintained dict, so
    # verify "msteams" resolves to the MSTEAMS member.
    import inspect
    from tools import send_message_tool
    from gateway.config import Platform
    assert Platform("msteams") is Platform.MSTEAMS

    dispatch_src = inspect.getsource(send_message_tool._send_to_platform)
    assert "Platform.MSTEAMS" in dispatch_src
    assert "_send_msteams" in dispatch_src


# ---------------------------------------------------------------------------
# Prompt hint
# ---------------------------------------------------------------------------

def test_platform_hints_include_msteams():
    from agent.prompt_builder import PLATFORM_HINTS
    hint = PLATFORM_HINTS.get("msteams")
    assert hint is not None
    assert "Microsoft Teams" in hint


# ---------------------------------------------------------------------------
# check_msteams_requirements
# ---------------------------------------------------------------------------

def test_check_msteams_requirements_returns_bool():
    # We don't assert True/False since the test environment may or may not
    # have the [msteams] extra installed; the contract is just that it
    # returns a bool without raising.
    assert isinstance(check_msteams_requirements(), bool)
