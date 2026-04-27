"""Configuration and gateway wiring tests for Microsoft Teams."""

from __future__ import annotations

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides, load_gateway_config


def _clear_msteams_env(monkeypatch):
    import os

    for key in list(os.environ):
        if key.startswith("MSTEAMS_"):
            monkeypatch.delenv(key, raising=False)


def test_platform_enum_has_msteams():
    assert Platform("msteams") is Platform.MSTEAMS
    assert Platform.MSTEAMS.value == "msteams"


def test_get_connected_platforms_requires_app_credentials():
    config = GatewayConfig(
        platforms={
            Platform.MSTEAMS: PlatformConfig(
                enabled=True,
                extra={"app_id": "app-id"},
            )
        }
    )
    assert Platform.MSTEAMS not in config.get_connected_platforms()

    config.platforms[Platform.MSTEAMS].extra["app_password"] = "secret"
    assert Platform.MSTEAMS in config.get_connected_platforms()


def test_env_overrides_populate_msteams_platform_config(monkeypatch):
    _clear_msteams_env(monkeypatch)

    monkeypatch.setenv("MSTEAMS_APP_ID", "app-123")
    monkeypatch.setenv("MSTEAMS_APP_PASSWORD", "secret-123")
    monkeypatch.setenv("MSTEAMS_TENANT_ID", "tenant-abc")
    monkeypatch.setenv("MSTEAMS_BOT_DISPLAY_NAME", "Hermes")
    monkeypatch.setenv("MSTEAMS_HOST", "127.0.0.1")
    monkeypatch.setenv("MSTEAMS_PORT", "4000")
    monkeypatch.setenv("MSTEAMS_PATH", "/teams/messages")
    monkeypatch.setenv("MSTEAMS_REQUIRE_MENTION", "false")
    monkeypatch.setenv("MSTEAMS_MENTION_PATTERNS", '["hermes", "bot"]')
    monkeypatch.setenv("MSTEAMS_FREE_RESPONSE_CONVERSATIONS", "19:free,team-1")
    monkeypatch.setenv("MSTEAMS_HOME_CHANNEL", "19:home")

    config = GatewayConfig()
    _apply_env_overrides(config)

    platform_config = config.platforms[Platform.MSTEAMS]
    assert platform_config.enabled is True
    assert platform_config.token == "app-123"
    assert platform_config.api_key == "secret-123"
    assert platform_config.home_channel.chat_id == "19:home"
    assert platform_config.extra["app_id"] == "app-123"
    assert platform_config.extra["app_password"] == "secret-123"
    assert platform_config.extra["tenant_id"] == "tenant-abc"
    assert platform_config.extra["bot_display_name"] == "Hermes"
    assert platform_config.extra["host"] == "127.0.0.1"
    assert platform_config.extra["port"] == 4000
    assert platform_config.extra["path"] == "/teams/messages"
    assert platform_config.extra["require_mention"] is False
    assert platform_config.extra["mention_patterns"] == ["hermes", "bot"]
    assert platform_config.extra["free_response_conversations"] == ["19:free", "team-1"]


def test_load_gateway_config_bridges_msteams_top_level_policy(tmp_path, monkeypatch):
    _clear_msteams_env(monkeypatch)
    monkeypatch.setenv("MSTEAMS_APP_PASSWORD", "env-password")
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n"
        "  msteams:\n"
        "    enabled: true\n"
        "    extra:\n"
        "      app_id: app-yaml\n"
        "msteams:\n"
        "  require_mention: false\n"
        "  mention_patterns:\n"
        "    - '^hermes[:, ]'\n"
        "  free_response_conversations:\n"
        "    - '19:free@thread.tacv2'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_gateway_config()

    platform_config = config.platforms[Platform.MSTEAMS]
    assert platform_config.enabled is True
    assert platform_config.extra["app_id"] == "app-yaml"
    assert platform_config.api_key == "env-password"
    assert platform_config.extra["app_password"] == "env-password"
    assert platform_config.extra["require_mention"] is False
    assert platform_config.extra["mention_patterns"] == ["^hermes[:, ]"]
    assert platform_config.extra["free_response_conversations"] == ["19:free@thread.tacv2"]


def test_runner_factory_references_msteams():
    import inspect
    from gateway.run import GatewayRunner

    source = inspect.getsource(GatewayRunner._create_adapter)
    assert "Platform.MSTEAMS" in source
    assert "MsTeamsAdapter" in source

    auth_source = inspect.getsource(GatewayRunner._is_user_authorized)
    assert "MSTEAMS_ALLOWED_USERS" in auth_source
    assert "MSTEAMS_ALLOW_ALL_USERS" in auth_source


def test_toolset_and_prompt_hint_include_msteams():
    import toolsets
    from hermes_cli.platforms import PLATFORMS
    from agent.prompt_builder import PLATFORM_HINTS

    assert PLATFORMS["msteams"].default_toolset == "hermes-msteams"
    assert "hermes-msteams" in toolsets.TOOLSETS
    assert "hermes-msteams" in toolsets.TOOLSETS["hermes-gateway"]["includes"]
    assert "Microsoft Teams" in PLATFORM_HINTS["msteams"]
