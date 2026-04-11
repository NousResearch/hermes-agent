from hermes_cli.platform_catalog import (
    configured_platform_keys,
    get_platform_spec,
    iter_setup_platform_specs,
    iter_skills_platform_specs,
    iter_tool_platform_specs,
)


def test_tool_platform_specs_include_default_toolsets():
    tool_specs = {spec.key: spec for spec in iter_tool_platform_specs()}
    assert tool_specs["cli"].default_toolset == "hermes-cli"
    assert tool_specs["telegram"].default_toolset == "hermes-telegram"
    assert tool_specs["webhook"].default_toolset == "hermes-webhook"


def test_skills_platform_specs_exclude_api_server():
    keys = {spec.key for spec in iter_skills_platform_specs()}
    assert "cli" in keys
    assert "telegram" in keys
    assert "api_server" not in keys


def test_setup_platform_specs_use_shared_setup_metadata():
    setup_specs = {spec.key: spec for spec in iter_setup_platform_specs()}
    assert setup_specs["weixin"].setup_display_label == "Weixin (WeChat)"
    assert setup_specs["webhook"].setup_display_label == "Webhooks (GitHub, GitLab, etc.)"


def test_configured_platform_keys_use_catalog_logic():
    env = {
        "DISCORD_BOT_TOKEN": "token",
        "MATRIX_HOMESERVER": "https://matrix.example.org",
        "MATRIX_PASSWORD": "pw",
        "SIGNAL_HTTP_URL": "http://signal",
        "SIGNAL_ACCOUNT": "+15551234567",
    }
    keys = configured_platform_keys(lambda name: env.get(name, ""))
    assert "cli" in keys
    assert "discord" in keys
    assert "matrix" in keys
    assert "signal" in keys
    assert "telegram" not in keys


def test_falsey_enable_flags_do_not_count_as_configured():
    env = {
        "WHATSAPP_ENABLED": "false",
        "WEBHOOK_ENABLED": "off",
        "API_SERVER_ENABLED": "0",
    }
    keys = configured_platform_keys(lambda name: env.get(name, ""))
    assert "whatsapp" not in keys
    assert "webhook" not in keys
    assert "api_server" not in keys


def test_get_platform_spec_exposes_home_channel_metadata():
    spec = get_platform_spec("telegram")
    assert spec is not None
    assert spec.home_channel_env == "TELEGRAM_HOME_CHANNEL"
    assert spec.warn_missing_home is True


def test_matrix_requires_homeserver_and_uses_room_home_env():
    spec = get_platform_spec("matrix")
    assert spec is not None
    assert spec.home_channel_env == "MATRIX_HOME_ROOM"
    assert spec.is_configured(lambda name: {"MATRIX_PASSWORD": "pw"}.get(name, "")) is False
    assert spec.is_configured(
        lambda name: {
            "MATRIX_HOMESERVER": "https://matrix.example.org",
            "MATRIX_PASSWORD": "pw",
        }.get(name, "")
    ) is True


def test_mattermost_requires_url_and_token():
    spec = get_platform_spec("mattermost")
    assert spec is not None
    assert spec.is_configured(lambda name: {"MATTERMOST_TOKEN": "token"}.get(name, "")) is False
    assert spec.is_configured(
        lambda name: {
            "MATTERMOST_URL": "https://mm.example.com",
            "MATTERMOST_TOKEN": "token",
        }.get(name, "")
    ) is True
