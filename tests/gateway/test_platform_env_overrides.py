"""Invariants for the env -> GatewayConfig bridge that platform plugins declare on their
``PlatformEntry`` (``env_overrides`` / ``env_enable_credentials``, applied by
``gateway.config_env._apply_env_overrides``)."""

from gateway.config import Platform
from gateway.platform_registry import platform_registry

# Core-only platforms keep their env rows in gateway/config_env.py.
_CORE_ONLY = {"local", "api_server", "webhook", "msgraph_webhook", "relay"}


def _entries():
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
    return platform_registry.all_entries()


def test_every_static_plugin_platform_declares_its_env_bridge():
    """A static ``Platform`` member served by a bundled plugin must carry its env rows: without them
    its env-only setup (``TELEGRAM_BOT_TOKEN`` in .env) would silently stop enabling the platform."""
    _entries()
    missing = []
    for member in Platform:
        if member.value in _CORE_ONLY:
            continue
        entry = platform_registry.get(member.value)
        if entry is None or not entry.env_overrides or not entry.env_enable_credentials:
            missing.append(member.value)
    assert missing == []


def test_env_override_secrets_stay_in_terminal_secret_blocklist():
    """The terminal tool's adapter-secret blocklist is built at import time and cannot load adapters,
    so every secret a plugin's env rows read must already be declared where that policy looks."""
    from hermes_cli.config import PLATFORM_SECRET_ENV_SUFFIXES
    from hermes_cli.profile_channels import config_env_table_keys
    from tools.environments.local_env_policy import _ADAPTER_SECRET_ENV

    secrets = {name.upper() for name in config_env_table_keys(_entries())
               if name.upper().endswith(PLATFORM_SECRET_ENV_SUFFIXES)}
    assert "WEIXIN_TOKEN" in secrets  # the rows were actually read
    assert sorted(secrets - _ADAPTER_SECRET_ENV) == []


def test_enable_plugin_platform_sets_token_and_api_key():
    """_enable_plugin_platform must bridge env-seeded token/api_key onto PlatformConfig."""
    from gateway.config import GatewayConfig, PlatformConfig, Platform
    from gateway.config_env import _enable_plugin_platform
    from gateway.platform_registry import platform_registry, PlatformEntry

    entry_name = "test_custom_creds_platform"
    # Register temporary entry in platform_registry so Platform(entry_name) accepts it
    entry = PlatformEntry(
        name=entry_name,
        label="Test Platform",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        env_enablement_fn=lambda: {
            "token": "test-token-xyz",
            "api_key": "test-api-key-123",
            "home_channel": {"chat_id": "998877", "name": "TestHome"},
        },
    )
    platform_registry.register(entry)
    try:
        cfg = GatewayConfig()
        _enable_plugin_platform(cfg, entry)

        plat = Platform(entry.name)
        pcfg = cfg.platforms.get(plat)
        assert pcfg is not None
        assert pcfg.enabled is True
        assert pcfg.token == "test-token-xyz"
        assert pcfg.api_key == "test-api-key-123"
        assert pcfg.home_channel is not None
        assert pcfg.home_channel.chat_id == "998877"
    finally:
        platform_registry._entries.pop(entry_name, None)

