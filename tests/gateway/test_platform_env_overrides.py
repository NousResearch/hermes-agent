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
