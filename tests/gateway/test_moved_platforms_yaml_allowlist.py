"""config.yaml ``platforms.<name>.allowed_users`` gates DMs on the platforms moved from core into plugins.

Their registry entries declare ``allowed_users_env``, so with no env allowlist the gateway reads the live
adapter's ``extra.allowed_users``, as it already does for Telegram, Discord, Slack and the other plugin
platforms. Listed senders are admitted; everyone else stays default-deny.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource

_MOVED = ("signal", "weixin", "bluebubbles", "qqbot", "yuanbao", "whatsapp_cloud")


@pytest.mark.parametrize("platform", _MOVED)
def test_yaml_allowed_users_admits_listed_dm_sender_only(monkeypatch, platform):
    import os

    from gateway.run import GatewayRunner
    from hermes_cli.plugins import discover_plugins

    for key in list(os.environ):
        if key.endswith(("_ALLOWED_USERS", "_ALLOW_ALL_USERS", "_ALLOWED_CHATS")):
            monkeypatch.delenv(key, raising=False)
    discover_plugins()

    p = Platform(platform)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {p: SimpleNamespace(config=PlatformConfig(enabled=True, extra={"allowed_users": ["listed-1"]}))}
    runner._profile_adapters = {}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False

    def dm(user_id):
        return SessionSource(platform=p, user_id=user_id, chat_id=user_id, user_name="u", chat_type="dm")

    assert runner._is_user_authorized(dm("listed-1")) is True
    assert runner._is_user_authorized(dm("stranger-9")) is False
