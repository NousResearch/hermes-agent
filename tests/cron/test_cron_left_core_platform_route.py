"""A cron route naming a platform that left core (``deliver: <platform>:<chat>``, saved while core shipped
it) keeps delivering once the plugin registers the platform, and names the install when it does not."""

from unittest.mock import patch

import pytest

from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from hermes_cli.left_core_migration import LEFT_CORE

# Platforms that left core with no static ``Platform`` member: only the plugin can register them.
_MOVED = [f for f in LEFT_CORE if f.platform and f.platform not in Platform._value2member_map_]


@pytest.fixture(autouse=True)
def _no_discovery(monkeypatch):
    # The test home has no plugins; discovery must not register (or drop) the stand-in entry below.
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda *a, **k: None)


def _config(name):
    cfg = type("Cfg", (), {})()
    cfg.platforms = {Platform(name): PlatformConfig(enabled=True)}
    cfg.get_home_channel = lambda _platform: None
    return cfg


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_route_delivers_through_the_plugins_sender(feature):
    sent = []

    async def sender(pconfig, chat_id, message, **_kw):
        sent.append((chat_id, message))
        return {"success": True, "platform": feature.platform, "chat_id": chat_id, "message_id": "m1"}

    platform_registry.register(PlatformEntry(
        name=feature.platform, label=feature.label, adapter_factory=lambda cfg: None, check_fn=lambda: True,
        standalone_sender_fn=sender, cron_deliver_env_var=f"{feature.platform.upper()}_HOME_CHANNEL"))
    try:
        with patch("gateway.config.load_gateway_config", return_value=_config(feature.platform)):
            error = _deliver_result({"id": "j1", "name": "Report", "deliver": f"{feature.platform}:chat-1"},
                                    "the report", adapters=None, loop=None)
    finally:
        platform_registry.unregister(feature.platform)
    assert error is None
    assert [chat for chat, _ in sent] == ["chat-1"] and "the report" in sent[0][1]


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_route_without_the_plugin_names_the_install(feature):
    assert platform_registry.get(feature.platform) is None
    error = _deliver_result({"id": "j1", "name": "Report", "deliver": f"{feature.platform}:chat-1"},
                            "the report", adapters=None, loop=None)
    assert error and f"hermes plugins install {feature.plugin}" in error


@pytest.mark.parametrize("feature", _MOVED, ids=[f.platform for f in _MOVED])
def test_webhook_route_stays_loaded_before_the_plugin_registers(feature):
    """Route load checks the launch registry; a secondary profile's plugin is not in it, and a skipped route is
    not re-read until its file changes."""
    from gateway.webhook_targets import is_dispatchable_target
    assert platform_registry.get(feature.platform) is None
    assert is_dispatchable_target(feature.platform) is True
