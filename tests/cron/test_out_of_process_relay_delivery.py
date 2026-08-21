"""Out-of-process cron delivery for relay-fronted platforms (#86249).

When cron runs outside the gateway process (``adapters=None``),
``resolve_delivery_transport`` cannot see the live relay socket. Previously
the native configured/enabled gate rejected Discord/Slack with
``platform 'discord' not configured/enabled`` even though
``GATEWAY_RELAY_PLATFORMS`` stamped those platforms as connector-fronted.

Fix (option 2): skip the native gate for relay-fronted platforms and POST
to the local gateway api_server's ``/api/delivery/send``, which delivers
via the live relay adapter — no second connector handshake.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig


def _clear_home_env(monkeypatch):
    for var in (
        "DISCORD_HOME_CHANNEL",
        "DISCORD_HOME_CHANNEL_THREAD_ID",
        "GATEWAY_RELAY_PLATFORMS",
        "API_SERVER_KEY",
        "API_SERVER_PORT",
    ):
        monkeypatch.delenv(var, raising=False)


def _relay_only_config():
    """Gateway config with no native Discord credentials (connector owns them)."""
    config = MagicMock()
    config.platforms = {
        Platform.RELAY: PlatformConfig(enabled=True),
    }
    config.get_home_channel = lambda p: None
    return config


def _job():
    return {
        "id": "c588c14f3abf",
        "name": "Nightly",
        "deliver": "origin",
        "origin": {"platform": "discord", "chat_id": "1517373704248758474"},
    }


class TestOutOfProcessRelayFrontedDelivery:
    def test_adapters_none_uses_gateway_loopback(self, monkeypatch):
        """adapters=None + GATEWAY_RELAY_PLATFORMS=discord → loopback, not gate."""
        _clear_home_env(monkeypatch)
        monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")

        with patch("gateway.config.load_gateway_config",
                   return_value=_relay_only_config()), \
             patch("cron.scheduler.load_config",
                   return_value={"cron": {"wrap_response": False}}), \
             patch(
                 "gateway.loopback_delivery.deliver_via_gateway_loopback",
                 return_value=None,
             ) as loopback:
            err = _deliver_result(_job(), "Nightly report.", adapters=None, loop=None)

        assert err is None
        loopback.assert_called_once()
        args, kwargs = loopback.call_args
        assert args[0] == "discord"
        assert args[1] == "1517373704248758474"
        assert "Nightly report" in args[2]

    def test_adapters_none_without_relay_stamp_still_rejects(self, monkeypatch):
        """No GATEWAY_RELAY_PLATFORMS → historical native gate still fires."""
        _clear_home_env(monkeypatch)

        with patch("gateway.config.load_gateway_config",
                   return_value=_relay_only_config()), \
             patch("cron.scheduler.load_config",
                   return_value={"cron": {"wrap_response": False}}), \
             patch(
                 "gateway.loopback_delivery.deliver_via_gateway_loopback",
             ) as loopback:
            err = _deliver_result(_job(), "Nightly report.", adapters=None, loop=None)

        assert err is not None
        assert "not configured/enabled" in err
        loopback.assert_not_called()

    def test_nonempty_adapters_without_relay_still_loopbacks(self, monkeypatch):
        """Non-empty adapters map without a Discord/relay handle → loopback.

        ``not adapters`` is too coarse: a process can hold other platform
        adapters and still lack a live transport for the relay-fronted
        logical platform.
        """
        _clear_home_env(monkeypatch)
        monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")
        # Other platform present, but no RELAY / Discord runtime adapter.
        adapters = {Platform.TELEGRAM: MagicMock()}

        with patch("gateway.config.load_gateway_config",
                   return_value=_relay_only_config()), \
             patch("cron.scheduler.load_config",
                   return_value={"cron": {"wrap_response": False}}), \
             patch(
                 "gateway.loopback_delivery.deliver_via_gateway_loopback",
                 return_value=None,
             ) as loopback:
            err = _deliver_result(
                _job(), "Nightly report.", adapters=adapters, loop=None,
            )

        assert err is None
        loopback.assert_called_once()

    def test_loopback_unreachable_fails_closed(self, monkeypatch):
        """Gateway down → clear error; never fall through to Discord HTTP."""
        _clear_home_env(monkeypatch)
        monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")

        with patch("gateway.config.load_gateway_config",
                   return_value=_relay_only_config()), \
             patch("cron.scheduler.load_config",
                   return_value={"cron": {"wrap_response": False}}), \
             patch(
                 "gateway.loopback_delivery.deliver_via_gateway_loopback",
                 return_value="gateway loopback unreachable at http://127.0.0.1:8642/api/delivery/send",
             ), \
             patch("tools.send_message_tool._send_to_platform") as standalone:
            err = _deliver_result(_job(), "Nightly report.", adapters=None, loop=None)

        assert err is not None
        assert "loopback" in err.lower() or "unreachable" in err.lower()
        standalone.assert_not_called()


class TestLoopbackDeliveryHelper:
    def test_resolve_default_port(self, monkeypatch):
        from gateway.loopback_delivery import resolve_api_server_loopback_base

        _clear_home_env(monkeypatch)
        with patch("hermes_cli.config.load_config", return_value={}):
            assert resolve_api_server_loopback_base() == "http://127.0.0.1:8642"

    def test_deliver_posts_bearer_and_payload(self, monkeypatch):
        from gateway.loopback_delivery import deliver_via_gateway_loopback

        _clear_home_env(monkeypatch)
        monkeypatch.setenv("API_SERVER_KEY", "test-key-sixteen-chars")
        monkeypatch.setenv("API_SERVER_PORT", "8700")

        class _Resp:
            status_code = 200

            def json(self):
                return {"success": True, "message_id": "m1"}

        posted = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            posted["url"] = url
            posted["json"] = json
            posted["headers"] = headers
            posted["timeout"] = timeout
            return _Resp()

        with patch("hermes_cli.config.load_config", return_value={}), \
             patch("httpx.post", side_effect=fake_post):
            err = deliver_via_gateway_loopback(
                "discord", "123", "hi", thread_id="t1",
            )

        assert err is None
        assert posted["url"] == "http://127.0.0.1:8700/api/delivery/send"
        assert posted["json"] == {
            "platform": "discord",
            "chat_id": "123",
            "content": "hi",
            "thread_id": "t1",
        }
        assert posted["headers"]["Authorization"] == "Bearer test-key-sixteen-chars"

    def test_deliver_connection_error_message(self, monkeypatch):
        from gateway.loopback_delivery import deliver_via_gateway_loopback

        _clear_home_env(monkeypatch)

        with patch("hermes_cli.config.load_config", return_value={}), \
             patch("httpx.post", side_effect=ConnectionError("refused")):
            err = deliver_via_gateway_loopback("discord", "123", "hi")

        assert err is not None
        assert "unreachable" in err.lower()


class TestRelayStandaloneSenderRegistration:
    def test_relay_entry_registers_standalone_sender(self):
        from gateway.platform_registry import platform_registry
        from gateway.relay import register_relay_adapter

        register_relay_adapter(force=True)
        entry = platform_registry.get("relay")
        assert entry is not None
        assert callable(entry.standalone_sender_fn)
