"""Tests for the MQTT platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_mqtt`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.

Tests target:
  - register() shape (env_enablement_fn, allowed_users_env, no core edits)
  - _env_enablement seeding
  - check_requirements / validate_config / is_connected
  - retained-message suppression (default: suppress, opt-in: tag)
  - observational mode (default: log, not invoke agent)
  - send() topic guard (suppress outside self_topic_prefix)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_mqtt = load_plugin_adapter("mqtt")

MQTTAdapter = _mqtt.MQTTAdapter
check_requirements = _mqtt.check_requirements
validate_config = _mqtt.validate_config
is_connected = _mqtt.is_connected
register = _mqtt.register
_env_enablement = _mqtt._env_enablement


# ---------------------------------------------------------------------------
# 1. Platform enum (plugin-discovered, not bundled)
# ---------------------------------------------------------------------------


def test_platform_enum_resolves_via_plugin_scan():
    """The plugin filesystem scan should expose Platform(\"mqtt\")."""
    from gateway.config import Platform
    p = Platform("mqtt")
    assert p.value == "mqtt"
    assert Platform("mqtt") is p


# ---------------------------------------------------------------------------
# 2. register() shape — no core edits
# ---------------------------------------------------------------------------


class TestRegisterShape:
    """Verify register() declares all hooks so no core edits are needed."""

    def test_register_calls_ctx_register_platform(self):
        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()

    def test_register_delegates_env_enablement(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs.get("env_enablement_fn") is _env_enablement

    def test_register_delegates_allowed_users_env(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs.get("allowed_users_env") == "MQTT_ALLOWED_USERS"

    def test_register_delegates_allow_all_env(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs.get("allow_all_env") == "MQTT_ALLOW_ALL_USERS"

    def test_register_delegates_cron_deliver_env(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs.get("cron_deliver_env_var") == "MQTT_HOME_CHANNEL"


# ---------------------------------------------------------------------------
# 3. _env_enablement
# ---------------------------------------------------------------------------


class TestEnvEnablement:
    def test_returns_none_without_credentials(self, monkeypatch):
        monkeypatch.delenv("MQTT_USER", raising=False)
        monkeypatch.delenv("MQTT_PASSWORD", raising=False)
        assert _env_enablement() is None

    def test_returns_none_with_only_user(self, monkeypatch):
        monkeypatch.setenv("MQTT_USER", "me")
        monkeypatch.delenv("MQTT_PASSWORD", raising=False)
        assert _env_enablement() is None

    def test_seeds_username_password(self, monkeypatch):
        monkeypatch.setenv("MQTT_USER", "me")
        monkeypatch.setenv("MQTT_PASSWORD", "secret")
        monkeypatch.delenv("MQTT_BROKER", raising=False)
        monkeypatch.delenv("MQTT_CA_CERT", raising=False)
        monkeypatch.delenv("MQTT_HOME_CHANNEL", raising=False)
        seed = _env_enablement()
        assert seed is not None
        assert seed["username"] == "me"
        assert seed["password"] == "secret"

    def test_seeds_broker_when_set(self, monkeypatch):
        monkeypatch.setenv("MQTT_USER", "me")
        monkeypatch.setenv("MQTT_PASSWORD", "secret")
        monkeypatch.setenv("MQTT_BROKER", "broker.local")
        seed = _env_enablement()
        assert seed is not None
        assert seed["broker_host"] == "broker.local"

    def test_seeds_ca_cert_when_set(self, monkeypatch):
        monkeypatch.setenv("MQTT_USER", "me")
        monkeypatch.setenv("MQTT_PASSWORD", "secret")
        monkeypatch.setenv("MQTT_CA_CERT", "/etc/ssl/ca.crt")
        seed = _env_enablement()
        assert seed is not None
        assert seed["ca_cert"] == "/etc/ssl/ca.crt"

    def test_seeds_home_channel_when_set(self, monkeypatch):
        monkeypatch.setenv("MQTT_USER", "me")
        monkeypatch.setenv("MQTT_PASSWORD", "secret")
        monkeypatch.setenv("MQTT_HOME_CHANNEL", "hermes/notify")
        seed = _env_enablement()
        assert seed is not None
        assert seed["home_channel"]["chat_id"] == "hermes/notify"


# ---------------------------------------------------------------------------
# 4. check_requirements / validate_config / is_connected
# ---------------------------------------------------------------------------


class TestRequirements:
    def test_check_requirements_false_without_paho(self, monkeypatch):
        monkeypatch.setattr(_mqtt, "PAHO_AVAILABLE", False)
        monkeypatch.setenv("MQTT_USER", "me")
        monkeypatch.setenv("MQTT_PASSWORD", "secret")
        assert check_requirements() is False

    def test_check_requirements_false_without_credentials(self, monkeypatch):
        monkeypatch.setattr(_mqtt, "PAHO_AVAILABLE", True)
        monkeypatch.delenv("MQTT_USER", raising=False)
        monkeypatch.delenv("MQTT_PASSWORD", raising=False)
        assert check_requirements() is False

    def test_check_requirements_true_with_paho_and_creds(self, monkeypatch):
        monkeypatch.setattr(_mqtt, "PAHO_AVAILABLE", True)
        monkeypatch.setenv("MQTT_USER", "me")
        monkeypatch.setenv("MQTT_PASSWORD", "secret")
        assert check_requirements() is True

    def test_validate_config_with_env(self, monkeypatch):
        monkeypatch.setenv("MQTT_USER", "me")
        monkeypatch.setenv("MQTT_PASSWORD", "secret")
        cfg = PlatformConfig()
        assert validate_config(cfg) is True

    def test_validate_config_without_credentials(self, monkeypatch):
        monkeypatch.delenv("MQTT_USER", raising=False)
        monkeypatch.delenv("MQTT_PASSWORD", raising=False)
        cfg = PlatformConfig()
        assert validate_config(cfg) is False

    def test_is_connected_with_env(self, monkeypatch):
        monkeypatch.setenv("MQTT_USER", "me")
        monkeypatch.setenv("MQTT_PASSWORD", "secret")
        cfg = PlatformConfig()
        assert is_connected(cfg) is True


# ---------------------------------------------------------------------------
# 5. Retained-message suppression (default: suppress; opt-in: tag)
# ---------------------------------------------------------------------------


class TestRetainedSuppression:
    """Retained messages should be suppressed by default (log_retained=False).
    With log_retained=True, they are tagged [retained] but still logged."""

    def _make_adapter(self, log_retained=False):
        cfg = PlatformConfig()
        cfg.extra = {
            "username": "u",
            "password": "p",
            "log_retained": log_retained,
            "observational": True,
        }
        adapter = MQTTAdapter(cfg)
        adapter._observe_log_path = MagicMock()
        return adapter

    def _make_msg(self, topic="test/topic", payload=b"hello", retain=False):
        msg = MagicMock()
        msg.topic = topic
        msg.payload = payload
        msg.retain = retain
        return msg

    def test_retained_suppressed_by_default(self):
        adapter = self._make_adapter(log_retained=False)
        adapter._on_message(None, None, self._make_msg(retain=True))
        adapter._observe_log_path.open.assert_not_called()

    def test_retained_logged_when_opted_in(self):
        adapter = self._make_adapter(log_retained=True)
        # Patch cooldown to 0 so it doesn't skip
        adapter._cooldown_seconds = 0
        adapter._on_message(None, None, self._make_msg(retain=True))
        # _log_event should have been called — check the file was opened
        adapter._observe_log_path.open.assert_called_once()
        # Verify the [retained] tag is in the written content
        write_call = adapter._observe_log_path.open.return_value.__enter__.return_value
        written = write_call.write.call_args[0][0]
        assert "[retained]" in written

    def test_non_retained_always_logged(self):
        adapter = self._make_adapter(log_retained=False)
        adapter._cooldown_seconds = 0
        adapter._on_message(None, None, self._make_msg(retain=False))
        adapter._observe_log_path.open.assert_called_once()
        write_call = adapter._observe_log_path.open.return_value.__enter__.return_value
        written = write_call.write.call_args[0][0]
        assert "[retained]" not in written


# ---------------------------------------------------------------------------
# 6. Observational mode (default: log, not invoke agent loop)
# ---------------------------------------------------------------------------


class TestObservationalMode:
    """In observational mode, _on_message logs the event and never invokes
    handle_message. In non-observational mode, it builds a MessageEvent
    and dispatches via run_coroutine_threadsafe."""

    def test_observational_logs_does_not_invoke_agent(self):
        cfg = PlatformConfig()
        cfg.extra = {"username": "u", "password": "p", "observational": True}
        adapter = MQTTAdapter(cfg)
        adapter._observe_log_path = MagicMock()
        adapter._cooldown_seconds = 0

        msg = MagicMock()
        msg.topic = "sensors/temp"
        msg.payload = b"22.5"
        msg.retain = False

        with patch.object(adapter, "handle_message") as mock_handle:
            adapter._on_message(None, None, msg)
            mock_handle.assert_not_called()
        adapter._observe_log_path.open.assert_called_once()

    def test_non_observational_invokes_agent(self):
        cfg = PlatformConfig()
        cfg.extra = {"username": "u", "password": "p", "observational": False}
        adapter = MQTTAdapter(cfg)
        adapter._cooldown_seconds = 0
        adapter._loop = MagicMock()
        adapter._loop.is_running.return_value = True

        msg = MagicMock()
        msg.topic = "chat/inbox"
        msg.payload = b"hello"
        msg.retain = False

        with patch("asyncio.run_coroutine_threadsafe") as mock_rcts:
            adapter._on_message(None, None, msg)
            mock_rcts.assert_called_once()


# ---------------------------------------------------------------------------
# 7. send() topic guard
# ---------------------------------------------------------------------------


class TestSendTopicGuard:
    """send() should suppress publishes outside self_topic_prefix."""

    def test_suppresses_non_prefixed_topic(self):
        import asyncio
        cfg = PlatformConfig()
        cfg.extra = {"username": "u", "password": "p", "self_topic_prefix": "hermes/"}
        adapter = MQTTAdapter(cfg)
        result = asyncio.get_event_loop().run_until_complete(adapter.send("sensors/temp", "reply"))
        assert result.success is True  # silent success, not a failure

    def test_allows_prefixed_topic_when_connected(self):
        import asyncio
        cfg = PlatformConfig()
        cfg.extra = {"username": "u", "password": "p", "self_topic_prefix": "hermes/"}
        adapter = MQTTAdapter(cfg)
        adapter._client = MagicMock()
        adapter._connected = True
        mock_info = MagicMock()
        mock_info.rc = 0  # MQTT_ERR_SUCCESS
        adapter._client.publish.return_value = mock_info

        result = asyncio.get_event_loop().run_until_complete(adapter.send("hermes/response", "hello"))
        assert result.success is True
        adapter._client.publish.assert_called_once()
