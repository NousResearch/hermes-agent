"""Telegram publishes a token-free receipt after real transport startup."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.secret_scope import reset_secret_scope, set_secret_scope
from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.status import read_runtime_status
from plugins.platforms.telegram.adapter import TelegramAdapter


def test_connected_telegram_adapter_publishes_runtime_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(
        enabled=True,
        token="123456:never-persist-this-secret",
        credential_source="profile_env",
    )
    adapter._bot = SimpleNamespace(id=123456789, username="fleet_test_bot")
    adapter._webhook_mode = False
    adapter._send_path_degraded = False
    adapter._drop_delayed_deliveries = True
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._status_write_logged = set()

    adapter._mark_connected()

    payload = read_runtime_status()
    platform = payload["platforms"]["telegram"]
    assert platform["state"] == "connected"
    assert platform["runtime"]["credential_source"] == "profile_env"
    assert platform["runtime"]["authenticated"] is True
    assert platform["runtime"]["bot_id"] == "123456789"
    assert platform["runtime"]["bot_username"] == "fleet_test_bot"
    assert platform["runtime"]["transport_mode"] == "polling"
    assert platform["runtime"]["transport_ready"] is True
    assert platform["runtime"]["verified_at"]
    assert "never-persist-this-secret" not in json.dumps(payload)

    adapter._mark_disconnected()

    disconnected = read_runtime_status()["platforms"]["telegram"]
    assert disconnected["state"] == "disconnected"
    assert "runtime" not in disconnected


def test_platform_config_tracks_token_origin_without_serializing_it(monkeypatch):
    configured = PlatformConfig.from_dict({"enabled": True, "token": "yaml-secret"})
    assert configured.credential_source == "config_file"
    assert "credential_source" not in configured.to_dict()


def test_profile_scoped_telegram_token_is_attributed_to_profile_env():
    scope_token = set_secret_scope({"TELEGRAM_BOT_TOKEN": "profile-secret"})
    try:
        config = GatewayConfig()
        _apply_env_overrides(config)
    finally:
        reset_secret_scope(scope_token)

    telegram = config.platforms[Platform.TELEGRAM]
    assert telegram.token == "profile-secret"
    assert telegram.credential_source == "profile_env"


def test_process_telegram_token_is_attributed_to_process_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "process-secret")
    config = GatewayConfig()

    _apply_env_overrides(config)

    telegram = config.platforms[Platform.TELEGRAM]
    assert telegram.token == "process-secret"
    assert telegram.credential_source == "process_env"


def test_polling_progress_refreshes_transport_readiness(monkeypatch):
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(
        enabled=True,
        token="123456:never-persist-this-secret",
        credential_source="profile_env",
    )
    adapter._bot = SimpleNamespace(id=123456789, username="fleet_test_bot")
    adapter._webhook_mode = False
    adapter._running = True
    adapter._polling_teardown_started = False
    adapter._polling_progress_accepting = True
    adapter._polling_generation = 7
    adapter._polling_progress_event = asyncio.Event()
    adapter._polling_network_error_count = 1
    adapter._polling_conflict_count = 1
    adapter._send_path_degraded = True
    status_writer = MagicMock()
    monkeypatch.setattr(adapter, "_write_runtime_status_safe", status_writer)

    adapter._record_polling_progress(7)

    status_writer.assert_called_once()
    receipt = status_writer.call_args.kwargs["platform_runtime"]
    assert receipt["transport_ready"] is True
    assert "never-persist-this-secret" not in json.dumps(receipt)
