from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "DISCORD_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "SLACK_ALLOWED_USERS",
        "SIGNAL_ALLOWED_USERS",
        "EMAIL_ALLOWED_USERS",
        "SMS_ALLOWED_USERS",
        "MATTERMOST_ALLOWED_USERS",
        "MATRIX_ALLOWED_USERS",
        "DINGTALK_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "DISCORD_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "SLACK_ALLOW_ALL_USERS",
        "SIGNAL_ALLOW_ALL_USERS",
        "EMAIL_ALLOW_ALL_USERS",
        "SMS_ALLOW_ALL_USERS",
        "MATTERMOST_ALLOW_ALL_USERS",
        "MATRIX_ALLOW_ALL_USERS",
        "DINGTALK_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_event(platform: Platform, user_id: str, chat_id: str) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_id="m1",
        source=SessionSource(
            platform=platform,
            user_id=user_id,
            chat_id=chat_id,
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner(platform: Platform, config: GatewayConfig):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner, adapter


def test_whatsapp_lid_user_matches_phone_allowlist_via_session_mapping(monkeypatch, tmp_path):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15550000001")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    session_dir = tmp_path / "whatsapp" / "session"
    session_dir.mkdir(parents=True)
    (session_dir / "lid-mapping-15550000001.json").write_text('"900000000000001"', encoding="utf-8")
    (session_dir / "lid-mapping-900000000000001_reverse.json").write_text('"15550000001"', encoding="utf-8")

    runner, _adapter = _make_runner(
        Platform.WHATSAPP,
        GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)}),
    )

    source = SessionSource(
        platform=Platform.WHATSAPP,
        user_id="900000000000001@lid",
        chat_id="900000000000001@lid",
        user_name="tester",
        chat_type="dm",
    )

    assert runner._is_user_authorized(source) is True


@pytest.mark.asyncio
async def test_unauthorized_dm_pairs_by_default(monkeypatch):
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)},
    )
    runner, adapter = _make_runner(Platform.WHATSAPP, config)
    runner.pairing_store.generate_code.return_value = "ABC12DEF"

    result = await runner._handle_message(
        _make_event(
            Platform.WHATSAPP,
            "15551234567@s.whatsapp.net",
            "15551234567@s.whatsapp.net",
        )
    )

    assert result is None
    runner.pairing_store.generate_code.assert_called_once_with(
        "whatsapp",
        "15551234567@s.whatsapp.net",
        "tester",
    )
    adapter.send.assert_awaited_once()
    assert "ABC12DEF" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_unauthorized_whatsapp_dm_can_be_ignored(monkeypatch):
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(
                enabled=True,
                extra={"unauthorized_dm_behavior": "ignore"},
            ),
        },
    )
    runner, adapter = _make_runner(Platform.WHATSAPP, config)

    result = await runner._handle_message(
        _make_event(
            Platform.WHATSAPP,
            "15551234567@s.whatsapp.net",
            "15551234567@s.whatsapp.net",
        )
    )

    assert result is None
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_ignore_suppresses_pairing_reply(monkeypatch):
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        unauthorized_dm_behavior="ignore",
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    runner, adapter = _make_runner(Platform.TELEGRAM, config)

    result = await runner._handle_message(
        _make_event(
            Platform.TELEGRAM,
            "12345",
            "12345",
        )
    )

    assert result is None
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()
