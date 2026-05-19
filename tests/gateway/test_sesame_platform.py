"""Native Sesame gateway platform tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _clear_sesame_env(monkeypatch):
    for key in (
        "SESAME_API_KEY",
        "SESAME_API_URL",
        "SESAME_WS_URL",
        "SESAME_ALLOWED_USERS",
        "SESAME_ALLOW_ALL_USERS",
        "SESAME_HOME_CHANNEL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_sesame_api_key_env_enables_platform(monkeypatch):
    """SESAME_API_KEY should make Sesame a first-class connected platform."""
    from gateway.config import Platform, load_gateway_config

    _clear_sesame_env(monkeypatch)
    monkeypatch.setenv("SESAME_API_KEY", "sesame_test_key")
    monkeypatch.setenv("SESAME_HOME_CHANNEL", "chan_123")

    cfg = load_gateway_config()

    assert Platform.SESAME in cfg.platforms
    pconfig = cfg.platforms[Platform.SESAME]
    assert pconfig.enabled is True
    assert pconfig.api_key == "sesame_test_key"
    assert pconfig.home_channel is not None
    assert pconfig.home_channel.chat_id == "chan_123"
    assert Platform.SESAME in cfg.get_connected_platforms()


def test_gateway_runner_creates_sesame_adapter(monkeypatch):
    """GatewayRunner should instantiate the native Sesame adapter."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner

    _clear_sesame_env(monkeypatch)
    monkeypatch.setenv("SESAME_API_KEY", "sesame_test_key")

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()

    adapter = runner._create_adapter(
        Platform.SESAME,
        PlatformConfig(enabled=True, api_key="sesame_test_key"),
    )

    assert adapter is not None
    assert adapter.platform == Platform.SESAME


def test_sesame_config_accepts_yaml_allowed_users(monkeypatch):
    """YAML/list allowlists should not be stringified into a single unusable ID."""
    from gateway.config import PlatformConfig
    from gateway.platforms.sesame import SesameAdapter

    _clear_sesame_env(monkeypatch)

    adapter = SesameAdapter(
        PlatformConfig(
            enabled=True,
            api_key="sesame_test_key",
            extra={"allowed_users": ["ryan-principal", "aiden-principal"]},
        )
    )

    assert adapter.allowed_users == {"ryan-principal", "aiden-principal"}


def test_sesame_cron_home_delivery_target(monkeypatch):
    """Cron delivery should recognize Sesame's configured home channel."""
    from cron.scheduler import _is_known_delivery_platform, _resolve_delivery_targets

    monkeypatch.setenv("SESAME_HOME_CHANNEL", "chan_home")

    assert _is_known_delivery_platform("sesame") is True
    assert _resolve_delivery_targets({"deliver": "sesame"}) == [
        {"platform": "sesame", "chat_id": "chan_home", "thread_id": None}
    ]


@pytest.mark.asyncio
async def test_sesame_adapter_maps_message_frame_to_hermes_event(monkeypatch):
    """Incoming Sesame WS message frames should become Hermes MessageEvents."""
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.sesame import SesameAdapter

    _clear_sesame_env(monkeypatch)
    monkeypatch.setenv("SESAME_ALLOW_ALL_USERS", "true")

    adapter = SesameAdapter(PlatformConfig(enabled=True, api_key="sesame_test_key"))
    adapter.client = SimpleNamespace(
        principal_id="agent-principal",
        get_channel_info=AsyncMock(return_value={"name": "Ryan DM", "kind": "dm"}),
    )
    adapter.handle_message = AsyncMock()

    await adapter._on_ws_event(
        {
            "type": "message",
            "message": {
                "id": "msg_1",
                "channelId": "chan_123",
                "senderId": "ryan-principal",
                "seq": 42,
                "plaintext": "ship it",
                "kind": "text",
                "metadata": {
                    "senderHandle": "ryan",
                    "senderDisplayName": "Ryan Hudson",
                    "senderKind": "human",
                },
            },
        }
    )

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "ship it"
    assert event.message_id == "msg_1"
    assert event.source.platform == Platform.SESAME
    assert event.source.chat_id == "sesame:chan_123"
    assert event.source.chat_type == "dm"
    assert event.source.user_id == "ryan-principal"
    assert event.source.user_name == "Ryan Hudson"
    assert adapter._load_cursors() == {"chan_123": 42}


@pytest.mark.asyncio
async def test_sesame_adapter_requests_replay_from_persisted_cursors(monkeypatch, tmp_path):
    """Native realtime integration should replay missed messages after reconnect."""
    from gateway.config import PlatformConfig
    from gateway.platforms.sesame import SesameAdapter

    _clear_sesame_env(monkeypatch)

    adapter = SesameAdapter(PlatformConfig(enabled=True, api_key="sesame_test_key"))
    adapter._cursor_path = tmp_path / "sesame_cursors.json"
    adapter.client = SimpleNamespace(principal_id="agent-principal", request_replay=AsyncMock())

    adapter._save_cursor("chan_123", 41)
    await adapter._request_replay_on_connect()

    adapter.client.request_replay.assert_awaited_once_with({"chan_123": 41})


@pytest.mark.asyncio
async def test_sesame_send_message_uses_attachment_kind_and_nonempty_content(monkeypatch):
    """Sesame message validation requires non-empty content and attachment kind for files."""
    from gateway.platforms.sesame import SesameClient

    client = SesameClient("https://api.sesame.space", "wss://ws.sesame.space", "sesame_test_key")
    client._request = AsyncMock(return_value={"data": {"id": "msg_1"}})

    await client.send_message("chan_123", "Attachment: report.pdf", attachment_ids=["file-uuid"])

    client._request.assert_awaited_once()
    kwargs = client._request.await_args.kwargs
    assert kwargs["json_body"]["content"] == "Attachment: report.pdf"
    assert kwargs["json_body"]["kind"] == "attachment"
    assert kwargs["json_body"]["attachmentIds"] == ["file-uuid"]
