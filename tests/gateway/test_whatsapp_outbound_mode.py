"""Outbound send-policy tests for the WhatsApp Baileys adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config
from plugins.platforms.whatsapp.adapter import (
    WhatsAppAdapter,
    _normalize_outbound_mode,
    _standalone_send,
)


@pytest.fixture
def read_only_adapter(tmp_path):
    adapter = WhatsAppAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "outbound_mode": "never",
                "session_path": str(tmp_path / "session"),
            },
        )
    )
    adapter._running = True
    adapter._http_session = MagicMock()
    adapter._send_read_receipts = True
    return adapter


def test_outbound_mode_defaults_normal_and_fails_closed():
    assert _normalize_outbound_mode(None) == "normal"
    assert _normalize_outbound_mode(" NORMAL ") == "normal"
    assert _normalize_outbound_mode("never") == "never"
    assert _normalize_outbound_mode("typo") == "never"


def test_yaml_config_seeds_outbound_mode_through_gateway_loader(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "whatsapp:\n  enabled: true\n  outbound_mode: never\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_gateway_config()

    assert config.platforms[Platform.WHATSAPP].extra["outbound_mode"] == "never"


@pytest.mark.asyncio
async def test_never_mode_refuses_every_adapter_outbound_primitive(
    read_only_adapter, tmp_path
):
    media = tmp_path / "photo.png"
    media.write_bytes(b"png")

    results = [
        await read_only_adapter.send("15551234567", "hello"),
        await read_only_adapter.edit_message("15551234567", "m1", "edited"),
        await read_only_adapter._send_media_to_bridge(
            "15551234567", str(media), "image"
        ),
        await read_only_adapter.send_poll(
            "15551234567", "Proceed?", ["Yes", "No"]
        ),
        await read_only_adapter.send_location("15551234567", 1.0, 2.0),
    ]
    await read_only_adapter.send_typing("15551234567")
    await read_only_adapter._send_read_receipt(
        {"readReceiptKey": {"id": "incoming-1"}}
    )

    assert all(result.success is False for result in results)
    assert all("outbound_mode=never" in result.error for result in results)
    read_only_adapter._http_session.post.assert_not_called()


@pytest.mark.asyncio
async def test_never_mode_refuses_standalone_delivery_before_http():
    config = SimpleNamespace(extra={"outbound_mode": "never", "bridge_port": 3000})
    with patch("aiohttp.ClientSession", new_callable=MagicMock) as client:
        result = await _standalone_send(config, "15551234567", "hello")

    assert "outbound_mode=never" in result["error"]
    client.assert_not_called()


@pytest.mark.asyncio
async def test_normal_mode_preserves_adapter_send_path(tmp_path):
    adapter = WhatsAppAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "outbound_mode": "normal",
                "session_path": str(tmp_path / "session"),
            },
        )
    )
    adapter._running = True
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)

    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={"messageId": "sent-1"})
    response_context = MagicMock()
    response_context.__aenter__ = AsyncMock(return_value=response)
    response_context.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = response_context
    adapter._http_session = session

    result = await adapter.send("15551234567", "hello")

    assert result.success is True
    assert result.message_id == "sent-1"
    session.post.assert_called_once()
