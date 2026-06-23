import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.photon.adapter import PhotonAdapter


@pytest.mark.asyncio
async def test_direct_send_restarts_sidecar_after_recoverable_upstream_drop(monkeypatch):
    adapter = PhotonAdapter(
        PlatformConfig(
            enabled=True,
            extra={"project_id": "pid", "project_secret": "secret"},
        )
    )

    calls = {"send": 0, "restart": 0}

    async def fake_sidecar_send(chat_id, text):
        calls["send"] += 1
        assert chat_id == "chat-1"
        assert text == "hello"
        if calls["send"] == 1:
            return SendResult(
                success=False,
                error='Photon sidecar /send returned 500 error_code=upstream_connection_dropped: {"ok":false,"error":"internal sidecar error","error_code":"upstream_connection_dropped"}',
            )
        return SendResult(success=True, message_id="m-direct-recovered")

    async def fake_restart(reason: str):
        calls["restart"] += 1
        assert "upstream_connection_dropped" in reason

    monkeypatch.setattr(adapter, "_sidecar_send", fake_sidecar_send)
    monkeypatch.setattr(adapter, "_restart_sidecar_for_outbound_error", fake_restart)

    result = await adapter.send(chat_id="chat-1", content="hello")

    assert result.success is True
    assert result.message_id == "m-direct-recovered"
    assert calls == {"send": 2, "restart": 1}


@pytest.mark.asyncio
async def test_direct_send_recovers_plain_upstream_drop_text(monkeypatch):
    adapter = PhotonAdapter(
        PlatformConfig(
            enabled=True,
            extra={"project_id": "pid", "project_secret": "secret"},
        )
    )

    calls = {"send": 0, "restart": 0}

    async def fake_sidecar_send(chat_id, text):
        calls["send"] += 1
        if calls["send"] == 1:
            return SendResult(
                success=False,
                error="Photon sidecar /send returned 500: ConnectionError: [upstream] Connection dropped",
            )
        return SendResult(success=True, message_id="m-plain-text-recovered")

    async def fake_restart(reason: str):
        calls["restart"] += 1
        assert "upstream_connection_dropped" in reason

    monkeypatch.setattr(adapter, "_sidecar_send", fake_sidecar_send)
    monkeypatch.setattr(adapter, "_restart_sidecar_for_outbound_error", fake_restart)

    result = await adapter.send(chat_id="chat-1", content="hello")

    assert result.success is True
    assert result.message_id == "m-plain-text-recovered"
    assert calls == {"send": 2, "restart": 1}


@pytest.mark.asyncio
async def test_send_with_retry_restarts_sidecar_after_recoverable_upstream_drop(monkeypatch):
    adapter = PhotonAdapter(
        PlatformConfig(
            enabled=True,
            extra={"project_id": "pid", "project_secret": "secret"},
        )
    )

    calls = {"send": 0, "restart": 0}

    async def fake_send(*, chat_id, content, reply_to=None, metadata=None):
        calls["send"] += 1
        if calls["send"] == 1:
            return SendResult(
                success=False,
                error='Photon sidecar /send returned 500 error_code=upstream_connection_dropped: {"ok":false,"error":"internal sidecar error","error_code":"upstream_connection_dropped"}',
            )
        return SendResult(success=True, message_id="m-recovered")

    async def fake_restart(reason: str):
        calls["restart"] += 1
        assert "upstream_connection_dropped" in reason

    monkeypatch.setattr(adapter, "send", fake_send)
    monkeypatch.setattr(adapter, "_restart_sidecar_for_outbound_error", fake_restart)

    result = await adapter._send_with_retry("chat-1", "/new")

    assert result.success is True
    assert result.message_id == "m-recovered"
    assert calls == {"send": 2, "restart": 1}
