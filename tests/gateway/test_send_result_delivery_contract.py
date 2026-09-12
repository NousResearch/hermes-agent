"""Production adapter contract for explicit pre-I/O delivery attestation."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.bluebubbles import BlueBubblesAdapter
from gateway.platforms.weixin import WeixinAdapter


def _weixin_adapter() -> WeixinAdapter:
    return WeixinAdapter(
        PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"account_id": "test-account"},
        )
    )


@pytest.mark.asyncio
async def test_disconnected_text_adapter_attests_no_delivery_attempt() -> None:
    adapter = _weixin_adapter()

    result = await adapter.send("target", "hello")

    assert result.success is False
    assert result.delivery_attempted is False


@pytest.mark.asyncio
async def test_local_validation_and_unsupported_send_attest_no_delivery_attempt() -> None:
    bluebubbles = BlueBubblesAdapter(
        PlatformConfig(
            enabled=True,
            extra={"server_url": "http://localhost:1234", "password": "test-password"},
        )
    )
    unsupported = object.__new__(APIServerAdapter)

    invalid_result = await bluebubbles.send("target", "")
    unsupported_result = await unsupported.send("target", "hello")

    assert invalid_result.delivery_attempted is False
    assert unsupported_result.delivery_attempted is False


@pytest.mark.asyncio
async def test_missing_local_attachment_attests_no_delivery_attempt(tmp_path) -> None:
    adapter = BlueBubblesAdapter(
        PlatformConfig(
            enabled=True,
            extra={"server_url": "http://localhost:1234", "password": "test-password"},
        )
    )
    adapter.client = object()

    result = await adapter._send_attachment("target", str(tmp_path / "missing.png"))

    assert result.success is False
    assert result.delivery_attempted is False


@pytest.mark.asyncio
async def test_partial_chunk_failure_remains_unattested(monkeypatch) -> None:
    adapter = _weixin_adapter()
    adapter._send_session = object()
    adapter._token = "test-token"
    monkeypatch.setattr(adapter, "_split_text", lambda _text: ["first", "second"])
    send_chunk = AsyncMock(side_effect=[None, RuntimeError("second chunk failed")])
    monkeypatch.setattr(adapter, "_send_text_chunk", send_chunk)

    result = await adapter.send("target", "two chunks")

    assert send_chunk.await_count == 2
    assert result.success is False
    assert result.delivery_attempted is None


@pytest.mark.asyncio
async def test_accepted_send_never_attests_no_delivery_attempt(monkeypatch) -> None:
    adapter = _weixin_adapter()
    adapter._send_session = object()
    adapter._token = "test-token"
    monkeypatch.setattr(adapter, "_split_text", lambda _text: ["only chunk"])
    send_chunk = AsyncMock(return_value=None)
    monkeypatch.setattr(adapter, "_send_text_chunk", send_chunk)

    result = await adapter.send("target", "hello")

    send_chunk.assert_awaited_once()
    assert result.success is True
    assert result.delivery_attempted is not False
