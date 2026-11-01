import pytest
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from plugins.platforms.rubika.adapter import RubikaAdapter


def _adapter() -> RubikaAdapter:
    cfg = PlatformConfig()
    cfg.extra = {"token": "TESTTOKEN"}
    return RubikaAdapter(cfg)


@pytest.mark.asyncio
async def test_send_image_file_uploads_then_sends(tmp_path):
    adapter = _adapter()
    image_path = tmp_path / "pic.png"
    image_path.write_bytes(b"fake-png")
    adapter._client.upload_file = AsyncMock(return_value="file-abc")
    adapter._client.call = AsyncMock(return_value={"message_id": "m11"})

    result = await adapter.send_image_file(chat_id="c1", image_path=str(image_path), caption="a pic")

    assert result.success is True
    adapter._client.upload_file.assert_awaited_once_with(str(image_path), file_type="Image")
    adapter._client.call.assert_awaited_once_with(
        "sendFile", chat_id="c1", file_id="file-abc", text="a pic")


@pytest.mark.asyncio
async def test_send_document_uploads_then_sends(tmp_path):
    adapter = _adapter()
    doc_path = tmp_path / "report.pdf"
    doc_path.write_bytes(b"fake-pdf")
    adapter._client.upload_file = AsyncMock(return_value="file-doc")
    adapter._client.call = AsyncMock(return_value={"message_id": "m12"})

    result = await adapter.send_document(chat_id="c1", file_path=str(doc_path))

    assert result.success is True
    adapter._client.upload_file.assert_awaited_once_with(str(doc_path), file_type="File")
    adapter._client.call.assert_awaited_once_with(
        "sendFile", chat_id="c1", file_id="file-doc", text="")
