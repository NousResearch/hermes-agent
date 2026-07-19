import logging

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class RecordingAdapter(BasePlatformAdapter):
    def __init__(self, result):
        super().__init__(config=PlatformConfig(extra={}), platform=Platform.FEISHU)
        self.result = result
        self.document_calls = []

    async def connect(self):  # pragma: no cover
        return True

    async def disconnect(self):  # pragma: no cover
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):  # pragma: no cover
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):  # pragma: no cover
        return {"chat_id": chat_id}

    async def send_document(self, chat_id, file_path, caption=None, file_name=None, reply_to=None, metadata=None, **kwargs):
        self.document_calls.append({"chat_id": chat_id, "file_path": file_path, "metadata": metadata})
        return self.result


class DummyRunner:
    def _thread_metadata_for_source(self, source, reply_anchor):
        return {"thread_id": source.thread_id} if source.thread_id else None

    def _reply_anchor_for_event(self, event):
        return event.message_id


@pytest.mark.asyncio
async def test_post_stream_media_delivery_logs_sendresult_failure(tmp_path, caplog):
    artifact = tmp_path / "report.md"
    artifact.write_text("hello", encoding="utf-8")
    adapter = RecordingAdapter(SendResult(success=False, error="file upload missing file_key"))
    event = MessageEvent(
        text="trigger",
        source=SessionSource(platform=Platform.FEISHU, chat_id="oc_test", message_id="om_test"),
        message_id="om_test",
    )

    with caplog.at_level(logging.WARNING):
        await GatewayRunner._deliver_media_from_response(
            DummyRunner(),
            f"done\nMEDIA:{artifact}",
            event,
            adapter,
        )

    assert adapter.document_calls
    assert "Post-stream media delivery failed" in caplog.text
    assert "file upload missing file_key" in caplog.text
