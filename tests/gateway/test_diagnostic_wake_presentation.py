"""Diagnostic-only wakes execute while their unsolicited free-form reply stays private."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class Adapter(BasePlatformAdapter):
    async def connect(self, **kwargs):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {"name": "fixture", "type": "dm"}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="sent")


@pytest.mark.asyncio
@pytest.mark.parametrize("suppressed", [False, True])
async def test_diagnostic_wake_executes_without_final_or_error_echo(tmp_path, monkeypatch, suppressed):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(suppressed).lower()}}}")
    adapter = Adapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    adapter.sent = []
    adapter._start_typing_refresh = lambda *a: None
    adapter._stop_typing_refresh = AsyncMock()
    adapter._run_processing_hook = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._flush_text_debounce_now = AsyncMock()
    adapter._finish_session_task = lambda *a: None
    # The normal lane calls the private bracket, which also returns the ledger row id.
    adapter._send_final_ledgered = AsyncMock(return_value=(SendResult(success=True), adapter, None))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat")
    event = MessageEvent(text="internal diagnostic", source=source, internal=True,
                         metadata={"notification_category": "diagnostic"})
    adapter._message_handler = AsyncMock(return_value="I repeat the technical diagnostic")
    await adapter._process_message_background(event, "session")
    assert adapter._message_handler.await_count == 1
    assert adapter._send_final_ledgered.await_count == (0 if suppressed else 1)
    adapter._message_handler = AsyncMock(side_effect=RuntimeError("provider detail"))
    await adapter._process_message_background(event, "session")
    assert bool(adapter.sent) is not suppressed
    assert adapter._run_processing_hook.await_args.args[-1].value == "failure"
