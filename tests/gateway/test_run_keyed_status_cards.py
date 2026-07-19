"""Gateway seam coverage for Discord's retained task-run status card."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import _discord_task_run_status_key, _send_or_update_status_coro
from gateway.session import SessionSource, build_session_key
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


@pytest.mark.asyncio
async def test_status_callback_prefers_routed_metadata_status_key():
    adapter = SimpleNamespace(
        send_or_update_status=AsyncMock(
            return_value=SendResult(success=True, message_id="7001")
        ),
        send=AsyncMock(),
    )

    result = await _send_or_update_status_coro(
        adapter,
        "555",
        "compression",
        "Still working",
        {"thread_id": "777", "status_key": "task_run"},
    )

    assert result.message_id == "7001"
    adapter.send_or_update_status.assert_awaited_once_with(
        "555",
        "task_run",
        "Still working",
        metadata={"thread_id": "777", "status_key": "task_run"},
    )
    adapter.send.assert_not_awaited()


def test_discord_task_run_key_is_stable_within_run_and_distinct_across_runs():
    first = _discord_task_run_status_key(
        Platform.DISCORD,
        event_message_id="42",
        session_key="discord:555:thread:777",
        run_generation=1,
    )
    same_first = _discord_task_run_status_key(
        Platform.DISCORD,
        event_message_id="42",
        session_key="discord:555:thread:777",
        run_generation=1,
    )
    second = _discord_task_run_status_key(
        Platform.DISCORD,
        event_message_id="43",
        session_key="discord:555:thread:777",
        run_generation=2,
    )

    assert first == same_first == "task_run:message:42"
    assert second == "task_run:message:43"
    assert first != second


def test_discord_task_run_key_fallback_is_generation_scoped():
    first = _discord_task_run_status_key(
        Platform.DISCORD,
        session_key="discord:555:thread:777",
        run_generation=1,
    )
    second = _discord_task_run_status_key(
        Platform.DISCORD,
        session_key="discord:555:thread:777",
        run_generation=2,
    )

    assert first.startswith("task_run:generation:")
    assert second.startswith("task_run:generation:")
    assert first != second
    assert _discord_task_run_status_key(Platform.TELEGRAM) is None


@pytest.mark.asyncio
async def test_stream_terminal_metadata_is_copied_not_shared_with_heartbeat():
    calls = []

    class _Adapter:
        async def edit_message(
            self,
            chat_id,
            message_id,
            content,
            *,
            finalize=False,
            metadata=None,
        ):
            calls.append(metadata)
            return SendResult(success=True, message_id=message_id)

    shared = {"thread_id": "777", "status_key": "task_run:message:42"}
    consumer = GatewayStreamConsumer(
        adapter=_Adapter(),
        chat_id="555",
        config=StreamConsumerConfig(),
        metadata=shared,
    )

    final_send_metadata = consumer._metadata_for_send(final=True)
    await consumer._edit_message(
        message_id="7001",
        content="Complete",
        finalize=True,
    )

    assert shared == {
        "thread_id": "777",
        "status_key": "task_run:message:42",
    }
    assert final_send_metadata["status_terminal"] is True
    assert calls == [
        {
            "thread_id": "777",
            "status_key": "task_run:message:42",
            "status_terminal": True,
        }
    ]


@pytest.mark.asyncio
async def test_identical_empty_cursor_final_still_terminalizes_keyed_card():
    calls = []

    class _Adapter:
        async def edit_message(
            self,
            chat_id,
            message_id,
            content,
            *,
            finalize=False,
            metadata=None,
        ):
            calls.append(
                {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "content": content,
                    "finalize": finalize,
                    "metadata": metadata,
                }
            )
            return SendResult(success=True, message_id=message_id)

    final_text = "Complete final result " * 40
    consumer = GatewayStreamConsumer(
        adapter=_Adapter(),
        chat_id="555",
        config=StreamConsumerConfig(cursor=""),
        metadata={"thread_id": "777", "status_key": "task_run:message:42"},
    )
    consumer._message_id = "7001"
    consumer._last_sent_text = final_text

    delivered = await consumer._send_or_edit(
        final_text,
        finalize=True,
        is_turn_final=True,
    )

    assert delivered is True
    assert calls == [
        {
            "chat_id": "555",
            "message_id": "7001",
            "content": final_text,
            "finalize": True,
            "metadata": {
                "thread_id": "777",
                "status_key": "task_run:message:42",
                "status_terminal": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_recovered_done_send_keeps_terminal_metadata():
    calls = []

    class _Adapter:
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            calls.append(metadata)
            if len(calls) == 1:
                return SendResult(success=False, error="temporary")
            return SendResult(success=True, message_id="7001")

    consumer = GatewayStreamConsumer(
        adapter=_Adapter(),
        chat_id="555",
        config=StreamConsumerConfig(cursor="", buffer_only=True),
        metadata={"thread_id": "777", "status_key": "task_run:message:42"},
    )
    consumer.on_delta("Complete final result")
    consumer.finish()

    await consumer.run()

    assert consumer.final_response_sent is True
    assert calls == [
        {
            "thread_id": "777",
            "status_key": "task_run:message:42",
            "expect_edits": True,
            "notify": True,
            "status_terminal": True,
        },
        {
            "thread_id": "777",
            "status_key": "task_run:message:42",
            "expect_edits": True,
            "notify": True,
            "status_terminal": True,
        },
    ]


class _FinalDeliveryAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="fake-token"),
            Platform.DISCORD,
        )
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="7001")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_base_final_delivery_carries_terminal_status_identity():
    adapter = _FinalDeliveryAdapter()

    async def handler(event):
        event._status_key = "task_run"
        return "Complete final result"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter._keep_typing = hold_typing
    event = MessageEvent(
        text="Do the work",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="555",
            chat_type="thread",
            thread_id="777",
        ),
        message_id="42",
    )

    await adapter._process_message_background(
        event,
        build_session_key(event.source),
    )

    assert adapter.sent == [
        {
            "chat_id": "555",
            "content": "Complete final result",
            "reply_to": "42",
            "metadata": {
                "thread_id": "777",
                "notify": True,
                "status_key": "task_run",
                "status_terminal": True,
            },
        }
    ]
