"""Reaction delivery keeps complete receipts within Matrix's byte budget."""

import asyncio
import json
from typing import Any
from unittest.mock import Mock, call

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.matrix.adapter import MatrixAdapter


class _RecordingAdapter(MatrixAdapter):
    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config)
        self.delivered: list[tuple[str, dict[str, Any], bool, str]] = []
        self.sent: list[tuple[str, dict[str, Any], bool]] = []

    async def _send_room_message(
        self, chat_id: str, msg_content: dict[str, Any], *, finalize: bool = True
    ) -> str:
        event_id = f"$event{len(self.delivered)}"
        self.delivered.append((chat_id, msg_content, finalize, event_id))
        return event_id

    async def _send_content_event(
        self, room_id: str, msg_content: dict[str, Any], *, finalize: bool = True
    ) -> SendResult:
        self.sent.append((room_id, msg_content, finalize))
        return SendResult(success=True, message_id="$edit")


def _adapter() -> _RecordingAdapter:
    return _RecordingAdapter(
        PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
            },
        )
    )


def test_multibyte_chunks_keep_every_reaction_delivery_receipt():
    async def exercise():
        adapter = _adapter()
        adapter.max_message_length = 12
        delivered = adapter.delivered
        result = await adapter.send("!room:example.org", "🦊" * 4)

        assert [item for item in delivered if len(item[1]["body"].encode()) > 12] == []
        event_ids = tuple(item[3] for item in delivered)
        assert result == SendResult(
            success=True,
            message_id=event_ids[-1],
            continuation_message_ids=event_ids[:-1],
        )
        assert [(item[0], item[2]) for item in delivered] == [
            ("!room:example.org", True) for _ in delivered
        ]

    asyncio.run(exercise())


def test_final_edit_caps_fallback_and_keeps_complete_effective_content():
    async def exercise():
        adapter = _adapter()
        adapter._event_context_cache = Mock()
        text = "**" + "é" * 7400 + "**"
        expected_content = adapter._build_text_message_content(text)
        sent = adapter.sent
        result = await adapter.edit_message(
            "!room:example.org", "$original", text, finalize=True
        )

        assert result == SendResult(success=True, message_id="$edit")
        assert len(sent) == 1
        chat_id, payload, finalize = sent[0]
        assert (chat_id, finalize, payload["m.new_content"]) == (
            "!room:example.org",
            True,
            expected_content,
        )
        assert (
            len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
            <= 45000
        )
        assert adapter._event_context_cache.apply_edit.call_args_list == [
            call(
                "!room:example.org", "@bot:example.org", payload, replacement_id="$edit"
            )
        ]

    asyncio.run(exercise())
