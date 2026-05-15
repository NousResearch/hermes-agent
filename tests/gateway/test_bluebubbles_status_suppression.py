from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.bluebubbles import BlueBubblesAdapter
from gateway.platforms.base import SendResult


class _RecordingBlueBubblesAdapter(BlueBubblesAdapter):
    def __init__(self, *, chat_type: str = "dm", show_status_in_groups: bool = False):
        super().__init__(
            PlatformConfig(
                enabled=True,
                extra={"show_status_in_groups": show_status_in_groups},
            )
        )
        self._chat_type = chat_type
        self.sent: list[dict[str, Any]] = []

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="bb-1")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": self._chat_type, "chat_id": chat_id}


@pytest.mark.asyncio
async def test_bluebubbles_suppresses_kind_status_in_groups_by_default():
    adapter = _RecordingBlueBubblesAdapter(chat_type="group")

    result = await adapter._send_with_retry(
        chat_id="chat;+;guid",
        content="⚡ Interrupting current task",
        kind="status",
    )

    assert result.success is True
    assert result.suppressed is True
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_bluebubbles_delivers_normal_replies_in_groups():
    adapter = _RecordingBlueBubblesAdapter(chat_type="group")

    result = await adapter._send_with_retry(
        chat_id="chat;+;guid",
        content="normal answer",
    )

    assert result.success is True
    assert result.suppressed is False
    assert len(adapter.sent) == 1
    assert adapter.sent[0]["content"] == "normal answer"


@pytest.mark.asyncio
async def test_bluebubbles_delivers_status_in_dms():
    adapter = _RecordingBlueBubblesAdapter(chat_type="dm")

    result = await adapter._send_with_retry(
        chat_id="person@example.com",
        content="⚡ Interrupting current task",
        kind="status",
    )

    assert result.success is True
    assert result.suppressed is False
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_bluebubbles_status_group_suppression_can_be_disabled():
    adapter = _RecordingBlueBubblesAdapter(
        chat_type="group",
        show_status_in_groups=True,
    )

    result = await adapter._send_with_retry(
        chat_id="chat;+;guid",
        content="⚡ Interrupting current task",
        kind="status",
    )

    assert result.success is True
    assert result.suppressed is False
    assert len(adapter.sent) == 1
