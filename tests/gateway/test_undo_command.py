import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_event(text="/undo"):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-1",
        chat_id="chat-1",
        user_name="tester",
    )
    return MessageEvent(text=text, source=source)


@pytest.mark.asyncio
async def test_handle_undo_command_reports_one_turn_and_internal_transcript_count():
    runner = object.__new__(GatewayRunner)
    entry = SimpleNamespace(session_id="sess-1", last_prompt_tokens=42)
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "fix /undo"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "search_files", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"success": true}'},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[Internal: inspect the attached image and continue.]"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
            "_native_image_attachment": True,
        },
        {"role": "assistant", "content": "intermediate note"},
        {"role": "assistant", "content": "final answer"},
    ]

    store = MagicMock()
    store.get_or_create_session.return_value = entry
    store.load_transcript.return_value = history
    runner.session_store = store

    result = await runner._handle_undo_command(_make_event())

    store.rewrite_transcript.assert_called_once_with("sess-1", history[:2])
    assert entry.last_prompt_tokens == 0
    assert "Undid 1 turn" in result
    assert "6 internal transcript messages" in result
    assert 'Removed your last message: "fix /undo"' in result
    assert "Undid 5 message(s)" not in result
