import asyncio
import sys
import types


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from cli import _format_process_notification
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _format_gateway_process_notification
from gateway.session import SessionSource


def _codex_completion(exit_code=0):
    return {
        "type": "completion",
        "session_id": "codex_turn_turn-123",
        "command": "codex app-server turn",
        "exit_code": exit_code,
        "output": "Codex turn completed via app-server",
    }


def test_cli_formats_codex_completion_as_codex_turn_notification():
    text = _format_process_notification(_codex_completion())

    assert "Codex app-server turn completed" in text
    assert "Session: codex_turn_turn-123" in text
    assert "Verify git diff and run relevant tests" in text


def test_gateway_formats_codex_completion_as_codex_turn_notification():
    text = _format_gateway_process_notification(_codex_completion(exit_code=1))

    assert "Codex app-server turn failed" in text
    assert "Session: codex_turn_turn-123" in text
    assert "Verify git diff and run relevant tests" in text


def test_gateway_leaves_regular_completion_to_process_watcher():
    text = _format_gateway_process_notification(
        {
            "type": "completion",
            "session_id": "proc_123",
            "command": "pytest",
            "exit_code": 0,
            "output": "ok",
        }
    )

    assert text is None


def test_gateway_inject_watch_notification_prefers_routing_event_metadata():
    class FakeAdapter:
        def __init__(self):
            self.events = []

        async def handle_message(self, event):
            self.events.append(event)

    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.LOCAL: adapter}

    original_event = MessageEvent(
        text="original",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.LOCAL, chat_id="original-chat", user_id="u1"),
    )

    routing_event = {
        "platform": "local",
        "chat_id": "target-chat",
        "thread_id": "thread-42",
        "user_id": "user-42",
        "user_name": "Dongwoo",
        "chat_name": "CLI",
    }

    asyncio.run(
        runner._inject_watch_notification(
            "[SYSTEM: Codex app-server turn completed]",
            original_event,
            routing_event,
        )
    )

    assert len(adapter.events) == 1
    injected = adapter.events[0]
    assert injected.internal is True
    assert injected.source.chat_id == "target-chat"
    assert injected.source.thread_id == "thread-42"
    assert injected.source.user_id == "user-42"
