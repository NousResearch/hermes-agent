from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _normalize_live_gateway_response
from gateway.session import SessionSource


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("(No message)", ""),
        ("[SILENT]", ""),
        ("`(No reply)`", ""),
        ("**(No response generated)**", ""),
        ("(empty)", ""),
        ("[SILENT] means stay quiet", "[SILENT] means stay quiet"),
        ("No message received from Discord", "No message received from Discord"),
    ],
)
def test_normalize_live_gateway_response(raw_text, expected):
    assert _normalize_live_gateway_response(raw_text) == expected


def test_normalize_live_gateway_response_preserves_failed_output():
    assert _normalize_live_gateway_response("[SILENT]", failed=True) == "[SILENT]"


def _make_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = MagicMock()
    runner.session_store = MagicMock()
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner.adapters = {}
    runner._show_reasoning = False
    runner._session_db = None
    runner._set_session_env = MagicMock(return_value=[])
    runner._clear_session_env = MagicMock()
    runner._should_send_voice_reply = MagicMock(return_value=False)
    runner._deliver_media_from_response = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_handle_message_with_agent_suppresses_placeholder(monkeypatch):
    runner = _make_runner()

    session_entry = SimpleNamespace(
        session_id="sess-1",
        session_key="key-1",
        created_at=1,
        updated_at=2,
        was_auto_reset=False,
        last_prompt_tokens=0,
    )
    history = [{"role": "assistant", "content": "Earlier reply"}]

    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(No message)",
            "messages": history,
            "api_calls": 1,
            "last_prompt_tokens": 0,
        }
    )

    monkeypatch.setattr("gateway.run.build_session_context", lambda *_a, **_kw: {})
    monkeypatch.setattr("gateway.run.build_session_context_prompt", lambda *_a, **_kw: "")

    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="chat-1",
        user_id="user-1",
        user_name="tester",
    )
    event = MessageEvent(text="test", message_type=MessageType.TEXT, source=source)

    result = await runner._handle_message_with_agent(event, source, "key-1")

    assert result == ""
    appended = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
    assert any(entry["role"] == "user" for entry in appended)
    assert not any(entry.get("content") == "(No message)" for entry in appended)
