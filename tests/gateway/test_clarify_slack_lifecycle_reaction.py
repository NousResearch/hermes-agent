"""Regression tests for Slack lifecycle reactions on typed clarify answers."""

from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


SESSION_KEY = "agent:main:slack:dm:D123:1111.2222"


def _event():
    return MessageEvent(
        text="2",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="D123",
            chat_type="dm",
            user_id="U1",
            thread_id="1111.2222",
        ),
        message_id="171.000002",
    )


def _runner(adapter):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._startup_restore_in_progress = False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized = lambda source: True
    runner._session_key_for_source = lambda source: SESSION_KEY
    runner._delivery_adapter_for = lambda source: adapter
    runner._update_prompt_pending = {}
    return runner


@pytest.mark.asyncio
async def test_typed_clarify_answer_gets_lifecycle_reactions():
    """A successful inline clarify reply gets eyes, then the success reaction."""
    from plugins.platforms.slack.adapter import SlackAdapter
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()
    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._reacting_message_ids = set()
    adapter._reactions_enabled = lambda: True
    adapter._workspace_message_marker = lambda team_id, ts: (team_id, ts)
    adapter._react = AsyncMock(return_value=True)
    adapter.resume_typing_for_chat = lambda chat_id: None
    adapter.retire_clarify_card = AsyncMock()
    async def run_hook(hook_name, *args, **kwargs):
        if hook_name == "on_processing_start":
            await SlackAdapter.on_processing_start(adapter, *args, **kwargs)
        else:
            await SlackAdapter.on_processing_complete(adapter, *args, **kwargs)
    adapter._run_processing_hook = run_hook
    event = _event()
    adapter._reacting_message_ids.add(("", event.message_id))
    runner = _runner(adapter)
    pending = cm.register("typed-slack-clarify", SESSION_KEY, "Pick an option", ["one", "two"])

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = await runner._handle_message(event)

    assert result == ""
    assert pending.response == "two"
    assert [call.args for call in adapter._react.await_args_list] == [
        ("D123", "171.000002", "eyes", ""),
        ("D123", "171.000002", "eyes", ""),
        ("D123", "171.000002", "white_check_mark", ""),
    ]
    assert [call.kwargs for call in adapter._react.await_args_list] == [
        {"remove": False}, {"remove": True}, {"remove": False},
    ]
    assert ("", event.message_id) not in adapter._reacting_message_ids
    adapter.retire_clarify_card.assert_awaited_once_with("typed-slack-clarify", "✅ answered: two")
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()
