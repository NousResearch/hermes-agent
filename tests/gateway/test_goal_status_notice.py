from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE


class FakeAdapter:
    def __init__(self):
        self.calls = []
        self.callbacks = {}
        self._active_sessions = {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SimpleNamespace(success=True)

    def register_post_delivery_callback(self, session_key, callback, *, generation=None):
        self.callbacks[session_key] = (generation, callback)


def _goal_continuation_event(source, goal="finish the task"):
    return MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal=goal),
        message_type=MessageType.TEXT,
        source=source,
    )


@pytest.mark.asyncio
async def test_goal_status_notice_uses_adapter_send_with_thread_metadata():
    """Regression: /goal judge status must use BasePlatformAdapter.send().

    The old implementation checked for a non-existent send_message() method,
    so the goal could be marked done in state_meta without the visible
    "✓ Goal achieved" status line being delivered to Discord/Telegram.
    """
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.DISCORD: adapter}

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
    )

    goal_event = {
        "contract": "hermes.goal_event.v1",
        "event_type": "done",
        "status": "done",
        "reason": "done",
        "raw_text": "✓ Goal achieved: done",
    }

    await runner._send_goal_status_notice(
        source,
        "✓ Goal achieved: done",
        goal_event=goal_event,
    )

    assert adapter.calls == [
        {
            "chat_id": "parent-channel",
            "content": "✓ Goal achieved: done",
            "reply_to": None,
            "metadata": {"thread_id": "thread-123", "goal_event": goal_event},
        }
    ]


@pytest.mark.asyncio
async def test_goal_status_notice_uses_profile_routed_adapter():
    """Multiplexed profile replies must use the source-routed adapter.

    A direct ``self.adapters.get(source.platform)`` lookup would pick the
    default-profile adapter and misroute the goal notice.
    """
    runner = GatewayRunner.__new__(GatewayRunner)
    default_adapter = FakeAdapter()
    profile_adapter = FakeAdapter()
    runner.adapters = {Platform.DISCORD: default_adapter}
    runner._adapter_for_source = lambda source: profile_adapter

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="profile-channel",
        thread_id="thread-123",
        profile="secondary",
    )

    await runner._send_goal_status_notice(source, "✓ Goal achieved: done")

    assert default_adapter.calls == []
    assert profile_adapter.calls == [
        {
            "chat_id": "profile-channel",
            "content": "✓ Goal achieved: done",
            "reply_to": None,
            "metadata": {"thread_id": "thread-123"},
        }
    ]


@pytest.mark.asyncio
async def test_goal_status_notice_defers_until_post_delivery_callback():
    """Regression: goal status must appear after the agent's visible reply.

    _post_turn_goal_continuation runs before BasePlatformAdapter sends the
    returned final response. It should therefore register a post-delivery
    callback, not send the judge status immediately.
    """
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False)

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
        user_id="user-1",
    )

    goal_event = {
        "contract": "hermes.goal_event.v1",
        "event_type": "done",
        "status": "done",
        "reason": "done",
        "raw_text": "✓ Goal achieved: done",
    }

    await runner._defer_goal_status_notice_after_delivery(
        source,
        "✓ Goal achieved: done",
        goal_event=goal_event,
    )

    assert adapter.calls == []
    assert len(adapter.callbacks) == 1

    _, callback = next(iter(adapter.callbacks.values()))
    result = callback()
    if hasattr(result, "__await__"):
        await result

    assert adapter.calls == [
        {
            "chat_id": "parent-channel",
            "content": "✓ Goal achieved: done",
            "reply_to": None,
            "metadata": {"thread_id": "thread-123", "goal_event": goal_event},
        }
    ]

