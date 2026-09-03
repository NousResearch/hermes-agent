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
        self.wisdom_calls = []
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

    async def send_wisdom_candidate_notifications(
        self, chat_id, session_id, *, metadata=None
    ):
        self.wisdom_calls.append(
            {"chat_id": chat_id, "session_id": session_id, "metadata": metadata}
        )


def _goal_continuation_event(source, goal="finish the task"):
    return MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal=goal),
        message_type=MessageType.TEXT,
        source=source,
    )


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

    await runner._defer_goal_status_notice_after_delivery(source, "✓ Goal achieved: done")

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
            "metadata": {"thread_id": "thread-123"},
        }
    ]


@pytest.mark.asyncio
async def test_wisdom_candidate_notice_defers_in_the_originating_telegram_session():
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.config = SimpleNamespace(
        group_sessions_per_user=True, thread_sessions_per_user=False
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="telegram-chat",
        thread_id="topic-7",
        user_id="user-1",
    )

    await runner._defer_wisdom_candidate_notice_after_delivery(
        source, "exact-session-id"
    )

    assert adapter.wisdom_calls == []
    assert len(adapter.callbacks) == 1
    _, callback = next(iter(adapter.callbacks.values()))
    result = callback()
    if hasattr(result, "__await__"):
        await result
    assert len(adapter.wisdom_calls) == 1
    assert adapter.wisdom_calls[0]["chat_id"] == "telegram-chat"
    assert adapter.wisdom_calls[0]["session_id"] == "exact-session-id"
    assert adapter.wisdom_calls[0]["metadata"]["thread_id"] == "topic-7"


@pytest.mark.asyncio
async def test_wisdom_candidate_notice_defers_to_slack_with_private_routing_identity():
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.SLACK: adapter}
    runner._profile_adapters = {
        "collective-demo": {Platform.SLACK: adapter},
    }
    runner.config = SimpleNamespace(
        group_sessions_per_user=True, thread_sessions_per_user=False
    )
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C_TEAM",
        thread_id="1720000000.0001",
        user_id="U_OWNER",
        scope_id="T_TEAM",
        profile="collective-demo",
    )

    await runner._defer_wisdom_candidate_notice_after_delivery(
        source, "exact-session-id"
    )

    assert adapter.wisdom_calls == []
    _, callback = next(iter(adapter.callbacks.values()))
    result = callback()
    if hasattr(result, "__await__"):
        await result
    assert adapter.wisdom_calls == [
        {
            "chat_id": "C_TEAM",
            "session_id": "exact-session-id",
            "metadata": {
                "thread_id": "1720000000.0001",
                "slack_team_id": "T_TEAM",
                "scope_id": "T_TEAM",
                "user_id": "U_OWNER",
                "profile": "collective-demo",
                "hermes_profile": "collective-demo",
            },
        }
    ]
