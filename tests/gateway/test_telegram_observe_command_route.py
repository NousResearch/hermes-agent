"""Telegram observe mode: an agent-producing command exports the shared route, not the sender.

``/queue`` in an observed group keeps ``source.user_id`` for slash access checks and keys the
shared group session, but its tools exported ``HERMES_SESSION_USER_ID=<sender>``; a Kanban
subscription saved that id and the completion wake rebuilt the per-user key, waking a different
conversation than the one that created the task.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hermes_agent.gateway.config import GatewayConfig
from hermes_agent.gateway.run import GatewayRunner
from hermes_agent.gateway.session import SessionSource, build_session_context, build_session_key
from hermes_agent.gateway.session_context import _UNSET, _VAR_MAP, get_session_env
from tests.gateway.test_telegram_group_gating import _bot_command_entity, _group_message, _make_adapter


@pytest.fixture(autouse=True)
def _reset_contextvars():
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


def test_observed_group_queue_command_wake_returns_to_the_shared_session():
    from tools.kanban_tools import _resolve_notify_target

    adapter = _make_adapter(require_mention=True, allowed_chats=["-100"], group_allowed_chats=["-100"],
                            observe_unmentioned_group_messages=True)
    seen = []
    adapter.handle_message = AsyncMock(side_effect=seen.append)
    adapter._enqueue_text_event = seen.append
    adapter._ensure_forum_commands = AsyncMock()
    text = "/queue@hermes_bot make a task"
    asyncio.run(adapter._handle_command(SimpleNamespace(
        update_id=1, message=_group_message(text, entities=[_bot_command_entity(text, "/queue@hermes_bot")]),
        effective_message=None), None))
    (event,) = seen
    creator_key = build_session_key(event.source)

    runner = object.__new__(GatewayRunner)
    tokens = runner._set_session_env(build_session_context(event.source, GatewayConfig(), event=event))
    try:
        assert get_session_env("HERMES_SESSION_ACTOR_ID") == "111"
        target = _resolve_notify_target()
    finally:
        runner._clear_session_env(tokens)
    # The notifier rebuilds the wake route from the saved subscription (kanban_watchers_notifier).
    wake = SessionSource(platform=adapter.platform, chat_id=target["chat_id"], chat_type=target["chat_type"],
                         thread_id=target["thread_id"], user_id=target["user_id"], user_id_alt=target["user_id_alt"])
    assert build_session_key(wake) == creator_key
