"""The DM-only blocked-overflow warning (#101003) reaches a one-to-one chat only: a Slack group DM (keyed
``dm`` for session scoping) and a handoff into a group home whose thread could not be created are shared
chats, so the warning stays out of them, while a real DM still gets it."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.conversation_compression import CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE
from hermes_agent.gateway.config import Platform
from hermes_agent.gateway.session import build_session_key
from tests.gateway.test_handoff_thread_session_key import _handoff_destination
from tests.gateway.test_slack import SlackAdapter  # installs the slack SDK stand-in

WARNING = CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(tokens=85_000, threshold=72_000, reason="cooldown:30")


def _delivered(source):
    """Run the production status callback for *source*; return what reached the chat."""
    from hermes_agent.gateway.run_turn_runner import TurnRunner
    from hermes_agent.gateway.turn_context import TurnContext

    sent = []

    class Transport:
        async def send(self, chat_id, text, metadata=None):
            sent.append(text)

    async def exercise():
        loop = asyncio.get_running_loop()
        ctx = TurnContext(source=source, user_config={}, _run_still_current=lambda: True, _loop_for_step=loop)
        ctx._status_adapter, ctx._status_chat_id, ctx._status_thread_metadata = Transport(), source.chat_id, None
        runner = TurnRunner(None, ctx)

        def emit():
            runner._status_callback_sync("warn", WARNING)
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=5)

        await asyncio.to_thread(emit)

    asyncio.run(exercise())
    return sent


@pytest.mark.parametrize(("channel_type", "shown"), [("mpim", False), ("im", True)])
def test_slack_group_dm_is_not_a_dm_for_the_warning(channel_type, shown):
    from hermes_agent.gateway.config import PlatformConfig

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id, adapter._running, adapter.handle_message = "U_BOT", True, AsyncMock()
    channel = "G123" if channel_type == "mpim" else "D123"
    asyncio.run(adapter._handle_slack_message({
        "text": "<@U_BOT> hi", "user": "U_USER", "channel": channel, "channel_type": channel_type,
        "team": "T1", "ts": "1.000001"}))
    source = adapter.handle_message.call_args[0][0].source
    assert source.chat_type == "dm"  # Slack's session scoping still treats both as DM-style
    assert build_session_key(source).split(":")[3] == "dm"
    assert _delivered(source) == ([WARNING] if shown else [])


@pytest.mark.parametrize(("platform", "home_type", "shown"), [
    (Platform.DISCORD, "channel", False), (Platform.DISCORD, "dm", True),
    (Platform.MATTERMOST, "group", False), (Platform.SIGNAL, "dm", True)])
def test_a_handoff_without_a_thread_keeps_the_home_chats_type(platform, home_type, shown):
    adapter = SimpleNamespace(get_chat_info=AsyncMock(return_value={"type": home_type}))
    dest, _ = _handoff_destination(platform, "home-1", None, None, adapter)
    assert _delivered(dest.source) == ([WARNING] if shown else [])
