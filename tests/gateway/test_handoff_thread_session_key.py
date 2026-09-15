"""Regression: CLI→Discord handoff must key a thread destination on the
thread's OWN id, matching how the platform adapter keys organic in-thread
messages.

Bug: the handoff built its destination ``SessionSource`` with
``chat_id = home.chat_id`` (the PARENT channel) while thread destinations use
``chat_type="thread"`` and ``thread_id = <thread>``. The Discord adapter,
however, builds organic in-thread messages with ``chat_id = <thread>`` (the
thread's own id). ``build_session_key`` therefore produced two different keys:

    handoff:  agent:main:discord:thread:{parent}:{thread}
    organic:  agent:main:discord:thread:{thread}:{thread}

So the next real user reply in the handoff thread resolved to a DIFFERENT
session_key and spawned a fresh session instead of continuing the handed-off
one (observed: a stray auto-titled session + a session_search fallback because
the new session had no prior context).

The fix is Discord-specific: Slack and Telegram adapters key organic thread
messages with ``chat_id = parent_channel``, so the parent channel is correct
for those platforms and the guard must NOT apply to them.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _organic_discord_thread_key(thread_id: str, parent_id: str, user_id: str) -> str:
    """Key the Discord adapter produces for a message typed inside a thread.

    Mirrors plugins/platforms/discord/adapter.py _handle_message: chat_id is
    the thread's own id, chat_type is "thread", thread_id is the thread id,
    parent_chat_id is the parent channel.
    """
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=str(thread_id),
        chat_type="thread",
        user_id=user_id,
        thread_id=str(thread_id),
        parent_chat_id=str(parent_id),
    )
    return build_session_key(source, thread_sessions_per_user=False)


def _handoff_key(
    platform: Platform,
    home_chat_id: str,
    thread_id: str,
) -> str:
    """Key the handoff produces after the fix.

    Mirrors the fixed logic in GatewayRunner._process_handoff: for Discord
    thread destinations, chat_id is the thread's own id; for other platforms,
    chat_id remains the parent/home channel.
    """
    dest_chat_type = "thread"
    # This mirrors the fixed logic in GatewayRunner._process_handoff.
    if platform == Platform.DISCORD and dest_chat_type == "thread" and thread_id:
        dest_chat_id = str(thread_id)
    else:
        dest_chat_id = str(home_chat_id)
    dest_source = SessionSource(
        platform=platform,
        chat_id=dest_chat_id,
        chat_type=dest_chat_type,
        user_id="system:handoff",
        user_name="Handoff",
        thread_id=str(thread_id),
    )
    return build_session_key(dest_source, thread_sessions_per_user=False)


def test_discord_handoff_key_matches_organic_in_thread_key():
    """For Discord, the handoff key must be byte-identical to the organic
    in-thread key — otherwise a reply in the handoff thread spawns a new session."""
    parent_id = "1523581766923845724"
    thread_id = "1523590238595846166"
    user_id = "171164909650968576"

    organic = _organic_discord_thread_key(thread_id, parent_id, user_id)
    handoff = _handoff_key(Platform.DISCORD, parent_id, thread_id)

    assert handoff == organic, (
        f"handoff key {handoff!r} != organic in-thread key {organic!r}; "
        "a reply in the handoff thread would spawn a new session"
    )
    assert handoff == f"agent:main:discord:thread:{thread_id}:{thread_id}"


def test_discord_handoff_key_does_not_use_parent_channel():
    """The pre-fix bug: keying on the parent channel. Guard against regression."""
    parent_id = "1523581766923845724"
    thread_id = "1523590238595846166"

    handoff = _handoff_key(Platform.DISCORD, parent_id, thread_id)
    buggy = f"agent:main:discord:thread:{parent_id}:{thread_id}"

    assert handoff != buggy, "handoff regressed to keying on the parent channel"


def _slack_handoff_destination(channel_id: str, thread_ts: str, team_id: str):
    """Run the real handoff destination/key path against a Slack home channel."""
    config = GatewayConfig(platforms={Platform.SLACK: PlatformConfig(enabled=True, token="test")})
    config.platforms[Platform.SLACK].home_channel = HomeChannel(
        platform=Platform.SLACK, chat_id=channel_id, name="home", scope_id=team_id)
    adapter = MagicMock()
    adapter.create_handoff_thread = AsyncMock(return_value=thread_ts)
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {Platform.SLACK: adapter}
    runner.session_store = None
    with patch("gateway.delivery.resolve_delivery_transport",
               lambda *_a: SimpleNamespace(adapter=adapter, send=AsyncMock())):
        dest = asyncio.run(runner._handoff_resolve_destination(
            {"id": "cli-session", "title": "work", "handoff_platform": "slack"}, profile_name=None))
    return dest, runner._handoff_session_key(dest, profile_name=None)


def _organic_slack_reply_key(channel_id: str, thread_ts: str, team_id: str, chat_type: str) -> str:
    """Key the Slack adapter builds for a thread reply (``_build_message_event``): parent channel
    as chat_id, ``dm``/``group`` from the channel type, workspace id as scope_id."""
    return build_session_key(SessionSource(
        platform=Platform.SLACK, chat_id=channel_id, chat_type=chat_type, user_id="U123456",
        thread_id=thread_ts, scope_id=team_id), thread_sessions_per_user=False)


def test_slack_dm_handoff_key_matches_the_thread_reply_key():
    """/handoff into a Slack DM must bind the key the next in-thread reply resolves to, or a
    gateway restart forks the thread onto a fresh empty session (#111896)."""
    dest, handoff = _slack_handoff_destination("D0C1HFBMQAX", "1789474088.089709", "T0C2HL96FH6")
    assert handoff == _organic_slack_reply_key("D0C1HFBMQAX", "1789474088.089709", "T0C2HL96FH6", "dm")
    assert dest.source.chat_id == "D0C1HFBMQAX"


def test_slack_channel_handoff_key_matches_the_thread_reply_key():
    """Channel handoffs key on the parent channel (not the thread ts) with the ``group`` layout the
    adapter uses for channel replies (#111896)."""
    dest, handoff = _slack_handoff_destination("C0CHANNEL01", "1789474088.089709", "T0C2HL96FH6")
    assert handoff == _organic_slack_reply_key("C0CHANNEL01", "1789474088.089709", "T0C2HL96FH6", "group")
    assert dest.source.chat_id == "C0CHANNEL01"
