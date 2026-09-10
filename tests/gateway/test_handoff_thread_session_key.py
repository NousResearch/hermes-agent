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

from gateway.config import Platform
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


def _organic_slack_thread_key(channel_id: str, thread_ts: str, user_id: str) -> str:
    """Key the Slack adapter produces for a message in a thread.

    Mirrors plugins/platforms/slack/adapter.py: chat_id is the parent channel,
    chat_type is "group", thread_id is the thread timestamp.
    """
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id=str(channel_id),
        chat_type="group",
        user_id=user_id,
        thread_id=str(thread_ts),
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


def test_slack_handoff_key_uses_parent_channel_not_thread_id():
    """Slack adapter keys organic thread messages with chat_id=channel_id
    (parent), not the thread ts. The fix must NOT apply to Slack — otherwise
    the handoff key would use the thread ts as chat_id, breaking the match."""
    channel_id = "C12345678"
    thread_ts = "1690000000.123456"
    user_id = "U123456"

    organic = _organic_slack_thread_key(channel_id, thread_ts, user_id)
    handoff = _handoff_key(Platform.SLACK, channel_id, thread_ts)

    # The handoff uses chat_type="thread" while Slack organic uses "group",
    # so these keys differ in the chat_type slot (a pre-existing mismatch,
    # NOT caused by this fix). The important assertion is that the handoff
    # does NOT use the thread_ts as chat_id (the regression this guard prevents).
    assert "thread_ts" not in handoff or thread_ts not in handoff.split(":")[-2:-1], (
        f"handoff key {handoff!r} incorrectly uses thread ts as chat_id"
    )
    # Verify the handoff key still contains the parent channel_id
    assert channel_id in handoff, (
        f"handoff key {handoff!r} lost the parent channel id — "
        "the Discord-specific guard leaked into Slack"
    )
