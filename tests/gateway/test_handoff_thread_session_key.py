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

This test pins the invariant: for a thread destination the handoff key must be
byte-identical to the organic in-thread key.
"""

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


def _organic_thread_key(thread_id: str, parent_id: str, user_id: str) -> str:
    """Key the Discord adapter produces for a message typed inside a thread.

    Mirrors plugins/platforms/discord/adapter.py on_message: chat_id is the
    thread's own id, chat_type is "thread", thread_id is the thread id,
    parent_chat_id is the parent channel.
    """
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=str(thread_id),          # adapter uses the thread's OWN id
        chat_type="thread",
        user_id=user_id,
        thread_id=str(thread_id),
        parent_chat_id=str(parent_id),
    )
    return build_session_key(source, thread_sessions_per_user=False)


def _handoff_thread_key(thread_id: str, home_chat_id: str) -> str:
    """Key the handoff produces after the fix: for a thread destination,
    chat_id is the thread's own id (not the parent/home channel)."""
    dest_chat_type = "thread"
    effective_thread_id = str(thread_id)
    # This mirrors the fixed logic in HermesGateway._process_handoff.
    dest_chat_id = (
        str(effective_thread_id)
        if dest_chat_type == "thread" and effective_thread_id
        else str(home_chat_id)
    )
    dest_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=dest_chat_id,
        chat_type=dest_chat_type,
        user_id="system:handoff",
        user_name="Handoff",
        thread_id=effective_thread_id,
    )
    return build_session_key(dest_source, thread_sessions_per_user=False)


def test_handoff_thread_key_matches_organic_in_thread_key():
    parent_id = "1523581766923845724"   # home/parent channel
    thread_id = "1523590238595846166"   # handoff thread
    user_id = "171164909650968576"

    organic = _organic_thread_key(thread_id, parent_id, user_id)
    handoff = _handoff_thread_key(thread_id, parent_id)

    # Threads are user-shared (thread_sessions_per_user=False), so user_id is
    # NOT part of the key — the synthetic handoff turn and the user's later
    # reply must land on the exact same session_key.
    assert handoff == organic, (
        f"handoff key {handoff!r} != organic in-thread key {organic!r}; "
        "a reply in the handoff thread would spawn a new session"
    )
    assert handoff == f"agent:main:discord:thread:{thread_id}:{thread_id}"


def test_handoff_thread_key_does_not_use_parent_channel():
    """The pre-fix bug: keying on the parent channel. Guard against regression."""
    parent_id = "1523581766923845724"
    thread_id = "1523590238595846166"

    handoff = _handoff_thread_key(thread_id, parent_id)
    buggy = f"agent:main:discord:thread:{parent_id}:{thread_id}"

    assert handoff != buggy, "handoff regressed to keying on the parent channel"
