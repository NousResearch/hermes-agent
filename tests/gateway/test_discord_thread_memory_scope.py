"""Pure session-key guards for Discord thread memory scope.

These tests intentionally avoid SessionStore and adapter setup so they do not
touch gateway state, transcripts, Discord APIs, or production memory files.
"""

from gateway.config import Platform
from gateway.session import (
    SessionSource,
    build_session_key,
    is_shared_multi_user_session,
)


def _discord_thread_source(
    *,
    thread_id: str,
    parent_chat_id: str = "111111111111111111",
    user_id: str = "333333333333333333",
) -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=thread_id,
        chat_type="thread",
        user_id=user_id,
        thread_id=thread_id,
        guild_id="999999999999999999",
        parent_chat_id=parent_chat_id,
    )


def test_same_discord_thread_returns_stable_shared_scope_key():
    first = _discord_thread_source(thread_id="222222222222222222")
    second = _discord_thread_source(thread_id="222222222222222222")

    assert build_session_key(first) == build_session_key(second)
    assert build_session_key(first) == (
        "agent:main:discord:thread:222222222222222222:222222222222222222"
    )
    assert is_shared_multi_user_session(first) is True


def test_different_discord_threads_in_same_parent_channel_do_not_collide():
    thread_a = _discord_thread_source(thread_id="222222222222222222")
    thread_b = _discord_thread_source(thread_id="444444444444444444")

    assert thread_a.parent_chat_id == thread_b.parent_chat_id
    assert build_session_key(thread_a) != build_session_key(thread_b)


def test_discord_thread_scope_is_shared_across_users_by_default():
    alice = _discord_thread_source(
        thread_id="222222222222222222",
        user_id="333333333333333333",
    )
    bob = _discord_thread_source(
        thread_id="222222222222222222",
        user_id="555555555555555555",
    )

    assert build_session_key(alice) == build_session_key(bob)
    assert is_shared_multi_user_session(alice) is True


def test_discord_thread_can_be_isolated_per_user_when_configured():
    alice = _discord_thread_source(
        thread_id="222222222222222222",
        user_id="333333333333333333",
    )
    bob = _discord_thread_source(
        thread_id="222222222222222222",
        user_id="555555555555555555",
    )

    assert build_session_key(
        alice,
        thread_sessions_per_user=True,
    ) != build_session_key(
        bob,
        thread_sessions_per_user=True,
    )
    assert is_shared_multi_user_session(alice, thread_sessions_per_user=True) is False


def test_missing_discord_thread_id_splits_scope_from_normal_thread_source():
    normal = _discord_thread_source(thread_id="222222222222222222")
    missing_thread_id = SessionSource(
        platform=Platform.DISCORD,
        chat_id="222222222222222222",
        chat_type="thread",
        user_id="333333333333333333",
        thread_id=None,
        guild_id="999999999999999999",
        parent_chat_id="111111111111111111",
    )

    assert build_session_key(missing_thread_id) == (
        "agent:main:discord:thread:222222222222222222:333333333333333333"
    )
    assert build_session_key(missing_thread_id) != build_session_key(normal)
