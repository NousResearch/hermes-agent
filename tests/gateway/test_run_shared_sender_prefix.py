from gateway.config import Platform
from gateway.run import _prefix_shared_session_sender
from gateway.session import SessionSource, _hash_sender_id


def test_shared_non_thread_group_prefixes_name_and_hashed_id():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="guild:channel",
        chat_type="group",
        user_id="1234567890",
        user_name="Alice",
    )

    result = _prefix_shared_session_sender(
        "hello",
        source,
        group_sessions_per_user=False,
        thread_sessions_per_user=False,
    )

    assert result == f"[Alice | {_hash_sender_id('1234567890')}] hello"


def test_shared_thread_prefixes_name_and_hashed_id():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="guild:channel",
        chat_type="group",
        thread_id="thread-1",
        user_id="thread-user",
        user_name="Bob",
    )

    result = _prefix_shared_session_sender(
        "reply",
        source,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )

    assert result == f"[Bob | {_hash_sender_id('thread-user')}] reply"


def test_non_shared_group_is_unchanged():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="guild:channel",
        chat_type="group",
        user_id="1234567890",
        user_name="Alice",
    )

    result = _prefix_shared_session_sender(
        "hello",
        source,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )

    assert result == "hello"


def test_dm_is_unchanged():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="dm-1",
        chat_type="dm",
        user_id="1234567890",
        user_name="Alice",
    )

    result = _prefix_shared_session_sender(
        "hello",
        source,
        group_sessions_per_user=False,
        thread_sessions_per_user=False,
    )

    assert result == "hello"


def test_name_only_shared_message_skips_missing_id():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="guild:channel",
        chat_type="group",
        user_name="Alice",
    )

    result = _prefix_shared_session_sender(
        "hello",
        source,
        group_sessions_per_user=False,
        thread_sessions_per_user=False,
    )

    assert result == "[Alice] hello"
