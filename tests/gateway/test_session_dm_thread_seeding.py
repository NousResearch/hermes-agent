"""Tests for DM thread session seeding.

When a bot reply creates a thread in a DM (e.g. Slack), the user's reply
in that thread gets a new session (keyed by thread_ts). The seeding logic
copies the parent DM session's transcript into the new thread session so
the bot retains context of the original conversation.

Covers:
- Basic seeding: parent transcript copied to new thread session
- No seeding for group/channel chats
- No seeding when parent session doesn't exist
- No seeding on auto-reset sessions
- No seeding on existing (non-new) thread sessions
- Parent transcript is not mutated by seeding
- Multiple threads from same parent each get independent copies
- Cross-platform: works for any platform with DM threads (Slack, Telegram, Discord)
"""

import pytest
from unittest.mock import patch

from gateway.config import Platform, GatewayConfig
from gateway.session import SessionSource, SessionStore, build_session_key


@pytest.fixture()
def store(tmp_path):
    """SessionStore with no SQLite, for fast unit tests."""
    config = GatewayConfig()
    with patch("gateway.session.SessionStore._ensure_loaded"):
        s = SessionStore(sessions_dir=tmp_path, config=config)
    s._db = None
    s._loaded = True
    return s


def _dm_source(platform=Platform.SLACK, chat_id="D123", thread_id=None, user_id="U1"):
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type="dm",
        user_id=user_id,
        thread_id=thread_id,
    )


def _group_source(platform=Platform.SLACK, chat_id="C456", thread_id=None, user_id="U1"):
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type="group",
        user_id=user_id,
        thread_id=thread_id,
    )


PARENT_HISTORY = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "It's sunny and 72°F."},
]


class TestDMThreadSeeding:
    """Core seeding behavior."""

    def test_thread_session_seeded_from_parent(self, store):
        """New DM thread session should contain the parent's transcript."""
        # Create parent DM session with history
        parent_source = _dm_source()
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        # Create thread session (user replied in thread)
        thread_source = _dm_source(thread_id="1234567890.000001")
        thread_entry = store.get_or_create_session(thread_source)

        # Thread should have parent's history
        thread_transcript = store.load_transcript(thread_entry.session_id)
        assert len(thread_transcript) == 2
        assert thread_transcript[0]["content"] == "What's the weather?"
        assert thread_transcript[1]["content"] == "It's sunny and 72°F."

    def test_parent_transcript_not_mutated(self, store):
        """Seeding should not alter the parent session's transcript."""
        parent_source = _dm_source()
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        # Create thread and add a message to it
        thread_source = _dm_source(thread_id="1234567890.000001")
        thread_entry = store.get_or_create_session(thread_source)
        store.append_to_transcript(thread_entry.session_id, {
            "role": "user", "content": "thread-only message"
        })

        # Parent should still have only its original messages
        parent_transcript = store.load_transcript(parent_entry.session_id)
        assert len(parent_transcript) == 2
        assert all(m["content"] != "thread-only message" for m in parent_transcript)

    def test_multiple_threads_get_independent_copies(self, store):
        """Each thread from the same parent gets its own copy."""
        parent_source = _dm_source()
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        # Thread A
        thread_a_source = _dm_source(thread_id="1111.000001")
        thread_a_entry = store.get_or_create_session(thread_a_source)
        store.append_to_transcript(thread_a_entry.session_id, {
            "role": "user", "content": "thread A message"
        })

        # Thread B
        thread_b_source = _dm_source(thread_id="2222.000002")
        thread_b_entry = store.get_or_create_session(thread_b_source)

        # Thread B should have parent history, not thread A's additions
        thread_b_transcript = store.load_transcript(thread_b_entry.session_id)
        assert len(thread_b_transcript) == 2
        assert all(m["content"] != "thread A message" for m in thread_b_transcript)

        # Thread A should have parent history + its own message
        thread_a_transcript = store.load_transcript(thread_a_entry.session_id)
        assert len(thread_a_transcript) == 3

    def test_existing_thread_session_not_reseeded(self, store):
        """Returning to an existing thread session should not re-copy parent history."""
        parent_source = _dm_source()
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        # Create thread session
        thread_source = _dm_source(thread_id="1234567890.000001")
        thread_entry = store.get_or_create_session(thread_source)
        store.append_to_transcript(thread_entry.session_id, {
            "role": "user", "content": "follow-up"
        })

        # Add more to parent after thread was created
        store.append_to_transcript(parent_entry.session_id, {
            "role": "user", "content": "new parent message"
        })

        # Get the same thread session again (not new — created_at != updated_at)
        thread_entry_again = store.get_or_create_session(thread_source)
        assert thread_entry_again.session_id == thread_entry.session_id

        # Should still have 3 messages (2 seeded + 1 follow-up), not re-seeded
        thread_transcript = store.load_transcript(thread_entry_again.session_id)
        assert len(thread_transcript) == 3
        assert thread_transcript[2]["content"] == "follow-up"


class TestDMThreadSeedingEdgeCases:
    """Edge cases and conditions where seeding should NOT happen."""

    def test_no_seeding_for_group_threads(self, store):
        """Group/channel threads should not trigger seeding."""
        parent_source = _group_source()
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        thread_source = _group_source(thread_id="1234567890.000001")
        thread_entry = store.get_or_create_session(thread_source)

        thread_transcript = store.load_transcript(thread_entry.session_id)
        assert len(thread_transcript) == 0

    def test_no_seeding_without_parent_session(self, store):
        """Thread session without a parent DM session should start empty."""
        thread_source = _dm_source(thread_id="1234567890.000001")
        thread_entry = store.get_or_create_session(thread_source)

        thread_transcript = store.load_transcript(thread_entry.session_id)
        assert len(thread_transcript) == 0

    def test_no_seeding_with_empty_parent(self, store):
        """If parent session exists but has no transcript, thread starts empty."""
        parent_source = _dm_source()
        store.get_or_create_session(parent_source)
        # No messages appended to parent

        thread_source = _dm_source(thread_id="1234567890.000001")
        thread_entry = store.get_or_create_session(thread_source)

        thread_transcript = store.load_transcript(thread_entry.session_id)
        assert len(thread_transcript) == 0

    def test_no_seeding_for_dm_without_thread_id(self, store):
        """Top-level DMs (no thread_id) should not trigger seeding."""
        source = _dm_source()
        entry = store.get_or_create_session(source)

        # Should just be a normal empty session
        transcript = store.load_transcript(entry.session_id)
        assert len(transcript) == 0


class TestDMThreadSeedingCrossPlatform:
    """Verify seeding works for platforms beyond Slack."""

    @pytest.mark.parametrize("platform", [Platform.SLACK, Platform.TELEGRAM, Platform.DISCORD])
    def test_seeding_works_across_platforms(self, store, platform):
        """DM thread seeding should work for any platform that uses thread_id."""
        parent_source = _dm_source(platform=platform)
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        thread_source = _dm_source(platform=platform, thread_id="thread_123")
        thread_entry = store.get_or_create_session(thread_source)

        thread_transcript = store.load_transcript(thread_entry.session_id)
        assert len(thread_transcript) == 2
        assert thread_transcript[0]["content"] == "What's the weather?"
