"""Tests for DM thread session isolation.

DM thread sessions must start empty — no parent transcript seeding.
Thread context is handled by platform adapters (e.g. Slack's
_fetch_thread_context fetches actual thread replies via the API).
Session-level seeding was removed because it copied the ENTIRE parent
DM transcript, causing unrelated conversations to bleed across threads.

Covers:
- Thread sessions start empty (no parent seeding)
- Group/channel thread sessions also start empty
- Multiple threads from same parent are independent
- Existing thread sessions are not mutated on re-access
- Cross-platform: consistent behavior for Slack, Telegram, Discord
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


class TestDMThreadIsolation:
    """Thread sessions must start empty — no parent transcript seeding."""

    def test_thread_session_starts_empty(self, store):
        """New DM thread session should NOT inherit parent's transcript."""
        parent_source = _dm_source()
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        thread_source = _dm_source(thread_id="1234567890.000001")
        thread_entry = store.get_or_create_session(thread_source)

        thread_transcript = store.load_transcript(thread_entry.session_id)
        assert len(thread_transcript) == 0

    def test_parent_transcript_unaffected_by_thread(self, store):
        """Creating a thread session should not alter parent's transcript."""
        parent_source = _dm_source()
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        thread_source = _dm_source(thread_id="1234567890.000001")
        thread_entry = store.get_or_create_session(thread_source)
        store.append_to_transcript(thread_entry.session_id, {
            "role": "user", "content": "thread-only message"
        })

        parent_transcript = store.load_transcript(parent_entry.session_id)
        assert len(parent_transcript) == 2
        assert all(m["content"] != "thread-only message" for m in parent_transcript)

    def test_multiple_threads_are_independent(self, store):
        """Each thread from the same parent starts empty and stays independent."""
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

        # Thread B starts empty
        thread_b_transcript = store.load_transcript(thread_b_entry.session_id)
        assert len(thread_b_transcript) == 0

        # Thread A has only its own message
        thread_a_transcript = store.load_transcript(thread_a_entry.session_id)
        assert len(thread_a_transcript) == 1
        assert thread_a_transcript[0]["content"] == "thread A message"

    def test_existing_thread_session_preserved(self, store):
        """Returning to an existing thread session should not reset it."""
        parent_source = _dm_source()
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        thread_source = _dm_source(thread_id="1234567890.000001")
        thread_entry = store.get_or_create_session(thread_source)
        store.append_to_transcript(thread_entry.session_id, {
            "role": "user", "content": "follow-up"
        })

        # Get the same thread session again
        thread_entry_again = store.get_or_create_session(thread_source)
        assert thread_entry_again.session_id == thread_entry.session_id

        # Should still have only its own message
        thread_transcript = store.load_transcript(thread_entry_again.session_id)
        assert len(thread_transcript) == 1
        assert thread_transcript[0]["content"] == "follow-up"


class TestDMThreadIsolationEdgeCases:
    """Edge cases — threads always start empty regardless of context."""

    def test_group_thread_starts_empty(self, store):
        """Group/channel threads should also start empty."""
        parent_source = _group_source()
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        thread_source = _group_source(thread_id="1234567890.000001")
        thread_entry = store.get_or_create_session(thread_source)

        thread_transcript = store.load_transcript(thread_entry.session_id)
        assert len(thread_transcript) == 0

    def test_thread_without_parent_session_starts_empty(self, store):
        """Thread session without a parent DM session should start empty."""
        thread_source = _dm_source(thread_id="1234567890.000001")
        thread_entry = store.get_or_create_session(thread_source)

        thread_transcript = store.load_transcript(thread_entry.session_id)
        assert len(thread_transcript) == 0

    def test_dm_without_thread_starts_empty(self, store):
        """Top-level DMs (no thread_id) should start empty as always."""
        source = _dm_source()
        entry = store.get_or_create_session(source)

        transcript = store.load_transcript(entry.session_id)
        assert len(transcript) == 0


class TestDMThreadIsolationCrossPlatform:
    """Verify thread isolation is consistent across all platforms."""

    @pytest.mark.parametrize("platform", [Platform.SLACK, Platform.TELEGRAM, Platform.DISCORD])
    def test_thread_starts_empty_across_platforms(self, store, platform):
        """DM thread sessions start empty regardless of platform."""
        parent_source = _dm_source(platform=platform)
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        thread_source = _dm_source(platform=platform, thread_id="thread_123")
        thread_entry = store.get_or_create_session(thread_source)

        thread_transcript = store.load_transcript(thread_entry.session_id)
        assert len(thread_transcript) == 0
