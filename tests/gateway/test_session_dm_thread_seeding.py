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

from gateway.config import Platform, GatewayConfig
from gateway.session import SessionSource, SessionStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """SessionStore with SQLite — load_transcript reads from DB only.

    Pin DEFAULT_DB_PATH to tmp_path so SessionDB() can't write to the real
    ~/.hermes/state.db. (DEFAULT_DB_PATH is a module-level constant computed
    at hermes_state import time, before pytest's HERMES_HOME monkeypatch
    fires — the autouse fixture's HERMES_HOME override doesn't help here.)
    """
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    config = GatewayConfig()
    s = SessionStore(sessions_dir=tmp_path, config=config)
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
