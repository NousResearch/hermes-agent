"""Tests for DM thread reply routing under the channel claim lock.

A thread reply on a live channel session folds into that session — the
claim lock keeps one session per channel (design/channel-claim-lock.md §6.1):
no second, empty thread-keyed session is minted while the channel's session
is live. Thread context still comes from platform adapters (e.g. Slack's
_fetch_thread_context fetches actual thread replies via the API), never from
transcript seeding, which previously copied the ENTIRE parent DM transcript
and let unrelated conversations bleed across threads.

Covers:
- DM thread replies fold into the live channel session
- Group thread replies fold the same way
- The session key stays channel-scoped after the fold
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


class TestDMThreadFoldingEdgeCases:
    """Edge cases — a live channel session absorbs thread replies."""

    def test_group_thread_reply_folds_into_channel_session(self, store):
        """Group thread replies join the live channel session (no new session)."""
        parent_source = _group_source()
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        thread_source = _group_source(thread_id="1234567890.000001")
        thread_entry = store.get_or_create_session(thread_source)

        assert thread_entry.session_id == parent_entry.session_id
        assert thread_entry.session_key == parent_entry.session_key
        assert len(store.load_transcript(thread_entry.session_id)) == len(PARENT_HISTORY)


class TestDMThreadFoldingCrossPlatform:
    """Verify thread-reply folding is consistent across all platforms."""

    @pytest.mark.parametrize("platform", [Platform.SLACK, Platform.TELEGRAM, Platform.DISCORD])
    def test_thread_reply_folds_into_channel_session_across_platforms(self, store, platform):
        """DM thread replies join the live channel session regardless of platform."""
        parent_source = _dm_source(platform=platform)
        parent_entry = store.get_or_create_session(parent_source)
        for msg in PARENT_HISTORY:
            store.append_to_transcript(parent_entry.session_id, msg)

        thread_source = _dm_source(platform=platform, thread_id="thread_123")
        thread_entry = store.get_or_create_session(thread_source)

        assert thread_entry.session_id == parent_entry.session_id
        assert thread_entry.session_key == parent_entry.session_key
        assert len(store.load_transcript(thread_entry.session_id)) == len(PARENT_HISTORY)
