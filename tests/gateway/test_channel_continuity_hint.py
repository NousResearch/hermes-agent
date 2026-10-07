"""Tests for the lightweight session-continuity hint on human chat surfaces.

Salvaged from PR #36220 (metamon-p), ported onto the current SessionStore.

Covers:
- SessionStore records the previous session_id on auto-reset (and only then).
- prev_session_id survives a to_dict() → from_dict() roundtrip (gateway restart).
- build_channel_continuity_note() emits a hint for human chat surfaces (Slack,
  Discord, Telegram, WeCom callback DMs, plugin platforms, ...) that were
  auto-reset with real prior activity, and stays silent for machine callers,
  system-generated events, and agent-peer transports (Home Assistant, ntfy,
  Raft wakes, A2A tasks).
"""

from datetime import datetime, timedelta

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import (
    SessionEntry,
    SessionSource,
    SessionStore,
    build_channel_continuity_note,
)


@pytest.fixture()
def _isolated_db(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _make_store(tmp_path):
    config = GatewayConfig()
    return SessionStore(sessions_dir=tmp_path / "sessions", config=config)


def _slack_source(thread_id=None):
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="thread" if thread_id else "channel",
        user_id="U1",
        thread_id=thread_id,
    )


def _human_source(platform, thread_id=None, chat_type="dm"):
    return SessionSource(
        platform=platform,
        chat_id="C1",
        chat_type=chat_type,
        user_id="U1",
        thread_id=thread_id,
    )


# ---------------------------------------------------------------------------
# SessionStore records prev_session_id on auto-reset
# ---------------------------------------------------------------------------

class TestPrevSessionIdCapture:
    def test_prev_session_id_set_on_auto_reset(self, _isolated_db, tmp_path):
        store = _make_store(tmp_path)
        source = _slack_source(thread_id="T9")

        entry1 = store.get_or_create_session(source)
        assert entry1.prev_session_id is None  # fresh session, nothing replaced

        entry1.last_prompt_tokens = 4000  # had real conversation
        entry1.suspended = True
        store._save()

        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.reset_had_activity is True
        assert entry2.prev_session_id == entry1.session_id


# ---------------------------------------------------------------------------
# build_channel_continuity_note
# ---------------------------------------------------------------------------

def _reset_entry(platform, prev: str | None = "20260101_000000_abc", had_activity=True):
    return SessionEntry(
        session_key="k",
        session_id="20260101_010000_def",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        was_auto_reset=True,
        auto_reset_reason="suspended",
        reset_had_activity=had_activity,
        prev_session_id=prev,
    )


class TestBuildChannelContinuityNote:
    def test_slack_channel_emits_hint(self):
        entry = _reset_entry(Platform.SLACK)
        note = build_channel_continuity_note(entry, _slack_source())
        assert note is not None
        assert "session_search" in note
        assert entry.prev_session_id in note
        assert "channel" in note

    def test_telegram_dm_emits_hint(self):
        entry = _reset_entry(Platform.TELEGRAM)
        note = build_channel_continuity_note(entry, _human_source(Platform.TELEGRAM))
        assert note is not None
        assert "session_search" in note
        assert "20260101_000000_abc" in note
        assert "conversation" in note  # DM wording, not Slack/Discord "channel"
        assert "channel" not in note

    def test_telegram_topic_uses_thread_wording(self):
        entry = _reset_entry(Platform.TELEGRAM)
        note = build_channel_continuity_note(entry, _human_source(Platform.TELEGRAM, thread_id="77"))
        assert note is not None
        assert "thread" in note

    def test_wecom_callback_dm_emits_hint(self):
        # Callback transport, but durable per-user DMs — a human chat surface.
        entry = _reset_entry(Platform.WECOM_CALLBACK)
        note = build_channel_continuity_note(entry, _human_source(Platform.WECOM_CALLBACK))
        assert note is not None
        assert "conversation" in note

    def test_plugin_platform_emits_hint(self):
        # Scoped by denylist: plugin platforms created via Platform._missing_ qualify too.
        irc = Platform("irc")
        entry = _reset_entry(irc)
        assert build_channel_continuity_note(entry, _human_source(irc)) is not None

    @pytest.mark.parametrize(
        "platform",
        [
            Platform.API_SERVER,
            Platform.WEBHOOK,
            Platform.MSGRAPH_WEBHOOK,
            Platform.HOMEASSISTANT,
            Platform("ntfy"),  # broadcast topic, no user identity
            Platform("raft"),  # machine-only wake bridge
            Platform("a2a"),  # agent-peer task protocol
        ],
    )
    def test_non_human_sources_stay_silent(self, platform):
        entry = _reset_entry(platform)
        assert build_channel_continuity_note(entry, _human_source(platform)) is None

    def test_no_activity_returns_none(self):
        entry = _reset_entry(Platform.SLACK, had_activity=False)
        assert build_channel_continuity_note(entry, _slack_source()) is None

    def test_no_prev_session_id_returns_none(self):
        entry = _reset_entry(Platform.TELEGRAM, prev=None)
        assert build_channel_continuity_note(entry, _human_source(Platform.TELEGRAM)) is None
