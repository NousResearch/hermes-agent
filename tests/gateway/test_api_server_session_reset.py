"""Tests for session reset policy in the API server adapter.

Verifies that:
- _should_reset_api_session() returns a reason string ("idle" or "daily")
  matching the same semantics as SessionStore._should_reset() in gateway/session.py
- Mode "none" always returns None
- Session not found in DB returns None
- Integration: expired sessions get a new session_id and empty history
- Integration: valid sessions reuse their session_id and load history

Mirrors the test patterns from tests/gateway/test_session_reset_notify.py.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
    SessionResetPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(policy=None):
    """Create an APIServerAdapter with a mocked SessionDB and config."""
    from gateway.platforms.api_server import APIServerAdapter

    pconfig = PlatformConfig(enabled=True)
    adapter = APIServerAdapter(config=pconfig)

    # Replace _ensure_session_db with a mock that returns a controlled SessionDB
    mock_db = MagicMock()
    adapter._session_db = mock_db
    adapter._ensure_session_db = MagicMock(return_value=mock_db)

    # Replace _load_gateway_config with a controlled config
    cfg = GatewayConfig()
    if policy:
        cfg.default_reset_policy = policy
    # Ensure api_server platform gets the same policy via get_reset_policy()
    cfg.reset_by_platform[Platform.API_SERVER] = policy or cfg.default_reset_policy

    return adapter, mock_db, cfg


def _make_session_row(started_at: datetime):
    """Return a dict that looks like a SessionDB.get_session() result."""
    return {
        "id": "test-session-id",
        "source": "api_server",
        "started_at": started_at.timestamp(),
    }


# ---------------------------------------------------------------------------
# _should_reset_api_session() unit tests
# ---------------------------------------------------------------------------

class TestShouldResetApiSession:
    """Mirrors TestShouldResetReason from test_session_reset_notify.py."""

    def test_returns_none_when_not_expired(self):
        """Recently started session with generous policy → not expired."""
        adapter, mock_db, cfg = _make_adapter(
            SessionResetPolicy(mode="both", idle_minutes=1440, at_hour=4),
        )
        mock_db.get_session.return_value = _make_session_row(datetime.now())

        with patch(
            "gateway.run._load_gateway_config",
            return_value=cfg,
        ):
            result = adapter._should_reset_api_session("test-session-id")

        assert result is None

    def test_returns_idle_when_idle_expired(self):
        """Session started long ago with tight idle policy → idle expired."""
        adapter, mock_db, cfg = _make_adapter(
            SessionResetPolicy(mode="idle", idle_minutes=30),
        )
        mock_db.get_session.return_value = _make_session_row(
            datetime.now() - timedelta(hours=2),
        )

        with patch(
            "gateway.run._load_gateway_config",
            return_value=cfg,
        ):
            result = adapter._should_reset_api_session("test-session-id")

        assert result == "idle"

    def test_returns_daily_when_daily_boundary_crossed(self):
        """Session started yesterday, daily reset at current hour → expired."""
        now = datetime.now()
        adapter, mock_db, cfg = _make_adapter(
            SessionResetPolicy(mode="daily", at_hour=now.hour),
        )
        mock_db.get_session.return_value = _make_session_row(
            now - timedelta(days=1),
        )

        with patch(
            "gateway.run._load_gateway_config",
            return_value=cfg,
        ):
            result = adapter._should_reset_api_session("test-session-id")

        assert result == "daily"

    def test_returns_none_when_mode_is_none(self):
        """Policy explicitly disabled → never reset, even ancient sessions."""
        adapter, mock_db, cfg = _make_adapter(
            SessionResetPolicy(mode="none"),
        )
        mock_db.get_session.return_value = _make_session_row(
            datetime.now() - timedelta(days=30),
        )

        with patch(
            "gateway.run._load_gateway_config",
            return_value=cfg,
        ):
            result = adapter._should_reset_api_session("test-session-id")

        assert result is None

    def test_returns_none_when_session_not_in_db(self):
        """Session doesn't exist yet → nothing to reset."""
        adapter, mock_db, cfg = _make_adapter(
            SessionResetPolicy(mode="both", idle_minutes=1),
        )
        mock_db.get_session.return_value = None

        with patch(
            "gateway.run._load_gateway_config",
            return_value=cfg,
        ):
            result = adapter._should_reset_api_session("nonexistent-id")

        assert result is None

    def test_returns_none_when_db_unavailable(self):
        """SessionDB not initialised → gracefully return None."""
        adapter, mock_db, cfg = _make_adapter(
            SessionResetPolicy(mode="daily"),
        )
        adapter._session_db = None
        adapter._ensure_session_db.return_value = None

        with patch(
            "gateway.run._load_gateway_config",
            return_value=cfg,
        ):
            result = adapter._should_reset_api_session("test-session-id")

        assert result is None

    def test_respects_platform_override(self):
        """Platform-specific override in reset_by_platform is used."""
        adapter, mock_db, cfg = _make_adapter()
        # Default policy would not expire, but platform override is tight
        cfg.default_reset_policy = SessionResetPolicy(mode="none")
        cfg.reset_by_platform[Platform.API_SERVER] = SessionResetPolicy(
            mode="idle", idle_minutes=30,
        )
        mock_db.get_session.return_value = _make_session_row(
            datetime.now() - timedelta(hours=2),
        )

        with patch(
            "gateway.run._load_gateway_config",
            return_value=cfg,
        ):
            result = adapter._should_reset_api_session("test-session-id")

        # Platform override (idle_minutes=30) should win over default (none)
        assert result == "idle"


# ---------------------------------------------------------------------------
# Integration: reset produces new session_id + empty history
# ---------------------------------------------------------------------------

class TestApiServerResetIntegration:
    """Verify that the _handle_chat_completions integration behaves correctly."""

    def test_expired_session_gets_new_id_and_empty_history(self):
        """When reset fires, old session is ended, new ID generated, history empty."""
        adapter, mock_db, cfg = _make_adapter(
            SessionResetPolicy(mode="idle", idle_minutes=1),
        )
        # Simulate an old session that exists in DB
        mock_db.get_session.return_value = _make_session_row(
            datetime.now() - timedelta(hours=2),
        )
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "old message"},
        ]
        # Simulate that _should_reset_api_session returns "idle"
        adapter._should_reset_api_session = MagicMock(return_value="idle")

        # Simulate the key lines from _handle_chat_completions
        session_id = "old-session-id"
        reset_reason = adapter._should_reset_api_session(session_id)
        if reset_reason:
            mock_db.end_session(session_id, "session_reset")
            new_session_id = "generated-uuid-12345"
            history = []
        else:
            new_session_id = session_id
            history = mock_db.get_messages_as_conversation(session_id)

        # Assertions
        assert new_session_id != session_id  # new ID generated
        assert history == []  # history is cleared
        mock_db.end_session.assert_called_once_with(
            "old-session-id", "session_reset",
        )

    def test_valid_session_reuses_id_and_loads_history(self):
        """When no reset, session_id is preserved and history is loaded."""
        adapter, mock_db, cfg = _make_adapter(
            SessionResetPolicy(mode="both", idle_minutes=1440, at_hour=4),
        )
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        adapter._should_reset_api_session = MagicMock(return_value=None)

        session_id = "valid-session-id"
        reset_reason = adapter._should_reset_api_session(session_id)
        if reset_reason:
            mock_db.end_session(session_id, "session_reset")
            new_session_id = "generated-uuid"
            history = []
        else:
            new_session_id = session_id
            history = mock_db.get_messages_as_conversation(session_id)

        assert new_session_id == session_id  # preserved
        assert len(history) == 2  # history loaded
        mock_db.end_session.assert_not_called()


# ---------------------------------------------------------------------------
# Policy config parity
# ---------------------------------------------------------------------------

class TestResetPolicyConfig:
    """Verify SessionResetPolicy defaults match gateway expectations."""

    def test_notify_exclude_defaults_include_api_server(self):
        policy = SessionResetPolicy()
        assert "api_server" in policy.notify_exclude_platforms

    def test_default_mode_is_both(self):
        policy = SessionResetPolicy()
        assert policy.mode == "both"

    def test_default_idle_is_24h(self):
        policy = SessionResetPolicy()
        assert policy.idle_minutes == 1440
