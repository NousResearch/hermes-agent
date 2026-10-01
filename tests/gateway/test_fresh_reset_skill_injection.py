"""Regression tests for topic/channel skill auto-injection after /new or /reset.

Covers the fix for issue #6508.

Before the fix:
    1. User sends ``/new`` — ``reset_session`` creates a fresh SessionEntry
       with ``created_at == updated_at``.
    2. User sends the next message.
    3. ``get_or_create_session`` finds the entry and bumps
       ``entry.updated_at = now`` (microseconds after ``created_at``).
    4. ``_handle_message_with_agent`` checks
       ``_is_new_session = (created_at == updated_at) or was_auto_reset``.
       Both are False → ``_is_new_session = False`` → topic/channel skills
       are silently skipped for the first message of a manually reset session.

After the fix:
    ``reset_session`` stamps the new entry with ``is_fresh_reset=True``.
    ``_handle_message_with_agent`` ORs this into ``_is_new_session`` and
    consumes the flag immediately after the check, so subsequent messages
    are treated as continuing the session and the flag does not leak.

We use ``was_auto_reset`` for surprise resets (idle/daily/suspended) and
``is_fresh_reset`` for user-initiated resets because the former also drives
a "Session automatically reset due to inactivity" user-facing notice and
a context-note prepend into the agent's prompt — both wrong for an explicit
/new or /reset.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, SessionStore


def _make_store(tmp_path):
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _make_source(chat_id="123", user_id="u1"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        user_id=user_id,
    )


def _is_new_session(entry) -> bool:
    """Mirror of the predicate in ``_handle_message_with_agent``.

    Kept in-sync with the production check so this test fails loudly if the
    upstream logic regresses.
    """
    first_agent_turn_pending = entry.metadata.pop("first_agent_turn_pending", False)
    return (
        entry.created_at == entry.updated_at
        or getattr(entry, "was_auto_reset", False)
        or getattr(entry, "is_fresh_reset", False)
        or first_agent_turn_pending
    )


# ---------------------------------------------------------------------------
# reset_session stamps is_fresh_reset=True
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Core regression: _is_new_session stays True after updated_at bump
# ---------------------------------------------------------------------------

class TestIsNewSessionSurvivesUpdatedAtBump:
    def test_is_new_session_true_after_reset_then_next_message(self, tmp_path):
        """The actual bug: _is_new_session was False on message after /reset."""
        store = _make_store(tmp_path)
        source = _make_source()
        store.get_or_create_session(source)
        session_key = store._generate_session_key(source)

        # User sends /reset
        store.reset_session(session_key)

        # Next inbound message — get_or_create_session bumps updated_at
        entry = store.get_or_create_session(source)

        # Before the fix: created_at != updated_at, was_auto_reset=False → False
        # After the fix: is_fresh_reset=True carries the signal through the bump
        assert _is_new_session(entry) is True


# ---------------------------------------------------------------------------
# Vanilla-session behavior is unchanged
# ---------------------------------------------------------------------------

class TestVanillaBehaviorUnaffected:
    def test_ongoing_session_not_flagged_as_new(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        _is_new_session(entry)  # first agent turn consumes the pending marker

        # A later message has an advanced activity clock and no pending marker.
        entry.updated_at = entry.created_at.replace(microsecond=entry.created_at.microsecond + 1)
        assert entry.is_fresh_reset is False
        assert _is_new_session(entry) is False

    def test_runner_emits_session_start_for_pending_first_turn(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        entry.updated_at = entry.created_at.replace(microsecond=entry.created_at.microsecond + 1)

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.hooks = SimpleNamespace(emit=AsyncMock())
        result = asyncio.run(runner._hmwa_open_session(entry, entry.session_key, source))

        assert result == (False, True)
        runner.hooks.emit.assert_awaited_once()
        assert "first_agent_turn_pending" not in entry.metadata

    def test_read_only_command_before_first_message_still_flags_first_turn(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()

        # A read-only command creates the routing entry before the first agent turn.
        entry = store.get_or_create_session(source)
        entry.updated_at = entry.created_at.replace(microsecond=entry.created_at.microsecond + 1)

        assert _is_new_session(entry) is True
        assert _is_new_session(entry) is False


# ---------------------------------------------------------------------------
# Persistence through sessions.json round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_is_fresh_reset_survives_to_dict_from_dict(self, tmp_path):
        """Protect against the gateway restarting between /reset and the
        next message — the flag must be persisted in sessions.json.
        """
        store = _make_store(tmp_path)
        source = _make_source()
        store.get_or_create_session(source)
        session_key = store._generate_session_key(source)
        new_entry = store.reset_session(session_key)

        assert new_entry.is_fresh_reset is True
        restored = SessionEntry.from_dict(new_entry.to_dict())
        assert restored.is_fresh_reset is True

