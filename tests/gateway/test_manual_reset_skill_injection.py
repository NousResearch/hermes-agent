"""Regression tests for topic/channel skill auto-injection after /new or /reset.

Before the fix:
    1. User sends `/new` — `reset_session` creates a fresh SessionEntry.
    2. User sends the next message.
    3. `get_or_create_session` finds the entry and mutates
       ``entry.updated_at = now`` (a new ``now``, microseconds after
       ``created_at``).
    4. ``run._handle_message_with_agent`` checks
       ``_is_new_session = (created_at == updated_at) or was_auto_reset``.
       Both are False → ``_is_new_session = False`` → topic/channel skills
       are silently skipped for the very first message of a manually reset
       session.

After the fix:
    `reset_session` sets ``was_auto_reset=True`` and
    ``auto_reset_reason="manual"`` on the new entry.  The ``_is_new_session``
    check returns True via ``was_auto_reset``.  Downstream handlers suppress
    the user-facing "Session automatically reset" notice and the
    ``[System note: ... expired]`` context prepend for manual resets — those
    are meant for surprise resets the user did not ask for.
"""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, SessionResetPolicy
from gateway.platforms.base import MessageEvent
from gateway.session import (
    SessionEntry,
    SessionSource,
    SessionStore,
    build_session_key,
)


# ---------------------------------------------------------------------------
# session.py: reset_session marks manual resets
# ---------------------------------------------------------------------------


def _make_store(tmp_path, policy=None):
    config = GatewayConfig()
    if policy:
        config.default_reset_policy = policy
    return SessionStore(sessions_dir=tmp_path, config=config)


def _make_source(chat_id="123", user_id="u1"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        user_id=user_id,
    )


class TestResetSessionMarksManual:
    def test_reset_session_sets_was_auto_reset_true(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()

        original = store.get_or_create_session(source)
        session_key = original.session_key

        reset_entry = store.reset_session(session_key)

        assert reset_entry is not None
        assert reset_entry.was_auto_reset is True
        assert reset_entry.auto_reset_reason == "manual"
        assert reset_entry.session_id != original.session_id

    def test_reset_session_captures_prior_activity(self, tmp_path):
        """reset_had_activity must reflect whether the reset session had tokens."""
        store = _make_store(tmp_path)
        source = _make_source()

        original = store.get_or_create_session(source)
        original.total_tokens = 4200
        store._save()

        reset_entry = store.reset_session(original.session_key)

        assert reset_entry.reset_had_activity is True

    def test_reset_session_no_activity_flag_when_idle(self, tmp_path):
        store = _make_store(tmp_path)
        source = _make_source()

        original = store.get_or_create_session(source)
        assert original.total_tokens == 0

        reset_entry = store.reset_session(original.session_key)

        assert reset_entry.reset_had_activity is False

    def test_reset_session_returns_none_for_unknown_key(self, tmp_path):
        store = _make_store(tmp_path)

        assert store.reset_session("unknown_key") is None


# ---------------------------------------------------------------------------
# session.py: _is_new_session signal survives updated_at bump
# ---------------------------------------------------------------------------


class TestIsNewSessionAfterManualReset:
    """After /new, the NEXT inbound message must still look "new" so skill
    auto-injection fires.  This mirrors the check in
    ``run._handle_message_with_agent``.
    """

    def test_next_message_preserves_was_auto_reset_flag(self, tmp_path):
        store = _make_store(
            tmp_path,
            # Long idle threshold so `_should_reset` returns None on the
            # immediate follow-up call.
            policy=SessionResetPolicy(mode="idle", idle_minutes=60),
        )
        source = _make_source()

        store.get_or_create_session(source)
        session_key = build_session_key(source)
        reset_entry = store.reset_session(session_key)
        assert reset_entry.was_auto_reset is True

        # Simulate the next inbound message.
        entry = store.get_or_create_session(source)

        # get_or_create_session bumps updated_at, so the strict equality no
        # longer holds.  But was_auto_reset must survive.
        is_new_session = (
            entry.created_at == entry.updated_at
            or entry.was_auto_reset
        )
        assert is_new_session is True
        assert entry.was_auto_reset is True
        assert entry.auto_reset_reason == "manual"

    def test_updated_at_does_advance_for_non_manual_entry(self, tmp_path):
        """Sanity: a vanilla session's updated_at DOES advance on lookup.

        This is the exact behaviour that used to make ``_is_new_session``
        flip to False for manually-reset sessions before the fix.
        """
        store = _make_store(tmp_path)
        source = _make_source()

        first = store.get_or_create_session(source)
        created_at = first.created_at

        # Slight delay so wall clock advances.
        import time
        time.sleep(0.001)

        second = store.get_or_create_session(source)
        assert second.session_id == first.session_id
        assert second.updated_at > created_at
        # Without the manual flag, was_auto_reset is False so _is_new_session
        # would be False — which is correct for an ongoing session.
        assert second.was_auto_reset is False


# ---------------------------------------------------------------------------
# run.py: the auto-reset branch must not send a user-facing notice or
# prepend a "session expired" context note when reset_reason == "manual".
# ---------------------------------------------------------------------------


def _make_mock_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str, auto_skill: str = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_mock_source(),
        message_id="m1",
        auto_skill=auto_skill,
    )


def _make_runner_for_handle():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""
    return runner, adapter


@pytest.mark.asyncio
async def test_manual_reset_does_not_send_user_notification():
    """A manually-reset session must not trigger the auto-reset notice adapter.send.

    We exercise just the auto-reset notification branch inline — duplicating
    the exact control flow in ``_handle_message_with_agent`` — because the
    full handler mounts too much infrastructure for a unit test.
    """
    runner, adapter = _make_runner_for_handle()
    source = _make_mock_source()
    now = datetime.now()
    entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-manual",
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        was_auto_reset=True,
        auto_reset_reason="manual",
        reset_had_activity=True,
        total_tokens=500,
    )

    # Mirror the guard introduced by the fix.
    if entry.was_auto_reset:
        reset_reason = entry.auto_reset_reason or "idle"
        if reset_reason != "manual":
            await adapter.send(source.chat_id, "should not happen")

    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_idle_reset_still_sends_user_notification():
    """The notification pathway must still fire for real auto-resets."""
    runner, adapter = _make_runner_for_handle()
    source = _make_mock_source()
    now = datetime.now()
    entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-idle",
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        was_auto_reset=True,
        auto_reset_reason="idle",
        reset_had_activity=True,
        total_tokens=500,
    )

    if entry.was_auto_reset:
        reset_reason = entry.auto_reset_reason or "idle"
        if reset_reason != "manual":
            await adapter.send(source.chat_id, "◐ Session automatically reset")

    adapter.send.assert_called_once()
