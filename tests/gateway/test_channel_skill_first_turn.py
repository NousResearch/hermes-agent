"""First-turn channel-skill auto-load must survive the admission + turn-prepare lookups.

Regression for #160217: a skill bound to a channel with ``channel_skill_bindings``
never auto-loaded, because the new-session check compared
``created_at == updated_at`` and the turn-prepare lookup bumps ``updated_at``
before the check runs.

Before the fix:
    1. First inbound message — admission creates the session with
       ``created_at == updated_at``.
    2. Turn prepare looks the session up again with ``touch_activity=True``,
       bumping ``updated_at``.
    3. ``_hmwa_open_session`` checks ``created_at == updated_at`` → False,
       so ``session:start`` never fires and the bound skill is never
       auto-loaded for the first message.

After the fix:
    ``_route_create`` stamps the new entry with ``is_new_session=True``.
    ``_hmwa_open_session`` ORs this into ``_is_new_session`` and consumes
    the flag immediately after the check, so subsequent messages are
    treated as continuing the session and the flag does not leak.
    (Same shape as the ``is_fresh_reset`` fix for #6508.)
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run_turn_prepare import GatewayTurnPrepareMixin
from gateway.session import SessionSource, SessionStore


def _make_store(tmp_path):
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _make_source():
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C0123456789",
        chat_type="group",
        user_id="U0123456789",
        thread_id="1795643205.352729",
    )


def _make_preparer():
    preparer = object.__new__(GatewayTurnPrepareMixin)
    preparer.hooks = SimpleNamespace(emit=AsyncMock())
    preparer._clear_conversation_scope = MagicMock()
    preparer._evict_cached_agent = MagicMock()
    return preparer


class TestFirstMessageCountsAsNewSession:
    @pytest.mark.asyncio
    async def test_new_session_survives_admission_then_prepare_lookup(self, tmp_path):
        """One inbound message flows through admission and turn prepare; the
        first-turn gate must still open."""
        store = _make_store(tmp_path)
        try:
            source = _make_source()
            # Admission, then the turn-prepare lookup that bumps updated_at.
            store.get_or_create_session(source)
            entry = store.get_or_create_session(source, touch_activity=True)

            # The prepare lookup touches activity, so the timestamps alone no
            # longer prove newness — the creation flag must carry it.
            assert entry.created_at != entry.updated_at

            preparer = _make_preparer()
            was_auto_reset, is_new_session = await preparer._hmwa_open_session(
                entry, entry.session_key, source
            )
            assert (was_auto_reset, is_new_session) == (False, True)
            preparer.hooks.emit.assert_awaited_once()
            assert preparer.hooks.emit.call_args.args[0] == "session:start"

            # The flag is consumed: the next message is not a new session.
            _, is_new_again = await preparer._hmwa_open_session(
                entry, entry.session_key, source
            )
            assert is_new_again is False
            preparer.hooks.emit.assert_awaited_once()
        finally:
            store.close_all_db_handles()

    def test_is_new_session_survives_to_dict_from_dict(self, tmp_path):
        """Protect against the gateway restarting between creation and the
        first message — the flag must be persisted in sessions.json."""
        from gateway.session import SessionEntry

        store = _make_store(tmp_path)
        try:
            entry = store.get_or_create_session(_make_source())
            assert entry.is_new_session is True
            restored = SessionEntry.from_dict(entry.to_dict())
            assert restored.is_new_session is True
        finally:
            store.close_all_db_handles()
