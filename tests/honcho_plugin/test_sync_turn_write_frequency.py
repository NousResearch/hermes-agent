"""Regression tests for sync_turn() writeFrequency routing.

sync_turn() must delegate persistence to manager.save() so the configured
writeFrequency mode controls whether messages flush immediately or later.
"""

from unittest.mock import MagicMock, patch

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


def _dummy_session() -> HonchoSession:
    return HonchoSession(
        key="test:1",
        user_peer_id="user-test-1",
        assistant_peer_id="hermes",
        honcho_session_id="test-1",
    )


def _configure_provider(provider: HonchoMemoryProvider, write_frequency):
    cfg = HonchoClientConfig(
        write_frequency=write_frequency,
        api_key="k",
        enabled=True,
    )
    mgr = HonchoSessionManager(config=cfg)
    mgr._honcho = MagicMock()
    session = _dummy_session()
    mgr._cache[session.key] = session
    provider._manager = mgr
    provider._config = cfg
    provider._session_key = session.key


class TestSyncTurnWriteFrequency:
    """sync_turn() delegates to manager.save(), honouring write_frequency."""

    def test_session_mode_sync_turn_does_not_flush(self):
        """write_frequency='session': sync_turn() adds messages; flush deferred."""
        provider = HonchoMemoryProvider()
        _configure_provider(provider, write_frequency="session")

        with patch.object(provider._manager, "_flush_session") as mock_flush:
            provider.sync_turn("user msg", "assistant msg")

            # sync_turn spawns a thread — join it so assertions are reliable
            if provider._sync_thread and provider._sync_thread.is_alive():
                provider._sync_thread.join(timeout=2.0)

            mock_flush.assert_not_called()

        session = provider._manager._cache["test:1"]
        assert len(session.messages) >= 2

    def test_turn_mode_sync_turn_flushes_via_save(self):
        """write_frequency='turn': sync_turn() still flushes immediately via save()."""
        provider = HonchoMemoryProvider()
        _configure_provider(provider, write_frequency="turn")

        with patch.object(provider._manager, "_flush_session") as mock_flush:
            provider.sync_turn("hello", "hi there")

            if provider._sync_thread and provider._sync_thread.is_alive():
                provider._sync_thread.join(timeout=2.0)

            mock_flush.assert_called_once()

    def test_integer_cadence_sync_turn_respects_cadence(self):
        """write_frequency=3: flush on every 3rd turn via save()."""
        provider = HonchoMemoryProvider()
        _configure_provider(provider, write_frequency=3)

        with patch.object(provider._manager, "_flush_session") as mock_flush:
            # Turns 1 and 2 — no flush
            provider.sync_turn("t1", "r1")
            if provider._sync_thread and provider._sync_thread.is_alive():
                provider._sync_thread.join(timeout=2.0)
            provider.sync_turn("t2", "r2")
            if provider._sync_thread and provider._sync_thread.is_alive():
                provider._sync_thread.join(timeout=2.0)

            assert mock_flush.call_count == 0

            # Turn 3 — flush fires
            provider.sync_turn("t3", "r3")
            if provider._sync_thread and provider._sync_thread.is_alive():
                provider._sync_thread.join(timeout=2.0)

            assert mock_flush.call_count == 1

    def test_async_mode_sync_turn_enqueues_without_inline_flush(self):
        """write_frequency='async': sync_turn() enqueues via save()."""
        provider = HonchoMemoryProvider()
        _configure_provider(provider, write_frequency="async")
        manager = provider._manager

        try:
            with patch.object(manager._async_queue, "put") as mock_put, \
                 patch.object(manager, "_flush_session") as mock_flush:
                provider.sync_turn("queued user", "queued assistant")

                if provider._sync_thread and provider._sync_thread.is_alive():
                    provider._sync_thread.join(timeout=2.0)

                mock_put.assert_called_once()
                mock_flush.assert_not_called()
        finally:
            manager.shutdown()

    def test_sync_turn_calls_save_not_flush_session_directly(self):
        """Sanity check: sync_turn() uses save(), not _flush_session directly."""
        provider = HonchoMemoryProvider()
        _configure_provider(provider, write_frequency="turn")

        with patch.object(provider._manager, "save") as mock_save, \
             patch.object(provider._manager, "_flush_session") as mock_flush:
            provider.sync_turn("u", "a")

            if provider._sync_thread and provider._sync_thread.is_alive():
                provider._sync_thread.join(timeout=2.0)

            mock_save.assert_called_once()
            mock_flush.assert_not_called()
