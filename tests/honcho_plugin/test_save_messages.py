"""Tests for the saveMessages knob: when false, the provider never writes to Honcho.

The knob has always been parsed by HonchoClientConfig but was not consumed by
the write paths (sync_turn / on_memory_write / on_session_end). These tests pin
the contract: saveMessages=false disables all automatic persistence while read
and tools paths remain untouched.
"""

from unittest.mock import MagicMock

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig


def _provider(save_messages: bool) -> HonchoMemoryProvider:
    p = HonchoMemoryProvider()
    p._config = HonchoClientConfig(save_messages=save_messages)
    p._manager = MagicMock()
    p._session_key = 'test-session'
    p._session_initialized = True
    return p


class TestSyncTurn:
    def test_disabled_writes_nothing(self):
        p = _provider(save_messages=False)
        p.sync_turn('user says', 'assistant says')
        p._manager.get_or_create.assert_not_called()
        p._manager._flush_session.assert_not_called()

    def test_enabled_writes(self):
        p = _provider(save_messages=True)
        p.sync_turn('user says', 'assistant says')
        if p._sync_thread is not None:
            p._sync_thread.join(timeout=5)
        p._manager.get_or_create.assert_called_once()


class TestOnMemoryWrite:
    def test_disabled_skips_conclusion_mirror(self):
        p = _provider(save_messages=False)
        p.on_memory_write('add', 'user', 'user likes coffee')
        p._manager.create_conclusion.assert_not_called()

    def test_enabled_mirrors(self):
        import time

        p = _provider(save_messages=True)
        p.on_memory_write('add', 'user', 'user likes coffee')
        deadline = time.time() + 5
        while time.time() < deadline and not p._manager.create_conclusion.called:
            time.sleep(0.05)
        p._manager.create_conclusion.assert_called_once()


class TestOnSessionEnd:
    def test_disabled_skips_flush(self):
        p = _provider(save_messages=False)
        p.on_session_end([])
        p._manager.flush_all.assert_not_called()

    def test_enabled_flushes(self):
        p = _provider(save_messages=True)
        p.on_session_end([])
        p._manager.flush_all.assert_called_once()
