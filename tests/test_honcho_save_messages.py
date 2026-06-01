"""Tests for Honcho saveMessages write behavior."""

from types import SimpleNamespace

from plugins.memory.honcho import HonchoMemoryProvider


class _RecordingSession:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append((role, content))


class _RecordingManager:
    def __init__(self):
        self.session = _RecordingSession()
        self.get_or_create_calls = []
        self.flush_calls = []

    def get_or_create(self, session_key):
        self.get_or_create_calls.append(session_key)
        return self.session

    def _flush_session(self, session):
        self.flush_calls.append(session)


def _provider_with_save_messages(save_messages):
    provider = HonchoMemoryProvider()
    provider._manager = _RecordingManager()
    provider._session_key = "test-session"
    provider._config = SimpleNamespace(
        save_messages=save_messages,
        message_max_chars=25000,
    )
    return provider


def test_sync_turn_skips_honcho_message_persistence_when_save_messages_false():
    provider = _provider_with_save_messages(False)

    provider.sync_turn("user message", "assistant message", session_id="test-session")
    if provider._sync_thread:
        provider._sync_thread.join(timeout=1)

    assert provider._manager.get_or_create_calls == []
    assert provider._manager.session.messages == []
    assert provider._manager.flush_calls == []


def test_sync_turn_persists_honcho_messages_when_save_messages_true():
    provider = _provider_with_save_messages(True)

    provider.sync_turn("user message", "assistant message", session_id="test-session")
    if provider._sync_thread:
        provider._sync_thread.join(timeout=1)

    assert provider._manager.get_or_create_calls == ["test-session"]
    assert provider._manager.session.messages == [
        ("user", "user message"),
        ("assistant", "assistant message"),
    ]
    assert provider._manager.flush_calls == [provider._manager.session]
