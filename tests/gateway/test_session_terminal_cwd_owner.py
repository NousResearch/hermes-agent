"""The terminal-cwd save checks the conversation in the same step as its write (#41128): a ``/new``
that lands between the turn's ownership check and the write leaves the fresh conversation alone."""
from datetime import datetime
from types import SimpleNamespace

from gateway.session import SessionEntry
from gateway.session_terminal_cwd import persist_turn_cwd, restore_turn_cwd

KEY = 'agent:main:telegram:dm:202'


class _Store:
    """SessionStore's two write paths: ``set_session_metadata`` writes whichever entry is current,
    ``_update_entry`` applies *mutate* to the current entry under the store lock (False: no save)."""

    def __init__(self):
        self._entries = {}

    def set_session_metadata(self, key, field, value):
        self._entries[key].metadata[field] = value

    def _update_entry(self, key, mutate):
        entry = self._entries.get(key)
        return entry is not None and mutate(entry) is not False


def _entry(session_id, created_at):
    return SessionEntry(session_key=KEY, session_id=session_id, created_at=created_at, updated_at=created_at)


def test_a_new_between_the_check_and_the_write_leaves_the_fresh_conversation_alone(tmp_path, monkeypatch):
    from tools import terminal_tool
    monkeypatch.setattr(terminal_tool, '_session_cwd', {})
    monkeypatch.setenv('TERMINAL_ENV', 'local')
    store = _Store()
    runner = SimpleNamespace(session_store=store)
    store._entries[KEY] = _entry('s1', datetime(2026, 11, 1))
    token = restore_turn_cwd(runner, KEY, None)
    fresh = _entry('s2', datetime(2026, 11, 2))

    def new_lands_after_the_check(key):  # /new publishes the fresh entry right after the turn's check
        store._entries[KEY] = fresh
        return str(tmp_path)
    monkeypatch.setattr(terminal_tool, 'get_session_cwd', new_lands_after_the_check)
    persist_turn_cwd(runner, KEY, token)
    assert 'terminal_cwd' not in fresh.metadata
