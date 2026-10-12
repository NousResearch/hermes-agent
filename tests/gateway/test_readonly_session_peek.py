"""Read-only gateway session routing must never initialize or repair storage."""

import hashlib
import json
import threading
from datetime import datetime, timezone, UTC

from gateway.session import SessionEntry, SessionStore
from hermes_state import SessionDB


def store(home):
    value = object.__new__(SessionStore)
    value.sessions_dir = home / 'sessions'
    value._entries = {}
    value._loaded = False
    value._lock = threading.Lock()
    value._routing_home = home
    return value


def entry(sid):
    now = datetime.now(UTC)
    return SessionEntry('route', sid, now, now)


def test_empty_peek_creates_no_home_or_sessions_directory(tmp_path):
    home = tmp_path / 'missing'
    value = store(home)
    assert value.peek_session_id('route', peek_only=True) is None
    assert not home.exists()
    assert value._entries == {}
    assert value._loaded is False


def test_persisted_peek_keeps_database_and_cache_unchanged(tmp_path, monkeypatch):
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    value = store(home)
    with_db = SessionDB(db_path=home / 'state.db')
    with_db.save_gateway_routing_entry('route', json.dumps(entry('canonical').to_dict()), scope=value._routing_scope())
    with_db.close()
    if value.sessions_dir.exists():
        value.sessions_dir.rmdir()
    before = hashlib.sha256((home / 'state.db').read_bytes()).hexdigest()
    assert value.peek_session_id('route', peek_only=True) == 'canonical'
    assert hashlib.sha256((home / 'state.db').read_bytes()).hexdigest() == before
    assert value._entries == {}
    assert value._loaded is False
    assert not value.sessions_dir.exists()


def test_readonly_legacy_lookup_does_not_import_or_rewrite(tmp_path):
    value = store(tmp_path)
    value.sessions_dir.mkdir()
    path = value.sessions_dir / 'sessions.json'
    text = json.dumps({'route': entry('legacy').to_dict()})
    path.write_text(text)
    assert value.peek_session_id('route', peek_only=True) == 'legacy'
    assert path.read_text() == text
    assert not (tmp_path / 'state.db').exists()
    assert value._entries == {}
