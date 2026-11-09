"""Upstream metadata/registry semantics remain behind cross-store ownership fencing."""
import json

import pytest

from hermes_state import SessionDB
from hermes_state_lineage import SessionLineageAmbiguousError, SessionIdOwnedElsewhereError


def test_user_rewrite_merges_metadata_and_refuses_ambiguous_owner(tmp_path, monkeypatch):
    root = tmp_path / 'hermes'
    monkeypatch.setenv('HERMES_HOME', str(root))
    sibling = root / 'profiles' / 'other'
    sibling.mkdir(parents=True)
    db, other = SessionDB(root / 'state.db'), SessionDB(sibling / 'state.db')
    try:
        db.create_session('s', 'cli')
        db.append_message('s', 'user', 'before')
        row_id = db._conn.execute('SELECT id FROM messages WHERE session_id=?', ('s',)).fetchone()[0]
        db._conn.execute('UPDATE messages SET api_content=?, display_metadata=? WHERE id=?',
                         ('stale', json.dumps({'reactions': {'heart': 1}, 'remove': True}), row_id))
        assert db.set_user_message_content('s', row_id, 'after', {'remove': None, 'added': True}) == 1
        row = db._conn.execute('SELECT content, api_content, display_metadata FROM messages WHERE id=?', (row_id,)).fetchone()
        assert row['content'] == 'after' and row['api_content'] is None
        assert json.loads(row['display_metadata']) == {'reactions': {'heart': 1}, 'added': True}
        # Simulate an existing cross-store duplicate, rather than bypassing the writer fence.
        other._conn.execute("INSERT INTO sessions(id, source, started_at) VALUES('s', 'cli', 1)")
        with pytest.raises(SessionLineageAmbiguousError):
            db.set_user_message_content('s', row_id, 'forbidden', {'reactions': None})
        unchanged = db._conn.execute('SELECT content, api_content, display_metadata FROM messages WHERE id=?', (row_id,)).fetchone()
        assert tuple(unchanged) == tuple(row)
    finally:
        other.close()
        db.close()


@pytest.mark.parametrize('hidden', [False, True])
def test_direct_archive_retires_bot_chat_only_under_ownership_guard(tmp_path, monkeypatch, hidden):
    root = tmp_path / 'hermes'
    monkeypatch.setenv('HERMES_HOME', str(root))
    sibling = root / 'profiles' / 'other'
    sibling.mkdir(parents=True)
    db, other = SessionDB(root / 'state.db'), SessionDB(sibling / 'state.db')
    try:
        db.create_session('bot', 'cli')
        db._conn.execute(
            "UPDATE sessions SET title='Bot Chat', title_source='user', hidden=?, archived=1, "
            "auto_archived=0, ended_at=123, end_reason='ws_orphan_reap' WHERE id='bot'", (int(hidden),))
        other._conn.execute("INSERT INTO sessions(id, source, started_at) VALUES('bot', 'cli', 1)")
        before = tuple(db._conn.execute('SELECT title, hidden, end_reason FROM sessions WHERE id=?', ('bot',)).fetchone())
        with pytest.raises(SessionIdOwnedElsewhereError):
            db.set_session_archived('bot', True, notify=False)
        assert tuple(db._conn.execute('SELECT title, hidden, end_reason FROM sessions WHERE id=?', ('bot',)).fetchone()) == before
        other._conn.execute("DELETE FROM sessions WHERE id='bot'")
        assert db.set_session_archived('bot', True, notify=False)
        row = db.get_session('bot')
        assert row['archived'] == 1 and row['auto_archived'] == 0
        assert row['title'] is None and row['title_source'] is None
        assert row['hidden'] == 0 and row['end_reason'] is None
        assert row['ended_at'] == 123
        assert db.get_session_by_title('Bot Chat') is None
    finally:
        other.close()
        db.close()
