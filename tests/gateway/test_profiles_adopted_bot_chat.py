"""Regression for #139227: adopted local chats stay visible in the bot roster."""
from types import SimpleNamespace

from gateway.session_contract import Principal
from gateway.session_group_controls import _profiles
from hermes_state import SessionDB
from hermes_state_local_migration import bind_legacy_target


def test_adopted_bot_chat_keeps_canonical_and_recent_discovery(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    sid = '20261006_105832_fixture'
    actor = Principal('fixture-owner', str(tmp_path), frozenset({'session:read'}), 'fixture')
    with SessionDB(tmp_path / 'state.db') as db:
        db.create_session(sid, source='cli')
        db.set_session_title(sid, 'Bot Chat')
        db.set_session_hidden(sid, True)
        db.append_message(sid, 'assistant', 'The retained bot reply')
        receipt = {'session_id': sid, 'legacy_session_id': sid, 'profile_id': str(tmp_path),
                   'principal_id': actor.subject, 'route': 'agent:main:local:dm:' + sid,
                   'entry': {'origin': {'platform': 'local', 'chat_id': sid, 'user_id': actor.subject}}}
        with db._lock:
            bind_legacy_target(db, db._conn, receipt)
            db._conn.commit()
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path))
        before = db._conn.total_changes
        row = _profiles(authority, actor, tmp_path, {})['profiles'][0]
        assert row['canonical_session'] is not None, 'adopted Bot Chat disappeared from roster'
        assert row['canonical_session']['resolved_id'] == sid
        assert row['canonical_session']['preview'] == 'The retained bot reply'
        assert row['last_session']['resolved_id'] == sid
        assert db._conn.total_changes == before, 'discovery must remain read-only'
        assert db.get_session(sid)['chat_id'] == sid

        foreign = Principal('other-owner', str(tmp_path), actor.capabilities, 'other')
        denied = _profiles(authority, foreign, tmp_path, {})['profiles'][0]
        assert denied['canonical_session'] is None
        assert denied['last_session'] is None
        operator = Principal('operator', str(tmp_path), actor.capabilities | {'session:operator'}, 'op')
        assert _profiles(authority, operator, tmp_path, {})['profiles'][0]['canonical_session']['id'] == sid
        light = _profiles(authority, actor, tmp_path, {'include_sessions': False})['profiles'][0]
        assert 'canonical_session' not in light

        db.set_session_archived(sid, True)
        archived = _profiles(authority, actor, tmp_path, {})['profiles'][0]
        assert archived['canonical_session'] is None
        assert archived['last_session'] is None


def test_unbound_native_chat_is_not_a_local_bot_chat(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    actor = Principal('owner', str(tmp_path), frozenset({'session:read', 'session:operator'}), 'fixture')
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path))
        empty = _profiles(authority, actor, tmp_path, {})['profiles'][0]
        assert empty['canonical_session'] is None
        assert empty['last_session'] is None
        db.create_session('native', source='telegram', user_id=actor.subject, chat_id='native')
        db.set_session_title('native', 'Bot Chat')
        unbound = _profiles(authority, actor, tmp_path, {})['profiles'][0]
        assert unbound['canonical_session'] is None
        assert unbound['last_session'] is None
        db.set_session_title('native', 'Native chat')
        db.create_session('local-current', source='cli', user_id=actor.subject, chat_id='local-current')
        db.set_session_title('local-current', 'Bot Chat')
        current = _profiles(authority, actor, tmp_path, {})['profiles'][0]
        assert current['canonical_session']['id'] == 'local-current'
        assert current['last_session']['id'] == 'local-current'
