"""Regression for #139227: adopted local chats stay visible in the bot roster."""
from types import SimpleNamespace

from gateway.session_contract import SessionActor
from gateway.session_group_controls import _profiles
from gateway.session_policy import created_surface
from persistence.authority.bindings import NewSessionBinding
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def test_adopted_hidden_bot_chat_is_canonical_but_not_recent(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    sid = '20261006_105832_fixture'
    actor = SessionActor('fixture-owner', str(tmp_path), frozenset({'session:read'}), 'fixture')
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, sid, 'cli', title='Bot Chat', hidden=True)
        seed_messages(db, sid, {'role': 'assistant', 'content': 'The retained bot reply'})
        stamp = '2026-10-06T10:58:32'
        binding = NewSessionBinding(
            profile_id=str(tmp_path), principal_id=actor.subject, request_id='adopt', session_id=sid,
            route='agent:main:local:dm:' + sid,
            entry={'session_id': sid, 'created_at': stamp, 'updated_at': stamp,
                   'origin': {'platform': 'local', 'chat_id': sid, 'user_id': actor.subject}},
            policy={'source': 'cli', 'request_json': '{}', 'model': None, 'cwd': None},
            created_source=created_surface('cli'), legacy_session_id=sid)
        db.bindings.reserve(epoch=db.runtime.begin_epoch(instance_id='fixture'), binding=binding, owner=None)
        authority = SimpleNamespace(profile_id=str(tmp_path), persistence=storage_handle(tmp_path))
        before = raw_holder(db)._conn.total_changes
        row = _profiles(authority, actor, tmp_path, {})['profiles'][0]
        assert row['canonical_session'] is not None, 'adopted Bot Chat disappeared from roster'
        assert row['canonical_session']['resolved_id'] == sid
        assert row['canonical_session']['preview'] == 'The retained bot reply'
        assert row['last_session'] is None, 'a hidden Bot Chat is never the newest listed conversation'
        assert raw_holder(db)._conn.total_changes == before, 'discovery must remain read-only'
        assert db.sessions.get(sid).chat_id == sid

        foreign = SessionActor('other-owner', str(tmp_path), actor.capabilities, 'other')
        denied = _profiles(authority, foreign, tmp_path, {})['profiles'][0]
        assert denied['canonical_session'] is None
        assert denied['last_session'] is None
        operator = SessionActor('operator', str(tmp_path), actor.capabilities | {'session:operator'}, 'op')
        assert _profiles(authority, operator, tmp_path, {})['profiles'][0]['canonical_session']['id'] == sid
        light = _profiles(authority, actor, tmp_path, {'include_sessions': False})['profiles'][0]
        assert 'canonical_session' not in light

        db.sessions.set_archived(sid, True)
        archived = _profiles(authority, actor, tmp_path, {})['profiles'][0]
        assert archived['canonical_session'] is None
        assert archived['last_session'] is None


def test_unbound_native_chat_is_not_a_local_bot_chat(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    actor = SessionActor('owner', str(tmp_path), frozenset({'session:read', 'session:operator'}), 'fixture')
    with open_persistence(profile_scope(home=tmp_path)) as db:
        authority = SimpleNamespace(profile_id=str(tmp_path), persistence=storage_handle(tmp_path))
        empty = _profiles(authority, actor, tmp_path, {})['profiles'][0]
        assert empty['canonical_session'] is None
        assert empty['last_session'] is None
        seed_session(db, 'native', 'telegram', title='Bot Chat', user_id=actor.subject, chat_id='native')
        unbound = _profiles(authority, actor, tmp_path, {})['profiles'][0]
        assert unbound['canonical_session'] is None
        assert unbound['last_session'] is None
        db.sessions.set_title('native', 'Native chat')
        seed_session(db, 'local-current', 'cli', title='Bot Chat', user_id=actor.subject, chat_id='local-current')
        current = _profiles(authority, actor, tmp_path, {})['profiles'][0]
        assert current['canonical_session']['id'] == 'local-current'
        assert current['last_session']['id'] == 'local-current'
