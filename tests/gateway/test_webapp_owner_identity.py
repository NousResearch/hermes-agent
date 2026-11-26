"""The Webapp is the machine owner: a signed-in tab and native Desktop share Group Chats and history.

Hermes is single-user. Sign-in only decides whether a browser may open the Webapp; it never names a
second owner. So a Webapp tab's Group Chats are native Desktop's (and the reverse), and the rooms an
older build stored under the tab's ``auth:v1`` login subject are still the owner's, while another OS
user's subject or a messaging identity stays foreign (gateway.owner_identity).
"""
import json
from types import SimpleNamespace

import pytest


from gateway.owner_identity import host_owner_subject, same_owner
from gateway.session_persistence import owner_rule
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import history_binding_key, hosted_owner_key, raw_holder
from tests.fixtures.storage_seed import seed_session

MEMBERS = [{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
           {'member_id': 'two', 'profile': 'two', 'handle': 'two'}]
NATIVE = {'provider': 'local', 'user_id': host_owner_subject(), 'profile_id': None, 'instance_id': 'owner',
          'capabilities': ['session:create', 'session:read', 'session:submit', 'session:control']}


@pytest.fixture
def authority(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'profiles' / 'two').mkdir(parents=True)
    (tmp_path / 'config.yaml').write_text(
        'hosted_rooms:\n  profiles:\n    two: ' + str((tmp_path / 'profiles' / 'two').resolve()) + '\n')
    from gateway.session_hosted_service import CanonicalHostedRoomService
    with open_persistence(profile_scope(home=tmp_path)) as db:
        authority = SimpleNamespace(persistence=db, profile_id=str(tmp_path), instance_id='owner', events={},
                                    epoch=db.runtime.begin_epoch(instance_id='test'))
        authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
        yield authority


async def _call(connection, method, **params):
    reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
    return reply.get('result', reply.get('error', {}).get('message'))


def _connect(authority, identity):
    from gateway.session_controls import AuthorityConnection
    if identity.get('provider') == 'local':
        identity = {**identity, 'profile_id': authority.profile_id}
    return AuthorityConnection(authority, object(), identity)


@pytest.mark.asyncio
async def test_a_webapp_tab_and_native_desktop_share_one_group_chat_list(authority):
    webapp = _connect(authority, {'user_id': 'admin', 'provider': 'basic'})
    desktop = _connect(authority, NATIVE)
    assert webapp.actor.subject == desktop.actor.subject == host_owner_subject()
    assert await _call(webapp, 'groups.create', room_id='web-room', name='Web', members=MEMBERS) is not None
    assert await _call(desktop, 'groups.create', room_id='desk-room', name='Desk', members=MEMBERS) is not None
    for surface in (webapp, desktop):
        listed = await _call(surface, 'groups.list')
        assert {room['room_id'] for room in listed['rooms']} == {'web-room', 'desk-room'}


@pytest.mark.asyncio
async def test_a_room_an_older_build_stored_under_the_login_stays_the_owners(authority):
    desktop = _connect(authority, NATIVE)
    assert await _call(desktop, 'groups.create', room_id='old-web-room', name='Old', members=MEMBERS) is not None
    raw_holder(authority.persistence)._execute_write(lambda conn: conn.execute(
        'UPDATE state_meta SET value=? WHERE key=?', ('auth:v1:["basic","","admin"]', hosted_owner_key('old-web-room'))))
    assert [r['room_id'] for r in (await _call(desktop, 'groups.list'))['rooms']] == ['old-web-room']
    stranger = _connect(authority, {**NATIVE, 'user_id': 'uid:999999'})
    assert (await _call(stranger, 'groups.list'))['rooms'] == []


OWNER = host_owner_subject()
WEBAPP_LOGIN = 'auth:v1:["basic","","admin"]'


@pytest.mark.parametrize('stamp, owned', [
    ('basic:admin', True), (f'local:{OWNER}', True), ('auth:v1:["self-hosted","https://idp","sub"]', True),
    ('auth:v1:["local","","' + OWNER + '"]', True),
    ('local:uid:999999', False), ('auth:v1:["local","","uid:999999"]', False),
    ('telegram:42', False), ('uid:999999', False), ('nous:account', False), ('auth:v1:["","","x"]', False),
])
def test_only_sign_in_stamps_and_this_machines_account_read_as_the_owners(stamp, owned):
    assert same_owner(stamp, OWNER) is owned
    assert same_owner(stamp, 'uid:999999') is (stamp == 'uid:999999')


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as db:
        yield SimpleNamespace(persistence=db, profile_id=str(tmp_path), sessions={},
                              _require_admission_open=lambda: None)


def _bind(store, sid, subject, source):
    seed_session(store.persistence, sid, source)
    binding = json.dumps([store.profile_id, subject], separators=(',', ':'))
    raw_holder(store.persistence)._execute_write(lambda conn: conn.execute(
        'INSERT INTO state_meta(key,value) VALUES(?,?)', (history_binding_key(sid), binding)))


def test_a_chat_a_webapp_tab_imported_before_the_upgrade_still_opens(store):
    """The sidebar (authorize_history) and the open (mount_history) read a binding by one rule."""
    from gateway.session_contract import SessionActor
    from gateway.session_history_mount import mount_history
    caps = frozenset({'session:create', 'session:read'})
    owner, stranger = (SessionActor(s, store.profile_id, caps, 't') for s in (OWNER, 'uid:999999'))
    _bind(store, 'imported-web', WEBAPP_LOGIN, 'webui')
    assert store.persistence.bindings.authorize_history(
        session_id='imported-web', profile_id=owner.profile_id, subject=owner.subject,
        may_create='session:create' in owner.capabilities, owner=owner_rule()) is True
    assert mount_history(store, owner, 'imported-web') is True
    store.sessions.clear()
    assert mount_history(store, stranger, 'imported-web') is False


def test_an_old_chat_a_webapp_tab_edited_is_adopted_by_the_owner(store):
    from gateway.session_policy import created_surface
    from persistence.authority.bindings import NewSessionBinding
    from persistence.core.errors import AuthorityRefusal
    _bind(store, 'legacy-web', WEBAPP_LOGIN, 'cli')
    epoch = store.persistence.runtime.begin_epoch(instance_id='test')

    def adopt(subject):
        stamp = '2026-10-06T10:58:32'
        binding = NewSessionBinding(
            profile_id=store.profile_id, principal_id=subject, request_id='adopt', session_id='legacy-web',
            route='agent:main:local:dm:legacy-web',
            entry={'session_id': 'legacy-web', 'created_at': stamp, 'updated_at': stamp,
                   'origin': {'platform': 'local', 'chat_id': 'legacy-web', 'user_id': subject}},
            policy={'source': 'cli', 'request_json': '{}', 'model': None, 'cwd': None},
            created_source=created_surface('cli'), legacy_session_id='legacy-web', lineage=('legacy-web',))
        store.persistence.bindings.reserve(epoch=epoch, binding=binding, owner=owner_rule())
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        adopt('uid:999999')
    adopt(OWNER)
    assert store.persistence.sessions.get('legacy-web').user_id == OWNER
