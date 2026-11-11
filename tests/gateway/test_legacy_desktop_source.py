"""Every old local row the native resume adopts also takes a sidebar edit (#106742).

The resume adopts main's ``desktop`` chats (including those stamped with the signed-in Nous
account) and ``oneshot``/``tool`` CLI runs, but ``authorize_history`` (pin/archive/hide/rename)
only knew ``cli``/``tui``/``gui``, so the same rows answered ``permission_denied`` on every
sidebar edit. A served remote backend's login label (``basic:alice``) is that login's ownership,
not a label: only the same login adopts and edits it (#139818; #140183 review F1).
"""
import json

import pytest

from gateway.session_contract import SessionActor
from persistence.core.errors import AuthorityRefusal
from persistence_assembly import open_persistence, profile_scope, store_owner_name
from session_rules.legacy_local import validate_legacy_row
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _login(user, issuer=''):
    """The subject a dashboard login (or its forwarded ticket) carries: ``authenticated_subject``."""
    return 'auth:v1:' + json.dumps(['basic', issuer, user], separators=(',', ':'))


def _authorize(store, actor, session_id):
    """Whether *actor* controls *session_id* as cold history (the authorization alone, no claim check)."""
    return store.bindings.authorize_history(session_id=session_id, profile_id=actor.profile_id, subject=actor.subject,
                                            may_create='session:create' in actor.capabilities, owner=None)


def test_rows_the_resume_adopts_take_sidebar_edits(tmp_path):
    store = storage_handle(tmp_path)
    for sid, source, user_id in (('old-desktop', 'desktop', None), ('old-oneshot', 'oneshot', None),
                                 ('old-desktop-nous', 'desktop', 'nous:account'),
                                 ('old-desktop-foreign', 'desktop', 'telegram:42'),
                                 ('old-cli-nous', 'cli', 'nous:account'), ('api-row', 'api_server', None)):
        seed_session(store, sid, source, user_id=user_id)
    actor = SessionActor('uid:1', str(tmp_path), frozenset({'session:read', 'session:create'}), 'native-http:x')
    own_profile = store_owner_name(store.scope.profile_dir)
    for sid in ('old-desktop', 'old-oneshot', 'old-desktop-nous'):
        validate_legacy_row(own_profile, session_row(store, sid))  # the resume adopts it ...
        assert _authorize(store, actor, sid) is True, sid  # ... so it takes an edit.
    # A real foreign identity, a Nous stamp on a non-Desktop row and API rows stay refused.
    for sid in ('old-desktop-foreign', 'old-cli-nous'):
        try:
            _authorize(store, actor, sid)
        except Exception as exc:
            assert getattr(exc, 'reason', None) == 'permission_denied', (sid, exc)
        else:
            raise AssertionError(sid)
    assert _authorize(store, actor, 'api-row') is False


def test_a_login_labelled_desktop_row_belongs_to_that_login_only(tmp_path):
    """#140183 review F1: reading ``basic:alice`` as a no-owner label let login bob rename, claim
    and delete alice's chats once dashboard writes reach the gateway as the dashboard login. The
    label is alice's ownership: alice adopts and edits it, bob and the host's own UID do not."""
    store = storage_handle(tmp_path)
    seed_session(store, 'alice-row', 'desktop', user_id='basic:alice')
    seed_session(store, 'oidc-row', 'desktop', user_id='oidc:alice')
    caps = frozenset({'session:read', 'session:create'})
    alice, bob = (SessionActor(_login(u), str(tmp_path), caps, 'native-http:x') for u in ('alice', 'bob'))
    issuer_alice = SessionActor(_login('alice', 'https://idp.example'), str(tmp_path), caps, 'native-http:x')
    host = SessionActor('uid:1', str(tmp_path), caps, 'native-http:x')
    own_profile = store_owner_name(store.scope.profile_dir)
    row = session_row(store, 'alice-row')
    validate_legacy_row(own_profile, row, subject=alice.subject)
    for stranger in (bob, issuer_alice, host):
        with pytest.raises(AuthorityRefusal, match='not_found'):
            validate_legacy_row(own_profile, row, subject=stranger.subject)
    with pytest.raises(AuthorityRefusal, match='not_found'):
        validate_legacy_row(own_profile, row)  # no caller named: never adoptable
    assert _authorize(store, alice, 'alice-row') is True
    for stranger in (bob, issuer_alice, host):
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            _authorize(store, stranger, 'alice-row')
    # A login label the caller's subject cannot spell (another provider) stays foreign.
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        _authorize(store, alice, 'oidc-row')


def test_a_served_login_controls_its_own_stamped_chat_over_http(tmp_path):
    """#143547 DB-15: a remote Desktop signed in as ``basic:alice`` archives, pins and renames the
    chats it created over plain HTTP (transport ``http``, no native ticket). The claim gate refused
    every non-native transport before ``authorize_history`` could read the login stamp."""
    from types import SimpleNamespace
    from gateway.session_contract import SessionRef
    from gateway.session_mutations import _authorize_control

    store = storage_handle(tmp_path)
    seed_session(store, 'alice-row', 'desktop', user_id='basic:alice')
    seed_session(store, 'unstamped', 'desktop')
    # Restored or copied from another profile: the adoption gate answers not_found for it.
    seed_session(store, 'alice-other-profile', 'desktop', user_id='basic:alice', profile_name='other')
    authority = SimpleNamespace(profile_id=str(tmp_path), sessions={}, persistence=store)
    caps = frozenset({'session:read', 'session:create', 'session:control'})
    alice, bob, host = (SessionActor(subject, str(tmp_path), caps, 'http')
                        for subject in (_login('alice'), _login('bob'), 'uid:1'))

    def control(actor, sid):
        params = {'session_id': sid, 'request_id': 'r-' + sid, 'operation': 'archive'}
        return _authorize_control(authority, actor, SessionRef(str(tmp_path), sid), params)

    assert control(alice, 'alice-row') is True
    for stranger, sid in ((bob, 'alice-row'), (host, 'alice-row'), (alice, 'unstamped'),
                          (alice, 'alice-other-profile')):
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            control(stranger, sid)


@pytest.mark.asyncio
@pytest.mark.parametrize('root_owner', ['basic:bob', None])
async def test_editing_ones_own_branch_leaves_another_logins_parent_alone(tmp_path, root_owner):
    """#144596 review F1: alice's branch of a compressed chat owned by bob (or by the host, unstamped)
    is another conversation. Her pin, hide or archive on it flagged that parent too, while a direct
    edit of the parent is refused. Her own chat's compression segments still move together."""
    from types import SimpleNamespace
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_mutations import mutate_session
    from tui_gateway import server

    caps = frozenset({'session:read', 'session:control', 'session:create', 'session:operator'})
    alice = SessionActor(_login('alice'), 'p', caps, 'http')

    async def edit(sid, rid, payload):
        params = {'session_id': sid, 'request_id': rid, 'payload': payload,
                  'expected_revision': authority.persistence.sessions.get(sid).runtime_revision,
                  'operation': 'archive' if 'archived' in payload else 'sidebar'}
        return await mutate_session(authority, alice, SessionRef(authority.profile_id, sid), params)

    with open_persistence(profile_scope(home=tmp_path)) as store:
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='p', instance_id='owner',
                                     epoch=store.runtime.begin_epoch(instance_id='owner'), persistence=storage_handle(tmp_path))
        seed_session(store, 'root', 'desktop', user_id=root_owner, end_reason='compression')
        seed_session(store, 'root-tip', 'desktop', user_id=root_owner, parent_session_id='root')
        server._persist_branch(store, 'alice-branch', 'root', 'Branch', [{'role': 'user', 'content': 'hi'}],
                        source='desktop', cwd=None, profile_name=None, model='m', user_id='basic:alice')
        # Her branch compressed: the continuation inherits the fork marker naming the old root.
        store.sessions.end('alice-branch', 'compression')
        seed_session(store, 'alice-tip', 'desktop', user_id='basic:alice', parent_session_id='alice-branch',
                     model_config={'_branched_from': 'root'})
        with pytest.raises(AuthorityRefusal):
            await edit('root', 'direct', {'pinned': True})
        await edit('alice-branch', 'pin', {'pinned': True, 'hidden': True})
        await edit('alice-tip', 'arch', {'archived': True})
        for sid in ('root', 'root-tip'):
            row = session_row(store, sid)
            assert (row['pinned'], row['hidden'], row['archived']) == (0, 0, 0), sid
        for sid in ('alice-branch', 'alice-tip'):
            row = session_row(store, sid)
            assert (row['pinned'], row['hidden'], row['archived']) == (1, 1, 1), sid


@pytest.mark.asyncio
@pytest.mark.parametrize('root_owner', ['basic:bob', None])
async def test_renaming_ones_own_branch_leaves_another_logins_parent_title_alone(tmp_path, root_owner):
    """#144596 round-2 review F1: a rename transfers a title off a compression ancestor. alice's branch
    of bob's compressed chat is not his continuation, so taking his title is a conflict, not a transfer."""
    from types import SimpleNamespace
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_mutations import mutate_session
    from tui_gateway import server

    caps = frozenset({'session:read', 'session:control', 'session:create', 'session:operator'})
    alice = SessionActor(_login('alice'), 'p', caps, 'http')
    with open_persistence(profile_scope(home=tmp_path)) as store:
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='p', instance_id='owner',
                                     epoch=store.runtime.begin_epoch(instance_id='owner'), persistence=storage_handle(tmp_path))
        seed_session(store, 'root', 'desktop', user_id=root_owner, title='Bob title')
        seed_messages(store, 'root', {'role': 'user', 'content': 'hello'})  # a real chat, not an empty ghost that yields its name
        store.sessions.end('root', 'compression')
        server._persist_branch(store, 'alice-branch', 'root', 'Branch', [{'role': 'user', 'content': 'hi'}],
                        source='desktop', cwd=None, profile_name=None, model='m', user_id='basic:alice')
        before = store.sessions.get('root').runtime_revision
        params = {'session_id': 'alice-branch', 'request_id': 'rename', 'operation': 'sidebar',
                  'payload': {'title': 'Bob title'},
                  'expected_revision': store.sessions.get('alice-branch').runtime_revision}
        with pytest.raises(Exception, match='already in use'):
            await mutate_session(authority, alice, SessionRef('p', 'alice-branch'), params)
        root = session_row(store, 'root')
        assert (root['title'], root['runtime_revision']) == ('Bob title', before)
