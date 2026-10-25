"""Every old local row the native resume adopts also takes a sidebar edit (#106742).

The resume adopts main's ``desktop`` chats (including those stamped with the signed-in Nous
account) and ``oneshot``/``tool`` CLI runs, but ``authorize_history`` (pin/archive/hide/rename)
only knew ``cli``/``tui``/``gui``, so the same rows answered ``permission_denied`` on every
sidebar edit. A served remote backend's login label (``basic:alice``) is that login's ownership,
not a label: only the same login adopts and edits it (#139818; #140183 review F1).
"""
import json

import pytest

from gateway.session_contract import Principal
from hermes_state import SessionDB
from hermes_state_local_migration import validate_legacy_row
from hermes_state_mutation_binding import authorize_history, same_history_owner
from hermes_state_runtime import RuntimeStoreError


def _login(user, issuer=''):
    """The subject a dashboard login (or its forwarded ticket) carries: ``authenticated_subject``."""
    return 'auth:v1:' + json.dumps(['basic', issuer, user], separators=(',', ':'))


def test_rows_the_resume_adopts_take_sidebar_edits(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        for sid, source, user_id in (('old-desktop', 'desktop', None), ('old-oneshot', 'oneshot', None),
                                     ('old-desktop-nous', 'desktop', 'nous:account'),
                                     ('old-desktop-foreign', 'desktop', 'telegram:42'),
                                     ('old-cli-nous', 'cli', 'nous:account'), ('api-row', 'api_server', None)):
            db.create_session(sid, source=source, user_id=user_id)
        actor = Principal('uid:1', str(tmp_path), frozenset({'session:read', 'session:create'}), 'native-http:x')
        with db._read_ctx() as conn:
            for sid in ('old-desktop', 'old-oneshot', 'old-desktop-nous'):
                validate_legacy_row(db, db.get_session(sid))  # the resume adopts it ...
                assert authorize_history(conn, actor, sid) is True, sid  # ... so it takes an edit.
            assert same_history_owner(conn, 'old-desktop', 'old-desktop-nous') is True
            # A real foreign identity, a Nous stamp on a non-Desktop row and API rows stay refused.
            for sid in ('old-desktop-foreign', 'old-cli-nous'):
                try:
                    authorize_history(conn, actor, sid)
                except Exception as exc:
                    assert getattr(exc, 'reason', None) == 'permission_denied', (sid, exc)
                else:
                    raise AssertionError(sid)
            assert authorize_history(conn, actor, 'api-row') is False
    finally:
        db.close()


def test_a_login_labelled_desktop_row_belongs_to_that_login_only(tmp_path):
    """#140183 review F1: reading ``basic:alice`` as a no-owner label let login bob rename, claim
    and delete alice's chats once dashboard writes reach the gateway as the dashboard login. The
    label is alice's ownership: alice adopts and edits it, bob and the host's own UID do not."""
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('alice-row', source='desktop', user_id='basic:alice')
        db.create_session('oidc-row', source='desktop', user_id='oidc:alice')
        caps = frozenset({'session:read', 'session:create'})
        alice, bob = (Principal(_login(u), str(tmp_path), caps, 'native-http:x') for u in ('alice', 'bob'))
        issuer_alice = Principal(_login('alice', 'https://idp.example'), str(tmp_path), caps, 'native-http:x')
        host = Principal('uid:1', str(tmp_path), caps, 'native-http:x')
        row = db.get_session('alice-row')
        validate_legacy_row(db, row, subject=alice.subject)
        for stranger in (bob, issuer_alice, host):
            with pytest.raises(RuntimeStoreError, match='not_found'):
                validate_legacy_row(db, row, subject=stranger.subject)
        with pytest.raises(RuntimeStoreError, match='not_found'):
            validate_legacy_row(db, row)  # no caller named: never adoptable
        with db._read_ctx() as conn:
            assert authorize_history(conn, alice, 'alice-row') is True
            for stranger in (bob, issuer_alice, host):
                with pytest.raises(RuntimeStoreError, match='permission_denied'):
                    authorize_history(conn, stranger, 'alice-row', claim=True)
            # A login label the caller's subject cannot spell (another provider) stays foreign.
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                authorize_history(conn, alice, 'oidc-row')
            # Not an anonymous chain member either: a no-owner anchor does not reach it.
            assert same_history_owner(conn, 'alice-row', 'alice-row') is True
        db.create_session('plain', source='desktop')
        with db._read_ctx() as conn:
            assert same_history_owner(conn, 'plain', 'alice-row') is False
    finally:
        db.close()


def test_a_served_login_controls_its_own_stamped_chat_over_http(tmp_path):
    """#143547 DB-15: a remote Desktop signed in as ``basic:alice`` archives, pins and renames the
    chats it created over plain HTTP (transport ``http``, no native ticket). The claim gate refused
    every non-native transport before ``authorize_history`` could read the login stamp."""
    from types import SimpleNamespace
    from gateway.session_contract import SessionRef
    from gateway.session_mutations import _authorize_control

    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('alice-row', source='desktop', user_id='basic:alice')
        db.create_session('unstamped', source='desktop')
        # Restored or copied from another profile: the adoption gate answers not_found for it.
        db.create_session('alice-other-profile', source='desktop', user_id='basic:alice', profile_name='other')
        authority = SimpleNamespace(profile_id=str(tmp_path), db=db, sessions={})
        caps = frozenset({'session:read', 'session:create', 'session:control'})
        alice, bob, host = (Principal(subject, str(tmp_path), caps, 'http')
                            for subject in (_login('alice'), _login('bob'), 'uid:1'))

        def control(actor, sid):
            params = {'session_id': sid, 'request_id': 'r-' + sid, 'operation': 'archive'}
            return _authorize_control(authority, actor, SessionRef(str(tmp_path), sid), params)

        assert control(alice, 'alice-row') is True
        for stranger, sid in ((bob, 'alice-row'), (host, 'alice-row'), (alice, 'unstamped'),
                              (alice, 'alice-other-profile')):
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                control(stranger, sid)
    finally:
        db.close()


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
    from hermes_state_runtime import begin_runtime_epoch
    from tui_gateway.methods_session import _persist_branch

    caps = frozenset({'session:read', 'session:control', 'session:create', 'session:operator'})
    alice = Principal(_login('alice'), 'p', caps, 'http')

    async def edit(sid, rid, payload):
        params = {'session_id': sid, 'request_id': rid, 'payload': payload,
                  'expected_revision': authority.db.get_session(sid)['runtime_revision'],
                  'operation': 'archive' if 'archived' in payload else 'sidebar'}
        return await mutate_session(authority, alice, SessionRef(authority.profile_id, sid), params)

    with SessionDB(db_path=tmp_path / 'state.db') as db:
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='p', instance_id='owner',
                                     db=db, epoch=begin_runtime_epoch(db, instance_id='owner'))
        db.create_session('root', source='desktop', user_id=root_owner)
        db.end_session('root', 'compression')
        db.create_session('root-tip', source='desktop', user_id=root_owner, parent_session_id='root')
        _persist_branch(db, 'alice-branch', 'root', 'Branch', [{'role': 'user', 'content': 'hi'}],
                        source='desktop', cwd=None, profile_name=None, model='m', user_id='basic:alice')
        # Her branch compressed: the continuation inherits the fork marker naming the old root.
        db.end_session('alice-branch', 'compression')
        db.create_session('alice-tip', source='desktop', user_id='basic:alice', parent_session_id='alice-branch',
                          model_config={'_branched_from': 'root'})
        with pytest.raises(RuntimeStoreError):
            await edit('root', 'direct', {'pinned': True})
        await edit('alice-branch', 'pin', {'pinned': True, 'hidden': True})
        await edit('alice-tip', 'arch', {'archived': True})
        for sid in ('root', 'root-tip'):
            row = db.get_session(sid)
            assert (row['pinned'], row['hidden'], row['archived']) == (0, 0, 0), sid
        for sid in ('alice-branch', 'alice-tip'):
            row = db.get_session(sid)
            assert (row['pinned'], row['hidden'], row['archived']) == (1, 1, 1), sid
