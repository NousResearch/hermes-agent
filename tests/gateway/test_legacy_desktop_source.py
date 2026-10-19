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
