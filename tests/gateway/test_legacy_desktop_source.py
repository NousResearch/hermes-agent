"""Every old local row the native resume adopts also takes a sidebar edit (#106742).

The resume adopts main's ``desktop`` chats (including those stamped with the signed-in Nous
account or with a served remote backend's login label) and ``oneshot``/``tool`` CLI runs, but
``authorize_history`` (pin/archive/hide/rename) only knew ``cli``/``tui``/``gui``, so the same
rows answered ``permission_denied`` on every sidebar edit.
"""
from gateway.session_contract import Principal
from hermes_state import SessionDB
from hermes_state_local_migration import validate_legacy_row
from hermes_state_mutation_binding import authorize_history, same_history_owner


def test_rows_the_resume_adopts_take_sidebar_edits(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        for sid, source, user_id in (('old-desktop', 'desktop', None), ('old-oneshot', 'oneshot', None),
                                     ('old-desktop-nous', 'desktop', 'nous:account'),
                                     ('old-desktop-remote', 'desktop', 'basic:hermes'),
                                     ('old-desktop-oidc', 'desktop', 'oidc:alice'),
                                     ('old-desktop-foreign', 'desktop', 'telegram:42'),
                                     ('old-cli-nous', 'cli', 'nous:account'), ('api-row', 'api_server', None)):
            db.create_session(sid, source=source, user_id=user_id)
        actor = Principal('uid:1', str(tmp_path), frozenset({'session:read', 'session:create'}), 'native-http:x')
        with db._read_ctx() as conn:
            for sid in ('old-desktop', 'old-oneshot', 'old-desktop-nous', 'old-desktop-remote',
                        'old-desktop-oidc'):
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
