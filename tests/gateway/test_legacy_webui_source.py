"""Relay APK (webui source) sessions must be adoptable as legacy local sessions.

These sessions were created by the Hermes relay APK (source='webui') and carry
a Nous account label. They must adopt like desktop sessions do: the Nous stamp is
a label, not an owner, so the native local connection can claim ownership.
Foreign user_ids on webui rows are still refused.
"""
from gateway.session_contract import Principal
from hermes_state import SessionDB
from hermes_state_local_migration import validate_legacy_row, legacy_account_label, LEGACY_LOCAL_SOURCES
from hermes_state_mutation_binding import authorize_history, _history_sources, _owner_user


def test_webui_sessions_are_legacy_local_and_adoptable(tmp_path):
    assert 'webui' in LEGACY_LOCAL_SOURCES
    assert 'webui' in _history_sources()

    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        # A relay APK session with a Nous account label.
        db.create_session('relay-session', source='webui',
                          user_id='nous:nas_user:abc-123')

        row = db.get_session('relay-session')

        # Nous label is recognized (not an owner).
        assert legacy_account_label(row) is True
        assert _owner_user(row) is None

        # validate_legacy_row must NOT reject it.
        validate_legacy_row(db, row)

        # authorize_history must allow the native owner to claim it.
        actor = Principal('uid:1000', str(tmp_path),
                          frozenset({'session:read', 'session:create'}), 'native')
        with db._read_ctx() as conn:
            assert authorize_history(conn, actor, 'relay-session') is True
    finally:
        db.close()


def test_webui_sessions_with_foreign_user_are_refused(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('foreign-relay', source='webui',
                          user_id='telegram:user:42')
        row = db.get_session('foreign-relay')

        # A non-Nous user_id on a webui row is a real owner, not a label.
        assert legacy_account_label(row) is False
        assert _owner_user(row) == 'telegram:user:42'

        # validate_legacy_row rejects foreign-owned webui rows.
        from hermes_state_runtime import RuntimeStoreError
        try:
            validate_legacy_row(db, row)
        except RuntimeStoreError as exc:
            assert exc.reason == 'not_found'
        else:
            raise AssertionError('foreign webui row should be rejected')

        # authorize_history raises permission_denied for non-matching owner.
        actor = Principal('uid:1000', str(tmp_path),
                          frozenset({'session:read', 'session:create'}), 'native')
        with db._read_ctx() as conn:
            try:
                authorize_history(conn, actor, 'foreign-relay')
            except Exception as exc:
                assert getattr(exc, 'reason', None) == 'permission_denied'
            else:
                raise AssertionError('foreign owner should be denied')
    finally:
        db.close()


def test_webui_source_is_mapped_to_gui_creation_source(tmp_path):
    from hermes_state_local_migration import LEGACY_CREATION_SOURCES
    assert LEGACY_CREATION_SOURCES['webui'] == 'gui'
