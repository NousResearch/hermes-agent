"""Relay APK (webui source) sessions must be adoptable as legacy local sessions.

These sessions were created by the Hermes relay APK (source='webui') and carry
a Nous account label. They must adopt like desktop sessions do: the Nous stamp is
a label, not an owner, so the native local connection can claim ownership.
Foreign user_ids on webui rows are still refused.
"""
import pytest

from gateway.session_contract import SessionActor
from persistence.core.errors import AuthorityRefusal
from persistence_assembly import store_owner_name
from persistence.authority.vocabulary import LEGACY_CREATION_SOURCES, LEGACY_LOCAL_SOURCES
from session_rules.legacy_local import legacy_account_label, legacy_creation_source, validate_legacy_row
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import storage_handle


def _authorize(store, actor, session_id):
    """Whether *actor* controls *session_id* as cold history (the authorization alone, no claim check)."""
    return store.bindings.authorize_history(session_id=session_id, profile_id=actor.profile_id, subject=actor.subject,
                                            may_create='session:create' in actor.capabilities, owner=None)


def test_webui_sessions_are_legacy_local_and_adoptable(tmp_path):
    assert 'webui' in LEGACY_LOCAL_SOURCES
    store = storage_handle(tmp_path)
    # A relay APK session with a Nous account label.
    seed_session(store, 'relay-session', 'webui', user_id='nous:nas_user:abc-123')
    row = session_row(store, 'relay-session')
    # Nous label is recognized (not an owner).
    assert legacy_account_label(row) is True
    own_profile = store_owner_name(store.scope.profile_dir)
    # validate_legacy_row must NOT reject it, so the native owner can claim it.
    validate_legacy_row(own_profile, row)
    actor = SessionActor('uid:1', str(tmp_path), frozenset({'session:read', 'session:create'}), 'native-http:x')
    assert _authorize(store, actor, 'relay-session') is True


def test_webui_sessions_with_foreign_user_are_refused(tmp_path):
    store = storage_handle(tmp_path)
    seed_session(store, 'foreign-relay', 'webui', user_id='telegram:user:42')
    row = session_row(store, 'foreign-relay')
    # A non-Nous user_id on a webui row is a real owner, not a label.
    assert legacy_account_label(row) is False
    own_profile = store_owner_name(store.scope.profile_dir)
    with pytest.raises(AuthorityRefusal, match='not_found'):
        validate_legacy_row(own_profile, row)
    # authorize_history refuses the non-matching owner.
    actor = SessionActor('uid:1', str(tmp_path), frozenset({'session:read', 'session:create'}), 'native-http:x')
    try:
        _authorize(store, actor, 'foreign-relay')
    except Exception as exc:
        assert getattr(exc, 'reason', None) == 'permission_denied'
    else:
        raise AssertionError('foreign owner should be denied')


def test_webui_source_is_mapped_to_gui_creation_source():
    assert LEGACY_CREATION_SOURCES['webui'] == 'gui'
    assert legacy_creation_source({'source': 'webui'}, None) == 'gui'
