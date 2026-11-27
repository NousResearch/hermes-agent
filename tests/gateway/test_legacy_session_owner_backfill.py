"""The owner migrates pre-ownership NULL session rows itself (#94724).

While a gateway owns a profile the offline REST backfill is refused, so authority bring-up is
the one place an upgraded install's legacy rows get their owner.
"""
import logging

from gateway.session_authority import backfill_legacy_session_owners
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _legacy(store, session_id, profile_name=None):
    seed_session(store, session_id, 'cli')
    raw_holder(store)._execute_write(lambda conn: conn.execute(
        'UPDATE sessions SET profile_name = ? WHERE id = ?', (profile_name, session_id)))


def _owners(store):
    with raw_holder(store)._read_ctx() as conn:
        return dict(conn.execute('SELECT id, profile_name FROM sessions ORDER BY id').fetchall())


def test_default_store_stamps_only_unowned_rows(_isolate_hermes_home):
    from hermes_constants import get_default_hermes_root
    store = storage_handle(get_default_hermes_root())
    _legacy(store, 'null-row')
    _legacy(store, 'blank-row', '  ')
    _legacy(store, 'owned-row', 'work')

    assert backfill_legacy_session_owners(store) == 2
    assert _owners(store) == {'blank-row': 'default', 'null-row': 'default', 'owned-row': 'work'}
    assert backfill_legacy_session_owners(store) == 0


def test_named_profile_store_stamps_its_own_name(_isolate_hermes_home):
    from hermes_constants import get_default_hermes_root
    home = get_default_hermes_root() / 'profiles' / 'work'
    home.mkdir(parents=True)
    store = storage_handle(home)
    _legacy(store, 'null-row')

    assert backfill_legacy_session_owners(store) == 1
    assert _owners(store) == {'null-row': 'work'}


def test_store_outside_the_profile_tree_keeps_null(tmp_path, _isolate_hermes_home):
    store = storage_handle(tmp_path / 'elsewhere')
    _legacy(store, 'null-row')

    assert backfill_legacy_session_owners(store) == 0
    assert _owners(store) == {'null-row': None}


def test_the_stamp_is_logged_with_the_owner_profile(_isolate_hermes_home, caplog):
    from hermes_constants import get_default_hermes_root
    home = get_default_hermes_root() / 'profiles' / 'work'
    home.mkdir(parents=True)
    store = storage_handle(home)
    _legacy(store, 'null-row')

    with caplog.at_level(logging.INFO, logger='gateway.session_authority'):
        assert backfill_legacy_session_owners(store) == 1
    assert "Stamped 1 legacy session row(s) with owner profile 'work'" in caplog.messages
