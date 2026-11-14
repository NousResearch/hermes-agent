"""The owner migrates pre-ownership NULL session rows itself (#94724).

While a gateway owns a profile the offline REST backfill is refused, so authority bring-up is
the one place an upgraded install's legacy rows get their owner.
"""
from hermes_state import SessionDB

from gateway.session_authority import backfill_legacy_session_owners


def _legacy(db, session_id, profile_name=None):
    db.create_session(session_id, source='cli')
    db._conn.execute('UPDATE sessions SET profile_name = ? WHERE id = ?', (profile_name, session_id))
    db._conn.commit()


def _owners(db):
    return dict(db._conn.execute('SELECT id, profile_name FROM sessions ORDER BY id').fetchall())


def test_default_store_stamps_only_unowned_rows(_isolate_hermes_home):
    from hermes_constants import get_default_hermes_root
    with SessionDB(db_path=get_default_hermes_root() / 'state.db') as db:
        _legacy(db, 'null-row')
        _legacy(db, 'blank-row', '  ')
        _legacy(db, 'owned-row', 'work')

        assert backfill_legacy_session_owners(db) == 2
        assert _owners(db) == {'blank-row': 'default', 'null-row': 'default', 'owned-row': 'work'}
        assert backfill_legacy_session_owners(db) == 0


def test_named_profile_store_stamps_its_own_name(_isolate_hermes_home):
    from hermes_constants import get_default_hermes_root
    home = get_default_hermes_root() / 'profiles' / 'work'
    home.mkdir(parents=True)
    with SessionDB(db_path=home / 'state.db') as db:
        _legacy(db, 'null-row')

        assert backfill_legacy_session_owners(db) == 1
        assert _owners(db) == {'null-row': 'work'}


def test_store_outside_the_profile_tree_keeps_null(tmp_path, _isolate_hermes_home):
    with SessionDB(db_path=tmp_path / 'elsewhere' / 'state.db') as db:
        _legacy(db, 'null-row')

        assert backfill_legacy_session_owners(db) == 0
        assert _owners(db) == {'null-row': None}
