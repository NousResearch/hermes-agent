"""``session.owner_backfill``: the live owner stamps its own legacy NULL-profile rows.

A dashboard cannot take the profile's exclusive-maintenance write while a standalone gateway owns
it (#148531), so the stamp is the owner's own — over the control socket, on the store only it may
write, with the profile name its bring-up derives.
"""
import sqlite3
from types import SimpleNamespace

import pytest

_OPERATOR = frozenset({'session:operator'})


def _connection(home, db, *, native=True, capabilities=_OPERATOR, ticket_profile=None):
    from gateway.session_contract import SessionActor

    actor = SessionActor('owner', ticket_profile or str(home), frozenset(capabilities), 'transport')
    return SimpleNamespace(authority=SimpleNamespace(profile_id=str(home), db=db),
                           actor=actor, native_owner=native)


def _profile_of(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT profile_name FROM sessions WHERE id = 'legacy-null'").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A profile home holding one row the pre-ownership install left unowned."""
    from hermes_state import SessionDB

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db_path = tmp_path / 'state.db'
    db = SessionDB(db_path=db_path)
    db.create_session('legacy-null', source='cli', profile_name='default')
    db._conn.execute("UPDATE sessions SET profile_name = NULL WHERE id = 'legacy-null'")
    db._conn.commit()
    try:
        yield db_path, db
    finally:
        db.close()


@pytest.mark.asyncio
async def test_owner_backfill_stamps_its_own_store_and_is_idempotent(store):
    from gateway.session_owner_backfill import backfill

    db_path, db = store
    connection = _connection(db_path.parent, db)
    assert await backfill(connection, None, {}) == {'stamped': 1, 'profile': 'default'}
    assert _profile_of(db_path) == 'default'
    assert await backfill(connection, None, {}) == {'stamped': 0, 'profile': 'default'}


@pytest.mark.asyncio
@pytest.mark.parametrize('refused', [
    {'native': False},                          # a remote or browser grant
    {'capabilities': {'session:read'}},         # not an operator ticket
    {'ticket_profile': '/elsewhere'},           # another profile's ticket
])
async def test_owner_backfill_refuses_anything_but_the_local_operator(store, refused):
    from gateway.session_owner_backfill import backfill
    from hermes_state_runtime import RuntimeStoreError

    db_path, db = store
    with pytest.raises(RuntimeStoreError):
        await backfill(_connection(db_path.parent, db, **refused), None, {})
    assert _profile_of(db_path) is None


@pytest.mark.asyncio
async def test_owner_backfill_rejects_unknown_params(store):
    from gateway.session_owner_backfill import backfill
    from hermes_state_runtime import RuntimeStoreError

    db_path, db = store
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        await backfill(_connection(db_path.parent, db), None, {'stamped': 5})
