"""A profile whose ``state.db`` is a symlink into another filesystem is still that profile's own store.

Operators routinely place the SQLite stores on a separate dataset (ZFS/NFS volume) and symlink
them into the profile directory. ``_build_profile_authority`` compared the *resolved* database
parent with the profile home, so such a store was rejected as "does not belong to the reserved
profile" and the gateway refused to boot. Ownership must be decided by file identity: the opened
database must be the same file as the profile's ``state.db``, wherever it physically lives.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import run_runtime


class _Registry:
    def __init__(self):
        self.added = []
        self.replaced = None

    def add(self, home, value, *, name):
        self.added.append((home, name))

    def remove(self, home):
        raise AssertionError('a store identical to the profile state.db must not be unregistered')

    def replace(self, home, authority):
        self.replaced = authority


def _runner(session_db, registry):
    return SimpleNamespace(_session_db=session_db, session_authorities=registry,
                           session_runtime_descriptor={'instance_id': 'owner'})


@pytest.mark.asyncio
async def test_a_symlinked_state_db_is_the_profiles_own_store(tmp_path, monkeypatch):
    from hermes_state_registry import acquire
    home = (tmp_path / 'profiles' / 'worker').resolve()
    home.mkdir(parents=True)
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (home / 'state.db').symlink_to(dataset / 'state.db')  # store lives on another volume
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    session_db = acquire(home / 'state.db')  # opens through the symlink, like the runner's cache
    sentinel = object()

    async def _fake_initialize(runner, **kwargs):
        return sentinel

    monkeypatch.setattr('gateway.session_authority.initialize_session_authority', _fake_initialize)
    registry = _Registry()
    authority = await run_runtime._build_profile_authority(_runner(session_db, registry), 'worker', home, register=False)
    assert authority is sentinel
    assert registry.added == [(home, 'worker')]
    assert registry.replaced is sentinel


@pytest.mark.asyncio
async def test_a_foreign_store_is_still_rejected(tmp_path, monkeypatch):
    from hermes_state_registry import acquire
    home = (tmp_path / 'profiles' / 'worker').resolve()
    home.mkdir(parents=True)
    (home / 'state.db').touch()  # a real file, so file identity is decidable
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    session_db = acquire(elsewhere / 'state.db')  # a DIFFERENT file than the profile's state.db
    registry = _Registry()
    with pytest.raises(RuntimeError) as exc:
        await run_runtime._build_profile_authority(_runner(session_db, registry), 'worker', home, register=False)
    assert 'does not belong' in str(exc.value)
    assert registry.added == []
