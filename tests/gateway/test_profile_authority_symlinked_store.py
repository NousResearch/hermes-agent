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
from hermes_state_registry import release


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

    def __iter__(self):
        return iter([self.replaced] if self.replaced is not None else [])


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
    try:
        # acquire() resolves the path: the handle names the link TARGET, never the profile home.
        assert Path(session_db.db_path).parent == dataset.resolve()

        async def _fake_initialize(runner, **kwargs):
            from gateway.session_authority import SessionAuthority
            return SessionAuthority(runner, profile_id=kwargs['profile_id'], instance_id=kwargs['instance_id'],
                                    db=kwargs['db'], epoch=None, home=kwargs['home'])

        monkeypatch.setattr('gateway.session_authority.initialize_session_authority', _fake_initialize)
        registry = _Registry()
        authority = await run_runtime._build_profile_authority(_runner(session_db, registry), 'worker', home, register=False)
        assert registry.added == [(home, 'worker')]
        assert registry.replaced is authority
        # Every profile-scoped read (cron scope, config, credentials) derives its home from here (#145075).
        from gateway.session_authorities import authority_home
        assert authority_home(authority) == home
    finally:
        release(session_db)


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
    try:
        registry = _Registry()
        with pytest.raises(RuntimeError) as exc:
            await run_runtime._build_profile_authority(_runner(session_db, registry), 'worker', home, register=False)
        assert 'does not belong' in str(exc.value)
        assert registry.added == []
    finally:
        release(session_db)


@pytest.mark.asyncio
async def test_a_store_already_served_for_another_profile_is_rejected(tmp_path, monkeypatch):
    """Two homes linking one state.db both pass the file-identity check; the second must not build
    an authority (its runtime epoch would stale the first one's admissions)."""
    from hermes_state_registry import acquire
    first = (tmp_path / 'profiles' / 'first').resolve()
    second = (tmp_path / 'profiles' / 'second').resolve()
    for home in (first, second):
        home.mkdir(parents=True)
    (first / 'state.db').touch()
    (second / 'state.db').symlink_to(first / 'state.db')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    session_db = acquire(second / 'state.db')
    try:
        registry = _Registry()
        registry.replaced = SimpleNamespace(profile_id=str(first), db=session_db)

        async def _never(runner, **kwargs):
            raise AssertionError('no authority may initialize over a store another profile serves')

        monkeypatch.setattr('gateway.session_authority.initialize_session_authority', _never)
        with pytest.raises(RuntimeError) as exc:
            await run_runtime._build_profile_authority(_runner(session_db, registry), 'second', second, register=False)
        assert 'already served' in str(exc.value)
        assert registry.added == []
    finally:
        release(session_db)
