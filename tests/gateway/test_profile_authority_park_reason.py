"""A reserved profile parked for an unusable store names the open failure, not an ownership mismatch.

The runner's scoped handle cache answers None when a profile's ``state.db`` fails to open, and
``_build_profile_authority`` reported every None as "session authority database does not belong to
the reserved profile": a legacy store crashing its migration on every open was parked under a
reason that sent the operator after the wrong problem (#141467). The reason is the exception the
failed open recorded; building it never opens (so never creates or migrates) the store.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import run_runtime
from gateway.session_db_recovery import RecoverableHandleCache


class _Registry:
    def add(self, *a, **kw):
        raise AssertionError('an unusable store must not reach registration')


def _runner(cache):
    return SimpleNamespace(_session_db=None, session_authorities=_Registry(), _session_db_handle_cache=cache,
                           session_runtime_descriptor={'instance_id': 'owner'})


@pytest.mark.asyncio
async def test_an_unopenable_store_is_parked_under_its_open_failure(tmp_path, monkeypatch):
    from hermes_state_registry import acquire
    home = (tmp_path / 'profiles' / 'worker').resolve()
    home.mkdir(parents=True)
    (home / 'state.db').mkdir()  # sqlite cannot open a directory: a store that fails on every open
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    cache = RecoverableHandleCache()
    assert cache.get(home / 'state.db', lambda: acquire(home / 'state.db')) is None  # the runner's failed open

    with pytest.raises(RuntimeError) as exc:
        await run_runtime._build_profile_authority(_runner(cache), 'worker', home, register=False)
    reason = str(exc.value)
    assert 'does not belong' not in reason
    assert str(Path(home) / 'state.db') in reason and 'did not open' in reason, reason


@pytest.mark.asyncio
async def test_the_park_reason_never_creates_the_store(tmp_path, monkeypatch):
    home = (tmp_path / 'profiles' / 'worker').resolve()
    home.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))

    with pytest.raises(RuntimeError) as exc:
        await run_runtime._build_profile_authority(_runner(RecoverableHandleCache()), 'worker', home, register=False)
    assert 'is unavailable' in str(exc.value)
    assert not (home / 'state.db').exists()
