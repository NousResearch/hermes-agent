"""A reserved profile parked for an unusable store names the open failure, not an ownership mismatch.

The runner's scoped handle cache answers None when a profile's ``state.db`` fails to open, and
``_build_profile_authority`` reported every None as "session authority database does not belong to
the reserved profile": a legacy store crashing its migration on every open was parked under a
reason that sent the operator after the wrong problem (#141467).
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import run_runtime


class _Registry:
    def add(self, *a, **kw):
        raise AssertionError('an unusable store must not reach registration')


@pytest.mark.asyncio
async def test_an_unopenable_store_is_parked_under_its_open_failure(tmp_path, monkeypatch):
    home = (tmp_path / 'profiles' / 'worker').resolve()
    home.mkdir(parents=True)
    (home / 'state.db').mkdir()  # sqlite cannot open a directory: a store that fails on every open
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    runner = SimpleNamespace(_session_db=None, session_authorities=_Registry(),
                             session_runtime_descriptor={'instance_id': 'owner'})

    with pytest.raises(RuntimeError) as exc:
        await run_runtime._build_profile_authority(runner, 'worker', home, register=False)
    reason = str(exc.value)
    assert 'does not belong' not in reason
    assert str(Path(home) / 'state.db') in reason and 'did not open' in reason, reason
