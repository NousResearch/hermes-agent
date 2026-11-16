"""Bring-up publishes a local authority only once its recovery is whole.

``recover_local_sessions`` yields between receipts: a cold boot's SQLite reads must not stall the
owner loop long enough for the out-of-loop liveness watchdog to hard-exit the process. Those yields
let other tasks run, so nothing may be published before the last receipt is restored — a task that
wakes mid-recovery must never see a half-registered authority. The publication is three writes
(``runner.session_authority``, the store's ``_local_authority_epoch`` fence and its per-home epoch
map), and this pins them behind the recovery.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

# Collection-time import: ``gateway.run`` runs hermes_bootstrap's payload probe, which reads
# ``manifest.json`` beside the checkout — on a default install that is the real home, which the
# per-test real-home guard refuses. Load it before the guard is installed, as gateway tests do.
from gateway import run

# ``_local_authority_epoch`` is only ever created by the publication itself, so an absent attribute
# and one explicitly set to None are different states: record the absence, not a None.
_UNSET = object()


async def _boot(tmp_path, monkeypatch, *, register, prior, check):
    """Boot one authority over a fresh runner/store under *tmp_path*, with *check* standing in for
    the local recovery: it runs inside the boot and records what the outside world sees then."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
                             session_authority=prior)

    async def recover_local_sessions(authority):
        check(runner, store, authority)

    monkeypatch.setattr('gateway.session_local_recovery.recover_local_sessions', recover_local_sessions)
    return await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner',
                                              register=register)


def _db_key(authority):
    return Path(authority.db.db_path).resolve()


@pytest.mark.asyncio
async def test_the_launch_authority_is_published_only_after_local_recovery(tmp_path, monkeypatch):
    """A task waking on the recovery's yields still sees the previous boot's authority and no
    epoch; the launch lookup, the epoch fence and the keyed map appear together afterwards."""
    prior, seen = SimpleNamespace(sessions={}), {}

    def check(runner, store, authority):
        seen['runner_authority_is_new'] = runner.session_authority is authority
        seen['runner_authority'] = runner.session_authority
        seen['epoch'] = getattr(store, '_local_authority_epoch', _UNSET)
        seen['epochs'] = dict(getattr(store, '_local_authority_epochs', {}))

    authority = await _boot(tmp_path, monkeypatch, register=True, prior=prior, check=check)

    assert seen['runner_authority_is_new'] is False, 'the new authority was visible mid-recovery'
    assert seen['runner_authority'] is prior
    assert seen['epoch'] is _UNSET, 'the epoch fence was published mid-recovery'
    assert _db_key(authority) not in seen['epochs'], 'the keyed epoch map was published mid-recovery'

    store = authority.runner.session_store
    assert store._local_authority_epoch == authority.epoch
    assert store._local_authority_epochs[_db_key(authority)] == authority.epoch


@pytest.mark.asyncio
async def test_a_secondary_authority_is_built_without_being_published(tmp_path, monkeypatch):
    """``register=False`` builds a served secondary's authority for the per-home registry only:
    neither the launch lookup nor the epoch fence may move, yet its home still gets its epoch."""
    prior, seen = SimpleNamespace(sessions={}), {}

    def check(runner, store, authority):
        seen['runner_authority_is_new'] = runner.session_authority is authority
        seen['epoch'] = getattr(store, '_local_authority_epoch', _UNSET)
        seen['epochs'] = dict(getattr(store, '_local_authority_epochs', {}))

    authority = await _boot(tmp_path, monkeypatch, register=False, prior=prior, check=check)

    assert seen['runner_authority_is_new'] is False, 'the new authority was visible mid-recovery'
    assert seen['epoch'] is _UNSET, 'the epoch fence was published mid-recovery'
    assert _db_key(authority) not in seen['epochs'], 'the keyed epoch map was published mid-recovery'
    store = authority.runner.session_store
    assert authority.runner.session_authority is prior
    assert getattr(store, '_local_authority_epoch', _UNSET) is _UNSET
    assert store._local_authority_epochs[_db_key(authority)] == authority.epoch
