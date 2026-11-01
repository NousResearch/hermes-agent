"""``runtime.describe`` names an update drain, so an attached Desktop or TUI renders "Updating to
v0.23.1, finishing 1 turn" instead of reading the owner's ``runtime_draining`` refusals as a fault."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.drain_control import mark_update_drain, open_restart_wait
from gateway.session_controls import AuthorityConnection
from tests.gateway.test_local_source_labels import _authority


@pytest.mark.asyncio
async def test_runtime_describe_carries_the_update_drain_and_nothing_for_a_plain_restart(tmp_path, monkeypatch):
    authority, _store = await _authority(tmp_path, monkeypatch)
    monkeypatch.setenv('HERMES_HOME', str(Path(authority.profile_id)))
    runner = authority.runner
    runner._describe_active_work = lambda: [{'kind': 'chat'}]
    runner.session_runtime_descriptor = {'state': 'ready'}
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    from tui_gateway.contracts.registry import CANONICAL_METHODS
    try:
        described = await conn.dispatch({'id': 0, 'method': 'runtime.describe', 'params': {}})
        assert described['result']['drain'] is None  # serving normally

        runner._draining = True
        open_restart_wait(runner)  # a restart (no update) names nothing a client renders
        described = await conn.dispatch({'id': 1, 'method': 'runtime.describe', 'params': {}})
        assert described['result']['drain'] is None

        mark_update_drain(runner, 'v0.23.1')
        described = await conn.dispatch({'id': 2, 'method': 'runtime.describe', 'params': {}})
        assert described['result']['drain'] == {'reason': 'update', 'target': 'v0.23.1', 'waiting_for': {'chat': 1}}
        CANONICAL_METHODS['runtime.describe'].result.model_validate(described['result'])
    finally:
        runner._draining = False
        await conn.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_runtime_describe_accepts_the_profile_routing_key(tmp_path, monkeypatch):
    """Desktop sends a shared-primary profile's reads with ``profile``; the closed describe params
    must take it as routing (stripped for the owner's own home), not refuse it as invalid_params."""
    authority, _store = await _authority(tmp_path, monkeypatch)
    root = tmp_path / 'root'
    (root / 'profiles').mkdir(parents=True)
    (root / 'profiles' / 'ops').symlink_to(Path(authority.profile_id), target_is_directory=True)
    monkeypatch.setattr('hermes_constants.get_default_hermes_root', lambda: root)
    monkeypatch.setenv('HERMES_HOME', str(Path(authority.profile_id)))
    authority.runner.session_runtime_descriptor = {'state': 'ready'}
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    try:
        described = await conn.dispatch({'id': 0, 'method': 'runtime.describe', 'params': {'profile': 'ops'}})
        assert described.get('result', {}).get('drain', 'missing') is None, described
    finally:
        await conn.close()
        authority.db.close()
