"""Session-row ``cwd`` is row metadata, split from the execution cwd: the classic CLI records what
main's own lazy create stamped (``row_cwd``: the launch dir, or none for ``/new`` and
``--create-if-missing``) while the frozen policy keeps the directory its tools run in."""
from types import SimpleNamespace

import pytest


def _authority_fixture(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    return store, runner


@pytest.mark.asyncio
async def test_row_cwd_is_recorded_apart_from_the_execution_cwd(tmp_path, monkeypatch):
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt

    store, runner = _authority_fixture(tmp_path, monkeypatch)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='owner')
    actor = AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': 'fixture'}).actor
    work = tmp_path / 'repo' / '.worktrees' / 'wt'
    work.mkdir(parents=True)
    try:
        rows = {}
        for name, extra in (('default', {}), ('blank', {'row_cwd': ''}),
                            ('repo', {'row_cwd': str(tmp_path / 'repo')})):
            ref = create_local_session(authority, actor, {'request_id': name, 'source': 'cli', 'cwd': str(work),
                                                          'model': 'm', 'toolsets': [], **extra})
            receipt = local_receipt(authority.db, ref.session_id)
            policy = restore_policy(receipt['policy'])
            assert policy.cwd == str(work)  # the tools still run in the launch/worktree dir
            rows[name] = authority.db.get_session(receipt['entry']['session_id'])['cwd']
        assert rows == {'default': str(work), 'blank': None, 'repo': str(tmp_path / 'repo')}
        with pytest.raises(Exception):
            create_local_session(authority, actor, {'request_id': 'bad', 'source': 'cli', 'cwd': str(work),
                                                    'model': 'm', 'toolsets': [], 'row_cwd': 'relative/dir'})
    finally:
        store.close_all_db_handles()


def test_canonical_create_contract_accepts_row_cwd():
    from tui_gateway.contracts.registry import CANONICAL_METHODS, canonical_param_problems
    contract = CANONICAL_METHODS['session.create']
    assert not canonical_param_problems(contract, {'source': 'cli', 'row_cwd': ''})
