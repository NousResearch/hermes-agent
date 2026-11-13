"""A /branch fork is a new conversation: it copies the transcript, never the parent's outcomes (#138653)."""
import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


@pytest.mark.asyncio
async def test_branch_does_not_copy_session_outcomes(tmp_path, monkeypatch):
    from gateway import run
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_mutations import mutate_session

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=())
    process_ownership.reserve([home for _, home in homes])
    runner = _runner(root, homes)
    try:
        await initialize_gateway_runtime(runner)
        _, home = homes[0]
        authority = runner.session_authorities.for_home(home)
        actor = Principal('owner', authority.profile_id,
                          frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
        with _profile_runtime_scope(home, {}):
            ref = create_local_session(authority, actor, {'request_id': 'r', 'source': 'gui', 'cwd': str(tmp_path),
                                                          'model': 'frozen', 'toolsets': []})
            authority.db.record_session_outcome(ref.session_id, signal='merged', source='github', recorder='cli')
            result = await mutate_session(authority, actor, ref, dict(
                session_id=ref.session_id, request_id='branch', expected_revision=0,
                expected_generation=0, operation='branch', payload={}))
        child = result['branched_session_id']
        assert authority.db.list_session_outcomes(child) == []
        assert [o['session_id'] for o in authority.db.list_session_outcomes()] == [ref.session_id]
    finally:
        for authority in list(getattr(runner, 'session_authorities', None) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)
