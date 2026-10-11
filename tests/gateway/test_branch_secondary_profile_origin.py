"""A branched local session keeps its owning profile's origin and resumes after restart."""
import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


@pytest.mark.asyncio
async def test_branch_restart_resume_keeps_source_and_route_for_default_and_secondary(tmp_path, monkeypatch):
    from gateway import run
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_mutations import mutate_session
    from hermes_state_local import local_receipt

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    process_ownership.reserve([home for _, home in homes])
    runners = []
    try:
        first = _runner(root, homes)
        runners.append(first)
        await initialize_gateway_runtime(first)
        branched = {}
        for name, home in homes:
            authority = first.session_authorities.for_home(home)
            actor = Principal('owner', authority.profile_id,
                              frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
            with _profile_runtime_scope(home, {}):
                ref = create_local_session(authority, actor, {'request_id': 'r', 'source': 'gui', 'cwd': str(tmp_path),
                                                              'model': 'frozen', 'toolsets': []})
                result = await mutate_session(authority, actor, ref, dict(
                    session_id=ref.session_id, request_id='branch', expected_revision=0,
                    expected_generation=0, operation='branch', payload={}))
            parent = local_receipt(authority.db, ref.session_id)
            child = local_receipt(authority.db, result['branched_session_id'])
            # Same owning profile on both halves of the identity: the stored origin and the route.
            assert child['entry']['origin'].get('profile') == parent['entry']['origin'].get('profile'), name
            assert child['route'].split(':')[:2] == parent['route'].split(':')[:2], name
            branched[name] = (home, result['branched_session_id'], child['route'], parent['entry']['origin'])
        for authority in list(first.session_authorities):
            authority.db.close()

        second = _runner(root, homes)  # restart: fresh authorities over the same state.db files
        runners.append(second)
        await initialize_gateway_runtime(second)
        for name, (home, sid, route, parent_origin) in branched.items():
            authority = second.session_authorities.for_home(home)
            assert sid in authority.sessions, f'{name}: branched session paused on cold recovery'
            with _profile_runtime_scope(home, {}):
                restore_local_session(authority, sid)
            live = authority.sessions[sid]
            assert live.route == route
            assert live.source.to_dict() == {**parent_origin, 'chat_id': sid}
    finally:
        for runner in runners:
            for authority in list(getattr(runner, 'session_authorities', None) or []):
                authority.db.close()
        for _, home in homes:
            process_ownership.release(home)
