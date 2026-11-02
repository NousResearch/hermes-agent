"""A cron owner reconstructs continuations from durable claims, not caller policy."""
import asyncio
import json
from types import SimpleNamespace


def test_owner_admits_claimed_completion_and_freezes_it_for_retries(tmp_path, monkeypatch):
    from cron import continuations, executions, jobs
    from gateway import run, session_cron
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    job = jobs.create_job(prompt='Inspect the build', schedule='every 1h', model='test-model')
    parent = executions.create_execution(job['id'], source='builtin')
    executions.finish_execution(parent['id'], success=True)
    continuations.retain_completion(
        {'job_id': job['id'], 'execution_id': parent['id'], 'profile_home': str(tmp_path)},
        {'id': 'build', 'output': 'BUILD_RECEIPT', 'exit_code': 7, 'completion_reason': 'exited'})
    candidate = continuations.pending_jobs()[0]
    execution = executions.create_execution(job['id'], source='continuation')
    candidate['execution_id'] = execution['id']
    assert continuations.claim_job(candidate)
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    db = SessionDB(tmp_path / 'state.db')
    runner = SimpleNamespace(_draining=False, session_store=SessionStore(tmp_path / 'sessions', GatewayConfig()),
                             adapters={})
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='test', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='test'))
    runner.session_authority = authority
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)

    async def probe():
        params = {'job_id': job['id'], 'request_id': execution['id'], 'extra_prompt': None}
        receipt = await session_cron.operation(authority, 'submit', params)
        policy = runner.adapters[next(iter(runner.adapters))].policies[receipt['session_id']]
        frozen = json.loads(policy.request_json)['cron_job']
        assert frozen['_process_continuation']['process_id'] == 'build'
        assert json.loads(frozen['_process_continuation']['result'])['output'] == 'BUILD_RECEIPT'
        assert frozen['prompt'].startswith('Continue the cron task')
        executions.finish_execution(execution['id'], success=True)
        assert continuations.pending_jobs() == []
        assert await session_cron.operation(authority, 'submit', params) == receipt
        # Another fire cannot borrow the claimed completion from this one.
        assert '_process_continuation' not in continuations.job_for_execution(job, parent['id'])

    try:
        asyncio.run(probe())
    finally:
        db.close()
