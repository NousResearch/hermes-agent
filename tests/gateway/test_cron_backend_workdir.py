"""Backend cron directories are not host session launch directories."""
import asyncio
import json
from pathlib import Path

import pytest

from tests.fakes.cron_owner import cron_owner


@pytest.mark.parametrize('backend', ['docker', 'ssh'])
def test_backend_workdir_survives_cron_admission_and_execution(tmp_path, monkeypatch, backend):
    from cron import jobs, scheduler
    from gateway import run, session_cron
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import claim_session_input
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope, terminal_env
    from agent.runtime_cwd import scoped_session_cwd

    home = tmp_path / 'profile'
    owner = cron_owner(home, monkeypatch)
    workdir = str(tmp_path / 'backend-only' / 'coupons')
    assert not Path(workdir).exists()
    config = {'model': {'default': 'test-model'}, 'platform_toolsets': {'cli': []},
              'terminal': {'backend': backend, 'cwd': '/workspace'}}
    (home / 'config.yaml').write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr(run, '_load_gateway_config', lambda: config)
    # A different launch profile's ambient settings must not choose the target.
    monkeypatch.setenv('TERMINAL_ENV', 'local')
    token = install_profile_terminal_scope(home)
    try:
        job = jobs.create_job(prompt='Review coupons', schedule='1d', target='backend', workdir=workdir)
    finally:
        reset_terminal_scope(token)

    observed = {}

    def run_job(frozen_job, *, extra_prompt, execution_id, cancel_event):
        assert frozen_job['workdir'] == workdir
        assert terminal_env('TERMINAL_ENV') == backend
        scope = scheduler._CronRunScope(frozen_job, job['id'], execution_id)
        try:
            scope.enter()
            observed['cwd'] = scoped_session_cwd()
            assert scope.workdir == workdir
        finally:
            scope.exit()
        return True, 'document', 'review completed', None

    monkeypatch.setattr(scheduler, 'run_job', run_job)

    async def probe():
        params = {'job_id': job['id'], 'request_id': 'fire', 'extra_prompt': None}
        receipt = await session_cron.operation(owner, 'submit', params)
        assert await session_cron.operation(owner, 'submit', params) == receipt
        sid = receipt['session_id']
        policy = next(iter(owner.runner.adapters.values())).policies[sid]
        assert Path(policy.cwd).is_dir()
        terminal = json.loads(policy.terminal_json)
        assert terminal['TERMINAL_ENV'] == backend
        assert terminal['TERMINAL_CWD'] == '/workspace'
        assert json.loads(policy.request_json)['cron_job']['workdir'] == workdir
        row = claim_session_input(owner.db, epoch=owner.epoch, session_id=sid)
        assert await session_cron.execute(owner, SessionRef(owner.profile_id, sid), row, policy) == 'review completed'

    try:
        asyncio.run(probe())
    finally:
        owner.db.close()
    assert observed == {'cwd': workdir}


@pytest.mark.parametrize('backend,mount_cwd', [('local', False), ('docker', True)])
@pytest.mark.parametrize('existing', [True, False])
def test_host_workdir_validation_is_preserved(tmp_path, monkeypatch, backend, mount_cwd, existing):
    from cron import jobs
    from gateway import run, session_cron
    from hermes_state_runtime import RuntimeStoreError

    home = tmp_path / 'profile'
    owner = cron_owner(home, monkeypatch)
    workdir = tmp_path / 'project'
    workdir.mkdir()
    job = jobs.create_job(prompt='Review', schedule='1d', workdir=str(workdir), model='test-model')
    config = {'model': {}, 'platform_toolsets': {'cli': []}, 'terminal': {
        'backend': backend, 'docker_mount_cwd_to_workspace': mount_cwd}}
    (home / 'config.yaml').write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr(run, '_load_gateway_config', lambda: config)
    if not existing:
        workdir.rmdir()

    async def probe():
        params = {'job_id': job['id'], 'request_id': 'fire', 'extra_prompt': None}
        if not existing:
            with pytest.raises(RuntimeStoreError, match='invalid_params') as error:
                await session_cron.operation(owner, 'submit', params)
            assert error.value.field == 'cwd'
            return
        receipt = await session_cron.operation(owner, 'submit', params)
        policy = next(iter(owner.runner.adapters.values())).policies[receipt['session_id']]
        assert policy.cwd == str(workdir)
        assert json.loads(policy.terminal_json)['TERMINAL_CWD'] == str(workdir)

    try:
        asyncio.run(probe())
    finally:
        owner.db.close()


def test_backend_prerun_script_can_suppress_agent_after_admission(tmp_path, monkeypatch):
    from cron import jobs, scheduler
    from gateway import run, session_cron
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import claim_session_input
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope

    home = tmp_path / 'profile'
    owner = cron_owner(home, monkeypatch)
    workdir = str(tmp_path / 'container-only')
    script = workdir + '/collect.py'
    config = {'model': {'default': 'test-model'}, 'platform_toolsets': {'cli': []},
              'terminal': {'backend': 'docker', 'cwd': '/workspace'}}
    (home / 'config.yaml').write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr(run, '_load_gateway_config', lambda: config)
    # Backend filesystem and script are simulated; admission/store/scheduler are real.
    monkeypatch.setattr('tools.terminal_tool.terminal_tool',
                        lambda **kwargs: json.dumps({'exit_code': 0, 'output': ''}))
    token = install_profile_terminal_scope(home)
    try:
        job = jobs.create_job(prompt='Review collected data', schedule='1d', target='backend',
                              workdir=workdir, script=script)
    finally:
        reset_terminal_scope(token)
    seen = []

    def collect(frozen_job, script_path, workdir=None, cancel_event=None):
        seen.append((frozen_job['target'], script_path, workdir))
        return True, '{"wakeAgent": false}'

    monkeypatch.setattr(scheduler, '_run_job_script_with_claim_heartbeat', collect)

    async def probe():
        receipt = await session_cron.operation(owner, 'submit', {
            'job_id': job['id'], 'request_id': 'fire', 'extra_prompt': None})
        sid = receipt['session_id']
        policy = next(iter(owner.runner.adapters.values())).policies[sid]
        row = claim_session_input(owner.db, epoch=owner.epoch, session_id=sid)
        assert await session_cron.execute(owner, SessionRef(owner.profile_id, sid), row, policy) == '[SILENT]'
        result = owner.pending_results[row['admission_id']]['result']
        assert result['completed'] and not result['failed']

    try:
        asyncio.run(probe())
    finally:
        owner.db.close()
    assert seen == [('backend', script, workdir)]
