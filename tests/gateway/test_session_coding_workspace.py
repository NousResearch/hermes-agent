"""Canonical workspace receipts stay with their authority-owned durable conversation."""
import copy
import json
import subprocess
from pathlib import Path

import pytest
import pytest_asyncio

from tests.gateway.test_local_create_env_refs import _owner as _base_owner


async def _owner(home, monkeypatch):
    owner, connection = await _base_owner(home, monkeypatch)
    owner.runner._adapter_for_source = lambda source: owner.runner.adapters.get(source.platform)
    return owner, connection


def git(path, *args):
    return subprocess.run(['git', '-C', str(path), *args], check=True, capture_output=True,
                          text=True, timeout=10).stdout.removesuffix('\n')


@pytest_asyncio.fixture
async def workspace(tmp_path, monkeypatch, request):
    from hermes_cli import projects_db as pdb
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway.coding_workspaces import prepare_workspace
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = tmp_path / '.hermes'
    if getattr(request, 'param', False):
        home = home / 'profiles' / 'alpha'
    home.mkdir(parents=True)
    default_cwd = tmp_path / 'inherited-profile-workspace'
    default_cwd.mkdir()
    (home / 'config.yaml').write_text('model: {provider: custom, default: m, base_url: http://unused.invalid/v1}\n'
                                     'platform_toolsets: {cli: [], desktop: []}\n'
                                     f'terminal: {{backend: local, cwd: {default_cwd}}}\n')
    source = tmp_path / 'source'
    source.mkdir()
    git(source, 'init')
    (source / 'sample.py').write_text('original bytes\n')
    git(source, 'add', '.')
    git(source, '-c', 'user.name=Test', '-c', 'user.email=test@localhost', 'commit', '-m', 'initial')
    owner, connection = await _owner(home, monkeypatch)
    token = set_hermes_home_override(home)
    try:
        with pdb.connect_closing(home / 'projects.db') as conn:
            binding = prepare_workspace(pdb, conn, {'path': str(source), 'mode': 'worktree', 'requestId': 'draft-one'})
        (Path(binding['cwd']) / 'sample.py').write_text('prepared checkout bytes\n')
        yield home, source, owner, connection, binding
    finally:
        await connection.close()
        owner.db.close()
        reset_hermes_home_override(token)


async def rpc(connection, method, **params):
    return await connection.dispatch(json.loads(json.dumps({'jsonrpc': '2.0', 'id': 'test', 'method': method,
                                                            'params': copy.deepcopy(params)})))


def create_params(binding):
    return {'source': 'gui', 'request_id': binding['requestId'], 'cwd': binding['cwd'],
            'cwd_explicit': False, 'coding_workspace': binding, 'toolsets': []}


@pytest.mark.asyncio
async def test_canonical_create_binds_raw_request_and_durable_server_artifacts(workspace, monkeypatch):
    from hermes_cli import projects_db as pdb
    from hermes_state_local import local_receipt
    home, _source, owner, connection, binding = workspace
    params = create_params(binding)
    response = await rpc(connection, 'session.create', **params)
    assert 'result' in response, response
    created = response['result']
    sid = created['session_id']
    bound = created['info']['coding_workspace']
    assert {k: bound[k] for k in binding} == binding
    assert Path(bound['artifactsPath']) == home / 'cache' / 'session-artifacts' / sid
    assert Path(bound['artifactsPath']).is_dir()
    assert created['info']['cwd'] == binding['cwd']
    assert created['info']['source'] == 'gui'
    assert created['info']['launch_request']['coding_workspace'] == binding
    assert created['info']['launch_request']['cwd_explicit'] is False
    saved = local_receipt(owner.db, sid)
    assert json.loads(saved['policy']['terminal_json'])['TERMINAL_CWD'] == binding['cwd']
    assert owner.db.get_session(sid)['cwd'] == binding['cwd']
    with pdb.connect_closing(home / 'projects.db') as conn:
        assert pdb.get_workspace_claim(conn, binding['requestId'])['state'] == 'bound'
    await connection.close()
    restarted, reconnected = await _owner(home, monkeypatch)
    try:
        retry = await rpc(reconnected, 'session.create', **params)
        assert 'result' in retry, retry
        assert retry['result']['session_id'] == sid
        assert retry['result']['info']['coding_workspace'] == bound
        assert local_receipt(restarted.db, sid)['policy'] == saved['policy']
        changed = {**params, 'coding_workspace': {**binding, 'branch': 'not-the-original-request'}}
        refused = await rpc(reconnected, 'session.create', **changed)
        assert refused['error']['message'] == 'invalid_params'
    finally:
        await reconnected.close()
        restarted.db.close()


@pytest.mark.asyncio
async def test_canonical_workspace_verify_and_references_keep_original_base(workspace):
    from agent.context_references import preprocess_context_references
    _home, source, _owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    verified = await rpc(connection, 'session.workspace.verify', session_id=sid, cwd=binding['cwd'])
    assert verified.get('result') == {'cwd': binding['cwd'], 'gatewayCwd': binding['cwd']}, verified
    checkout = Path(binding['cwd'])
    plain = await rpc(connection, 'session.workspace.references', session_id=sid, paths=[], text='ordinary text')
    assert plain.get('result') == {'paths': [], 'text': 'ordinary text'}, plain
    external = source.parent / 'external.txt'
    external.write_text('external original bytes')
    remapped = await rpc(connection, 'session.workspace.references', session_id=sid, paths=['sample.py', '../external.txt'],
                         text='Read @file:sample.py:1-2, and @file:../external.txt', reference_cwd=str(source))
    assert 'result' in remapped, remapped
    assert remapped['result']['paths'] == [str(checkout / 'sample.py'), str(external)]
    text = remapped['result']['text']
    assert f'@file:{checkout}/sample.py:1-2,' in text
    assert f'@file:{external}' in text
    expanded = preprocess_context_references(text, cwd=str(checkout), context_length=8192)
    assert 'prepared checkout bytes' in expanded.message
    assert 'original bytes' not in expanded.message.replace('external original bytes', '')
    # Branch changes do not change the immutable receipt or the repository/checkout identity.
    git(checkout, 'switch', '-c', 'valid-new-branch')
    switched = await rpc(connection, 'session.workspace.verify', session_id=sid, cwd=binding['cwd'])
    assert switched.get('result') == verified['result'], switched
    missing_base = await rpc(connection, 'session.workspace.references', session_id=sid, paths=[], text='@file:sample.py')
    assert missing_base['error']['message'] == 'invalid_params'
    (source / 'escape').symlink_to(external)
    escaped = await rpc(connection, 'session.workspace.references', session_id=sid, paths=['escape'], reference_cwd=str(source))
    assert escaped['error']['message'] == 'invalid_params'
    (checkout / 'target-escape').symlink_to(external)
    (source / 'target-escape').write_text('source safe')
    target_escape = await rpc(connection, 'session.workspace.references', session_id=sid,
                              paths=['target-escape'], reference_cwd=str(source))
    assert target_escape['error']['message'] == 'invalid_params'


def replace_checkout(binding):
    checkout = Path(binding['cwd'])
    moved = checkout.with_name('moved-checkout')
    checkout.rename(moved)
    checkout.mkdir()
    # Same path and .git contents are not the same checkout: directory inode is pinned.
    (checkout / '.git').write_bytes((moved / '.git').read_bytes())
    (checkout / 'sample.py').write_text('replacement bytes')


@pytest.mark.asyncio
@pytest.mark.parametrize('prior_prompt', [False, True])
async def test_plain_prompt_admission_rejects_replaced_checkout_without_preflight(workspace, monkeypatch, prior_prompt):
    from hermes_state_runtime import list_session_admissions
    _home, _source, owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    monkeypatch.setattr(owner, '_schedule_admitted', lambda ref, row: None)  # no model execution in this admission test
    if prior_prompt:
        admitted = await rpc(connection, 'prompt.submit', session_id=sid, text='first valid prompt', submission_id='prior')
        assert admitted.get('result', {}).get('status') == 'queued', admitted
    replace_checkout(binding)
    refused = await rpc(connection, 'prompt.submit', session_id=sid, text='ordinary text', submission_id='first')
    assert refused.get('error', {}).get('message') == 'workspace_unavailable', refused
    rows = list_session_admissions(owner.db, session_id=sid)
    assert [row['request_id'] for row in rows] == (['prior'] if prior_prompt else [])


@pytest.mark.asyncio
async def test_actual_turn_scope_rechecks_checkout_and_keeps_frozen_instructions(workspace, monkeypatch):
    from types import SimpleNamespace
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_policy import restore_policy
    from gateway.session_worker_construct import construct_inputs
    from hermes_state_local import local_receipt
    _home, _source, owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    policy = restore_policy(local_receipt(owner.db, sid)['policy'])
    construct = construct_inputs(owner, policy, sid)
    assert binding['cwd'] in construct['ephemeral_system_prompt']
    assert created['result']['info']['coding_workspace']['artifactsPath'] in construct['ephemeral_system_prompt']
    turn = TurnRunner(owner.runner, SimpleNamespace(source=owner.sessions[sid].source, session_id=sid))
    monkeypatch.setattr(turn, '_run_sync_scoped', lambda: {'ran_model': True})
    # Enqueue/preflight is not an execution lease: a replacement between them must still fail.
    replace_checkout(binding)
    result = turn.run_sync()
    assert result.get('ran_model') is not True, result
    assert result.get('failed') is True, result


@pytest.mark.asyncio
@pytest.mark.parametrize('workspace', [True], indirect=True)
async def test_prepared_cwd_wins_over_named_profile_inherited_flag(workspace):
    _home, _source, _owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    info = created['result']['info']
    assert info['profile_name'] == 'alpha'
    assert info['cwd'] == binding['cwd']
    verified = await rpc(connection, 'session.workspace.verify', session_id=created['result']['session_id'])
    assert verified.get('result', {}).get('cwd') == binding['cwd'], verified


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [('cwd', '/not-the-prepared-checkout'), ('repoRoot', '/wrong-repository'),
                                       ('sourcePath', '/missing-source'), ('projectId', 'foreign-project'),
                                       ('artifactsPath', '/client-owned-artifacts'), ('requestId', ''),
                                       ('unexpected', True)])
async def test_invalid_binding_never_mints_a_durable_session(workspace, field, value):
    from gateway.session_local_recovery import local_identity
    home, _source, owner, connection, binding = workspace
    sid = local_identity(owner.profile_id, connection.actor.subject, binding['requestId'])
    params = create_params(binding)
    params['coding_workspace'] = {**binding, field: value}
    refused = await rpc(connection, 'session.create', **params)
    assert refused.get('error', {}).get('message') == 'invalid_params', refused
    assert owner.db.get_session(sid) is None
    assert not (home / 'cache' / 'session-artifacts' / sid).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('method,params', [('session.workspace.verify', {}),
                                         ('session.workspace.references', {'paths': [], 'text': 'plain'})])
async def test_workspace_reads_require_exact_principal_profile_and_read_capability(workspace, method, params):
    from dataclasses import replace
    from tui_gateway import server
    home, _source, _owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    actor = connection.actor
    before = set(server._sessions)
    try:
        for changed, expected in [
            (replace(actor, capabilities=frozenset({'session:submit'})), 'permission_denied'),
            (replace(actor, subject='foreign', capabilities=frozenset({'session:read', 'session:operator'})), 'permission_denied'),
            (replace(actor, profile_id=str(home.parent / 'foreign')), 'profile_mismatch'),
        ]:
            connection.actor = changed
            refused = await rpc(connection, method, session_id=sid, **params)
            assert refused.get('error', {}).get('message') == expected, refused
    finally:
        connection.actor = actor
    assert set(server._sessions) == before


@pytest.mark.asyncio
async def test_binding_rejects_remote_terminal_and_mismatched_row_cwd(workspace):
    from gateway.session_local_recovery import local_identity
    home, source, owner, connection, binding = workspace
    sid = local_identity(owner.profile_id, connection.actor.subject, binding['requestId'])
    remote_config = (home / 'config.yaml').read_text().replace('backend: local', 'backend: ssh')
    (home / 'config.yaml').write_text(remote_config)
    remote = await rpc(connection, 'session.create', **create_params(binding))
    assert remote.get('error', {}).get('message') == 'invalid_params', remote
    assert owner.db.get_session(sid) is None
    (home / 'config.yaml').write_text(remote_config.replace('backend: ssh', 'backend: local'))
    mismatched = await rpc(connection, 'session.create', **{**create_params(binding), 'row_cwd': str(source)})
    assert mismatched.get('error', {}).get('message') == 'invalid_params', mismatched
    assert owner.db.get_session(sid) is None


@pytest.mark.asyncio
async def test_managed_execution_refuses_replaced_checkout_without_lost_worker(workspace, monkeypatch):
    from gateway import session_managed_worker as managed
    from gateway.session_contract import SessionRef
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    _home, _source, owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    policy = restore_policy(local_receipt(owner.db, sid)['policy'])
    async def would_spawn(*args):
        return 'worker would execute'
    monkeypatch.setattr(managed, 'execute_managed', would_spawn)
    replace_checkout(binding)
    result = await managed.admit_managed(owner, SessionRef(owner.profile_id, sid), {'admission_id': 'queued'}, policy)
    assert result != 'worker would execute'
    assert owner.pending_results['queued']['result']['failed'] is True


@pytest.mark.asyncio
async def test_overlapping_principals_cannot_bind_one_project_claim(workspace):
    from dataclasses import replace
    from gateway.session_local import stage_local_session
    from gateway.session_authorities import owner_scope
    from hermes_state_local import commit_local_session
    from hermes_state_runtime import RuntimeStoreError
    _home, _source, owner, connection, binding = workspace
    with owner_scope(owner):
        first = stage_local_session(owner, connection.actor, create_params(binding))
        foreign = stage_local_session(owner, replace(connection.actor, subject='another-principal'), create_params(binding))
    commit_local_session(owner.db, epoch=owner.epoch, receipt=first)
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        commit_local_session(owner.db, epoch=owner.epoch, receipt=foreign)
    assert owner.db.get_session(first['session_id']) is not None
    assert owner.db.get_session(foreign['session_id']) is None


@pytest.mark.asyncio
async def test_create_retry_repairs_post_commit_claim_projection_without_rebinding(workspace, monkeypatch):
    import sqlite3
    from hermes_cli import projects_db as pdb
    from hermes_state_local import local_receipt
    from gateway.session_local_recovery import local_identity
    home, _source, owner, connection, binding = workspace
    original = pdb.set_workspace_claim_state
    def fail_projection(*args):
        raise sqlite3.OperationalError('synthetic post-commit storage fault')
    monkeypatch.setattr(pdb, 'set_workspace_claim_state', fail_projection)
    failed = await rpc(connection, 'session.create', **create_params(binding))
    assert failed.get('error', {}).get('message') == 'storage_unavailable', failed
    sid = local_identity(owner.profile_id, connection.actor.subject, binding['requestId'])
    saved = local_receipt(owner.db, sid)
    monkeypatch.setattr(pdb, 'set_workspace_claim_state', original)
    retry = await rpc(connection, 'session.create', **create_params(binding))
    assert retry.get('result', {}).get('session_id') == sid, retry
    assert local_receipt(owner.db, sid)['policy'] == saved['policy']
    with pdb.connect_closing(home / 'projects.db') as conn:
        assert pdb.get_workspace_claim(conn, binding['requestId'])['state'] == 'bound'


@pytest.mark.asyncio
async def test_relative_path_preflight_without_original_base_is_refused(workspace):
    _home, _source, _owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    refused = await rpc(connection, 'session.workspace.references', session_id=sid, paths=['sample.py'])
    assert refused.get('error', {}).get('message') == 'invalid_params', refused


@pytest.mark.asyncio
async def test_terminal_cwd_retarget_is_denied_at_plain_prompt_admission(workspace, monkeypatch):
    from gateway.session_authorities import owner_scope
    from tools.terminal_tool import clear_task_env_overrides, register_task_env_overrides
    _home, source, owner, connection, binding = workspace
    created = await rpc(connection, 'session.create', **create_params(binding))
    assert 'result' in created, created
    sid = created['result']['session_id']
    monkeypatch.setattr(owner, '_schedule_admitted', lambda ref, row: None)
    with owner_scope(owner):
        register_task_env_overrides(sid, {'cwd': str(source)})
    try:
        refused = await rpc(connection, 'prompt.submit', session_id=sid, text='ordinary text', submission_id='retargeted')
        assert refused.get('error', {}).get('message') == 'workspace_unavailable', refused
    finally:
        with owner_scope(owner):
            clear_task_env_overrides(sid)
