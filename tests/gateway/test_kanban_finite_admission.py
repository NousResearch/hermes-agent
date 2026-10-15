"""Regression for #138992: dispatched cards retain headless approval semantics."""
import asyncio
from contextlib import closing
from types import SimpleNamespace

import pytest


def test_kanban_submission_enters_single_query_approval_context(tmp_path, monkeypatch):
    from gateway.session_contract import AdmissionReceipt, Principal, SessionRef
    from gateway.session_finite import execute_finite_admission, finite_turn_required, unattended_turn
    from gateway.session_kanban import run_task
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from tools.approval_context import _is_single_query_approval_context

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    for name in ('HERMES_KANBAN_DB', 'HERMES_DELEGATED_CHILD', 'HERMES_SINGLE_QUERY_SESSION'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    kb.create_board('owned')
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Headless work', assignee='default',
                                workspace_kind='dir', workspace_path=str(workspace))
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    config = {'model': {'default': 'loop-model', 'provider': 'custom'},
              'platform_toolsets': {'cli': ['terminal']}}
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: config)
    ref = SessionRef(str(tmp_path), 'owner-session')
    policies = []

    def create(authority, actor, params, *, trusted_policy, trusted_secrets):
        policies.append(trusted_policy)
        return ref

    monkeypatch.setattr('gateway.session_local.create_local_session', create)
    observed = []

    async def execute(authority, ref, row):
        observed.append((_is_single_query_approval_context(), finite_turn_required(), unattended_turn()))

    monkeypatch.setattr('gateway.session_ingress.execute_admission', execute)

    async def submit(actor, submission):
        await execute_finite_admission(authority, ref, {'admission_id': 'admission', 'payload': submission.payload})
        return AdmissionReceipt('admission', ref, 1, 'queued', None, 1, None)

    async def resume(ref, params):
        pass

    authority = SimpleNamespace(profile_id=str(tmp_path), submit=submit,
                                db=SimpleNamespace(get_session=lambda sid: None, db_path=str(tmp_path / 'state.db')))
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=authority, actor=actor, native_owner=True, resume=resume)
    params = dict(board='owned', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    result = asyncio.run(run_task(connection, params))
    assert result['session_id'] == ref.session_id
    assert policies[0].source == 'kanban'
    assert observed == [(True, True, False)], 'card turns must be finite without granting unattended auto-approval'
    assert finite_turn_required() is None


@pytest.mark.parametrize('mode', ['deny', 'approve'])
def test_durable_kanban_admission_preserves_headless_policy_in_worker(tmp_path, monkeypatch, mode):
    import json
    from gateway import run
    from gateway.session_controls import AuthorityConnection
    from gateway.session_contract import SessionRef
    from gateway.session_kanban import run_task
    from gateway.session_managed_worker import _bootstrap, worker_turn_scope
    from gateway.session_policy import policy_for_source
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_state_runtime import list_session_admissions
    from tests.gateway.test_session_follow_profile_config import _owner
    from tools.approval import request_tool_approval
    from tools.approval_context import _is_single_query_approval_context

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    for name in ('HERMES_KANBAN_DB', 'HERMES_DELEGATED_CHILD', 'HERMES_SINGLE_QUERY_SESSION', 'HERMES_YOLO_MODE'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(run, '_hermes_home', tmp_path)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path))
    config = {'model': {'default': 'loop-model', 'provider': 'custom'},
              'platform_toolsets': {'cli': ['terminal']},
              'approvals': {'mode': 'smart', 'single_query_mode': mode}}
    (tmp_path / 'config.yaml').write_text(json.dumps(config))
    kb.create_board('owned')
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Headless work', assignee='default',
                                workspace_kind='dir', workspace_path=str(tmp_path))
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    store, runner, authority, dashboard = _owner(tmp_path)
    authority.profile_id = str(tmp_path)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    connection = AuthorityConnection(authority, object(), {
        'user_id': 'owner', 'provider': 'local', 'profile_id': authority.profile_id,
        'instance_id': authority.instance_id, 'native_bootstrap': True,
        'capabilities': ['session:create', 'session:read', 'session:submit', 'session:control']})

    async def scenario():
        try:
            params = dict(board='owned', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
            first = await run_task(connection, params)
            retry = await run_task(connection, params)
            assert retry == first
            sid = first['session_id']
            rows = list_session_admissions(authority.db, session_id=sid)
            assert len(rows) == 1
            policy = policy_for_source(runner, authority.sessions[sid].source)
            assert policy.source == 'kanban'
            frame = _bootstrap(authority, SessionRef(authority.profile_id, sid), rows[0], policy,
                               {'execution_id': 'probe'})
            with worker_turn_scope(frame):
                assert _is_single_query_approval_context(), 'persisted card must reach worker as single-query'
                result = request_tool_approval('execute_code', 'headless test',
                    approval_callback=lambda *args: (_ for _ in ()).throw(AssertionError('must not prompt')))
                assert result['approved'] is (mode == 'approve')
                if mode == 'deny':
                    assert 'single' in result['message'].lower()
        finally:
            await connection.close()
            await dashboard.close()
            store.close_all_db_handles()

    asyncio.run(scenario())
