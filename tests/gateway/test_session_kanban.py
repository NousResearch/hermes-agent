"""Task claims, not caller-supplied launch policy, authorize Kanban execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_claim_freezes_task_policy_and_rejects_forgery(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    from contextlib import closing
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Owned acceptance', body='Goal criteria', assignee='default',
            workspace_kind='dir', workspace_path=str(workspace), skills=['owned-skill'], goal_mode=True,
            goal_max_turns=3, model_override='loop-model', provider_override='custom')
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board='owned', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    from gateway.session_kanban import build_kanban_policy
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal', 'file']}}
    policy, secrets = build_kanban_policy(connection, params, config)
    context = json.loads(policy.kanban_json)
    assert policy.source == 'kanban' and policy.cwd == str(workspace)
    assert policy.model == task.model_override and policy.provider == task.provider_override
    assert {'kanban', 'terminal', 'file'} <= set(policy.toolsets)
    assert context['task_id'] == task.id and context['run_id'] == task.current_run_id
    assert context['board'] == 'owned' and context['claim_lock'] == task.claim_lock
    assert context['skills'] == ['owned-skill'] and context['goal_mode'] and context['goal_max_turns'] == 3
    assert context['accept_hooks'] is True and 'Goal criteria' in context['goal_text']
    for changed in ({'run_id': task.current_run_id + 1}, {'board': 'foreign'}, {'claim_lock': 'forged'}, {'cwd': '/tmp'}):
        with pytest.raises(RuntimeStoreError):
            build_kanban_policy(connection, params | changed, config)
    connection.native_owner = False
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        build_kanban_policy(connection, params, config)
    from gateway.session_policy import build_policy
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        build_policy({'source': 'kanban', 'cwd': str(workspace)}, config)


@pytest.mark.parametrize('mode', ['complete', 'timeout', 'rate_limit', 'billing', 'crash'])
def test_real_dispatcher_lifecycle(tmp_path, mode):
    import os
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'SYSTEMROOT') if k in os.environ}
    env.update(HOME=str(tmp_path / 'home'), HERMES_HOME=str(tmp_path / 'state'), PYTHONPATH=str(repo), KANBAN_PROBE_MODE=mode)
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'kanban_owner_probe.py')],
        env=env, cwd=repo, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'state' / 'receipt.json').read_text())
    print(json.dumps(receipt))
    if mode != 'complete':
        assert receipt['task_status'] == ('running' if mode == 'crash' else 'ready')
        return
    assert receipt['task_status'] == 'done' and receipt['source'] == 'kanban'
    assert receipt['tools'] and receipt['task_context'] and receipt['skill_context'] and receipt['hook_effect']
    assert receipt['admissions'] == 1 and receipt['retry_same_session']
    assert receipt['goal_continuation'] and receipt['stable_prefix']
    print(json.dumps(receipt))


def test_late_launcher_cannot_stamp_replacement_claim(tmp_path, monkeypatch):
    from contextlib import closing
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    with closing(connect(board='owned')) as conn:
        tid = kb.create_task(conn, title='Owned race', assignee='default')
        old = kb.claim_task(conn, tid)
        kb.block_task(conn, tid, reason='replace claim', expected_run_id=old.current_run_id)
        kb.unblock_task(conn, tid)
        new = kb.claim_task(conn, tid)
        dispatch._set_worker_pid(conn, tid, 424242, run_id=old.current_run_id, claim_lock=old.claim_lock)
        assert kb.get_task(conn, tid).worker_pid is None
        assert conn.execute('SELECT worker_pid FROM task_runs WHERE id=?', (new.current_run_id,)).fetchone()[0] is None


def test_claim_reclaimed_before_admission_marker_is_refused(tmp_path, monkeypatch):
    """A claim that stops being current between policy/session creation and the marker
    transaction is refused: no owner_admitted row, no resume, no submit."""
    import asyncio
    from contextlib import closing
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal
    from hermes_state_runtime import RuntimeStoreError
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Owned race', assignee='default', workspace_kind='dir', workspace_path=str(workspace))
        kb.recompute_ready(conn)
        old = kb.claim_task(conn, task_id)
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal']}}
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: config)
    calls = []
    def create_local_session(authority, actor, params, *, trusted_policy=None, trusted_secrets=None):
        # Another process reclaims the task while the owner session is being created.
        with closing(connect(board='owned')) as conn:
            kb.block_task(conn, task_id, reason='replace claim', expected_run_id=old.current_run_id)
            kb.unblock_task(conn, task_id)
            kb.claim_task(conn, task_id)
        return SimpleNamespace(session_id='owner-session', profile_id=str(tmp_path))
    monkeypatch.setattr('gateway.session_local.create_local_session', create_local_session)
    async def submit(actor, submission):
        calls.append(('submit', submission))
    async def resume(ref, params):
        calls.append(('resume', ref))
    db = SimpleNamespace(get_session=lambda sid: None, db_path=str(tmp_path / 'state.db'))
    authority = SimpleNamespace(profile_id=str(tmp_path), db=db, submit=submit)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=authority, actor=actor, native_owner=True, resume=resume)
    params = dict(board='owned', task_id=task_id, run_id=old.current_run_id, claim_lock=old.claim_lock)
    from gateway.session_kanban import run_task
    with pytest.raises(RuntimeStoreError, match='stale_kanban_claim'):
        asyncio.run(run_task(connection, params))
    assert calls == []
    with closing(connect(board='owned')) as conn:
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='owner_admitted'", (task_id,)).fetchone()[0] == 0
