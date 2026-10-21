"""Task claims, not caller-supplied launch policy, authorize Kanban execution."""
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_github_delivery_binding_is_owner_loaded_and_body_pinned():
    from gateway.session_kanban import _github_delivery_binding
    from hermes_state_runtime import RuntimeStoreError

    task = SimpleNamespace(
        id='task-22', created_by='github:ISVGitHub/autonomous-entity#22',
        body='Build the ARCH-001 reuse index.',
    )
    binding = {
        'repository': 'ISVGitHub/autonomous-entity', 'repository_id': 123,
        'issue_number': 22, 'comment_id': 456, 'app_id': 7,
        'installation_id': 8, 'actor_id': 9, 'nonce': '0123456789abcdef',
        'task_sha256': hashlib.sha256(task.body.encode()).hexdigest(),
        'result_files': ['ARCH001_H1_REUSE_INDEX.json'], 'recorded_at': 100,
    }
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute('CREATE TABLE task_events (id INTEGER PRIMARY KEY, task_id TEXT, run_id INTEGER, kind TEXT, payload TEXT)')
    conn.execute('INSERT INTO task_events (task_id,run_id,kind,payload) VALUES (?, NULL, ?, ?)',
                 (task.id, 'github_task_binding', json.dumps(binding)))

    assert _github_delivery_binding(conn, task) == binding
    # Delivery artifacts are frozen in the authenticated task event, never
    # accepted as caller-supplied kanban.run parameters.
    assert _github_delivery_binding(conn, SimpleNamespace(
        id='ordinary-task', created_by='operator', body=task.body)) is None

    conn.execute('UPDATE task_events SET payload=?',
                 (json.dumps(binding | {'task_sha256': '0' * 64}),))
    with pytest.raises(RuntimeStoreError, match='github_task_binding_invalid'):
        _github_delivery_binding(conn, task)
    conn.execute('UPDATE task_events SET payload=?', (json.dumps(binding),))
    conn.execute('INSERT INTO task_events (task_id,run_id,kind,payload) VALUES (?, NULL, ?, ?)',
                 (task.id, 'github_task_binding', json.dumps(binding)))
    with pytest.raises(RuntimeStoreError, match='github_task_binding_invalid'):
        _github_delivery_binding(conn, task)
    conn.close()


def test_github_running_evidence_is_recorded_before_the_first_managed_turn(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from hermes_cli import issue_delivery, version_info
    from gateway import session_kanban

    events = []

    @contextmanager
    def connection(_path):
        yield object()

    monkeypatch.setattr('hermes_cli.kanban_db_connect.connect_closing', connection)
    monkeypatch.setattr(version_info, '_resolve_repo_dir', lambda: tmp_path)
    monkeypatch.setattr(version_info, 'get_code_identity', lambda: {'sha': 'a' * 40})
    monkeypatch.setattr(issue_delivery, 'record_worker_delivery',
        lambda conn, task_id, **kwargs: events.append(('running', task_id, kwargs)))
    monkeypatch.setattr('agent.skill_commands.build_preloaded_skills_prompt',
        lambda skills, task_id: ('', [], []))

    class Agent:
        session_id = 'owner-created-session'

        def run_conversation(self, prompt, **kwargs):
            events.append(('turn', prompt))
            return {'final_response': 'done'}

    context = {
        'skills': [], 'context': 'trusted task context', 'db': str(tmp_path / 'kanban.db'),
        'task_id': 'task-22', 'run_id': 11, 'claim_lock': 'claim-lock',
        'request_id': 'kanban:["missions","task-22",11]',
        'goal_mode': False, 'goal_max_turns': None,
        'github_delivery': {'result_files': ['ARCH001_H1_REUSE_INDEX.json']},
    }
    frame = {'text': 'Do the work.'}
    session_kanban._run_task_turns(Agent(), frame, [], context)

    assert events[0][0] == 'running'
    assert events[0][1] == 'task-22'
    assert events[0][2]['session_id'] == 'owner-created-session'
    assert events[0][2]['runtime_identity'] == {'runtime_root': str(tmp_path), 'runtime_commit': 'a' * 40}
    assert events[1][0] == 'turn'
    assert 'ARCH001_H1_REUSE_INDEX.json' in events[1][1]


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
    (workspace / 'pyproject.toml').write_text('[project]\nname = "task"\n')  # a code worktree
    from contextlib import closing
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
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
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal', 'file']},
              'agent': {'coding_context': 'focus'}}  # the worktree must not collapse a worker into the coding posture
    policy, _secrets = build_kanban_policy(connection, params, config)
    context = json.loads(policy.kanban_json)
    assert policy.source == 'kanban' and policy.cwd == str(workspace)
    assert policy.model == task.model_override and policy.provider == task.provider_override
    assert {'kanban', 'terminal', 'file'} <= set(policy.toolsets)
    assert context['task_id'] == task.id and context['run_id'] == task.current_run_id
    assert context['board'] == 'owned' and context['claim_lock'] == task.claim_lock
    assert context['skills'] == ['owned-skill'] and context['goal_mode'] and context['goal_max_turns'] == 3
    assert context['accept_hooks'] is True and 'Goal criteria' in context['goal_text']
    for changed in ({'run_id': task.current_run_id + 1}, {'board': 'foreign'}, {'claim_lock': 'forged'},
                    {'cwd': '/tmp'}, {'result_files': ['caller.json']}, {'session_id': 'caller-session'}):
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
    from tests.gateway.fixtures.local_recovery_probe import child_env
    env = child_env()
    env.update(HOME=str(tmp_path / 'home'), USERPROFILE=str(tmp_path / 'home'), HERMES_HOME=str(tmp_path / 'state'), PYTHONPATH=str(repo), KANBAN_PROBE_MODE=mode)
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' / 'kanban_owner_probe.py')],
        env=env, cwd=repo, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=150, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'state' / 'receipt.json').read_text())
    print(json.dumps(receipt))
    if mode != 'complete':
        assert receipt['task_status'] == 'ready'
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
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
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
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
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
    async def submit(actor, submission, **kwargs):
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
