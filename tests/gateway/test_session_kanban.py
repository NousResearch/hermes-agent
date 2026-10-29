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


class _TurnAgent:
    """The least agent _run_task_turns needs: a session id and one recorded conversation."""

    session_id = 's1'

    def __init__(self):
        self.calls = []

    def run_conversation(self, message, conversation_history=None):
        self.calls.append((message, conversation_history))
        return {'final_response': 'checked', 'messages': []}


def test_worker_restores_lane_bundled_skill_deleted_from_profile(tmp_path, monkeypatch):
    """A review-lane worker whose profile lacks bundled sdlc-review (a manual delete or curator
    prune the seed sync deliberately never re-adds) restores it from the bundled source like the
    connector catalog does. Without the restore the worker died before its first model call and
    the dispatcher could only book "execution unknown"."""
    from gateway import session_kanban
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    probes, restored = [], []

    def probe(identifiers, task_id=None, **kwargs):
        probes.append(list(identifiers))
        return ('', [], ['sdlc-review']) if len(probes) == 1 else ('skill: sdlc-review', ['sdlc-review'], [])

    monkeypatch.setattr('agent.skill_commands.build_preloaded_skills_prompt', probe)
    monkeypatch.setattr('tools.skills_sync_bundled_ops.ensure_bundled_skill',
                        lambda name: restored.append(name) or {'ok': True, 'action': 'restored',
                                                                'path': None, 'message': ''})
    recovered = []
    monkeypatch.setattr('agent.kanban_turn_recovery.recover_failed_kanban_turns',
                        lambda recover_fn, get_result, **kwargs: recovered.append(kwargs) or 0)
    agent = _TurnAgent()
    context = {'skills': ['sdlc-review'], 'context': '', 'goal_mode': False, 'task_id': 't1',
               'run_id': 1, 'claim_lock': 'lock', 'db': 'unused'}
    result = session_kanban._run_task_turns(agent, {'text': 'review it'}, [], context)
    assert result['final_response'] == 'checked'
    assert restored == ['sdlc-review'] and len(probes) == 2
    assert 'skill: sdlc-review' in agent.calls[0][0]
    assert recovered and recovered[0]['task_id'] == 't1'


def test_worker_still_refuses_missing_unbundled_skill(tmp_path, monkeypatch):
    """Only bundled names revive: a card-listed skill with no bundled source keeps the hard
    refusal, so a worker never claims a task whose instructions it could not load."""
    from gateway import session_kanban
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr('agent.skill_commands.build_preloaded_skills_prompt',
                        lambda identifiers, task_id=None, **kwargs: ('', [], ['owned-skill']))
    ensured = []
    monkeypatch.setattr('tools.skills_sync_bundled_ops.ensure_bundled_skill',
                        lambda name: ensured.append(name) or {'ok': False, 'action': 'not_restored'})
    context = {'skills': ['owned-skill'], 'context': '', 'goal_mode': False, 'task_id': 't1',
               'run_id': 1, 'claim_lock': 'lock', 'db': 'unused'}
    with pytest.raises(ValueError, match='kanban_skills_unavailable'):
        session_kanban._run_task_turns(SimpleNamespace(session_id='s1'), {'text': 'work'}, [], context)
    assert ensured == []


def test_worker_restores_lane_skill_even_when_other_skills_load(tmp_path, monkeypatch):
    """The restore gate is `missing`, not `missing and not loaded`: when a card skill loads but
    the lane's sdlc-review is out of the profile, the review must not silently run without the
    lane-required skill - the contract build_kanban_policy states when it injects sdlc-review."""
    from gateway import session_kanban
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    probes, restored = [], []

    def probe(identifiers, task_id=None, **kwargs):
        probes.append(list(identifiers))
        return (('skill: owned-skill', ['owned-skill'], ['sdlc-review']) if len(probes) == 1
                else ('skill: both', ['owned-skill', 'sdlc-review'], []))

    monkeypatch.setattr('agent.skill_commands.build_preloaded_skills_prompt', probe)
    monkeypatch.setattr('tools.skills_sync_bundled_ops.ensure_bundled_skill',
                        lambda name: restored.append(name) or {'ok': True, 'action': 'restored',
                                                                'path': None, 'message': ''})
    monkeypatch.setattr('agent.kanban_turn_recovery.recover_failed_kanban_turns',
                        lambda recover_fn, get_result, **kwargs: 0)
    agent = _TurnAgent()
    context = {'skills': ['owned-skill', 'sdlc-review'], 'context': '', 'goal_mode': False, 'task_id': 't1',
               'run_id': 1, 'claim_lock': 'lock', 'db': 'unused'}
    result = session_kanban._run_task_turns(agent, {'text': 'review it'}, [], context)
    assert result['final_response'] == 'checked'
    assert restored == ['sdlc-review'] and len(probes) == 2
    assert 'skill: both' in agent.calls[0][0]


def test_worker_still_refuses_when_bundled_restore_fails(tmp_path, monkeypatch):
    """A bundled restore that cannot write the profile keeps the deterministic pre-model refusal
    instead of half-claiming the task - and re-probes nothing after a failed restore."""
    from gateway import session_kanban
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    probes, ensured = [], []

    def probe(identifiers, task_id=None, **kwargs):
        probes.append(list(identifiers))
        return '', [], ['sdlc-review']

    monkeypatch.setattr('agent.skill_commands.build_preloaded_skills_prompt', probe)
    monkeypatch.setattr('tools.skills_sync_bundled_ops.ensure_bundled_skill',
                        lambda name: ensured.append(name) or {'ok': False, 'action': 'not_restored',
                                                               'path': None, 'message': 'blocked'})
    context = {'skills': ['sdlc-review'], 'context': '', 'goal_mode': False, 'task_id': 't1',
               'run_id': 1, 'claim_lock': 'lock', 'db': 'unused'}
    with pytest.raises(ValueError, match='kanban_skills_unavailable'):
        session_kanban._run_task_turns(_TurnAgent(), {'text': 'review it'}, [], context)
    assert ensured == ['sdlc-review'] and len(probes) == 1


def test_safe_worker_skips_bundled_restore(tmp_path, monkeypatch):
    """The safe worker reports every listed skill missing; it must not mutate the profile trying
    to revive them. The refusal stays, and no restore is attempted."""
    from gateway import session_kanban
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.setattr('agent.safe_worker_policy.safe_worker_enabled', lambda: True)
    monkeypatch.setattr('agent.skill_commands.build_preloaded_skills_prompt',
                        lambda identifiers, task_id=None, **kwargs: ('', [], list(identifiers)))
    ensured = []
    monkeypatch.setattr('tools.skills_sync_bundled_ops.ensure_bundled_skill',
                        lambda name: ensured.append(name) or {'ok': True, 'action': 'restored',
                                                                'path': None, 'message': ''})
    context = {'skills': ['sdlc-review'], 'context': '', 'goal_mode': False, 'task_id': 't1',
               'run_id': 1, 'claim_lock': 'lock', 'db': 'unused'}
    with pytest.raises(ValueError, match='kanban_skills_unavailable'):
        session_kanban._run_task_turns(SimpleNamespace(session_id='s1'), {'text': 'review it'}, [], context)
    assert ensured == []


def test_restore_makes_real_probe_reload_in_same_process(tmp_path, monkeypatch):
    """Pins the unmocked chain under a real scratch home: the probe misses, the restore lands on
    disk, and the SAME probe then loads it. A probe-level cache that ignored the restore would
    silently turn every revive back into the pre-model crash."""
    from gateway import session_kanban
    from agent.skill_commands import build_preloaded_skills_prompt
    from tools import skill_usage
    from tools.skills_sync_bundled_ops import bundled_skill_for_install, ensure_bundled_skill
    home = tmp_path / 'home'
    monkeypatch.setenv('HERMES_HOME', str(home))
    p1, l1, m1 = build_preloaded_skills_prompt(['sdlc-review'])
    assert m1 == ['sdlc-review'] and not l1
    assert bundled_skill_for_install('sdlc-review') == 'sdlc-review'
    assert ensure_bundled_skill('sdlc-review')['ok']
    assert (skill_usage._skills_dir() / 'devops' / 'sdlc-review' / 'SKILL.md').is_file()
    p2, l2, m2 = build_preloaded_skills_prompt(['sdlc-review'])
    assert l2 == ['sdlc-review'] and not m2 and 'sdlc-review' in p2


def test_operator_disabled_bundled_skill_keeps_the_refusal(tmp_path, monkeypatch):
    """An operator skills.disabled bundled skill reports as missing but stays disabled after the
    restore, so the worker keeps the refusal: the disable is the operator's choice to honor."""
    from gateway import session_kanban
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    probes, ensured = [], []

    def probe(identifiers, task_id=None, **kwargs):
        probes.append(list(identifiers))
        return '', [], ['sdlc-review']  # disabled_as_missing: stays missing after any restore

    monkeypatch.setattr('agent.skill_commands.build_preloaded_skills_prompt', probe)
    monkeypatch.setattr('tools.skills_sync_bundled_ops.ensure_bundled_skill',
                        lambda name: ensured.append(name) or {'ok': True, 'action': 'present',
                                                                'path': None, 'message': ''})
    context = {'skills': ['sdlc-review'], 'context': '', 'goal_mode': False, 'task_id': 't1',
               'run_id': 1, 'claim_lock': 'lock', 'db': 'unused'}
    with pytest.raises(ValueError, match='kanban_skills_unavailable'):
        session_kanban._run_task_turns(SimpleNamespace(session_id='s1'), {'text': 'review it'}, [], context)
    assert ensured == ['sdlc-review'] and len(probes) == 2
