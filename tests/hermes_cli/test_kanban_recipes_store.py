"""Durable recipe receipts compose native task construction, never a second DAG."""
import hashlib
import importlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_recipes import RecipeError, canonical, prepare_definition


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    for key in ('HERMES_KANBAN_DB', 'HERMES_KANBAN_BOARD', 'HERMES_KANBAN_TASK',
                'HERMES_DELEGATE_DEPTH', 'HERMES_DELEGATE_TASK_ID'):
        monkeypatch.delenv(key, raising=False)
    return home


@pytest.fixture
def conn(env):
    c = kbc.connect(board='default')
    yield c
    c.close()


def api():
    # Assertion, not collection failure, makes the initial missing-feature RED explicit.
    assert importlib.util.find_spec('hermes_cli.kanban_recipes_store'), 'recipe persistence is missing'
    assert importlib.util.find_spec('hermes_cli.kanban_recipes_bindings'), 'recipe resolver is missing'
    return (importlib.import_module('hermes_cli.kanban_recipes_store'),
            importlib.import_module('hermes_cli.kanban_recipes_bindings'))


def prepared(task=None):
    return prepare_definition({'schema_version': 1, 'recipe_id': 'sample', 'roles': ['builder'],
        'nodes': [{'key': 'child', 'role': 'builder', 'title': '  Child  ', 'needs': ['root'],
                   'task': task or {}},
                  {'key': 'root', 'role': 'builder', 'title': 'Root', 'task': task or {}}]}, {})


BINDINGS = {'roles': {'builder': 'default'}}


def test_receipt_hashes_native_topology_immutable_replay(conn, monkeypatch):
    store, resolver = api()
    p = prepared({'goal_mode': True, 'model': '  model  ', 'provider': ' provider ',
                  'priority': -1, 'max_runtime_seconds': 1, 'max_retries': 1,
                  'skills': ['github-code-review']})
    result = store.instantiate(conn, p, BINDINGS, 'request', board='default')
    assert result['replayed'] is False
    tasks = result['tasks']
    root, child = [kb.get_task(conn, tasks[key]) for key in ('root', 'child')]
    assert (root.status, child.status) == ('ready', 'todo')
    assert child.title == 'Child'
    assert (child.model_override, child.provider_override) == ('model', 'provider')
    assert child.max_runtime_seconds == child.max_retries == 1
    assert child.goal_max_turns > 0 and child.goal_mode
    assert child.idempotency_key is None
    assert conn.execute('SELECT parent_id,child_id FROM task_links').fetchall()[0][:] == (root.id, child.id)
    shown = store.show_instance(conn, result['instance_id'])
    request = {k: shown[k] for k in ('definition_digest', 'inputs', 'bindings', 'effective_plan')}
    assert hashlib.sha256(canonical(request).encode()).hexdigest() == result['request_digest']
    assert shown['effective_plan']['nodes'][0]['title'] == child.title
    assert shown['definition'] == p['definition']
    conn.execute("UPDATE tasks SET title='edited',status='archived'")
    monkeypatch.setattr(resolver, 'resolve_plan', lambda *a: pytest.fail('replay resolved ambient state'))
    replay = store.instantiate(conn, p, {'roles': {'builder': ' DEFAULT '}}, 'request')
    assert replay == dict(result, replayed=True)
    assert store.show_instance(conn, result['instance_id'])['request_digest'] == result['request_digest']
    with pytest.raises(RecipeError) as e:
        store.instantiate(conn, p, dict(BINDINGS, tenant='other'), 'request')
    assert e.value.code == 'IDEMPOTENCY_CONFLICT'


def test_atomic_failure_leaves_no_receipt_tasks_links_or_events(conn, monkeypatch):
    store, _ = api()
    original = kb.create_task
    calls = []
    reader = sqlite3.connect(kb.kanban_db_path('default'))
    def fail_second(c, **kwargs):
        assert reader.execute('SELECT count(*) FROM tasks').fetchone()[0] == 0
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError('injected')
        return original(c, **kwargs)
    monkeypatch.setattr(kb, 'create_task', fail_second)
    with pytest.raises(RuntimeError, match='injected'):
        store.instantiate(conn, prepared(), BINDINGS, 'rollback')
    for table in ('tasks', 'task_links', 'task_events', 'recipe_instances', 'recipe_instance_tasks'):
        assert conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] == 0
    reader.close()


@pytest.mark.parametrize('key', [None, '', 'a b', '\n', 'é', 'x' * 129, 1])
def test_key_validation_before_writes(conn, key):
    store, _ = api()
    with pytest.raises(RecipeError) as e:
        store.instantiate(conn, prepared(), BINDINGS, key)
    assert e.value.code == 'RECIPE_INVALID'
    assert e.value.path == '/key'
    assert conn.execute('SELECT count(*) FROM tasks').fetchone()[0] == 0


@pytest.mark.parametrize('bindings', [{}, {'roles': {}}, {'roles': {'builder': 2}},
    {'roles': {'builder': '../escape'}}, {'roles': {'builder': 'default', 'other': 'default'}},
    dict(BINDINGS, unknown=True), dict(BINDINGS, project=None), dict(BINDINGS, tenant=2)])
def test_bindings_shape_is_pure(env, bindings):
    _, resolver = api()
    with pytest.raises(RecipeError) as e:
        resolver.normalize_bindings(prepared(), bindings)
    assert e.value.code == 'BINDING_INVALID'


def test_normalization_does_not_check_existence_and_resolution_does(env):
    _, resolver = api()
    p = prepared()
    b = resolver.normalize_bindings(p, {'roles': {'builder': ' Uninstalled '}, 'project': 'Case-Slug'})
    assert b['roles']['builder'] == 'uninstalled'
    assert b['project'] == 'Case-Slug'
    before = set(env.rglob('*'))
    with pytest.raises(RecipeError) as e:
        resolver.resolve_plan(p, b, 'default')
    assert e.value.code == 'BINDING_INVALID'
    assert set(env.rglob('*')) == before


@pytest.mark.parametrize('damage', ['membership', 'task', 'imported'])
def test_damage_and_import_never_reconstruct(conn, damage):
    store, _ = api()
    p = prepared()
    result = store.instantiate(conn, p, BINDINGS, 'same')
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute('DELETE FROM tasks WHERE id=?', (result['tasks']['root'],))
    if damage == 'membership':
        conn.execute('DELETE FROM recipe_instance_tasks WHERE node_key=?', ('root',))
    elif damage == 'task':
        conn.execute('PRAGMA foreign_keys=OFF')
        conn.execute('DELETE FROM tasks WHERE id=?', (result['tasks']['root'],))
        conn.execute('PRAGMA foreign_keys=ON')
    else:
        conn.execute('UPDATE recipe_instances SET imported=1')
    before = conn.total_changes
    with pytest.raises(RecipeError) as e:
        store.instantiate(conn, p, BINDINGS, 'same')
    assert e.value.code == ('IMPORTED_INSTANCE_REBIND_REQUIRED' if damage == 'imported' else 'INSTANCE_DAMAGED')
    assert conn.total_changes == before


def test_project_resolution_frozen_fresh_paths_no_registry_reopen(conn, env, monkeypatch):
    store, resolver = api()
    from hermes_cli import projects_db as pdb
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name='Demo', primary_path=str(env / 'repo'))
    kb.write_board_metadata('default', project_id=pid, default_workdir='/never-share')
    p = prepared()
    b = resolver.normalize_bindings(p, BINDINGS)
    plan = resolver.resolve_plan(p, b, 'default')
    assert plan['project'] == {'id': pid, 'slug': 'demo', 'repo': str(env / 'repo')}
    monkeypatch.setattr(pdb, 'connect', lambda *a, **kw: pytest.fail('project validation migrated'))
    result = store.instantiate(conn, p, BINDINGS, 'first')
    second = store.instantiate(conn, p, BINDINGS, 'second')
    tasks = [kb.get_task(conn, tid) for r in (result, second) for tid in r['tasks'].values()]
    assert len({t.workspace_path for t in tasks}) == 4
    assert len({t.branch_name for t in tasks}) == 4
    assert all(t.workspace_path == str(env / 'repo' / '.worktrees' / t.id) for t in tasks)
    assert all(t.project_id == pid and t.workspace_kind == 'worktree' for t in tasks)
    assert not (env / 'repo').exists()
    (env / 'projects.db').unlink()
    kb.write_board_metadata('default', project_id='missing')
    assert store.instantiate(conn, p, BINDINGS, 'first')['tasks'] == result['tasks']


def test_native_goal_default_frozen_and_invalid_default_rejected(conn, monkeypatch):
    store, _ = api()
    from hermes_cli import goals
    monkeypatch.setattr(goals, 'DEFAULT_MAX_TURNS', 37)
    p = prepared({'goal_mode': True})
    r = store.instantiate(conn, p, BINDINGS, 'goal')
    assert all(kb.get_task(conn, t).goal_max_turns == 37 for t in r['tasks'].values())
    monkeypatch.setattr(goals, 'DEFAULT_MAX_TURNS', True)
    assert store.instantiate(conn, p, BINDINGS, 'goal')['tasks'] == r['tasks']
    with pytest.raises(RecipeError) as e:
        store.instantiate(conn, p, BINDINGS, 'invalid-default')
    assert e.value.code == 'BINDING_INVALID'
    plain = store.instantiate(conn, prepared(), BINDINGS, 'plain')
    assert all(kb.get_task(conn, t).goal_max_turns is None for t in plain['tasks'].values())


def test_migration_is_composable_and_idempotent(conn):
    store, _ = api()
    conn.execute('DROP TABLE recipe_instance_tasks')
    conn.execute('DROP TABLE recipe_instances')
    conn.execute('BEGIN IMMEDIATE')
    store.migrate_recipe_tables(conn)
    store.migrate_recipe_tables(conn)
    assert conn.in_transaction
    conn.execute('ROLLBACK')
    assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='recipe_instances'").fetchone()
    kbc._migrate_add_optional_columns(conn)
    assert conn.execute('SELECT count(*) FROM recipe_instances').fetchone()[0] == 0


def test_standalone_task_key_lookup_is_under_lock(conn):
    queries = []
    conn.set_trace_callback(queries.append)
    first = kb.create_task(conn, title='first', idempotency_key='native')
    assert kb.create_task(conn, title='second', idempotency_key='native') == first
    begins = [i for i, sql in enumerate(queries) if sql == 'BEGIN IMMEDIATE']
    lookups = [i for i, sql in enumerate(queries) if sql.startswith('SELECT id FROM tasks WHERE idempotency_key')]
    assert len(begins) == len(lookups) == 2
    assert all(begin < lookup for begin, lookup in zip(begins, lookups))


def test_native_creator_origin_without_dependency_and_frozen_context(conn, monkeypatch):
    store, resolver = api()
    creator = kb.create_task(conn, title='Creator', session_id='origin')
    conn.execute("INSERT INTO kanban_notify_subs(task_id,platform,chat_id,created_at) VALUES (?,?,?,?)",
                 (creator, 'telegram', 'fixture', 1))
    original = resolver.resolve_plan
    def freeze_then_drift(*args):
        plan = original(*args)
        monkeypatch.setattr(kb, '_board_meta_for', lambda *a: pytest.fail('second default lookup'))
        return plan
    monkeypatch.setattr(resolver, 'resolve_plan', freeze_then_drift)
    result = store.instantiate(conn, prepared(), BINDINGS, 'origin', creator_task_id=creator)
    for task_id in result['tasks'].values():
        task = kb.get_task(conn, task_id)
        assert task.session_id == 'origin'
        assert task.workspace_kind == 'scratch' and task.project_id is None
        assert conn.execute('SELECT chat_id FROM kanban_notify_subs WHERE task_id=?', (task_id,)).fetchone()[0] == 'fixture'
    assert not conn.execute('SELECT 1 FROM task_links WHERE parent_id=?', (creator,)).fetchone()


def test_readonly_resolve_never_creates_project_registry(env):
    _, resolver = api()
    p = prepared({'workspace_kind': 'worktree'})
    before = set(env.rglob('*'))
    with pytest.raises(RecipeError) as e:
        resolver.resolve_plan(p, resolver.normalize_bindings(p, dict(BINDINGS, project='missing')), 'default')
    assert e.value.code == 'BINDING_INVALID'
    assert set(env.rglob('*')) == before


def test_preflight_plan_freezes_defaults_before_opening_writable_db(conn, monkeypatch):
    store, resolver = api()
    p = prepared()
    plan = resolver.resolve_plan(p, resolver.normalize_bindings(p, BINDINGS), 'default')
    p['effective_plan'] = plan
    monkeypatch.setattr(resolver, 'resolve_plan', lambda *a: pytest.fail('preflight plan re-resolved'))
    result = store.instantiate(conn, p, BINDINGS, 'preflight')
    assert store.show_instance(conn, result['instance_id'])['effective_plan'] == plan


def test_postcommit_invariant_error_preserves_recoverable_receipt(conn, monkeypatch):
    store, _ = api()
    def fail_check(_):
        raise sqlite3.DatabaseError('injected postcommit invariant')
    with monkeypatch.context() as patcher:
        patcher.setattr(kbc, '_check_file_length_invariant', fail_check)
        with pytest.raises(sqlite3.DatabaseError):
            store.instantiate(conn, prepared(), BINDINGS, 'uncertain')
    reader = sqlite3.connect(kb.kanban_db_path('default'))
    assert reader.execute('SELECT count(*) FROM recipe_instances').fetchone()[0] == 1
    assert reader.execute('SELECT count(*) FROM tasks').fetchone()[0] == 2
    reader.close()
    replay = store.instantiate(conn, prepared(), BINDINGS, 'uncertain')
    assert replay['replayed'] and len(replay['tasks']) == 2


_PROCESS_SCRIPT = '''
import json, pathlib, sys, time
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli.kanban_recipes import prepare_definition
from hermes_cli.kanban_recipes_store import instantiate
db, mode, marker, definition = sys.argv[1:]
conn = kbc.connect(pathlib.Path(db))
original = kb.create_task
def pause_during(c, **kwargs):
    result = original(c, **kwargs)
    pathlib.Path(marker).touch()
    time.sleep(60)
    return result
if mode == 'before':
    kb.create_task = pause_during
if mode == 'native':
    result = {'task': kb.create_task(conn, title='native', idempotency_key='race')}
else:
    result = instantiate(conn, prepare_definition(json.loads(definition), {}),
                         {'roles': {'builder': 'default'}}, 'race', board='default')
if mode == 'after':
    pathlib.Path(marker).touch()
    time.sleep(60)
print(json.dumps(result), flush=True)
conn.close()
'''


def _start_creator(env, mode, marker):
    child_env = os.environ.copy()
    child_env['HOME'] = str(env.parent)
    return subprocess.Popen([sys.executable, '-c', _PROCESS_SCRIPT,
                             str(kb.kanban_db_path('default')), mode, str(marker),
                             json.dumps(prepared()['definition'])],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=child_env)


@pytest.mark.parametrize('mode', ['recipe', 'native'])
def test_separate_process_same_key_race(conn, env, mode):
    api()
    processes = [_start_creator(env, mode, env / 'unused') for _ in range(4)]
    try:
        results = []
        for proc in processes:
            stdout, stderr = proc.communicate(timeout=30)
            assert proc.returncode == 0, stderr
            results.append(json.loads(stdout))
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    if mode == 'native':
        assert len({r['task'] for r in results}) == 1
        assert conn.execute('SELECT count(*) FROM tasks').fetchone()[0] == 1
    else:
        assert len({r['instance_id'] for r in results}) == 1
        assert sum(not r['replayed'] for r in results) == 1
        assert all(r['tasks'] == results[0]['tasks'] for r in results)
        assert conn.execute('SELECT count(*) FROM tasks').fetchone()[0] == 2
        assert conn.execute('SELECT count(*) FROM task_links').fetchone()[0] == 1


@pytest.mark.parametrize('mode,visible', [('before', 0), ('after', 2)])
def test_killed_creator_is_atomic_and_replayable(conn, env, mode, visible):
    store, _ = api()
    marker = env / 'creator-reached'
    proc = _start_creator(env, mode, marker)
    try:
        deadline = time.monotonic() + 20
        while not marker.exists() and time.monotonic() < deadline and proc.poll() is None:
            time.sleep(.02)
        assert marker.exists()
        assert conn.execute('SELECT count(*) FROM tasks').fetchone()[0] == visible
    finally:
        proc.kill()
        proc.communicate(timeout=10)
    original = conn.execute('SELECT id FROM tasks ORDER BY id').fetchall()
    result = store.instantiate(conn, prepared(), BINDINGS, 'race')
    assert result['replayed'] == (mode == 'after')
    if mode == 'after':
        assert sorted(result['tasks'].values()) == [r[0] for r in original]
    assert conn.execute('SELECT count(*) FROM tasks').fetchone()[0] == 2
    assert conn.execute('SELECT count(*) FROM task_links').fetchone()[0] == 1
