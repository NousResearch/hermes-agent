"""Composed CLI -> SQLite -> canonical dispatcher -> local worker acceptance.

No LLM, gateway, real profiles, network, or production monkeypatches. CLI
subprocesses call the shipped parser/kanban_command, not the store directly.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc


ROOT = Path(__file__).resolve().parents[2]
PROCESS = Path(__file__).parent / 'fixtures' / 'recipe_process.py'
TABLES = ('tasks', 'task_links', 'task_events', 'task_runs',
          'recipe_instances', 'recipe_instance_tasks')


def wait_for(path, process):
    deadline = time.monotonic() + 25
    while not path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f'process exited {process.returncode}: {stdout}\n{stderr}')
        if time.monotonic() >= deadline:
            pytest.fail(f'process did not publish {path.name}')
        time.sleep(0.01)


class Runtime:
    def __init__(self, directory):
        self.directory = directory
        self.home = directory / '.hermes'
        self.home.mkdir()
        (self.home / 'config.yaml').write_text('kanban:\n  review_dispatch: true\n')
        reviewer = self.home / 'profiles' / 'reviewer'
        reviewer.mkdir(parents=True)
        (reviewer / 'config.yaml').write_text('{}')
        # Allowlist rather than trying to enumerate every inherited credential,
        # delegated-worker marker, board pin, proxy, or profile environment key.
        self.env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC')
                    if k in os.environ}
        tmux_home = directory / 'tmux'
        tmux_home.mkdir()
        self.env.update(HOME=str(directory), USERPROFILE=str(directory),
                        TMUX_TMPDIR=str(tmux_home),
                        HERMES_HOME=str(self.home), HERMES_KANBAN_HOME=str(self.home),
                        PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1',
                        LANG='C.UTF-8', TZ='UTC', GIT_CONFIG_NOSYSTEM='1',
                        GIT_CONFIG_GLOBAL=os.devnull)
        self.processes = []
        self.serial = 0
        self.db = None

    def path(self, suffix):
        self.serial += 1
        return self.directory / f'{self.serial}-{suffix}'

    def start(self, mode, options=None, words=()):
        process = subprocess.Popen(
            [sys.executable, str(PROCESS), mode, json.dumps(options or {}), *map(str, words)],
            cwd=ROOT, env=self.env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(process)
        return process

    def finish(self, process, code=0):
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == code, (stdout, stderr)
        return json.loads(stdout or stderr)

    def cli(self, words, code=0):
        return self.finish(self.start('cli', words=words), code)

    def observer(self):
        ready = self.path('observer-ready')
        process = self.start('observer', {'db': str(self.db), 'ready': str(ready)})
        wait_for(ready, process)
        return process

    def command(self, process, action, **kwargs):
        output = self.path(action + '.json')
        process.stdin.write(json.dumps({'action': action, 'output': str(output), **kwargs}) + '\n')
        process.stdin.flush()
        return output

    def receive(self, process, path):
        wait_for(path, process)
        return json.loads(path.read_text())

    def snapshot(self, process):
        return self.receive(process, self.command(process, 'snapshot'))

    def tick(self, process, **kwargs):
        return self.receive(process, self.command(process, 'dispatch', **kwargs))

    def stop(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    directory = tmp_path / 'runtime'
    directory.mkdir()
    rt = Runtime(directory)
    # In-process native setup uses the same isolated roots as the subprocesses.
    for name in tuple(os.environ):
        if name.startswith(('HERMES_', 'TERMINAL_')) or any(
                token in name for token in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            monkeypatch.delenv(name, raising=False)
    for key, value in rt.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(Path, 'home', lambda: directory)
    with kb.scoped_current_board(None):
        rt.db = kb.kanban_db_path('default')
        with kbc.connect_closing(board='default'):
            pass
        try:
            yield rt
        finally:
            rt.stop()


def recipe(runtime, *, worktree=False):
    task = {'workspace_kind': 'worktree' if worktree else 'scratch',
            'model': 'fixture-model', 'provider': 'fixture-provider',
            'reasoning_effort': 'high', 'skills': ['github-code-review'],
            'goal_mode': True, 'goal_max_turns': 7, 'max_runtime_seconds': 120,
            'max_retries': 2}
    # Intentionally not topologically ordered. Two roots fan into one join.
    definition = {'schema_version': 1, 'recipe_id': 'composed', 'roles': ['builder'],
                  'inputs': {'topic': {'type': 'string', 'required': True}},
                  'nodes': [
                      {'key': 'join', 'role': 'builder', 'title': 'Join',
                       'needs': ['left', 'right'], 'task': task},
                      {'key': 'left', 'role': 'builder', 'title': 'Left {{input.topic}}',
                       'task': task},
                      {'key': 'right', 'role': 'builder', 'title': 'Right', 'task': task}]}
    paths = [runtime.path(n + '.json') for n in ('recipe', 'inputs', 'bindings')]
    bindings = {'roles': {'builder': 'default'}}
    if worktree:
        bindings['project'] = 'fixture'
    for path, value in zip(paths, (definition, {'topic': 'isolated'}, bindings)):
        path.write_text(json.dumps(value))
    return paths


def run_args(paths, key='same'):
    definition, inputs, bindings = paths
    return ['recipe', 'run', definition, '--inputs', inputs, '--bindings', bindings,
            '--key', key, '--json']


def assert_empty(snapshot):
    assert {name: len(snapshot[name]) for name in TABLES} == dict.fromkeys(TABLES, 0)


def assert_graph(snapshot, results):
    mappings = [r['tasks'] for r in results]
    ids = {tid for mapping in mappings for tid in mapping.values()}
    assert len(ids) == 3 * len(results)
    assert {row['id'] for row in snapshot['tasks']} == ids
    assert len(snapshot['recipe_instances']) == len(results)
    assert len(snapshot['recipe_instance_tasks']) == len(ids)
    assert {(row['instance_id'], row['node_key'], row['task_id'])
            for row in snapshot['recipe_instance_tasks']} == {
                (result['instance_id'], key, tid) for result in results
                for key, tid in result['tasks'].items()}
    expected_edges = {(mapping[parent], mapping['join']) for mapping in mappings
                      for parent in ('left', 'right')}
    assert {(row['parent_id'], row['child_id']) for row in snapshot['task_links']} == expected_edges
    assert len(snapshot['task_links']) == len(expected_edges)
    for kind in ('created', 'recipe_instantiated'):
        events = [row for row in snapshot['task_events'] if row['kind'] == kind]
        assert len(events) == len(ids)
        assert {row['task_id'] for row in events} == ids


def test_concurrent_cli_same_key_distinct_invocation_and_invalid_no_partial(runtime):
    paths = recipe(runtime)
    observer = runtime.observer()
    barrier = runtime.path('start')
    processes = []
    for _ in range(4):
        ready = runtime.path('ready')
        process = runtime.start('cli', {'start': str(barrier), 'ready': str(ready)}, run_args(paths))
        processes.append((process, ready))
    for process, ready in processes:
        wait_for(ready, process)
    barrier.touch()
    results = [runtime.finish(process) for process, _ in processes]
    assert len({r['instance_id'] for r in results}) == 1
    assert sum(not r['replayed'] for r in results) == 1
    assert all(r == dict(results[0], replayed=r['replayed']) for r in results)
    assert_graph(runtime.snapshot(observer), results[:1])
    distinct = runtime.cli(run_args(paths, 'distinct'))
    before = runtime.snapshot(observer)
    assert_graph(before, [results[0], distinct])
    paths[1].write_text('{"topic": false}')
    error = runtime.cli(run_args(paths, 'bad'), code=2)
    assert error['error']['code'] == 'RECIPE_INVALID'
    assert runtime.snapshot(observer) == before
    paths[1].write_text('{"topic": "changed"}')
    conflict = runtime.cli(run_args(paths), code=3)
    assert conflict['error']['code'] == 'IDEMPOTENCY_CONFLICT'
    assert runtime.snapshot(observer) == before


@pytest.mark.parametrize('mutate_outer_transaction', [False, True], ids=['atomic', 'mutation-detected'])
def test_process_reader_and_dispatcher_observe_only_committed_graph(runtime, mutate_outer_transaction):
    paths = recipe(runtime)
    reader = runtime.observer()
    dispatcher = runtime.observer()
    paused, release = runtime.path('paused'), runtime.path('release')
    creator = runtime.start('cli', {'pause': 'before', 'paused': str(paused),
                                    'release': str(release),
                                    'mutate_outer_transaction': mutate_outer_transaction}, run_args(paths))
    wait_for(paused, creator)
    visible = runtime.snapshot(reader)
    if mutate_outer_transaction:
        # Same zero-visibility oracle MUST fail when ONLY the outer recipe txn
        # is removed; this mutation never edits a module on disk.
        with pytest.raises(AssertionError):
            assert_empty(visible)
        assert len(visible['tasks']) == 1
        assert len(visible['recipe_instances']) == 1
        assert not visible['recipe_instance_tasks'] and not visible['task_links']
        unsafe = runtime.tick(dispatcher)
        assert len(unsafe['records']) == 1
        assert len(unsafe['records'][0]['visible_at_spawn']['tasks']) == 1
        assert not unsafe['records'][0]['visible_at_spawn']['recipe_instance_tasks']
        print('OUTER-TXN MUTATION CAUGHT: reader and real dispatcher saw 1/3 tasks')
        creator.kill()
        creator.communicate(timeout=5)
        return
    assert_empty(visible)
    assert_empty(runtime.snapshot(dispatcher))
    output = runtime.command(dispatcher, 'dispatch')
    wait_for(Path(str(output) + '.entered'), dispatcher)
    # Dispatcher is blocked at its native write boundary, not at connection init.
    assert_empty(runtime.snapshot(reader))
    release.touch()
    result = runtime.finish(creator)
    tick = runtime.receive(dispatcher, output)
    assert_empty(tick['before'])
    assert_graph(tick['claimed'], [result])
    assert {r['task'] for r in tick['records']} == {result['tasks'][key] for key in ('left', 'right')}
    for record in tick['records']:
        assert_graph(record['visible_at_spawn'], [result])
    assert all(row['status'] == 'done' for row in tick['after']['tasks']
               if row['id'] != result['tasks']['join'])
    join = runtime.tick(dispatcher)
    assert [w['task'] for w in join['workers']] == [result['tasks']['join']]
    assert {row['status'] for row in join['after']['tasks']} == {'done'}


@pytest.mark.parametrize('pause', ['before', 'after'])
def test_killed_cli_before_commit_or_before_stdout_recovers_by_replay(runtime, pause):
    paths = recipe(runtime)
    observer = runtime.observer()
    paused, release = runtime.path('paused'), runtime.path('release')
    creator = runtime.start('cli', {'pause': pause, 'paused': str(paused), 'release': str(release)}, run_args(paths))
    wait_for(paused, creator)
    before = runtime.snapshot(observer)
    if pause == 'before':
        assert_empty(before)
    else:
        assert len(before['tasks']) == 3
    creator.kill()
    stdout, _ = creator.communicate(timeout=5)
    assert stdout == ''
    result = runtime.cli(run_args(paths))
    assert result['replayed'] == (pause == 'after')
    after = runtime.snapshot(observer)
    assert_graph(after, [result])
    if pause == 'after':
        assert after == before
    assert runtime.cli(run_args(paths)) == dict(result, replayed=True)


def test_same_card_review_changes_new_run_rereview_and_join(runtime):
    paths = recipe(runtime)
    result = runtime.cli(run_args(paths))
    ids = result['tasks']
    observer = runtime.observer()
    first = runtime.tick(observer, actions={ids['left']: 'review'})
    assert {r['task'] for r in first['records']} == {ids['left'], ids['right']}
    left = next(row for row in first['after']['tasks'] if row['id'] == ids['left'])
    assert (left['status'], left['assignee']) == ('review', 'reviewer')
    review = runtime.tick(observer, actions={ids['left']: 'changes'})
    assert [r['assignee'] for r in review['records']] == ['reviewer']
    assert 'sdlc-review' in review['records'][0]['skills']
    left = next(row for row in review['after']['tasks'] if row['id'] == ids['left'])
    assert (left['status'], left['assignee']) == ('ready', 'default')
    correction = runtime.tick(observer, actions={ids['left']: 'rereview'})
    assert [r['assignee'] for r in correction['records']] == ['default']
    accepted = runtime.tick(observer)
    assert [r['assignee'] for r in accepted['records']] == ['reviewer']
    runs = [row for row in accepted['after']['task_runs'] if row['task_id'] == ids['left']]
    assert [row['profile'] for row in runs] == ['default', 'reviewer', 'default', 'reviewer']
    assert [row['outcome'] for row in runs] == ['review_requested', 'changes_requested', 'review_requested', 'completed']
    assert len({row['id'] for row in runs}) == 4
    for tick in (first, review, correction, accepted):
        assert ids['join'] not in {r['task'] for r in tick['records']}
        for record in tick['records']:
            argv = record['argv']
            assert argv[argv.index('-p') + 1] == record['assignee']
            assert argv[argv.index('-m') + 1] == record['model'] == 'fixture-model'
            assert argv[argv.index('--provider') + 1] == record['provider'] == 'fixture-provider'
            assert argv[argv.index('--reasoning') + 1] == record['reasoning'] == 'high'
            assert 'github-code-review' in argv and '--cli' in argv and '-Q' in argv
            assert record['goal_mode'] and record['goal_max_turns'] == 7
            assert record['pid'] in {w['pid'] for w in tick['workers']}
            assert record['run'] in {w['run'] for w in tick['workers']}
    final = runtime.tick(observer)
    assert [r['task'] for r in final['records']] == [ids['join']]
    assert {row['status'] for row in final['after']['tasks']} == {'done'}
    assert_graph(final['after'], [result])


def test_worktree_project_materializes_unique_native_paths_across_invocations(runtime):
    from hermes_cli import projects_db
    repo = runtime.directory / 'repo'
    repo.mkdir()
    for words in (['init'], ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                            'commit', '--allow-empty', '-m', 'local fixture']):
        subprocess.run(['git', '-C', str(repo), *words], env=runtime.env,
                       check=True, capture_output=True, text=True, timeout=15)
    with projects_db.connect_closing() as conn:
        project = projects_db.create_project(conn, name='Fixture', primary_path=str(repo))
    paths = recipe(runtime, worktree=True)
    results = [runtime.cli(run_args(paths, key)) for key in ('first', 'second')]
    observer = runtime.observer()
    before = runtime.snapshot(observer)
    assert_graph(before, results)
    tasks = before['tasks']
    assert len({row['workspace_path'] for row in tasks}) == len(tasks)
    assert len({row['branch_name'] for row in tasks}) == len(tasks)
    assert all(row['project_id'] == project for row in tasks)
    assert all(not Path(row['workspace_path']).exists() for row in tasks)
    roots, joins = runtime.tick(observer), runtime.tick(observer)
    records = roots['records'] + joins['records']
    assert {row['task'] for row in records} == {row['id'] for row in tasks}
    # Native completion may prune a clean worktree. Capture actual git state
    # inside spawn_fn while the harmless child waits, not after cleanup.
    for row in tasks:
        record = next(r for r in records if r['task'] == row['id'])
        assert record['workspace'] == row['workspace_path']
        assert record['git_branch_at_spawn'] == row['branch_name']
    assert {row['status'] for row in joins['after']['tasks']} == {'done'}


def test_archived_parent_and_nonrecipe_board_keep_native_dispatch(runtime):
    observer = runtime.observer()
    with kbc.connect_closing(board='default') as conn:
        parent = kb.create_task(conn, title='Native parent', assignee='default')
        child = kb.create_task(conn, title='Native child', assignee='default', parents=[parent])
        external = kb.create_task(conn, title='External control lane', assignee='external-lane')
        assert kb.get_task(conn, child).status == 'todo'
        assert kb.archive_task(conn, parent)
        assert kb.get_task(conn, child).status == 'ready'
    native = runtime.tick(observer)
    assert [r['task'] for r in native['records']] == [child]
    assert not native['after']['recipe_instances'] and not native['after']['recipe_instance_tasks']
    assert next(row for row in native['after']['tasks'] if row['id'] == external)['status'] == 'ready'
    result = runtime.cli(run_args(recipe(runtime)))
    with kbc.connect_closing(board='default') as conn:
        for key in ('left', 'right'):
            assert kb.archive_task(conn, result['tasks'][key])
        assert kb.get_task(conn, result['tasks']['join']).status == 'ready'
    tick = runtime.tick(observer)
    assert [r['task'] for r in tick['records']] == [result['tasks']['join']]
    assert runtime.cli(['recipe', 'show', result['instance_id'], '--json'])['tasks'] == result['tasks']
