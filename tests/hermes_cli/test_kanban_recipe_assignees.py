"""Native assignment through the real CLI, with optional profile aliases."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli.kanban_recipes import RecipeError, prepare_definition
from hermes_cli.kanban_recipes_bindings import normalize_bindings, resolve_plan
from hermes_cli.kanban_recipes_store import instantiate


@pytest.fixture
def assignment_env(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    profile = home / 'profiles' / 'analyst'
    profile.mkdir(parents=True)
    (profile / 'config.yaml').write_text('{}')
    for name in tuple(os.environ):
        if name.startswith('HERMES_KANBAN_'):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    with kb.scoped_current_board(None):
        yield home


@pytest.mark.parametrize('aliases', [None, {}, {'researcher': 'analyst', 'writer': 'analyst'}])
def test_real_cli_assignment_freezes_native_profiles_and_replays(assignment_env, aliases):
    home = assignment_env
    root = Path(__file__).resolve().parents[2]
    env = {key: os.environ[key] for key in ('PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC')
           if key in os.environ}
    env.update(HOME=str(home.parent), USERPROFILE=str(home.parent), HERMES_HOME=str(home),
               HERMES_KANBAN_HOME=str(home), PYTHONPATH=str(root), TZ='UTC', LANG='C.UTF-8')
    references = ['researcher', 'writer'] if aliases else ['analyst', 'analyst']
    definition = {'schema_version': 1, 'recipe_id': 'assignment', 'nodes': [
        {'key': 'first', 'assignee': references[0], 'title': 'First'},
        {'key': 'second', 'assignee': references[1], 'title': 'Second', 'needs': ['first']},
        {'key': 'direct', 'assignee': 'default', 'title': 'Direct'}]}
    source = home.parent / 'recipe.json'
    source.write_text(json.dumps(definition))
    options = [source]
    bindings = home.parent / 'bindings.json'
    if aliases is not None:
        bindings.write_text(json.dumps({'profiles': aliases}))
        options += ['--bindings', bindings]

    def cli(*args, code=0):
        process = subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'kanban',
                                  'recipe', *map(str, args), '--json'],
                                 cwd=root, env=env, capture_output=True, text=True, timeout=45)
        assert process.returncode == code, process.stdout + process.stderr
        return json.loads(process.stdout)

    validated = cli('validate', *options)
    assert not (home / 'kanban.db').exists()
    first = cli('run', *options, '--key', 'same')
    shown = cli('show', first['instance_id'])
    assert shown['definition'] == definition
    assert shown['effective_plan'] == validated['effective_plan']
    assert shown['bindings'] == {'profiles': aliases or {}}
    expected = {'first': 'analyst', 'second': 'analyst', 'direct': 'default'}
    assert {key: task['assignee'] for key, task in shown['current_tasks'].items()} == expected
    assert {node['key']: node['assignee'] for node in shown['effective_plan']['nodes']} == expected
    exported = home.parent / 'export.json'
    cli('export', first['instance_id'], '--output', exported)
    assert json.loads(exported.read_text()) == definition
    assert cli('run', *options, '--key', 'same') == dict(first, replayed=True)
    if aliases:
        bindings.write_text(json.dumps({'profiles': dict(aliases, researcher='default')}))
        assert cli('run', *options, '--key', 'same', code=3)['error']['code'] == 'IDEMPOTENCY_CONFLICT'
        bindings.write_text(json.dumps({'profiles': aliases}))
    profile = home / 'profiles' / 'analyst'
    (profile / 'config.yaml').unlink()
    profile.rmdir()
    assert cli('run', *options, '--key', 'same') == dict(first, replayed=True)
    assert cli('run', *options, '--key', 'new', code=2)['error']['code'] == 'BINDING_INVALID'
    with sqlite3.connect(home / 'kanban.db') as conn:
        assert conn.execute('SELECT count(*) FROM tasks').fetchone()[0] == len(expected)
        assert conn.execute('SELECT count(*) FROM recipe_instances').fetchone()[0] == 1


@pytest.mark.parametrize('problem', ['missing', 'null', 'number', 'list', 'invalid-name',
    'uninstalled', 'map-null', 'map-list', 'target-null', 'target-number', 'target-list',
    'target-invalid', 'target-uninstalled', 'unused', 'unknown-definition', 'unknown-node',
    'unknown-binding', 'deleted-after-preflight', 'boolean', 'target-boolean', 'reserved'])
def test_invalid_assignment_never_publishes_partial_graph(assignment_env, problem):
    definition = {'schema_version': 1, 'recipe_id': 'assignment', 'nodes': [
        {'key': 'valid', 'assignee': 'default', 'title': 'Valid'},
        {'key': 'subject', 'assignee': 'analyst', 'title': 'Subject'}]}
    bindings = {}
    node = definition['nodes'][1]
    bad_values = {'null': None, 'number': 1, 'list': [], 'invalid-name': '../escape',
                  'uninstalled': 'not-installed', 'boolean': True, 'reserved': 'hermes'}
    if problem == 'missing':
        del node['assignee']
    elif problem in bad_values:
        node['assignee'] = bad_values[problem]
    elif problem.startswith('map-'):
        bindings['profiles'] = None if problem == 'map-null' else []
    elif problem.startswith('target-'):
        value = {'null': None, 'number': 1, 'list': [], 'invalid': '../escape',
                 'uninstalled': 'not-installed', 'boolean': True}[problem.removeprefix('target-')]
        bindings['profiles'] = {'analyst': value}
    elif problem == 'unused':
        bindings['profiles'] = {'unused': 'default'}
    elif problem.startswith('unknown-'):
        target = {'definition': definition, 'node': node, 'binding': bindings}[problem.removeprefix('unknown-')]
        target['obsolete_assignment'] = 'default'
    with kbc.connect_closing() as conn:
        with pytest.raises(RecipeError) as error:
            prepared = prepare_definition(definition, {})
            if problem == 'deleted-after-preflight':
                prepared['effective_plan'] = resolve_plan(prepared, normalize_bindings(prepared, bindings), 'default')
                profile = assignment_env / 'profiles' / 'analyst'
                (profile / 'config.yaml').unlink()
                profile.rmdir()
            instantiate(conn, prepared, bindings, 'invalid', board='default')
        if problem.startswith('unknown-'):
            assert error.value.message == 'Unknown field'
        for table in ('tasks', 'task_links', 'task_events', 'recipe_instances', 'recipe_instance_tasks'):
            assert conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] == 0


def test_exact_alias_overrides_installed_name_without_chaining(assignment_env):
    definition = {'schema_version': 1, 'recipe_id': 'override', 'nodes': [
        {'key': 'first', 'assignee': 'analyst', 'title': 'First'},
        {'key': 'second', 'assignee': 'default', 'title': 'Second'}]}
    prepared = prepare_definition(definition, {})
    with kbc.connect_closing() as conn:
        direct = instantiate(conn, prepared, {}, 'direct')
        aliases = {'profiles': {'analyst': 'default', 'default': 'analyst'}}
        mapped = instantiate(conn, prepared, aliases, 'mapped')
        assert kb.get_task(conn, mapped['tasks']['first']).assignee == 'default'
        assert kb.get_task(conn, mapped['tasks']['second']).assignee == 'analyst'
        assert kb.get_task(conn, direct['tasks']['first']).assignee == 'analyst'
        with pytest.raises(RecipeError) as error:
            instantiate(conn, prepared, aliases, 'direct')
        assert error.value.code == 'IDEMPOTENCY_CONFLICT'
