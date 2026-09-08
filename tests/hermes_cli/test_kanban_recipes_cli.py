"""Recipe command contracts across the shared CLI/slash boundary."""
import argparse
import json
import os
from pathlib import Path
import sqlite3

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith('HERMES_KANBAN_') or name == 'HERMES_DELEGATED_CHILD_CONTEXT':
            monkeypatch.delenv(name, raising=False)
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    with kb.scoped_current_board(None):
        yield home


def parse(*words):
    parser = argparse.ArgumentParser()
    cli.build_parser(parser.add_subparsers())
    return parser.parse_args(['kanban', *map(str, words)])


def invoke(capsys, *words):
    rc = cli.kanban_command(parse(*words))
    captured = capsys.readouterr()
    return rc, json.loads(captured.out or captured.err)


def files(home):
    return {str(p.relative_to(home)): p.read_bytes() for p in home.rglob('*') if p.is_file()}


@pytest.fixture
def recipe_files(tmp_path):
    definition = {'schema_version': 1, 'recipe_id': 'brief', 'roles': ['writer'],
                  'nodes': [{'key': 'draft', 'role': 'writer', 'title': 'Original {{input.topic}}'},
                            {'key': 'check', 'role': 'writer', 'title': 'Check', 'needs': ['draft']}],
                  'inputs': {'topic': {'type': 'string', 'required': True}}}
    values = [definition, {'topic': 'private-input'}, {'roles': {'writer': 'default'}}]
    paths = [tmp_path / name for name in ['recipe.json', 'inputs.json', 'bindings.json']]
    for path, value in zip(paths, values):
        path.write_text(json.dumps(value))
    return paths


def request(recipe_files, action='validate', key=None):
    definition, inputs, bindings = recipe_files
    args = ['recipe', action, definition, '--inputs', inputs, '--bindings', bindings, '--json']
    return args + (['--key', key] if key is not None else [])


def test_shared_parser_requires_run_key_and_export_output(isolated):
    for words in [('recipe', 'run', 'r.json'), ('recipe', 'export', 'ri_abc')]:
        with pytest.raises(SystemExit) as exc:
            parse(*words)
        assert exc.value.code == 2
    args = parse('recipe', 'run', 'r.json', '--key', 'once', '--inputs', 'i.json', '--bindings', 'b.json', '--json')
    assert args.key == 'once' and args.inputs == 'i.json' and args.bindings == 'b.json'


@pytest.mark.parametrize('action', ['validate', 'run'])
def test_invalid_definition_never_initializes_storage(isolated, recipe_files, capsys, action):
    recipe_files[0].write_text('{"secret-value": NaN}')
    before = files(isolated)
    rc, result = invoke(capsys, *request(recipe_files, action, 'once' if action == 'run' else None))
    assert rc == 2 and result['error']['code'] == 'RECIPE_INVALID'
    assert 'secret-value' not in json.dumps(result)
    assert files(isolated) == before


def test_validate_default_absent_db_and_slash_are_read_only(isolated, recipe_files, capsys):
    before = files(isolated)
    rc, result = invoke(capsys, *request(recipe_files))
    assert rc == 0 and result['definition_digest']
    assert result['effective_plan']
    assert files(isolated) == before
    import shlex
    slash = cli.run_slash(shlex.join(map(str, request(recipe_files))))
    assert json.loads(slash)['definition_digest'] == result['definition_digest']
    assert files(isolated) == before


@pytest.mark.parametrize('action', ['validate', 'run'])
def test_missing_profile_and_explicit_board_do_not_create_storage(isolated, recipe_files, capsys, action):
    args = request(recipe_files, action, 'once' if action == 'run' else None)
    before = files(isolated)
    rc, result = invoke(capsys, '--board', 'absent', *args)
    assert rc == 2 and result['error']['code'] == 'BINDING_INVALID'
    recipe_files[2].write_text('{"roles":{"writer":"missing-private-profile"}}')
    rc, result = invoke(capsys, *args)
    assert rc == 2 and result['error']['code'] == 'BINDING_INVALID'
    assert 'missing-private-profile' not in json.dumps(result)
    assert files(isolated) == before


@pytest.mark.parametrize('conflict', ['context-board', 'context-db', 'explicit-db', 'environment-board'])
def test_conflicting_board_context_is_typed_and_read_only(isolated, recipe_files, capsys, monkeypatch, conflict):
    kb.create_board('alpha')
    kb.create_board('beta')
    args = request(recipe_files)
    scope = None
    if conflict == 'context-board':
        scope = 'alpha'
        args = ['--board', 'beta', *args]
    elif conflict == 'context-db':
        scope = 'alpha'
        monkeypatch.setenv('HERMES_KANBAN_DB', str(kb.board_dir('beta') / 'kanban.db'))
    elif conflict == 'explicit-db':
        args = ['--board', 'alpha', *args]
        monkeypatch.setenv('HERMES_KANBAN_DB', str(kb.board_dir('beta') / 'kanban.db'))
    else:
        args = ['--board', 'alpha', *args]
        monkeypatch.setenv('HERMES_KANBAN_BOARD', 'beta')
    before = files(isolated)
    with kb.scoped_current_board(scope):
        rc, result = invoke(capsys, *args)
    assert rc == 5 and result['error']['code'] == 'BOARD_CONTEXT_CONFLICT'
    assert files(isolated) == before


@pytest.mark.parametrize('action', ['run', 'export'])
def test_child_mutation_denied_before_files_or_db(isolated, capsys, monkeypatch, action):
    monkeypatch.setenv('HERMES_DELEGATED_CHILD_CONTEXT', '1')
    args = ['recipe', action, 'missing', '--json']
    args += ['--key', 'once'] if action == 'run' else ['--output', str(isolated / 'export.json')]
    before = files(isolated)
    rc, result = invoke(capsys, *args)
    assert rc == 5 and result['error']['code'] == 'AUTHORITY_DENIED'
    assert files(isolated) == before


@pytest.mark.parametrize('key', ['', 'white space', 'é', 'x' * 129, '\n'])
def test_bad_key_rejected_before_migration(isolated, recipe_files, capsys, key):
    before = files(isolated)
    rc, result = invoke(capsys, *request(recipe_files, 'run', key))
    assert rc == 2 and result['error']['code'] == 'RECIPE_INVALID'
    assert files(isolated) == before


def test_run_replay_show_and_canonical_export(isolated, recipe_files, capsys, monkeypatch):
    rc, created = invoke(capsys, *request(recipe_files, 'run', 'once'))
    assert rc == 0 and created['replayed'] is False
    assert set(created['tasks']) == {'draft', 'check'}
    rc, replayed = invoke(capsys, *request(recipe_files, 'run', 'once'))
    assert rc == 0 and replayed['replayed'] is True and replayed['tasks'] == created['tasks']
    monkeypatch.setenv('HERMES_DELEGATED_CHILD_CONTEXT', '1')
    rc, shown = invoke(capsys, 'recipe', 'show', created['instance_id'], '--json')
    assert rc == 0 and shown['tasks'] == created['tasks']
    rc, _ = invoke(capsys, *request(recipe_files))
    assert rc == 0
    monkeypatch.delenv('HERMES_DELEGATED_CHILD_CONTEXT')
    output = isolated / 'export.json'
    rc, _ = invoke(capsys, 'recipe', 'export', created['instance_id'], '--output', output, '--json')
    from hermes_cli.kanban_recipes import canonical
    assert rc == 0 and output.read_text() == canonical(json.loads(recipe_files[0].read_text()))
    output.write_text('keep-me')
    rc, _ = invoke(capsys, 'recipe', 'export', created['instance_id'], '--output', output, '--json')
    assert rc == 2 and output.read_text() == 'keep-me'
    rc, _ = invoke(capsys, 'recipe', 'export', created['instance_id'], '--output', output, '--overwrite', '--json')
    assert rc == 0 and json.loads(output.read_text()) == shown['definition']
    assert 'private-input' not in output.read_text()


def test_invalid_run_does_not_migrate_existing_legacy_database(isolated, recipe_files, capsys):
    path = isolated / 'kanban.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE legacy (value TEXT)')
    before = files(isolated)
    recipe_files[1].write_text('{"topic":false}')
    rc, _ = invoke(capsys, *request(recipe_files, 'run', 'once'))
    assert rc == 2 and files(isolated) == before


def test_export_no_overwrite_race_preserves_competing_file(isolated, recipe_files, capsys, monkeypatch):
    rc, created = invoke(capsys, *request(recipe_files, 'run', 'once'))
    assert rc == 0
    import hermes_cli.kanban_recipes_cli as surface
    output = isolated / 'export.json'
    original_link = os.link
    def racing_link(src, dst, **kwargs):
        Path(dst).write_text('competitor')
        return original_link(src, dst, **kwargs)
    monkeypatch.setattr(surface.os, 'link', racing_link)
    rc, result = invoke(capsys, 'recipe', 'export', created['instance_id'], '--output', output, '--json')
    assert rc == 2 and output.read_text() == 'competitor'
    assert not list(isolated.glob('.export.json.*'))


@pytest.mark.parametrize('problem', ['changed-inputs', 'damaged', 'imported'])
def test_invalid_replay_fails_before_migration(isolated, recipe_files, capsys, monkeypatch, problem):
    rc, created = invoke(capsys, *request(recipe_files, 'run', 'once'))
    assert rc == 0
    expected = 'IDEMPOTENCY_CONFLICT'
    if problem == 'changed-inputs':
        recipe_files[1].write_text('{"topic":"changed"}')
    else:
        with sqlite3.connect(isolated / 'kanban.db') as conn:
            if problem == 'damaged':
                conn.execute('DELETE FROM recipe_instance_tasks')
                expected = 'INSTANCE_DAMAGED'
            else:
                conn.execute('UPDATE recipe_instances SET imported=1')
                expected = 'IMPORTED_INSTANCE_REBIND_REQUIRED'
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid replay must not migrate storage')
    monkeypatch.setattr(kb, 'init_db', forbidden)
    rc, result = invoke(capsys, *request(recipe_files, 'run', 'once'))
    assert rc == 3 and result['error']['code'] == expected


def test_run_freezes_plan_once_before_migration(isolated, recipe_files, capsys, monkeypatch):
    from hermes_cli import kanban_recipes_bindings as bindings
    original = bindings.resolve_plan
    calls = []
    def resolve(*args):
        calls.append(True)
        assert not (isolated / 'kanban.db').exists(), 'Defaults resolved after migration'
        return original(*args)
    monkeypatch.setattr(bindings, 'resolve_plan', resolve)
    rc, result = invoke(capsys, *request(recipe_files, 'run', 'once'))
    assert rc == 0, result
    assert len(calls) == 1


def test_replay_does_not_check_removed_profile(isolated, recipe_files, capsys):
    profile = isolated / 'profiles' / 'writer'
    profile.mkdir(parents=True)
    (profile / 'config.yaml').write_text('{}')
    recipe_files[2].write_text('{"roles":{"writer":"writer"}}')
    rc, first = invoke(capsys, *request(recipe_files, 'run', 'once'))
    assert rc == 0, first
    (profile / 'config.yaml').unlink()
    profile.rmdir()
    rc, replay = invoke(capsys, *request(recipe_files, 'run', 'once'))
    assert rc == 0 and replay['tasks'] == first['tasks'] and replay['replayed']
    recipe_files[1].write_text('{"topic":"different-private-value"}')
    rc, conflict = invoke(capsys, *request(recipe_files, 'run', 'once'))
    assert rc == 3 and conflict['error']['code'] == 'IDEMPOTENCY_CONFLICT'
    assert 'different-private-value' not in json.dumps(conflict)


@pytest.mark.parametrize('exception,code,exit_code', [
    (sqlite3.OperationalError('private-storage-value'), 'STORAGE_UNAVAILABLE', 4),
    (RuntimeError('private-commit-value'), 'COMMIT_UNCERTAIN', 4),
    (PermissionError('private-authority-value'), 'AUTHORITY_DENIED', 5),
    (ValueError('private-binding-value'), 'BINDING_INVALID', 2),
])
def test_native_failures_are_typed_without_values(isolated, recipe_files, capsys, monkeypatch, exception, code, exit_code):
    def fail(*args, **kwargs):
        raise exception
    monkeypatch.setattr(kb, 'init_db', fail)
    rc, result = invoke(capsys, *request(recipe_files, 'run', 'once'))
    assert rc == exit_code and result['error']['code'] == code
    assert 'private-' not in json.dumps(result)


def test_real_cli_processes_run_show_export_and_replay(isolated, recipe_files, tmp_path):
    import subprocess
    import sys
    env = dict(os.environ, HOME=str(tmp_path))
    root = Path(__file__).resolve().parents[2]
    def run(*args):
        result = subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'kanban', *map(str, args)],
                                cwd=root, env=env, text=True, capture_output=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)
    first = run(*request(recipe_files, 'run', 'process-key'))
    shown = run('recipe', 'show', first['instance_id'], '--json')
    assert shown['current_tasks']['draft']['status'] == 'ready'
    assert shown['current_tasks']['check']['status'] == 'todo'
    output = tmp_path / 'portable.json'
    run('recipe', 'export', first['instance_id'], '--output', output, '--json')
    assert json.loads(output.read_text()) == json.loads(recipe_files[0].read_text())
    replay = run(*request(recipe_files, 'run', 'process-key'))
    assert replay['replayed'] and replay['tasks'] == first['tasks']
    with sqlite3.connect(isolated / 'kanban.db') as conn:
        assert conn.execute('SELECT COUNT(*) FROM recipe_instances').fetchone()[0] == 1
        assert conn.execute('SELECT COUNT(*) FROM tasks').fetchone()[0] == len(first['tasks'])
