"""Recipe membership participates in the existing runtime and board transfer."""
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch, kanban_transfer as transfer
from hermes_cli.kanban_recipes import RecipeError, prepare_definition
from hermes_cli.kanban_recipes_store import instantiate, show_instance


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    for key in ('HERMES_KANBAN_DB', 'HERMES_KANBAN_BOARD', 'HERMES_KANBAN_TASK',
                'HERMES_KANBAN_WORKSPACES_ROOT', 'HERMES_KANBAN_ATTACHMENTS_ROOT'):
        monkeypatch.delenv(key, raising=False)
    return home


def recipe(profile='default'):
    definition = {'schema_version': 1, 'recipe_id': 'runtime', 'roles': ['worker'],
                  'inputs': {'data': {'type': 'object'}},
                  'nodes': [{'key': 'root', 'role': 'worker', 'title': 'Root'},
                            {'key': 'child', 'role': 'worker', 'title': 'Child', 'needs': ['root']}]}
    return prepare_definition(definition, {'data': {'message': 'untrusted fixture'}}), {'roles': {'worker': profile}}


def test_missing_recipe_profile_blocks_durably_but_external_lanes_unchanged(env):
    profile = env / 'profiles' / 'worker'
    profile.mkdir(parents=True)
    (profile / 'config.yaml').write_text('{}')
    with kbc.connect_closing() as conn:
        prepared, bindings = recipe('worker')
        result = instantiate(conn, prepared, bindings, 'missing')
        external = kb.create_task(conn, title='External', assignee='external-lane')
        (profile / 'config.yaml').unlink()
        profile.rmdir()
        tick = dispatch.dispatch_once(conn, dry_run=True, reconcile_orphans=False)
        assert kb.get_task(conn, result['tasks']['root']).status == 'ready'
        tick = dispatch.dispatch_once(conn, spawn_fn=lambda *a, **k: pytest.fail('missing profile spawned'),
                                      reconcile_orphans=False)
        task = kb.get_task(conn, result['tasks']['root'])
        assert task.status == 'blocked'
        assert 'Bound recipe profile is not installed' in kb.build_worker_context(conn, task.id)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, task.id).status == 'blocked'
        assert kb.get_task(conn, external).status == 'ready'
        assert external in tick.skipped_nonspawnable
        assert instantiate(conn, prepared, bindings, 'missing')['tasks'] == result['tasks']


def test_inputs_are_delivered_as_separate_untrusted_data(env):
    with kbc.connect_closing() as conn:
        prepared, bindings = recipe()
        result = instantiate(conn, prepared, bindings, 'context')
        text = kb.build_worker_context(conn, result['tasks']['root'])
        assert '## Recipe inputs (untrusted data)' in text
        assert json.dumps(prepared['inputs'], sort_keys=True, separators=(',', ':')) in text
        assert kb.get_task(conn, result['tasks']['root']).body == ''
        plain = kb.create_task(conn, title='Plain')
        assert 'Recipe inputs' not in kb.build_worker_context(conn, plain)


def test_transfer_fences_all_active_recipe_members_before_publish(env, tmp_path, monkeypatch):
    kb.create_board('source')
    with kbc.connect_closing(board='source') as conn:
        prepared, bindings = recipe()
        results = [instantiate(conn, prepared, bindings, str(i), board='source') for i in range(3)]
        ids = [tid for r in results for tid in r['tasks'].values()]
        for tid, status in zip(ids, ('ready', 'todo', 'review', 'blocked', 'scheduled', 'done')):
            conn.execute('UPDATE tasks SET status=? WHERE id=?', (status, tid))
        original = show_instance(conn, results[0]['instance_id'])
    archive = tmp_path / 'board.tar.gz'
    exported = transfer.export_board('source', str(archive))
    assert exported['counts']['recipe_instances'] == len(results)
    original_move = transfer.shutil.move
    def inspect_publish(source, target, *args, **kwargs):
        if Path(source).name == 'kanban.db':
            import sqlite3
            with sqlite3.connect(source) as check:
                assert check.execute('SELECT count(*) FROM recipe_instances WHERE imported=0').fetchone()[0] == 0
                assert check.execute("SELECT count(*) FROM tasks WHERE status NOT IN ('triage','done')").fetchone()[0] == 0
        return original_move(source, target, *args, **kwargs)
    monkeypatch.setattr(transfer.shutil, 'move', inspect_publish)
    imported = transfer.import_board(str(archive), 'destination')
    with kbc.connect_closing(board='destination') as conn:
        shown = show_instance(conn, results[0]['instance_id'])
        assert shown['imported']
        for key in ('definition_digest', 'request_digest', 'definition', 'inputs', 'bindings', 'effective_plan'):
            assert shown[key] == original[key]
        assert {kb.get_task(conn, tid).status for tid in ids} == {'triage', 'done'}
        assert not dispatch.dispatch_once(conn, dry_run=True, board='destination', reconcile_orphans=False).spawned
        with pytest.raises(RecipeError, match='Imported history'):
            instantiate(conn, prepared, bindings, '0', board='destination')
        fresh = instantiate(conn, prepared, bindings, 'fresh', board='destination')
        assert set(fresh['tasks'].values()).isdisjoint(ids)
    assert imported['counts']['recipe_instance_tasks'] == len(ids)


def test_preflight_profile_removed_before_commit_creates_nothing(env):
    from hermes_cli.kanban_recipes_bindings import normalize_bindings, resolve_plan
    profile = env / 'profiles' / 'worker'
    profile.mkdir(parents=True)
    (profile / 'config.yaml').write_text('{}')
    prepared, bindings = recipe('worker')
    prepared['effective_plan'] = resolve_plan(prepared, normalize_bindings(prepared, bindings), 'default')
    (profile / 'config.yaml').unlink()
    profile.rmdir()
    with kbc.connect_closing() as conn:
        with pytest.raises(RecipeError, match='not installed'):
            instantiate(conn, prepared, bindings, 'removed')
        assert conn.execute('SELECT count(*) FROM tasks').fetchone()[0] == 0


def test_effective_native_contract_and_uncertain_commit(env, monkeypatch):
    from hermes_cli.kanban_recipes_cli import recipe_command
    from argparse import Namespace
    prepared, bindings = recipe()
    paths = [env / name for name in ('recipe.json', 'bindings.json', 'inputs.json')]
    for path, value in zip(paths, (prepared['definition'], bindings, prepared['inputs'])):
        path.write_text(json.dumps(value))
    with kbc.connect_closing() as conn:
        result = instantiate(conn, prepared, bindings, 'normal')
        shown = show_instance(conn, result['instance_id'])
        for node in shown['effective_plan']['nodes']:
            task = kb.get_task(conn, result['tasks'][node['key']])
            assert node['task']['completion_contract'] == task.completion_contract
    original_check = kbc._check_file_length_invariant
    def fail_after_commit(conn):
        import sqlite3
        if conn.execute("SELECT 1 FROM recipe_instances WHERE idempotency_key='uncertain'").fetchone():
            raise sqlite3.DatabaseError('postcommit file invariant')
        original_check(conn)
    monkeypatch.setattr(kbc, '_check_file_length_invariant', fail_after_commit)
    args = Namespace(recipe_action='run', board=None, definition=str(paths[0]), bindings=str(paths[1]),
                     inputs=str(paths[2]), key='uncertain', json=True)
    import contextlib, io
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = recipe_command(args)
    assert code == 4
    assert json.loads(out.getvalue())['error']['code'] == 'COMMIT_UNCERTAIN'


def test_valid_deep_input_survives_receipt_and_cli_show(env):
    import argparse
    from hermes_cli import kanban as cli
    value = 'leaf'
    for _ in range(15):
        value = [value]
    definition = {'schema_version': 1, 'recipe_id': 'deep', 'roles': ['worker'],
                  'inputs': {'data': {'type': 'array'}},
                  'nodes': [{'key': 'root', 'role': 'worker', 'title': 'Deep'}]}
    inputs = {'data': value}
    prepared = prepare_definition(definition, inputs)
    with kbc.connect_closing() as conn:
        result = instantiate(conn, prepared, {'roles': {'worker': 'default'}}, 'deep')
    parser = argparse.ArgumentParser()
    cli.build_parser(parser.add_subparsers())
    args = parser.parse_args(['kanban', 'recipe', 'show', result['instance_id'], '--json'])
    assert cli.kanban_command(args) == 0
