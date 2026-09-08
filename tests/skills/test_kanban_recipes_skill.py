"""Execute the shipped recipe template through the shared command surface."""
import argparse
import json
import os
from pathlib import Path


def test_bundled_recipe_validate_run_show_export(tmp_path, monkeypatch, capsys):
    from hermes_cli import kanban as cli, kanban_db as kb
    root = Path(__file__).resolve().parents[2]
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    for name in tuple(os.environ):
        if name.startswith('HERMES_KANBAN_'):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    definition = root / 'skills/devops/kanban-recipes/templates/brief.json'
    inputs, bindings = tmp_path / 'inputs.json', tmp_path / 'bindings.json'
    inputs.write_text('{"topic":"SQLite transactions"}')
    bindings.write_text('{"roles":{"researcher":"default","writer":"default"}}')
    def invoke(*words):
        parser = argparse.ArgumentParser()
        cli.build_parser(parser.add_subparsers())
        args = parser.parse_args(['kanban', '--board', 'default', 'recipe', *map(str, words), '--json'])
        with kb.scoped_current_board(None):
            assert cli.kanban_command(args) == 0
        return json.loads(capsys.readouterr().out)
    options = [definition, '--inputs', inputs, '--bindings', bindings]
    assert invoke('validate', *options)['valid']
    assert not (home / 'kanban.db').exists()
    result = invoke('run', *options, '--key', 'documented-example')
    assert invoke('show', result['instance_id'])['tasks'] == result['tasks']
    assert invoke('run', *options, '--key', 'documented-example')['replayed']
    target = tmp_path / 'export.json'
    invoke('export', result['instance_id'], '--output', target)
    assert json.loads(target.read_text()) == json.loads(definition.read_text())
