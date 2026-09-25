"""Only declared Python dependencies require Python install consent."""
import json
import os
import subprocess

import pytest
import hermes_yaml as yaml

from hermes_cli import plugins_cmd
from tests.pm.test_plugin_survival_contract import admission_env  # noqa: F401


@pytest.mark.parametrize('python_surface', [None, 'pyproject', 'legacy'])
def test_install_only_requests_relevant_python_consent(admission_env, monkeypatch, capsys, python_surface):
    root, home = admission_env
    source = root / 'plugin-source'
    source.mkdir()
    manifest = {'name': 'sidecar-plugin'}
    if python_surface == 'legacy':
        manifest['python_dependencies'] = ['never-install-without-consent==1']
    (source / 'plugin.yaml').write_text(yaml.safe_dump(manifest), encoding='utf-8')
    (source / '__init__.py').write_text('def register(ctx):\n    pass\n', encoding='utf-8')
    (source / 'package.json').write_text(json.dumps({'name': 'sidecar-plugin', 'version': '1.0.0'}), encoding='utf-8')
    if python_surface == 'pyproject':
        (source / 'pyproject.toml').write_text(
            '[project]\nname="sidecar-plugin"\nversion="1.0"\nrequires-python=">=3.14"\n', encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if k not in {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'}}
    for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')]:
        subprocess.run(['git', *args], cwd=source, env=env, check=True, capture_output=True, timeout=30)
    monkeypatch.setattr(plugins_cmd.sys.stdin, 'isatty', lambda: False)
    config_path = home / 'config.yaml'
    before = config_path.read_bytes()
    plugins_cmd.cmd_install(source.as_uri(), enable=True, allow_removed=True)
    installed = home / 'plugins' / 'sidecar-plugin'
    assert (installed / 'package.json').is_file()
    assert not (installed / 'node_modules').exists()
    enabled = yaml.safe_load(config_path.read_text(encoding='utf-8'))['plugins']['enabled']
    output = capsys.readouterr().out
    if python_surface is None:
        assert 'sidecar-plugin' in enabled
        assert 'declares Python dependencies' not in output
    else:
        assert 'sidecar-plugin' not in enabled
        assert config_path.read_bytes() == before
        assert 'dependency install skipped (non-interactive)' in output


def test_node_sidecar_question_stays_independent(tmp_path, monkeypatch):
    from pm import workspace

    target = tmp_path / 'plugin'
    target.mkdir()
    (target / 'package.json').write_text('{}', encoding='utf-8')
    prompts, installs, lines = [], [], []
    monkeypatch.setattr(plugins_cmd.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(plugins_cmd.sys.stdout, 'isatty', lambda: True)
    monkeypatch.setattr('builtins.input', lambda prompt: prompts.append(prompt) or 'yes')
    monkeypatch.setattr(workspace, 'install_node_sidecar', lambda path, **kwargs: installs.append((path, kwargs)))
    from types import SimpleNamespace

    result = plugins_cmd._install_plugin_python_deps(
        {'name': 'sidecar'}, target, SimpleNamespace(print=lambda *args, **kwargs: lines.extend(args)))
    assert result == (True, None)
    assert installs == [(target, {'explicit': True})]
    assert len(prompts) == 1 and 'node_modules' in prompts[0]
    assert not any('Python dependencies' in str(line) for line in lines)


def test_yes_deps_carries_consent_through_non_interactive_install(admission_env, monkeypatch, capsys):
    """--yes-deps (#122134): the flag is the user's own answer to the consent
    question, so a non-interactive install publishes AND enables in one run
    instead of being refused (the `enable` recovery cannot publish a plugin)."""
    root, home = admission_env
    source = root / 'plugin-source'
    source.mkdir()
    (source / 'plugin.yaml').write_text(yaml.safe_dump({'name': 'yes-deps-plugin'}), encoding='utf-8')
    (source / '__init__.py').write_text('def register(ctx):\n    pass\n', encoding='utf-8')
    (source / 'pyproject.toml').write_text(
        '[project]\nname="yes-deps-plugin"\nversion="1.0"\nrequires-python=">=3.11"\n', encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if k not in {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'}}
    for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')]:
        subprocess.run(['git', *args], cwd=source, env=env, check=True, capture_output=True, timeout=30)
    monkeypatch.setattr(plugins_cmd.sys.stdin, 'isatty', lambda: False)

    plugins_cmd.cmd_install(source.as_uri(), enable=True, allow_removed=True, yes_deps=True)

    enabled = yaml.safe_load((home / 'config.yaml').read_text(encoding='utf-8'))['plugins']['enabled']
    output = capsys.readouterr().out
    assert 'yes-deps-plugin' in enabled
    assert 'dependency install skipped (non-interactive)' not in output


def test_publish_assume_consent_skips_replacement_veto(tmp_path, monkeypatch):
    """The active-replacement veto at publication honors --yes-deps: with the
    explicit answer the PM handoff runs; the default without it stays the
    non-interactive refusal (#122134's 'Reinstall declined' half)."""
    from hermes_cli import plugins_transaction
    from pm import workspace as ws

    staged, target = tmp_path / 'staged', tmp_path / 'target'
    staged.mkdir()
    target.mkdir()
    (staged / 'plugin.yaml').write_text(yaml.safe_dump({'name': 'active-plugin'}), encoding='utf-8')
    (staged / 'pyproject.toml').write_text(
        '[project]\nname="active-plugin"\nversion="1.0"\nrequires-python=">=3.11"\n', encoding='utf-8')
    monkeypatch.setattr(ws, 'enabled_plugin_dirs', lambda **kwargs: [target])
    syncs = []
    monkeypatch.setattr('pm.client.sync_venv', lambda **kwargs: syncs.append(kwargs))
    monkeypatch.setattr(plugins_cmd.sys.stdin, 'isatty', lambda: False)

    plugins_transaction.publish_plugin(
        staged, target, {}, {'active-plugin': {}}, require_consent=True, assume_consent=True)
    assert len(syncs) == 1

    with pytest.raises(plugins_cmd.PluginOperationError, match='Reinstall declined'):
        plugins_transaction.publish_plugin(staged, target, {}, {'active-plugin': {}}, require_consent=True)


def test_install_parser_offers_yes_deps_exclusive_with_no_deps():
    import argparse

    from hermes_cli.subcommands.plugins import build_plugins_parser

    parser = argparse.ArgumentParser(prog='hermes')
    build_plugins_parser(parser.add_subparsers(), cmd_plugins=lambda args: None)
    ns = parser.parse_args(['plugins', 'install', 'x', '--yes-deps'])
    assert ns.yes_deps is True
    assert ns.no_deps is False
    with pytest.raises(SystemExit):
        parser.parse_args(['plugins', 'install', 'x', '--no-deps', '--yes-deps'])
