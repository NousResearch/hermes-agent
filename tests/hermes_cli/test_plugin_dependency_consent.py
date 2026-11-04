"""Only declared Python dependencies require Python install consent."""
import json
import os
import subprocess

import pytest
import hermes_yaml as yaml

from hermes_cli import plugins_cmd
from tests.pm.test_plugin_survival_contract import admission_env


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
            '[project]\nname="sidecar-plugin"\nversion="1.0"\nrequires-python=">=3.14"\n'
            'dependencies=["never-install-without-consent==1"]\n', encoding='utf-8')
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


def test_empty_python_dependencies_install_and_enable_without_consent(admission_env, monkeypatch, capsys):  # health: allow F811 -- pytest injects the imported admission_env fixture by parameter name
    """``dependencies = []`` installs nothing, so a headless ``install --enable`` enables the plugin
    exactly like one without a pyproject instead of refusing at an empty consent question."""
    root, home = admission_env
    source = root / 'plugin-source'
    source.mkdir()
    (source / 'plugin.yaml').write_text(yaml.safe_dump({'name': 'no-deps-plugin'}), encoding='utf-8')
    (source / '__init__.py').write_text('def register(ctx):\n    pass\n', encoding='utf-8')
    (source / 'pyproject.toml').write_text(
        '[project]\nname="no-deps-plugin"\nversion="1.0"\nrequires-python=">=3.11"\ndependencies=[]\n',
        encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if k not in {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'}}
    for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')]:
        subprocess.run(['git', *args], cwd=source, env=env, check=True, capture_output=True, timeout=30)
    monkeypatch.setattr(plugins_cmd.sys.stdin, 'isatty', lambda: False)

    plugins_cmd.cmd_install(source.as_uri(), enable=True, allow_removed=True)

    enabled = yaml.safe_load((home / 'config.yaml').read_text(encoding='utf-8'))['plugins']['enabled']
    output = capsys.readouterr().out
    assert 'no-deps-plugin' in enabled, 'install completed but the plugin was not enabled; CLI output:\n' + output
    assert 'declares Python dependencies' not in output


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


def _tty_install_recording_npm(root, monkeypatch, files: dict, npm_error=None):
    """A committed local plugin made of *files*, a TTY that answers yes, and an ``npm`` stub that
    writes a marker into the tree it runs in. Returns ``(uri, calls)``: ``(tree, live_tree_existed)``."""
    from pm import workspace

    source = root / 'plugin-source'
    source.mkdir()
    for name, text in files.items():
        (source / name).write_text(text, encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if k not in {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'}}
    for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')]:
        subprocess.run(['git', *args], cwd=source, env=env, check=True, capture_output=True, timeout=30)
    calls = []

    def fake_npm(path, **_kwargs):
        calls.append((path, (plugins_cmd._plugins_dir() / 'node-plugin').exists()))
        (path / 'node_modules').mkdir()
        (path / 'node_modules' / 'marker').write_text('ok', encoding='utf-8')
        return npm_error

    monkeypatch.setattr(workspace, 'install_node_sidecar', fake_npm)
    monkeypatch.setattr(plugins_cmd.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(plugins_cmd.sys.stdout, 'isatty', lambda: True)
    monkeypatch.setattr('builtins.input', lambda _prompt: 'y')
    return source.as_uri(), calls


_NODE_PLUGIN = {'plugin.yaml': yaml.safe_dump({'name': 'node-plugin'}), '__init__.py': 'def register(ctx):\n    pass\n'}


_BUILD_ONLY_PACKAGE_JSON = json.dumps({'name': 'node-plugin', 'private': True, 'devDependencies': {'esbuild': '^0.20.0'}})


@pytest.mark.parametrize('package_json, npm_error', [
    (json.dumps({'name': 'node-plugin', 'version': '1.0.0'}), None),
    (json.dumps({'name': 'node-plugin', 'version': '1.0.0'}), 'npm ci exited 1: boom'),
    (_BUILD_ONLY_PACKAGE_JSON, 'npm install exited 1: no network'),
], ids=['npm-ok', 'npm-fails', 'build-only-npm-fails'])
def test_cli_install_runs_node_deps_in_the_staged_tree_before_publication(admission_env, monkeypatch, capsys, package_json, npm_error):  # health: allow F811 -- pytest injects the imported admission_env fixture by parameter name
    """``hermes plugins install`` ran npm in the live plugin tree after publication. npm now runs in
    the staged tree before publication, so its node_modules ship with the plugin. An npm failure
    (e.g. a build-only package.json with prebuilt files committed) still only warns: the plugin is
    published and enabled, as before."""
    root, home = admission_env
    uri, calls = _tty_install_recording_npm(root, monkeypatch, {**_NODE_PLUGIN, 'package.json': package_json}, npm_error)
    installed = home / 'plugins' / 'node-plugin'

    plugins_cmd.cmd_install(uri, enable=True, allow_removed=True)

    assert len(calls) == 1
    tree, live_tree_existed = calls[0]
    assert tree != installed and not live_tree_existed, f'npm ran in {tree} after publication'
    assert (installed / 'node_modules' / 'marker').read_text(encoding='utf-8') == 'ok'
    assert 'node-plugin' in yaml.safe_load((home / 'config.yaml').read_text(encoding='utf-8'))['plugins']['enabled']
    out = capsys.readouterr().out
    assert ('Node deps: npm ' in out) is bool(npm_error), out


def test_cli_install_without_package_json_runs_no_node_step(admission_env, monkeypatch):  # health: allow F811 -- pytest injects the imported admission_env fixture by parameter name
    root, home = admission_env
    uri, calls = _tty_install_recording_npm(root, monkeypatch, _NODE_PLUGIN)

    plugins_cmd.cmd_install(uri, enable=True, allow_removed=True)

    assert calls == []
    assert (home / 'plugins' / 'node-plugin' / 'plugin.yaml').is_file()


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
    # The install publishes into plugins/ and THEN a second PM admission transaction
    # enables it; that admission catches every exception as a refusal, so an
    # environment-sensitive failure there leaves the plugin installed-but-disabled
    # with exit 0. Carry the CLI output in the failure message so the refusal's
    # cause is visible on any machine instead of a bare `in []`.
    assert 'yes-deps-plugin' in enabled, (
        'install completed but the plugin was not enabled; CLI output:\n' + output)
    assert 'dependency install skipped (non-interactive)' not in output


def test_dashboard_install_publishes_a_plugin_the_config_already_selects(admission_env, monkeypatch):  # health: allow F811 -- pytest injects the imported admission_env fixture by parameter name
    """The Desktop/dashboard Install click on a plugin memory.provider already names (a provider that
    left core, a synced config, a re-install after remove) publishes like any fresh install instead of
    the non-interactive 'Install declined' that a retry can never get past. A forced replacement of an
    installed plugin keeps the veto: that is the reinstall the click did not review."""
    root, home = admission_env
    source = root / 'mem-source'
    source.mkdir()
    (source / 'plugin.yaml').write_text(yaml.safe_dump({'name': 'mem-twin'}), encoding='utf-8')
    (source / '__init__.py').write_text('class Twin(MemoryProvider): ...\n', encoding='utf-8')
    (source / 'pyproject.toml').write_text(
        '[project]\nname="mem-twin"\nversion="1.0"\nrequires-python=">=3.11"\n', encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if k not in {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'}}
    for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture')]:
        subprocess.run(['git', *args], cwd=source, env=env, check=True, capture_output=True, timeout=30)
    config = yaml.safe_load((home / 'config.yaml').read_text(encoding='utf-8'))
    config['memory'] = {'provider': 'mem-twin'}
    (home / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    monkeypatch.setattr(plugins_cmd.sys.stdin, 'isatty', lambda: False)

    result = plugins_cmd.dashboard_install_plugin(source.as_uri(), force=False, enable=True)
    assert result['ok'], result
    assert (home / 'plugins' / 'mem-twin' / 'plugin.yaml').is_file()
    # The forced replacement now declares a dependency: installing it is what the veto guards.
    (source / 'pyproject.toml').write_text(
        '[project]\nname="mem-twin"\nversion="1.1"\nrequires-python=">=3.11"\n'
        'dependencies=["never-install-without-consent==1"]\n', encoding='utf-8')
    subprocess.run(['git', '-c', 'user.name=test', '-c', 'user.email=test@example.invalid', 'commit', '-qam', 'deps'],
                   cwd=source, env=env, check=True, capture_output=True, timeout=30)
    again = plugins_cmd.dashboard_install_plugin(source.as_uri(), force=True, enable=True)
    assert not again['ok'] and 'Reinstall declined' in again['error']


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
        '[project]\nname="active-plugin"\nversion="1.0"\nrequires-python=">=3.11"\n'
        'dependencies=["never-install-without-consent==1"]\n', encoding='utf-8')
    monkeypatch.setattr(ws, 'enabled_plugin_dirs', lambda **kwargs: [target])
    syncs = []
    monkeypatch.setattr('pm.client.sync_venv', lambda **kwargs: syncs.append(kwargs))
    monkeypatch.setattr(plugins_cmd.sys.stdin, 'isatty', lambda: False)

    plugins_transaction.publish_plugin(
        staged, target, {}, {'active-plugin': {}}, require_consent=True, assume_consent=True)
    assert len(syncs) == 1

    with pytest.raises(plugins_cmd.PluginOperationError, match='Reinstall declined'):
        plugins_transaction.publish_plugin(staged, target, {}, {'active-plugin': {}}, require_consent=True)


@pytest.mark.parametrize('yes_deps', [False, True])
def test_active_replacement_npm_failure_warns_and_publishes(tmp_path, monkeypatch, capsys, yes_deps):
    """Replacing an active plugin runs npm once, in the staged tree before publication; an npm
    failure warns and the replacement is still published, with and without --yes-deps."""
    from hermes_cli import plugins_transaction
    from pm import workspace as ws

    staged, target = tmp_path / 'staged', tmp_path / 'target'
    for tree in (staged, target):
        tree.mkdir()
        (tree / 'plugin.yaml').write_text(yaml.safe_dump({'name': 'active-plugin'}), encoding='utf-8')
    (staged / 'package.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(ws, 'enabled_plugin_dirs', lambda **kwargs: [target])
    events, prompts = [], []
    monkeypatch.setattr('pm.client.sync_venv', lambda **kwargs: events.append('publish'))
    monkeypatch.setattr(ws, 'install_node_sidecar', lambda path, **kwargs: events.append(path) or 'npm ci exited 1')
    monkeypatch.setattr(plugins_cmd.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(plugins_cmd.sys.stdout, 'isatty', lambda: True)
    monkeypatch.setattr('builtins.input', lambda prompt: prompts.append(prompt) or 'y')

    plugins_transaction.publish_plugin(staged, target, {}, {'active-plugin': {}}, require_consent=True,
                                       assume_consent=yes_deps, node_deps=True)
    assert (events, len(prompts)) == ([staged, 'publish'], 1)
    assert 'Node deps: npm ci exited' in capsys.readouterr().out


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
