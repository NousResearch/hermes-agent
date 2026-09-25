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
    if python_surface == 'legacy':
        # --enable now implies dependency consent, but the fixture's fake
        # dependency cannot resolve, so admission refuses and the plugin
        # stays installed but disabled.
        assert 'sidecar-plugin' not in enabled
        assert config_path.read_bytes() == before
        assert 'dependency install skipped (non-interactive)' not in output
    else:
        assert 'sidecar-plugin' in enabled
        assert 'dependency install skipped (non-interactive)' not in output
        if python_surface is None:
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


def _dep_consent_target(tmp_path):
    target = tmp_path / 'plugin'
    target.mkdir()
    (target / 'plugin.yaml').write_text('name: implied-consent\n', encoding='utf-8')
    (target / 'pyproject.toml').write_text(
        '[project]\nname="implied-consent"\nversion="1.0"\nrequires-python=">=3.14"\ndependencies=["requests>=2"]\n',
        encoding='utf-8',
    )
    return target


def _stub_console(lines):
    from types import SimpleNamespace

    return SimpleNamespace(print=lambda *args, **kwargs: lines.append(' '.join(str(a) for a in args)))


def test_explicit_enable_implies_python_dep_consent_noninteractive(tmp_path, monkeypatch):
    """#122134: an explicit --enable is dependency consent, even with no TTY."""
    lines = []
    monkeypatch.setattr(plugins_cmd.sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr(plugins_cmd.sys.stdout, 'isatty', lambda: False)
    result = plugins_cmd._install_plugin_python_deps(
        {'name': 'implied-consent'}, _dep_consent_target(tmp_path),
        _stub_console(lines), preconsented=True)
    assert result == (True, None)
    assert any('implied-consent' in line for line in lines)


def test_dep_consent_stays_fail_closed_without_explicit_enable(tmp_path, monkeypatch):
    """Without --enable, a non-interactive install still refuses consent."""
    lines = []
    monkeypatch.setattr(plugins_cmd.sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr(plugins_cmd.sys.stdout, 'isatty', lambda: False)
    result = plugins_cmd._install_plugin_python_deps(
        {'name': 'implied-consent'}, _dep_consent_target(tmp_path), _stub_console(lines))
    assert result == (False, 'dependency install skipped (non-interactive)')
