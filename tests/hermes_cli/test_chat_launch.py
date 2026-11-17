"""``hermes_cli.chat_launch``: what a plugin's own chat launcher calls around its client process."""
import json
import os
import subprocess

import pytest

from hermes_cli.main import PROJECT_ROOT


def test_exit_summary_names_the_launchers_command(tmp_path, monkeypatch, capsys):
    from hermes_state import SessionDB
    from hermes_cli.chat_launch import print_exit_summary
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    db = SessionDB(db_path=home / 'state.db')
    db.create_session('20261110_120000_abc123', source='tui')
    db.set_session_title('20261110_120000_abc123', 'Demo chat')
    db.append_message('20261110_120000_abc123', 'user', 'q')
    db.append_message('20261110_120000_abc123', 'assistant', 'a')
    db.create_session('20261110_120001_empty0', source='tui')
    db.close()

    print_exit_summary('20261110_120000_abc123', command='hermes demo')
    out = capsys.readouterr().out
    assert '  hermes demo --resume 20261110_120000_abc123\n' in out
    assert '  hermes demo -c "Demo chat"\n' in out
    assert '--tui' not in out

    print_exit_summary('20261110_120001_empty0', command='hermes demo')
    assert capsys.readouterr().out == ''


def test_hermes_command_runs_this_install_without_pythonpath(tmp_path):
    from hermes_cli.chat_launch import hermes_command
    home = tmp_path / '.hermes'
    home.mkdir()
    (home / 'config.yaml').write_text('updates:\n  check: false\n', encoding='utf-8')
    env = {'HERMES_HOME': str(home), 'HOME': str(tmp_path), 'PATH': os.environ.get('PATH', '')}
    child = subprocess.run(hermes_command(['--version']), env=env, cwd=str(tmp_path), capture_output=True,
                           text=True, timeout=60, check=False)
    assert child.returncode == 0, child.stderr
    assert child.stdout.startswith('Hermes Agent v')
    assert f'Install directory: {PROJECT_ROOT}\n' in child.stdout


@pytest.mark.platforms('windows')  # only Windows publishes the launcher as hermes.exe or hermes.cmd
def test_hermes_command_starts_with_a_file_windows_can_start(tmp_path, monkeypatch):
    from hermes_cli import main
    from hermes_cli.chat_launch import hermes_command
    store = tmp_path / 'store'
    python = store / 'py' / 'python.exe'
    python.parent.mkdir(parents=True)
    python.write_bytes(b'')
    (store / 'facts.json').write_text(json.dumps({'packages': {'python': {'entry': 'py'}}}), encoding='utf-8')
    monkeypatch.setenv('HERMES_RUNTIME_DIR', str(store))
    launchers = tmp_path / 'install' / '.hermes' / 'bin'
    launchers.mkdir(parents=True)
    monkeypatch.setattr(main, 'PROJECT_ROOT', tmp_path / 'install')

    # Only the batch fallback is published: no child process starts it without a shell.
    (launchers / 'hermes.cmd').write_text('@echo off\r\n', encoding='utf-8')
    command = hermes_command(['--version'])
    assert os.path.samefile(command[0], python)
    assert command[-1] == '--version'

    (launchers / 'hermes.exe').write_bytes(b'MZ')
    command = hermes_command(['--version'])
    assert os.path.samefile(command[0], launchers / 'hermes.exe')
    assert command[1:] == ['--version']
