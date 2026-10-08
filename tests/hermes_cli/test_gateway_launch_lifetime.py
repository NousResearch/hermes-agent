"""Detached owners must not lose the checkout their session still uses."""
from pathlib import Path
import subprocess
import sys

import pytest


def test_tui_detach_preserves_real_clean_worktree(tmp_path, monkeypatch):
    from hermes_cli import main_tui_launch
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '--allow-empty', '-qm', 'fixture'], check=True)
    checkout = tmp_path / 'worktree'
    subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '-qb', 'fixture-work', str(checkout)], check=True)
    monkeypatch.chdir(repo)
    monkeypatch.setenv('HERMES_PYTHON', sys.executable)
    monkeypatch.setattr(main_tui_launch, '_setup_tui_worktree', lambda: {
        'path': str(checkout), 'repo_root': str(repo), 'branch': 'fixture-work'})
    monkeypatch.setattr(main_tui_launch, '_make_tui_argv', lambda *_: (['node'], Path('.')))
    monkeypatch.setattr(main_tui_launch.subprocess, 'call', lambda *_, **__: 0)
    monkeypatch.setattr(main_tui_launch, '_print_tui_exit_summary', lambda *_: None)
    with pytest.raises(SystemExit) as exited:
        main_tui_launch._launch_tui(worktree=True)
    assert exited.value.code == 0
    assert checkout.is_dir()
    assert (checkout / '.git').is_file()
