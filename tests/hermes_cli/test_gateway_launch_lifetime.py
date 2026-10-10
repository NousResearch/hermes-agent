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


@pytest.mark.parametrize('resumable', [True, False])
def test_retained_tui_worktree_survives_stale_prune_while_its_session_resumes(tmp_path, monkeypatch, resumable):
    """The launch lock names the exiting launcher's pid; once it is gone a later launch's 24 h
    prune must still keep a clean retained tree whose session can be resumed (and only then)."""
    import json
    import os
    import time

    from hermes_cli import main_tui_launch
    from hermes_cli.worktree_ops import _prune_stale_worktrees, _setup_worktree
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    repo = tmp_path / 'repo'
    git = ['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid']
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True)
    (repo / '.gitignore').write_text('.worktrees/\n')
    subprocess.run([*git, '-C', str(repo), 'add', '.gitignore'], check=True)
    subprocess.run([*git, '-C', str(repo), 'commit', '-qm', 'fixture'], check=True)
    monkeypatch.chdir(repo)
    monkeypatch.setenv('HERMES_PYTHON', sys.executable)
    wt_info = _setup_worktree(str(repo), sync_base=False)
    # The real launcher exits with the TUI, so its `hermes pid=<pid>` launch lock is dead by the next launch.
    subprocess.run(['git', '-C', str(repo), 'worktree', 'unlock', wt_info['path']], check=True)
    subprocess.run(['git', '-C', str(repo), 'worktree', 'lock', '--reason', 'hermes pid=999999999',
                    wt_info['path']], check=True)
    monkeypatch.setattr(main_tui_launch, '_setup_tui_worktree', lambda: wt_info)
    monkeypatch.setattr(main_tui_launch, '_make_tui_argv', lambda *_: (['node'], Path('.')))
    monkeypatch.setattr(main_tui_launch, '_print_tui_exit_summary', lambda *_: None)

    def ink(_argv, cwd=None, env=None):
        db = SessionDB(db_path=get_hermes_home() / 'state.db')
        db.create_session('tui-wt', 'tui', cwd=env['HERMES_TUI_CWD'])
        if resumable:
            db.append_message('tui-wt', 'user', 'keep my checkout')
        db.close()
        Path(env['HERMES_TUI_ACTIVE_SESSION_FILE']).write_text(json.dumps({'session_id': 'tui-wt'}))
        ink.checkout = Path(env['HERMES_TUI_CWD'])
        return 0

    monkeypatch.setattr(main_tui_launch.subprocess, 'call', ink)
    with pytest.raises(SystemExit):
        main_tui_launch._launch_tui(worktree=True)
    # The launcher is gone; the next launch runs its prune a day later.
    old = time.time() - 25 * 3600
    os.utime(ink.checkout, (old, old))
    _prune_stale_worktrees(str(repo))
    assert ink.checkout.is_dir() is resumable


@pytest.mark.parametrize('case', ['quoted_home', 'after_new', 'terminal_closed'])
def test_retained_tui_worktree_kept_while_any_session_in_it_resumes(tmp_path, monkeypatch, case):
    """N5: liveness comes from the store, for every way the lock used to read as dead — a home
    git C-quotes (non-ASCII / quote / backslash, every Windows path), ``/new`` leaving the older
    session's history behind the named empty one, and a launcher killed with its terminal
    (SIGHUP skips the exit re-lock)."""
    import json
    import os
    import re
    import time

    from hermes_cli import main_tui_launch, worktree_ops
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    home = tmp_path / ('Zoë "q" \\ home' if case == 'quoted_home' else 'home')
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    repo = tmp_path / 'repo'
    git = ['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid']
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True)
    (repo / '.gitignore').write_text('.worktrees/\n')
    subprocess.run([*git, '-C', str(repo), 'add', '.gitignore'], check=True)
    subprocess.run([*git, '-C', str(repo), 'commit', '-qm', 'fixture'], check=True)
    monkeypatch.chdir(repo)
    monkeypatch.setenv('HERMES_PYTHON', sys.executable)
    monkeypatch.setattr(main_tui_launch, '_make_tui_argv', lambda *_: (['node'], Path('.')))
    monkeypatch.setattr(main_tui_launch, '_print_tui_exit_summary', lambda *_: None)
    if case == 'terminal_closed':
        monkeypatch.setattr(worktree_ops, '_retain_worktree_for_session', lambda *_: False)

    def ink(_argv, cwd=None, env=None):
        db = SessionDB(db_path=get_hermes_home() / 'state.db')
        db.create_session('tui-a', 'tui', cwd=env['HERMES_TUI_CWD'])
        db.append_message('tui-a', 'user', 'keep my checkout')
        attached = 'tui-a'
        if case == 'after_new':  # /new: the attached session has nothing sent yet
            db.create_session(attached := 'tui-b', 'tui', cwd=env['HERMES_TUI_CWD'])
        db.close()
        Path(env['HERMES_TUI_ACTIVE_SESSION_FILE']).write_text(json.dumps({'session_id': attached}))
        ink.checkout = Path(env['HERMES_TUI_CWD'])
        return 0

    monkeypatch.setattr(main_tui_launch.subprocess, 'call', ink)
    with pytest.raises(SystemExit):
        main_tui_launch._launch_tui(worktree=True)
    listing = subprocess.run(['git', '-C', str(repo), 'worktree', 'list', '--porcelain', '-z'],
                             capture_output=True, text=True, check=True).stdout
    reason = next(f[len('locked '):] for f in listing.split('\0') if f.startswith('locked '))
    # The launcher process is gone by the next launch, whichever lock it left.
    subprocess.run(['git', '-C', str(repo), 'worktree', 'unlock', str(ink.checkout)], check=True)
    subprocess.run(['git', '-C', str(repo), 'worktree', 'lock', '--reason',
                    re.sub(r'pid=\d+', 'pid=999999999', reason), str(ink.checkout)], check=True)
    old = time.time() - 25 * 3600
    os.utime(ink.checkout, (old, old))
    worktree_ops._prune_stale_worktrees(str(repo))
    assert ink.checkout.is_dir()
