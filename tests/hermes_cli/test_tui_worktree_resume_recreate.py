"""N8 review 2.2: quitting a clean ``hermes --tui -w`` session that had a conversation removes its
tree and branch, but the session's policy froze that cwd, so a later resume had every turn refused
``workspace_unavailable``. A resume (``--tui --resume``, ``-c``, classic CLI; all go through
``_resolve_chat_session_args``) now recreates a missing tree the ``--tui -w`` launcher created, at
the same path, before attaching. Other missing directories are never created."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

GIT = ['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid']


def _repo(tmp_path):
    repo = tmp_path / 'repo'
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True)
    (repo / '.gitignore').write_text('.worktrees/\n')
    subprocess.run([*GIT, '-C', str(repo), 'add', '.gitignore'], check=True)
    subprocess.run([*GIT, '-C', str(repo), 'commit', '-qm', 'fixture'], check=True)
    return repo


def _branches(repo):
    return subprocess.run(['git', '-C', str(repo), 'branch', '--format=%(refname:short)'],
                          capture_output=True, text=True, check=True).stdout.split()


def _quit_clean_session_with_history(tmp_path, monkeypatch):
    """``hermes --tui -w``, one user + assistant exchange, ``/quit`` (row ended ``tui_shutdown``)."""
    from hermes_cli import main_tui_launch
    from hermes_cli.worktree_ops import _setup_worktree
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setenv('HERMES_PYTHON', sys.executable)
    wt_info = _setup_worktree(str(repo), sync_base=False, retain_db=get_hermes_home() / 'state.db')
    monkeypatch.setattr(main_tui_launch, '_setup_tui_worktree', lambda: wt_info)
    monkeypatch.setattr(main_tui_launch, '_make_tui_argv', lambda *_: (['node'], Path('.')))
    monkeypatch.setattr(main_tui_launch, '_print_tui_exit_summary', lambda *_: None)

    def ink(_argv, cwd=None, env=None):
        db = SessionDB(db_path=get_hermes_home() / 'state.db')
        db.create_session('tui-wt', 'tui', cwd=env['HERMES_TUI_CWD'])
        db.append_message('tui-wt', 'user', 'hello')
        db.append_message('tui-wt', 'assistant', 'hi')
        db.end_session('tui-wt', 'tui_shutdown')
        db.close()
        Path(env['HERMES_TUI_ACTIVE_SESSION_FILE']).write_text(json.dumps({'session_id': 'tui-wt'}))
        return 0

    monkeypatch.setattr(main_tui_launch.subprocess, 'call', ink)
    with pytest.raises(SystemExit):
        main_tui_launch._launch_tui(worktree=True)
    tree = Path(wt_info['path'])
    assert not tree.exists(), 'precondition: the clean idle tree is removed on quit (N8)'
    return repo, tree, wt_info


def _resume(session_id, **extra):
    from hermes_cli import main as m
    args = SimpleNamespace(resume=session_id, continue_last=None, in_dir=None, worktree=False,
                           no_restore_cwd=False, create_if_missing=False, **extra)
    m._resolve_chat_session_args(args, use_tui=True)
    return args


def _turn_runs_in(cwd):
    """The owner's readiness gate a resumed turn passes through (``policy_scope``)."""
    from gateway.session_policy import policy_scope
    from hermes_state_runtime import RuntimeStoreError
    try:
        with policy_scope(SimpleNamespace(cwd=str(cwd))):
            pass
    except RuntimeStoreError as exc:
        return str(exc) != 'workspace_unavailable'
    except Exception:  # health: allow BLE001 -- past the readiness gate; the stub policy is incomplete
        return True
    return True


@pytest.mark.parametrize('no_restore_cwd', [False, True])
def test_resume_recreates_the_removed_tui_worktree_so_turns_run(tmp_path, monkeypatch, no_restore_cwd):
    repo, tree, wt_info = _quit_clean_session_with_history(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    from hermes_cli import main as m
    args = SimpleNamespace(resume='tui-wt', continue_last=None, in_dir=None, worktree=False,
                           no_restore_cwd=no_restore_cwd, create_if_missing=False)
    m._resolve_chat_session_args(args, use_tui=True)
    assert tree.is_dir(), ('quit removed the resumable session\'s worktree; the resumed session\'s '
                           'turns are refused workspace_unavailable')
    assert _turn_runs_in(tree)
    assert wt_info['branch'] in _branches(repo)
    head = subprocess.run(['git', '-C', str(tree), 'rev-parse', '--abbrev-ref', 'HEAD'],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == wt_info['branch']
    if not no_restore_cwd:
        assert Path.cwd().resolve() == tree.resolve()


def test_resume_reuses_the_surviving_branch(tmp_path, monkeypatch):
    """A tree removed while its branch survived comes back on that branch, at its commit."""
    from hermes_cli.worktree_ops import _setup_worktree
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    repo = _repo(tmp_path)
    monkeypatch.chdir(repo)
    wt_info = _setup_worktree(str(repo), sync_base=False, retain_db=get_hermes_home() / 'state.db')
    tree = Path(wt_info['path'])
    (tree / 'f.txt').write_text('work')
    subprocess.run([*GIT, '-C', str(tree), 'add', 'f.txt'], check=True)
    subprocess.run([*GIT, '-C', str(tree), 'commit', '-qm', 'work'], check=True)
    tip = subprocess.run(['git', '-C', str(tree), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                         check=True).stdout.strip()
    db = SessionDB(db_path=get_hermes_home() / 'state.db')
    db.create_session('tui-wt', 'tui', cwd=str(tree))
    db.append_message('tui-wt', 'user', 'hello')
    db.close()
    subprocess.run(['git', '-C', str(repo), 'worktree', 'remove', '--force', '--force', str(tree)], check=True)
    assert not tree.exists() and wt_info['branch'] in _branches(repo)

    _resume('tui-wt')
    assert (tree / 'f.txt').read_text() == 'work'
    assert subprocess.run(['git', '-C', str(tree), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip() == tip


def test_resume_never_creates_a_missing_dir_the_launcher_did_not_make(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    repo = _repo(tmp_path)
    foreign = repo / '.worktrees' / 'hermes-deadbeef'  # looks like ours; no launcher record
    db = SessionDB(db_path=get_hermes_home() / 'state.db')
    db.create_session('other', 'tui', cwd=str(foreign))
    db.append_message('other', 'user', 'hello')
    db.create_session('plain', 'tui', cwd=str(tmp_path / 'gone'))
    db.append_message('plain', 'user', 'hello')
    db.close()
    monkeypatch.chdir(tmp_path)
    _resume('other')
    _resume('plain')
    assert not foreign.exists()
    assert not (tmp_path / 'gone').exists()
