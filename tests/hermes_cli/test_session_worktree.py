"""The ``-w`` worktree of a chat session through the functions any launcher calls
(``hermes_cli.session_worktree``): ``create`` makes a tree the stale-tree sweep keeps, ``finish``
removes a clean idle one and hands a kept one to its session. ``restore`` is covered through
``hermes --resume`` in ``test_tui_worktree_resume_recreate.py``."""
import subprocess
from pathlib import Path

GIT = ['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid']


def _repo(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True)
    (repo / '.gitignore').write_text('.worktrees/\n')
    subprocess.run([*GIT, '-C', str(repo), 'add', '.gitignore'], check=True)
    subprocess.run([*GIT, '-C', str(repo), 'commit', '-qm', 'fixture'], check=True)
    monkeypatch.chdir(repo)
    return repo


def _branches(repo):
    return subprocess.run(['git', '-C', str(repo), 'branch', '--format=%(refname:short)'],
                          capture_output=True, text=True, check=True).stdout.split()


def _lock_reason(info):
    from hermes_cli.worktree_ops import _worktree_lock_reason
    found, reason = _worktree_lock_reason(info['repo_root'], info['path'], 10)
    assert found
    return reason


def test_create_makes_a_hermes_locked_tree_and_finish_removes_it_when_clean_and_idle(tmp_path, monkeypatch):
    from hermes_cli.session_worktree import create, finish

    repo = _repo(tmp_path, monkeypatch)
    info = create()
    tree = Path(info['path'])
    assert tree.is_dir()
    assert tree.resolve().parent == (repo / '.worktrees').resolve()
    assert _lock_reason(info).startswith('hermes ')
    assert info['branch'] in _branches(repo)

    finish(info, None, ask=False)
    assert not tree.exists()
    assert info['branch'] not in _branches(repo)


def test_finish_keeps_a_dirty_tree_and_hands_its_lock_to_the_session(tmp_path, monkeypatch, capsys):
    from hermes_cli.session_worktree import create, finish

    repo = _repo(tmp_path, monkeypatch)
    info = create()
    tree = Path(info['path'])
    (tree / 'work.txt').write_text('uncommitted')

    finish(info, 'sess-1', ask=False)
    assert (tree / 'work.txt').read_text() == 'uncommitted'
    assert info['branch'] in _branches(repo)
    assert _lock_reason(info).startswith('hermes session=sess-1 ')
    assert f"Worktree retained for this session: {info['path']}" in capsys.readouterr().err
