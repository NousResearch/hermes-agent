"""Failed creation must not publish a partial Kanban checkout (#126004)."""
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db_workspace as workspace


@pytest.mark.parametrize('failure', ['timeout', 'nonzero'])
@pytest.mark.parametrize('explicit', [False, True])
def test_failed_checkout_retry_materializes_tracked_files(tmp_path, monkeypatch, failure, explicit):
    repo = tmp_path / 'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test',
            '-c', 'user.email=test@example.com', '-c', 'commit.gpgsign=false', *args],
            check=True, capture_output=True, text=True)
    git('init', '-b', 'main')
    (repo / 'payload.txt').write_text('required tracked payload', encoding='utf8')
    git('add', 'payload.txt')
    git('commit', '-m', 'fixture')
    target = repo / '.worktrees' / 'task'
    task = SimpleNamespace(id='task', workspace_kind='worktree',
        workspace_path=str(target if explicit else repo), branch_name='wt/task')
    original = workspace._git
    def interrupted(root, *args, **kwargs):
        if args[:2] == ('worktree', 'add'):
            # Real Git metadata with no populated files models the interrupted
            # checkout deterministically, without a slow monorepo or timing race.
            original(root, 'worktree', 'add', '--no-checkout', *args[2:], **kwargs)
            if failure == 'timeout':
                raise subprocess.TimeoutExpired(['git', *args], 60)
            return subprocess.CompletedProcess(args, 1, '', 'checkout interrupted')
        return original(root, *args, **kwargs)
    monkeypatch.setattr(workspace, '_git', interrupted)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        workspace.resolve_workspace(task)
    monkeypatch.setattr(workspace, '_git', original)
    actual = workspace.resolve_workspace(task)
    assert (actual / 'payload.txt').is_file(), 'retry returned a partial checkout as usable'
    assert (actual / 'payload.txt').read_text(encoding='utf8') == 'required tracked payload'
    (actual / 'local.txt').write_text('keep my changes', encoding='utf8')
    assert workspace.resolve_workspace(task) == actual
    assert (actual / 'local.txt').read_text(encoding='utf8') == 'keep my changes'


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('mode', ['cleanup-refused', 'in-progress', 'existing-directory'])
def test_unpublished_checkout_never_becomes_usable(tmp_path, monkeypatch, explicit, mode):
    repo = tmp_path / 'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test',
            '-c', 'user.email=test@example.com', '-c', 'commit.gpgsign=false', *args],
            check=True, capture_output=True, text=True)
    git('init', '-b', 'main')
    (repo / 'payload.txt').write_text('tracked', encoding='utf8')
    git('add', 'payload.txt')
    git('commit', '-m', 'fixture')
    target = repo / '.worktrees' / 'task'
    task = SimpleNamespace(id='task', workspace_kind='worktree',
        workspace_path=str(target if explicit else repo), branch_name='wt/task')
    if mode == 'existing-directory':
        target = tmp_path / 'external-target'
        target.mkdir()
        (target / 'precious.txt').write_text('user data', encoding='utf8')
    def resolve():
        if mode == 'existing-directory':
            return workspace._ensure_git_worktree(repo, target, 'wt/task')
        return workspace.resolve_workspace(task)
    original = workspace._git
    def interrupted(root, *args, **kwargs):
        if args[:2] == ('worktree', 'add'):
            if mode == 'existing-directory':
                return original(root, *args, **kwargs)
            result = original(root, 'worktree', 'add', '--no-checkout', *args[2:], **kwargs)
            assert result.returncode == 0
            if mode == 'in-progress':
                # Re-enter while the real Git metadata exists but creation has
                # not returned: neither public acceptance path may publish it.
                with pytest.raises(RuntimeError, match='unfinished'):
                    workspace.resolve_workspace(task)
            raise subprocess.TimeoutExpired(['git', *args], 60)
        if args[:2] == ('worktree', 'remove'):
            return subprocess.CompletedProcess(args, 1, '', 'fixture: handle still open')
        return original(root, *args, **kwargs)
    monkeypatch.setattr(workspace, '_git', interrupted)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        resolve()
    monkeypatch.setattr(workspace, '_git', original)
    with pytest.raises(RuntimeError, match='unfinished'):
        resolve()
    assert target.is_dir()
    if mode == 'existing-directory':
        assert (target / 'precious.txt').read_text(encoding='utf8') == 'user data'
    else:
        assert not (target / 'payload.txt').exists()
        assert git('show-ref', '--verify', 'refs/heads/wt/task').returncode == 0
