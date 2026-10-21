"""The launcher tells the owner before it removes a session's worktree (#134889 TUI-27).

A submit that lands while ``git worktree remove`` runs would be admitted and only fail later,
when the turn tries to run in a directory that is gone. The mark refuses it up front, by name,
and comes down once the removal finished.
"""
import pytest


def _info(tmp_path, session_id='sess-1'):
    return {'path': str(tmp_path / 'wt'), 'repo_root': str(tmp_path / 'repo'), 'branch': 'feature',
            'exit_choice': 'discard', 'session_id': session_id}


def test_discarding_a_tree_marks_it_retiring_first_and_clears_it_after(tmp_path, monkeypatch):
    from hermes_cli import worktree_exit, worktree_ops

    calls = []
    monkeypatch.setattr(worktree_ops, 'mark_workspace_retiring',
                        lambda path, session_id=None, *, retiring=True:
                        calls.append((path, session_id, retiring)))
    removed = []
    monkeypatch.setattr(worktree_ops, '_git_quiet', lambda args, *a, **k: removed.append(args))

    assert worktree_exit.apply_exit_choice(_info(tmp_path), lambda text: None) is True
    assert calls == [(str(tmp_path / 'wt'), 'sess-1', True), (str(tmp_path / 'wt'), 'sess-1', False)]
    assert removed[0][:2] == ['worktree', 'unlock']
    assert removed[1][:2] == ['worktree', 'remove']
    # The mark goes up before the tree is gone and comes down after it is.
    assert calls[0][2] is True and calls[1][2] is False


def test_a_tree_without_a_session_is_never_marked(tmp_path, monkeypatch):
    from hermes_cli import worktree_exit, worktree_ops

    calls = []
    monkeypatch.setattr(worktree_ops, 'mark_workspace_retiring',
                        lambda path, session_id=None, *, retiring=True:
                        calls.append((path, session_id, retiring)))
    monkeypatch.setattr(worktree_ops, '_git_quiet', lambda args, *a, **k: None)

    info = _info(tmp_path)
    info.pop('session_id')
    assert worktree_exit.apply_exit_choice(info, lambda text: None) is True
    assert calls == []


def test_the_mark_never_starts_a_gateway(tmp_path, monkeypatch):
    """The launcher attaches to an owner that is already running; it never starts one."""
    from hermes_cli import worktree_ops

    def explode(*args, **kwargs):
        raise AssertionError('the launcher must not start a gateway')

    monkeypatch.setattr('hermes_cli.gateway_runtime.ensure_gateway_runtime', explode)
    assert worktree_ops.mark_workspace_retiring(str(tmp_path / 'wt'), 'sess-1') is False


@pytest.mark.parametrize('path,session_id', [('/tmp/wt', None), ('', 'sess-1')])
def test_a_mark_without_a_tree_or_a_session_is_skipped(path, session_id):
    from hermes_cli import worktree_ops

    assert worktree_ops.mark_workspace_retiring(path, session_id) is False
