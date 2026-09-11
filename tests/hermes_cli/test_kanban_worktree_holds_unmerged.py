"""Guard 2 of the auto-decomposer (t_c52b9bc3): a child inherits the parent's worktree only
when that worktree holds unmerged work. 2026-09-11: the guard imported its predicates from
``cli``, which stopped exporting ``_worktree_is_dirty`` after upstream moved both predicates to
``hermes_cli.worktree_ops`` — so the import always failed and the guard always said "unmerged"."""
from hermes_cli import kanban_db, worktree_ops


def test_clean_pushed_worktree_is_not_unmerged(monkeypatch):
    monkeypatch.setattr(worktree_ops, "_worktree_is_dirty", lambda p, timeout=10: False)
    monkeypatch.setattr(worktree_ops, "_worktree_has_unpushed_commits", lambda p, timeout=10: False)
    assert kanban_db._worktree_holds_unmerged("/tmp/does-not-matter") is False


def test_dirty_worktree_is_unmerged(monkeypatch):
    monkeypatch.setattr(worktree_ops, "_worktree_is_dirty", lambda p, timeout=10: True)
    monkeypatch.setattr(worktree_ops, "_worktree_has_unpushed_commits", lambda p, timeout=10: False)
    assert kanban_db._worktree_holds_unmerged("/tmp/does-not-matter") is True


def test_unpushed_worktree_is_unmerged(monkeypatch):
    monkeypatch.setattr(worktree_ops, "_worktree_is_dirty", lambda p, timeout=10: False)
    monkeypatch.setattr(worktree_ops, "_worktree_has_unpushed_commits", lambda p, timeout=10: True)
    assert kanban_db._worktree_holds_unmerged("/tmp/does-not-matter") is True


def test_predicate_error_fails_safe_to_inherit(monkeypatch):
    def boom(p, timeout=10):
        raise RuntimeError("git exploded")
    monkeypatch.setattr(worktree_ops, "_worktree_is_dirty", boom)
    assert kanban_db._worktree_holds_unmerged("/tmp/does-not-matter") is True
