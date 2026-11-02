"""Invariant tests for the opt-in git-branch status-bar field."""

from datetime import datetime, timedelta

from cli import HermesCLI
from hermes_cli import status_bar_git
from hermes_cli.status_bar_git import current_git_branch


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "anthropic/claude-sonnet-4-20250514"
    cli_obj.session_start = datetime.now() - timedelta(minutes=3)
    cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
    cli_obj.agent = None
    return cli_obj


def test_current_git_branch_reads_head_and_worktree_pointer(tmp_path):
    """Branch resolves from .git/HEAD in a plain repo AND through a gitdir: pointer
    (worktree layout); a non-repo dir yields ''."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/minimax-inspired/status-bar-git-branch\n")
    status_bar_git._cache.clear()
    assert current_git_branch(str(repo)) == "minimax-inspired/status-bar-git-branch"

    # Worktree: .git is a file pointing at a private git dir with its own HEAD.
    private = tmp_path / "gitdir-store"
    private.mkdir()
    (private / "HEAD").write_text("ref: refs/heads/feature-x\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {private}\n")
    status_bar_git._cache.clear()
    assert current_git_branch(str(wt)) == "feature-x"

    plain = tmp_path / "plain"
    plain.mkdir()
    status_bar_git._cache.clear()
    assert current_git_branch(str(plain)) == ""


def test_git_branch_segment_is_opt_in(monkeypatch, tmp_path):
    """The ⎇ segment renders only when 'git_branch' is in the configured field list —
    default field set (None) never probes or shows it."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    monkeypatch.chdir(repo)
    status_bar_git._cache.clear()

    cli_obj = _make_cli()
    cli_obj._status_bar_field_set_cache = None  # default set
    snapshot = cli_obj._get_status_bar_snapshot()
    assert snapshot["git_branch"] == ""
    text = "".join(
        t for seg in cli_obj._status_bar_segments(
            snapshot, 120, None, False, styled=False) for _, t in seg)
    assert "⎇" not in text

    cli_obj2 = _make_cli()
    fields = frozenset({"model", "git_branch"})
    cli_obj2._status_bar_field_set_cache = fields
    snapshot2 = cli_obj2._get_status_bar_snapshot()
    assert snapshot2["git_branch"] == "main"
    text2 = "".join(
        t for seg in cli_obj2._status_bar_segments(
            snapshot2, 120, fields, False, styled=False) for _, t in seg)
    assert "⎇ main" in text2
