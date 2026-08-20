"""Tests for _restore_feature_branch_after_update."""

from __future__ import annotations
import subprocess
import pytest
from types import SimpleNamespace
from hermes_cli.update_cmd import _restore_feature_branch_after_update

GIT = ["git"]

def _git(repo, *argv, check=True):
    return subprocess.run(
        GIT + list(argv), cwd=repo, capture_output=True, text=True, check=check
    )

def _current_branch(repo) -> str:
    return _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "feature")
    (repo / "feature.txt").write_text("feature\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "feature work")
    _git(repo, "checkout", "main")
    (repo / "upstream.txt").write_text("upstream\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "upstream update")
    return repo

def _patch_syntax_guard(monkeypatch, ok=True, failing="hermes_cli/config.py"):
    import hermes_cli.update_cmd as uc
    monkeypatch.setattr(
        uc,
        "_validate_critical_files_syntax",
        lambda root: (True, None, None) if ok else (False, failing, "boom"),
    )

def test_returns_to_feature_branch_rebased(repo, monkeypatch):
    _patch_syntax_guard(monkeypatch, ok=True)
    _restore_feature_branch_after_update(GIT, repo, "feature", "main")
    assert _current_branch(repo) == "feature"
    merge_base = _git(repo, "merge-base", "feature", "main").stdout.strip()
    main_sha = _git(repo, "rev-parse", "main").stdout.strip()
    assert merge_base == main_sha
    assert (repo / "feature.txt").exists()
    assert (repo / "upstream.txt").exists()

def test_noop_when_already_on_target(repo, monkeypatch):
    _patch_syntax_guard(monkeypatch, ok=True)
    _restore_feature_branch_after_update(GIT, repo, "main", "main")
    assert _current_branch(repo) == "main"

def test_noop_for_detached_head_marker(repo, monkeypatch):
    _patch_syntax_guard(monkeypatch, ok=True)
    _restore_feature_branch_after_update(GIT, repo, "HEAD", "main")
    assert _current_branch(repo) == "main"

def test_conflict_falls_back_to_target_and_preserves_branch(repo, monkeypatch):
    _patch_syntax_guard(monkeypatch, ok=True)
    _git(repo, "checkout", "feature")
    (repo / "base.txt").write_text("feature version\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "feature edits base")
    feature_sha = _git(repo, "rev-parse", "feature").stdout.strip()
    _git(repo, "checkout", "main")
    (repo / "base.txt").write_text("main version\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "main edits base")
    _restore_feature_branch_after_update(GIT, repo, "feature", "main")
    assert _current_branch(repo) == "main"
    assert _git(repo, "rev-parse", "feature").stdout.strip() == feature_sha
    assert not (repo / ".git" / "rebase-merge").exists()
    assert not (repo / ".git" / "rebase-apply").exists()

def test_syntax_guard_failure_returns_to_target(repo, monkeypatch):
    _patch_syntax_guard(monkeypatch, ok=False)
    _restore_feature_branch_after_update(GIT, repo, "feature", "main")
    assert _current_branch(repo) == "main"
    merge_base = _git(repo, "merge-base", "feature", "main").stdout.strip()
    assert merge_base == _git(repo, "rev-parse", "main").stdout.strip()

def test_end_to_end_dirty_feature_branch_stash_conflict(repo, monkeypatch, capsys):
    import hermes_cli.update_cmd as uc
    from types import SimpleNamespace
    
    # 1. We are on feature branch
    _git(repo, "checkout", "feature")
    
    # 2. Main branch has an edit on base.txt
    _git(repo, "checkout", "main")
    (repo / "base.txt").write_text("upstream main edits\n")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "advance main")
    
    # 3. We are back on feature branch
    _git(repo, "checkout", "feature")
    
    # 4. We make a dirty edit to base.txt
    (repo / "base.txt").write_text("feature uncommitted\n")

    # Mocks
    monkeypatch.setattr(uc, "_is_fork", lambda url: False)
    monkeypatch.setattr(uc, "_venv_core_imports_healthy", lambda: (True, None))
    monkeypatch.setattr(uc, "_desktop_app_present", lambda d: False)
    
    class _MockM:
        def __init__(self):
            self.PROJECT_ROOT = repo
        def __getattr__(self, name):
            if name == 'PROJECT_ROOT': return getattr(self, name)
            if name == '_stash_local_changes_if_needed':
                from hermes_cli.update_cmd import _stash_local_changes_if_needed as f
                return f
            if name == '_restore_stashed_changes':
                from hermes_cli.update_cmd import _restore_stashed_changes as f
                return f
            if name == '_discard_stashed_changes':
                from hermes_cli.update_cmd import _discard_stashed_changes as f
                return f
            if name == '_is_windows': return lambda *a, **k: False
            if name == '_get_origin_url': return lambda *a, **k: 'http'
            if name == '_resolve_update_branch': return lambda *a, **k: 'main'
            if name == '_capture_active_lazy_features': return lambda *a, **k: []
            if name == '_capture_active_tool_dependencies': return lambda *a, **k: []
            if name == '_update_marker_path': return lambda *a, **k: self.PROJECT_ROOT / ".update_marker"
            if name == '_lazy_refresh_marker_path': return lambda *a, **k: self.PROJECT_ROOT / ".lazy_marker"
            if name == '_pytest_owns_live_checkout': return lambda *a, **k: True
            if name == '_kill_stale_dashboard_processes': return lambda *a, **k: {}
            return lambda *a, **k: None
    monkeypatch.setattr(uc, "_m", _MockM)

    real_run = subprocess.run
    def mock_run(cmd, *args, **kwargs):
        if cmd[:2] == ["git", "fetch"]:
            _git(repo, "branch", "origin/main", "main")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "rev-list"]:
            return SimpleNamespace(returncode=0, stdout="1\n", stderr="")
        if cmd[:2] == ["git", "rev-parse"] and "--is-shallow-repository" in cmd:
            return SimpleNamespace(returncode=0, stdout="false\n", stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr("sys.exit", lambda code: None)

    args = SimpleNamespace(yes=True, force=True, dev=False)
    uc._cmd_update_impl(args, gateway_mode=False)

    out = capsys.readouterr().out
    
    assert "Returning to your branch 'feature'" in out
    assert "rebased onto main and checked out" in out
    assert "restoring local changes hit conflicts" in out
    assert _current_branch(repo) == "feature"
    content = (repo / "base.txt").read_text()
    assert content == "upstream main edits\n"

