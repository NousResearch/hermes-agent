import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.kanban_ceiling import RoleCeiling, ScopeError, resolve_scope
from hermes_cli.kanban_projection import build_projection


def git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src/pkg").mkdir(parents=True)
    (repo / "src/a.py").write_text("a")
    (repo / "src/pkg/b.py").write_text("b")
    (repo / "tests/x.py").parent.mkdir()
    (repo / "tests/x.py").write_text("x")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=test", "commit", "-qm", "initial"],
        cwd=repo, check=True,
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return repo, sha


def test_scope_clips_wider_request_and_nested_paths(tmp_path):
    repo, sha = git_repo(tmp_path)
    ceiling = RoleCeiling(frozenset({"src/**"}), frozenset({"src/pkg/**"}))
    scope = resolve_scope(SimpleNamespace(scope_paths=["**"]), ceiling, repo=repo, base_sha=sha)
    assert scope.read == frozenset({"src/a.py", "src/pkg/b.py"})
    assert scope.write == frozenset({"src/pkg/b.py"})


def test_empty_intersection_is_empty(tmp_path):
    repo, sha = git_repo(tmp_path)
    scope = resolve_scope(SimpleNamespace(scope_paths=["tests/**"]), RoleCeiling(frozenset({"src/**"})), repo=repo, base_sha=sha)
    assert not scope.paths
    with pytest.raises(ScopeError, match="empty scope"):
        build_projection(sha, scope, tmp_path / "projection", repo=repo)


def test_projection_has_only_granted_files_and_no_git(tmp_path):
    repo, sha = git_repo(tmp_path)
    scope = resolve_scope(SimpleNamespace(scope_paths=["src/pkg/**"]), RoleCeiling(frozenset({"src/**"})), repo=repo, base_sha=sha)
    dest = tmp_path / "projection"
    handle = build_projection(sha, scope, dest, repo=repo)
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*")) == ["src", "src/pkg", "src/pkg/b.py"]
    assert not (dest / ".git").exists()
    # The hash manifest a later .git-free ingest diff needs as a baseline is
    # returned to the caller, never written inside the worker-controlled
    # destination (a worker that can edit a file could edit an in-workspace
    # manifest just as easily, hiding the tamper from the diff).
    assert handle.manifest == {"src/pkg/b.py": hashlib.sha256(b"b").hexdigest()}


def test_no_repo_clip_never_widens_past_ceiling():
    # Without repo/base_sha, a request wider than the ceiling ("**") must clip
    # down to the ceiling's own pattern, never keep the wider request.
    scope = resolve_scope(SimpleNamespace(scope_paths=["**"]), RoleCeiling(frozenset({"src/**"})))
    assert scope.read == frozenset({"src/**"})


def test_path_traversal_is_rejected(tmp_path):
    with pytest.raises(ScopeError, match="traversal"):
        resolve_scope(SimpleNamespace(scope_paths=["../outside"]), RoleCeiling(frozenset({"**"})))


def test_symlink_is_rejected(tmp_path):
    repo, sha = git_repo(tmp_path)
    (repo / "src/link").symlink_to("a.py")
    subprocess.run(["git", "add", "src/link"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "link"], cwd=repo, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    with pytest.raises(ScopeError, match="symlink"):
        resolve_scope(SimpleNamespace(scope_paths=["src/link"]), RoleCeiling(frozenset({"src/**"})), repo=repo, base_sha=sha)
