"""A removed live-checkout guard can damage only disposable repositories here."""
from pathlib import Path
import shlex
import subprocess

import pytest


@pytest.mark.platforms("posix")
def test_guard_blocks_native_and_shell_git_mutations_without_touching_checkout(tmp_path, monkeypatch):
    from tests._fixtures import live_system_guard

    def git(repo, *args):
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()

    protected, ordinary = tmp_path / "protected", tmp_path / "ordinary"
    for repo in (protected, ordinary):
        repo.mkdir()
        git(repo, "init")
        git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m", "first")
        (repo / "sentinel").write_text("committed", encoding="utf-8")
        git(repo, "add", "sentinel")
        git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "second")
    monkeypatch.setattr(live_system_guard, "_LIVE_GUARD_PROTECTED_GIT_ROOTS", (protected,))
    head = git(protected, "rev-parse", "HEAD")
    (protected / "sentinel").write_bytes(b"uncommitted user data")
    for command in (
        ["git", "-C", str(protected), "reset", "--hard", "HEAD~1"],
        ["sh", "-c", f"git -C {shlex.quote(str(protected))} checkout -- sentinel"],
        *(["git", "-C", str(protected), *args] for args in (
            ("commit", "-am", "overwrite"), ("add", "-A"), ("rm", "-r", "."),
            ("config", "url.https://example.invalid/.insteadOf", "git@example.invalid:"),
            ("update-ref", "refs/heads/main", head), ("branch", "-f", "main", head),
            ("worktree", "add", str(tmp_path / "new-worktree")), ("tag", "unsafe-tag"),
        )),
    ):
        with pytest.raises(RuntimeError, match="live-system guard"):
            subprocess.run(command, check=True)
        assert git(protected, "rev-parse", "HEAD") == head
        assert (protected / "sentinel").read_bytes() == b"uncommitted user data"
    assert not (tmp_path / "new-worktree").exists()
    assert git(protected, "tag", "--list", "unsafe-tag") == ""
    assert "url.https://example.invalid/.insteadof" not in git(protected, "config", "--local", "--list")
    old = git(ordinary, "rev-parse", "HEAD~1")
    git(ordinary, "reset", "--hard", old)
    assert git(ordinary, "rev-parse", "HEAD") == old
    assert not (ordinary / "sentinel").exists()


def test_checkout_guard_covers_write_verbs_without_blocking_queries(tmp_path):
    from tests.git_safety import blocked_git_mutation

    root = tmp_path / "checkout"
    root.mkdir()
    options = {"cwd": root}
    for argv in (
        ["commit", "-am", "change"], ["add", "-A"], ["rm", "-r", "."],
        ["config", "url.https://example.invalid/.insteadOf", "git@example.invalid:"],
        ["config", "--file", ".git/config", "url.https://example.invalid/.pushInsteadOf", "git@x:"],
        ["config", "set", "url.https://example.invalid/.insteadOf", "git@example.invalid:"],
        ["update-ref", "refs/heads/main", "abc"], ["branch", "-f", "main"],
        ["worktree", "add", "../other"], ["tag", "v1"],
        ["tag", "--delete", "--list", "v1"], ["tag", "--list", "--force", "v1"],
    ):
        assert blocked_git_mutation(["git", *argv], options, (root,)) == argv[0]
        assert blocked_git_mutation(["git", *argv], {"cwd": tmp_path}, (root,)) is None
    for argv in (["status"], ["config", "--get", "url.x.insteadOf"], ["config", "get", "url.x.insteadOf"],
                 ["worktree", "list"], ["branch", "--show-current"], ["tag", "-l"],
                 ["tag", "--merged", "HEAD", "--list", "v[0-9]*"]):
        assert blocked_git_mutation(["git", *argv], options, (root,)) is None
