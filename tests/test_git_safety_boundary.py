"""A removed live-checkout guard can damage only disposable repositories here."""
from pathlib import Path
import shlex
import subprocess

import pytest


@pytest.mark.platforms("posix")
def test_guard_blocks_native_and_shell_git_mutations_without_touching_checkout(tmp_path, monkeypatch):
    from tests import conftest

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
    monkeypatch.setattr(conftest, "_LIVE_GUARD_PROTECTED_GIT_ROOTS", (protected,))
    head = git(protected, "rev-parse", "HEAD")
    (protected / "sentinel").write_bytes(b"uncommitted user data")
    for command in (["git", "-C", str(protected), "reset", "--hard", "HEAD~1"],
                    ["sh", "-c", f"git -C {shlex.quote(str(protected))} checkout -- sentinel"]):
        with pytest.raises(RuntimeError, match="live-system guard"):
            subprocess.run(command, check=True)
        assert git(protected, "rev-parse", "HEAD") == head
        assert (protected / "sentinel").read_bytes() == b"uncommitted user data"
    old = git(ordinary, "rev-parse", "HEAD~1")
    git(ordinary, "reset", "--hard", old)
    assert git(ordinary, "rev-parse", "HEAD") == old
    assert not (ordinary / "sentinel").exists()
