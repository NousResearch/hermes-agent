import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli.update_cmd import (
    _reconcile_removed_lazy_refresh_marker_conflict,
    _restore_stashed_changes,
    _stash_local_changes_if_needed,
)


MARKER = ".lazy-refresh-incomplete"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "hermes@example.invalid")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_historical_tracked_marker_restores_as_ignored_runtime_state(tmp_path):
    """Updating from the accidental tracked-marker release stays conflict-free."""
    repo = tmp_path / "repo"
    _init_repo(repo)

    marker_payload = b"started=live\npid=999\n"
    (repo / MARKER).write_text("started=release\npid=1\n", encoding="utf-8")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", MARKER, "README.md")
    _git(repo, "commit", "-qm", "affected release")
    legacy_head = _git(repo, "rev-parse", "HEAD").stdout.strip()

    _git(repo, "rm", "-q", MARKER)
    (repo / ".gitignore").write_text(f"{MARKER}\n", encoding="utf-8")
    (repo / "release.txt").write_text("new release\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "release.txt")
    _git(repo, "commit", "-qm", "remove and ignore runtime marker")
    update_head = _git(repo, "rev-parse", "HEAD").stdout.strip()

    _git(repo, "checkout", "-q", legacy_head)
    (repo / MARKER).write_bytes(marker_payload)
    (repo / "README.md").write_text("user edit\n", encoding="utf-8")

    stash_ref = _stash_local_changes_if_needed(["git"], repo)
    assert stash_ref
    _git(repo, "reset", "--hard", update_head)

    assert _restore_stashed_changes(["git"], repo, stash_ref) is True
    assert (repo / MARKER).read_bytes() == marker_payload
    assert (repo / "README.md").read_text(encoding="utf-8") == "user edit\n"
    assert (repo / "release.txt").read_text(encoding="utf-8") == "new release\n"
    assert _git(repo, "diff", "--name-only", "--diff-filter=U").stdout == ""
    assert _git(repo, "ls-files", "--", MARKER).stdout == ""
    assert _git(repo, "check-ignore", "-q", "--", MARKER, check=False).returncode == 0
    assert _git(repo, "stash", "list").stdout == ""


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_marker_reconciliation_refuses_any_additional_conflict(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)

    assert (
        _reconcile_removed_lazy_refresh_marker_conflict(
            ["git"], repo, "stash@{0}", [MARKER, "README.md"]
        )
        is False
    )
