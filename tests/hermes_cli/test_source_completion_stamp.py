"""A source checkout publishes identity only after successful completion."""

import json
import os
from pathlib import Path
import subprocess
import sys

from hermes_cli.source_completion import complete_source_checkout
from hermes_cli.source_stamp import write_source_stamp


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes@example.invalid")
    (root / "tracked").write_text("release\n", encoding="utf-8")
    git("add", "tracked")
    git("commit", "-qm", "release")
    git("tag", "v0.21.4")
    return root


def _completion_dependencies(monkeypatch, maintenance):
    monkeypatch.setattr("hermes_cli.venv_sync.publish_launchers", lambda root: None)
    monkeypatch.setattr("hermes_cli.source_build.build_update_products", lambda root, *, desktop: None)
    monkeypatch.setattr("hermes_cli.update_cmd_maint._run_post_update_maintenance", maintenance)


def test_successful_source_completion_writes_checkout_identity(tmp_path, monkeypatch):
    root = _repo(tmp_path)

    def maintenance(**_kwargs):
        assert not (root / "install-stamp.json").exists()
        return True

    _completion_dependencies(monkeypatch, maintenance)

    assert complete_source_checkout(root, desktop=False, assume_yes=True)
    assert (root / "install-stamp.json").is_file()


def test_failed_source_completion_does_not_publish_identity(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    _completion_dependencies(monkeypatch, lambda **_kwargs: False)

    assert not complete_source_checkout(root, desktop=False, assume_yes=True)
    assert not (root / "install-stamp.json").exists()


def _verify_bootstrap_receipt(root: Path) -> subprocess.CompletedProcess:
    """The same verifier the Windows install/update E2E runs after an update."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "verify-bootstrap-version-stamp.py"
    return subprocess.run([sys.executable, "-B", str(script), "--stamp", str(root / ".hermes-bootstrap-complete"),
                           "--repo", str(root)], capture_output=True, text=True, encoding="utf-8")


def test_publishing_checkout_identity_moves_an_installer_receipt_to_head(tmp_path):
    # write_source_stamp is the one seam: the completion handoff, the PM updater's finish and
    # boot-time adoption all publish identity through it, and the receipt must follow every one.
    root = _repo(tmp_path)
    release = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, encoding="utf-8", check=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, capture_output=True, text=True, encoding="utf-8", check=True).stdout.strip()
    # What install.sh / install.ps1's complete stage leaves behind at the installed release.
    (root / ".hermes-bootstrap-complete").write_text(json.dumps({
        "schemaVersion": 1, "pinnedCommit": release, "pinnedBranch": branch, "completedAt": "2026-06-19T00:00:00.000Z",
    }), encoding="utf-8")
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "update"], cwd=root, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"})
    assert write_source_stamp(root) is not None
    result = _verify_bootstrap_receipt(root)
    assert result.returncode == 0, result.stdout + result.stderr


def test_publishing_checkout_identity_never_invents_an_installer_receipt(tmp_path):
    # The receipt's presence is what marks a script install; a manual clone stays one.
    root = _repo(tmp_path)

    assert write_source_stamp(root) is not None
    assert not (root / ".hermes-bootstrap-complete").exists()


def test_shallow_checkout_publishes_its_release_after_fetching_the_commit_graph(tmp_path):
    # Pre-PM installers cloned --depth 1; the completion fetches commits (not trees) so
    # the stamp can still name the release the checkout is built on.
    from hermes_cli.gitlock import fetch_full_commit_graph

    server = _repo(tmp_path)
    subprocess.run(["git", "config", "uploadpack.allowFilter", "true"], cwd=server, check=True)
    for message in ("one", "two", "three"):
        subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", message], cwd=server, check=True,
                       capture_output=True, env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                                                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"})
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "-q", "--depth", "1", server.as_uri(), str(checkout)], check=True, capture_output=True)

    assert fetch_full_commit_graph(checkout)
    stamp = write_source_stamp(checkout)

    assert stamp is not None
    assert (stamp["baseVersion"], stamp["distance"]) == ("0.21.4", 3)
    assert not fetch_full_commit_graph(checkout)
