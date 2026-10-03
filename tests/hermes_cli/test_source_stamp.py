"""Source checkout identity is written only from the checkout itself."""

import json
import os
from pathlib import Path
import subprocess


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
        env={"HOME": str(repo.parent), "PATH": os.environ["PATH"]},
    )
    return result.stdout.strip()


def _init_repo(repo: Path, message: str) -> None:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "hermes@example.invalid")
    (repo / "tracked").write_text(message + "\n", encoding="utf-8")
    _git(repo, "add", "tracked")
    _git(repo, "commit", "-qm", message)


def test_write_source_stamp_records_live_checkout_identity_atomically(tmp_path):
    from hermes_cli.source_stamp import write_source_stamp

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "hermes@example.invalid")
    (repo / "tracked").write_text("release\n", encoding="utf-8")
    _git(repo, "add", "tracked")
    _git(repo, "commit", "-qm", "release")
    _git(repo, "tag", "v0.21.4")
    (repo / "tracked").write_text("next\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "next")

    written = write_source_stamp(repo)
    stored = json.loads((repo / "install-stamp.json").read_text(encoding="utf-8-sig"))

    assert stored == written
    assert stored["commit"] == _git(repo, "rev-parse", "HEAD")
    assert stored["baseVersion"] == "0.21.4"
    assert stored["displayVersion"].startswith("0.21.4+1.g")
    assert stored["source"] == "git"
    assert stored["distribution"] is None
    assert stored["updateMechanism"] == "self"
    assert not list(repo.glob(".install-stamp.*.tmp"))


def test_stale_source_stamp_defers_to_live_checkout(tmp_path, monkeypatch):
    from hermes_cli.source_stamp import write_source_stamp
    from hermes_cli.version_info import _reset_version_info_cache, get_version_info

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "hermes@example.invalid")
    (repo / "tracked").write_text("release\n", encoding="utf-8")
    _git(repo, "add", "tracked")
    _git(repo, "commit", "-qm", "release")
    _git(repo, "tag", "v0.21.4")
    write_source_stamp(repo)

    (repo / "tracked").write_text("manual pull\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "manual pull")
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(repo))
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: repo)
    _reset_version_info_cache()

    info = get_version_info()

    assert info.commit == _git(repo, "rev-parse", "HEAD")
    assert info.derived_version.startswith("0.21.4+1.g")
    assert info.source == "git"


def test_unresolvable_release_keeps_the_previous_one(tmp_path):
    """One unreadable release must not replace a known one with ``unknown``."""
    from hermes_cli.source_stamp import write_source_stamp

    repo = tmp_path / "repo"
    _init_repo(repo, "release")
    _git(repo, "tag", "v0.21.4")
    (repo / "tracked").write_text("next\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "next")
    first = write_source_stamp(repo)
    assert first["baseVersion"] == "0.21.4"
    assert first["distance"] == 1

    # No release is reachable any more: the live read can only answer "unknown".
    _git(repo, "tag", "-d", "v0.21.4")
    (repo / "tracked").write_text("another\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "another")

    second = write_source_stamp(repo)

    assert second is not None
    assert second["commit"] == _git(repo, "rev-parse", "HEAD")
    assert second["baseVersion"] == "0.21.4"
    assert second["displayVersion"].startswith("0.21.4+1.g")
    assert second["distance"] == 1
    stored = json.loads((repo / "install-stamp.json").read_text(encoding="utf-8-sig"))
    assert stored == second


def test_unresolvable_release_without_a_prior_stamp_still_records_the_commit(tmp_path):
    """A checkout that never had a release keeps the documented ``unknown`` shape."""
    from hermes_cli.source_stamp import write_source_stamp

    repo = tmp_path / "repo"
    _init_repo(repo, "no tags here")

    written = write_source_stamp(repo)

    assert written is not None
    assert written["baseVersion"] == "unknown"
    assert written["distance"] is None
    assert written["displayVersion"] == f"git.{_git(repo, 'rev-parse', 'HEAD')[:7]}"


def test_prior_stamp_that_is_itself_unresolvable_is_not_reused(tmp_path):
    """A stamped ``unknown`` is not an identity to carry forward."""
    from hermes_cli.source_stamp import write_source_stamp

    repo = tmp_path / "repo"
    _init_repo(repo, "no tags here")
    (repo / "install-stamp.json").write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "commit": "0" * 40,
                "baseVersion": "unknown",
                "displayVersion": "git.deadbee",
                "distance": None,
            }
        ),
        encoding="utf-8",
    )

    written = write_source_stamp(repo)

    assert written is not None
    assert written["baseVersion"] == "unknown"
    assert written["displayVersion"].startswith(f"git.{_git(repo, 'rev-parse', 'HEAD')[:7]}")
