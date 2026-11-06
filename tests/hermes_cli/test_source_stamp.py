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