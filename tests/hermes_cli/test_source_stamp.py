"""Source checkout identity is written only from the checkout itself."""

import json
import os
from pathlib import Path
import subprocess

import pytest


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


@pytest.mark.parametrize("mechanism", ["self", "external"])
def test_source_stamp_keeps_only_source_runtime_binding(tmp_path, monkeypatch, mechanism):
    from hermes_cli.source_stamp import write_source_stamp

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "hermes@example.invalid")
    (repo / "tracked").write_text("release\n", encoding="utf-8")
    _git(repo, "add", "tracked")
    _git(repo, "commit", "-qm", "release")
    persistent = tmp_path / "persistent-runtime"
    scratch = tmp_path / "scratch-runtime"
    (repo / "install-stamp.json").write_text(json.dumps({
        "updateMechanism": mechanism, "runtimeDir": str(persistent),
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(scratch))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "scratch-home"))

    written = write_source_stamp(repo)

    assert written is not None
    stored = json.loads((repo / "install-stamp.json").read_text())
    if mechanism == "self":
        assert written["runtimeDir"] == stored["runtimeDir"] == str(persistent)
    else:
        assert "runtimeDir" not in written
        assert "runtimeDir" not in stored


@pytest.mark.parametrize("failure", ["unreadable", "malformed"])
def test_source_stamp_does_not_erase_unreadable_runtime_binding(tmp_path, monkeypatch, failure):
    from hermes_cli.source_stamp import write_source_stamp

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "hermes@example.invalid")
    (repo / "tracked").write_text("release\n", encoding="utf-8")
    _git(repo, "add", "tracked")
    _git(repo, "commit", "-qm", "release")
    stamp = repo / "install-stamp.json"
    stamp.write_text("{not readable json", encoding="utf-8")
    original = stamp.read_bytes()
    if failure == "unreadable":
        read_text = Path.read_text

        def deny_stamp_read(path, *args, **kwargs):
            if path == stamp:
                raise PermissionError("source stamp cannot read prior binding")
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", deny_stamp_read)

    with pytest.raises((OSError, ValueError)):
        write_source_stamp(repo)
    assert stamp.read_bytes() == original


@pytest.mark.parametrize("bad_binding", ["relative/path", 17, ""])
def test_source_stamp_does_not_discard_invalid_existing_runtime_binding(tmp_path, bad_binding):
    from hermes_cli.source_stamp import write_source_stamp

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "hermes@example.invalid")
    (repo / "tracked").write_text("release\n", encoding="utf-8")
    _git(repo, "add", "tracked")
    _git(repo, "commit", "-qm", "release")
    stamp = repo / "install-stamp.json"
    stamp.write_text(json.dumps({
        "updateMechanism": "self", "runtimeDir": bad_binding,
    }), encoding="utf-8")
    original = stamp.read_bytes()

    with pytest.raises(ValueError, match="runtimeDir"):
        write_source_stamp(repo)
    assert stamp.read_bytes() == original


@pytest.mark.parametrize("pinned", [True, False])
def test_gitless_source_stamp_drops_stale_commit_but_keeps_runtime_ownership(tmp_path, pinned):
    from hermes_cli.source_stamp import write_source_stamp

    repo = tmp_path / "gitless-source"
    repo.mkdir()
    previous = {"updateMechanism": "self", "commit": "stale-commit"}
    if pinned:
        previous["runtimeDir"] = str(tmp_path / "persistent-store")
    stamp = repo / "install-stamp.json"
    stamp.write_text(json.dumps(previous), encoding="utf-8")

    assert write_source_stamp(repo) is None
    retained = json.loads(stamp.read_text(encoding="utf-8"))
    assert retained["updateMechanism"] == "self"
    assert "commit" not in retained
    assert retained.get("runtimeDir") == previous.get("runtimeDir")