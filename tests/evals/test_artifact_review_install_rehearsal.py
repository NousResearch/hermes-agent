from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from evals.artifact_review_install_rehearsal import _write_receipt_exclusive, rehearse


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "component.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "component.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "fixture"], check=True)
    return tmp_path


def test_rehearsal_is_deterministic_and_reports_non_activation(repo: Path):
    commit = _git(repo, "rev-parse", "HEAD")
    first = rehearse(repo, commit, ["component.py"])
    second = rehearse(repo, commit, ["component.py"])

    assert first == second
    assert first["checks"]["rollback_restored_prior_bytes"] is True
    assert first["checks"]["live_install_touched"] is False
    unsigned = {key: value for key, value in first.items() if key != "receipt_sha256"}
    assert first["receipt_sha256"] == hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_rehearsal_refuses_wrong_revision_and_dirty_checkout(repo: Path):
    commit = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="expected"):
        rehearse(repo, "0" * 40, ["component.py"])

    (repo / "component.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        rehearse(repo, commit, ["component.py"])


@pytest.mark.parametrize("component", ["../outside", "/etc/passwd", "a/../../outside", r"a\outside"])
def test_rehearsal_refuses_component_path_escape(repo: Path, component: str):
    with pytest.raises(ValueError, match="unsafe|escapes"):
        rehearse(repo, _git(repo, "rev-parse", "HEAD"), [component])


def test_rehearsal_requires_exact_tracked_regular_blob(repo: Path):
    (repo / "link.py").symlink_to("component.py")
    subprocess.run(["git", "-C", str(repo), "add", "link.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "add symlink"], check=True)
    with pytest.raises(ValueError, match="tracked regular Git file"):
        rehearse(repo, _git(repo, "rev-parse", "HEAD"), ["link.py"])


def test_rehearsal_scrubs_caller_git_environment_and_disables_fsmonitor(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    marker = outside / "fsmonitor-ran"
    monitor = outside / "monitor.sh"
    monitor.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    monitor.chmod(0o755)
    subprocess.run(["git", "-C", str(repo), "config", "core.fsmonitor", str(monitor)], check=True)
    commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(outside / "missing-objects"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(outside / "caller-index"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(outside / "caller-config"))

    receipt = rehearse(repo, commit, ["component.py"])

    assert receipt["checks"]["caller_git_environment_scrubbed"] is True
    assert not marker.exists()
    assert not (outside / "caller-index").exists()


def test_receipt_writer_never_replaces_file_or_follows_final_symlink(tmp_path: Path):
    output = tmp_path / "receipt.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _write_receipt_exclusive(output, {"ok": True})
    assert output.read_text(encoding="utf-8") == "keep"

    target = tmp_path / "target.json"
    target.write_text("target", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        _write_receipt_exclusive(link, {"ok": True})
    assert target.read_text(encoding="utf-8") == "target"

