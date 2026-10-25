from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from evals.artifact_review_install_rehearsal import rehearse


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

