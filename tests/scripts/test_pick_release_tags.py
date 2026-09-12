"""Reachability filter for scripts/sandbox/pick-release-tags.sh (#100947)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "sandbox" / "pick-release-tags.sh"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "pick-release-tags-test")
    _git(repo, "config", "user.email", "pick-release-tags-test@example.test")
    return repo


def _commit(repo: Path, name: str) -> str:
    (repo / "marker").write_text(name + "\n", encoding="utf-8")
    _git(repo, "add", "marker")
    _git(repo, "commit", "-m", name)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _pick(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "--repo", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "TZ": "UTC"},
    )


def test_drops_tags_that_are_not_ancestors_of_head(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _commit(repo, "oldest")
    _git(repo, "tag", "v2026.1.1")
    _commit(repo, "newest")
    _git(repo, "tag", "v2026.2.1")

    _git(repo, "checkout", "--orphan", "rewritten")
    _git(repo, "rm", "-rf", ".", check=False)
    _commit(repo, "orphaned-history")
    _git(repo, "tag", "v2026.0.9")
    _git(repo, "checkout", "main")

    result = _pick(repo, "--count", "5")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["v2026.1.1", "v2026.2.1"]


def test_annotated_and_lightweight_reachable_tags(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _commit(repo, "light")
    _git(repo, "tag", "v2026.3.1")
    _commit(repo, "annotated")
    _git(repo, "tag", "-a", "v2026.3.2", "-m", "release")

    result = _pick(repo, "--count", "5")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["v2026.3.1", "v2026.3.2"]


def test_errors_when_every_release_tag_is_unreachable(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _commit(repo, "mainline")
    _git(repo, "checkout", "--orphan", "other")
    _git(repo, "rm", "-rf", ".", check=False)
    _commit(repo, "side")
    _git(repo, "tag", "v2026.4.1")
    _git(repo, "checkout", "main")

    result = _pick(repo, "--count", "5")
    assert result.returncode == 1
    assert "no reachable release tags" in result.stderr


def test_shallow_clone_without_history_does_not_emit_a_false_empty_matrix(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    _commit(repo, "root")
    _git(repo, "tag", "v2026.5.1")
    _commit(repo, "tip")
    _git(repo, "tag", "v2026.5.2")

    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "--depth", "1", f"file://{repo}", str(shallow)],
        check=True,
        capture_output=True,
        text=True,
    )
    # Match the GHA picker: tag names are fetched, commit objects may not be.
    subprocess.run(
        ["git", "-C", str(shallow), "fetch", "--tags", "--force", f"file://{repo}"],
        check=True,
        capture_output=True,
        text=True,
    )
    # Depth-1 + tag names: the oldest tag is listed but not ancestral until
    # the script deepens. After that both tags are ancestors of HEAD.
    result = _pick(shallow, "--count", "5")
    assert result.returncode == 0, result.stderr
    picked = json.loads(result.stdout)
    assert "v2026.5.2" in picked
    assert "v2026.5.1" in picked
