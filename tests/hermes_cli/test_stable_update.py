"""Tests for stable-tag update helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, name: str) -> str:
    (repo / f"{name}.txt").write_text(name)
    _git(repo, "add", f"{name}.txt")
    _git(repo, "commit", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


def test_stable_update_status_reports_latest_tag(tmp_path):
    from hermes_cli.stable_update import stable_update_status

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")

    _commit(repo, "old")
    _git(repo, "tag", "v2026.5.7")
    newest = _commit(repo, "new")
    _git(repo, "tag", "v2026.5.16")

    _git(repo, "checkout", "-q", "v2026.5.7")
    status = stable_update_status(repo, fetch=False)

    assert status["latest_tag"] == "v2026.5.16"
    assert status["target_tag"] == "v2026.5.16"
    assert status["target_commit"] == newest
    assert status["current_tag"] == "v2026.5.7"
    assert status["current_release_tag"] == "v2026.5.7"
    assert status["releases_behind"] == 1
    assert status["up_to_date"] is False
    assert status["update_available"] is True


def test_stable_update_status_up_to_date_on_latest_tag(tmp_path):
    from hermes_cli.stable_update import stable_update_status

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")

    _commit(repo, "release")
    _git(repo, "tag", "v2026.5.16")

    status = stable_update_status(repo, fetch=False)

    assert status["latest_tag"] == "v2026.5.16"
    assert status["releases_behind"] == 0
    assert status["up_to_date"] is True
    assert status["update_available"] is False


def test_stable_updates_enabled_accepts_strategy_and_legacy_bool():
    from hermes_cli.stable_update import configured_update_channel, stable_updates_enabled

    assert stable_updates_enabled({"updates": {"channel": "release"}})
    assert stable_updates_enabled({"updates": {"check_strategy": "stable-tags"}})
    assert stable_updates_enabled({"updates": {"stable_tags": True}})
    assert not stable_updates_enabled({"updates": {"check_strategy": "branch"}})
    assert configured_update_channel({"updates": {"channel": "fast-track"}}) == "main"


def test_overlay_commit_on_latest_release_is_not_exact_pin(tmp_path):
    from hermes_cli.stable_update import stable_update_status

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")

    _commit(repo, "release")
    _git(repo, "tag", "v2026.5.16")
    _git(repo, "checkout", "-q", "-b", "local-overlay")
    _commit(repo, "overlay")

    status = stable_update_status(repo, fetch=False)

    assert status["current_branch"] == "local-overlay"
    assert status["current_release_tag"] == "v2026.5.16"
    assert status["releases_behind"] == 0
    assert status["up_to_date"] is False
    assert status["update_available"] is True


def test_moving_main_after_latest_release_still_offers_release_pin(tmp_path):
    from hermes_cli.stable_update import stable_update_status

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "checkout", "-q", "-B", "main")

    _commit(repo, "release")
    _git(repo, "tag", "v2026.5.16")
    _commit(repo, "unreleased-main")

    status = stable_update_status(repo, fetch=False)

    assert status["current_branch"] == "main"
    assert status["current_release_tag"] == "v2026.5.16"
    assert status["releases_behind"] == 0
    assert status["up_to_date"] is False
    assert status["update_available"] is True
