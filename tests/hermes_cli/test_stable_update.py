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
    assert status["up_to_date"] is True
    assert status["update_available"] is False


def test_stable_updates_enabled_accepts_strategy_and_legacy_bool():
    from hermes_cli.stable_update import stable_updates_enabled

    assert stable_updates_enabled({"updates": {"check_strategy": "stable-tags"}})
    assert stable_updates_enabled({"updates": {"stable_tags": True}})
    assert not stable_updates_enabled({"updates": {"check_strategy": "branch"}})


def test_stable_update_status_overlay_commit_is_not_an_update(tmp_path):
    """An overlay commit on top of the newest stable tag is NOT an update.

    Customized installs carry local commits on top of the latest stable tag.
    HEAD then differs from the tag commit, so an equality check would falsely
    report an update. Freshness must be reachability: the tag is still an
    ancestor of HEAD, so the release is present and up_to_date is True.
    """
    from hermes_cli.stable_update import stable_update_status

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")

    tagged = _commit(repo, "release")
    _git(repo, "tag", "v2026.5.16")
    overlay = _commit(repo, "local-overlay")  # HEAD now sits above the tag

    status = stable_update_status(repo, fetch=False)

    # HEAD is the overlay commit, NOT the tag commit: equality would fail here.
    assert status["head"] == overlay
    assert status["target_commit"] == tagged
    assert status["head"] != status["target_commit"]
    # But the tag is reachable from HEAD, so it is up to date, not an update.
    assert status["reachable"] is True
    assert status["up_to_date"] is True
    assert status["update_available"] is False


def test_stable_update_status_newer_tag_unreachable_is_an_update(tmp_path):
    """A newer stable tag that is NOT reachable from HEAD is a real update."""
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

    # Sit on the older tag with a local overlay so HEAD != tag commit but the
    # newest tag is genuinely ahead (unreachable) and must report an update.
    _git(repo, "checkout", "-q", "v2026.5.7")
    _commit(repo, "overlay-on-old")

    status = stable_update_status(repo, fetch=False)

    assert status["target_tag"] == "v2026.5.16"
    assert status["target_commit"] == newest
    assert status["reachable"] is False
    assert status["up_to_date"] is False
    assert status["update_available"] is True
