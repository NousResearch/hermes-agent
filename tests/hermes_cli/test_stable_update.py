"""Tests for the opt-in stable-tag update check.

Covers the pure helpers in ``hermes_cli.stable_update`` plus the wiring in
``hermes_cli.banner.check_for_updates`` that routes to stable-tag mode only
when config opts in. Default (config absent) behaviour must stay the legacy
branch/commit-distance check, unchanged.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import stable_update


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)


def _commit(path: Path, msg: str) -> None:
    (path / "f").write_text(msg, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Hermes Test",
            "-c",
            "user.email=hermes-test@example.invalid",
            "commit",
            "-q",
            "-m",
            msg,
        ],
        cwd=path,
        check=True,
    )


# --- config gate ---------------------------------------------------------

def test_disabled_by_default():
    assert stable_updates_absent() is False


def stable_updates_absent() -> bool:
    return stable_update.stable_updates_enabled({})


def test_enabled_via_check_strategy():
    cfg = {"updates": {"check_strategy": "stable-tags"}}
    assert stable_update.stable_updates_enabled(cfg) is True


def test_enabled_via_bool_flag():
    cfg = {"updates": {"stable_tags": True}}
    assert stable_update.stable_updates_enabled(cfg) is True


def test_branch_strategy_stays_disabled():
    cfg = {"updates": {"check_strategy": "branch"}}
    assert stable_update.stable_updates_enabled(cfg) is False


def test_settings_defaults():
    s = stable_update.stable_tag_settings(None)
    assert s["pattern"] == stable_update.DEFAULT_STABLE_TAG_PATTERN
    assert s["remote"] == stable_update.DEFAULT_STABLE_TAG_REMOTE


def test_settings_overrides():
    cfg = {
        "updates": {
            "stable_tag_pattern": "release-*",
            "stable_tag_remote": "upstream",
        }
    }
    s = stable_update.stable_tag_settings(cfg)
    assert s["pattern"] == "release-*"
    assert s["remote"] == "upstream"


# --- version parsing -----------------------------------------------------

@pytest.mark.parametrize(
    "tag,expected",
    [
        ("v1.2.3", (1, 2, 3)),
        ("v2.0", (2, 0, 0)),
        ("v10.4.1", (10, 4, 1)),
        ("v1.2.3-rc1", None),
        ("nightly", None),
        ("1.2.3", None),
    ],
)
def test_parse_version(tag, expected):
    assert stable_update._parse_version(tag) == expected


# --- tag resolution against a real local remote --------------------------

def test_latest_stable_tag_picks_highest(tmp_path):
    remote = tmp_path / "remote"
    _init_repo(remote)
    _commit(remote, "one")
    for tag in ["v1.0.0", "v1.2.0", "v1.10.0", "v2.0.0-rc1"]:
        subprocess.run(["git", "tag", tag], cwd=remote, check=True)

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(remote), str(clone)], check=True
    )
    # v1.10.0 > v1.2.0 numerically; the rc is excluded.
    latest = stable_update.latest_stable_tag(clone, remote="origin")
    assert latest == "v1.10.0"


def test_update_available_when_behind(tmp_path):
    remote = tmp_path / "remote"
    _init_repo(remote)
    _commit(remote, "one")
    subprocess.run(["git", "tag", "v1.0.0"], cwd=remote, check=True)

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(remote), str(clone)], check=True
    )
    subprocess.run(["git", "checkout", "-q", "v1.0.0"], cwd=clone, check=True)

    # Remote gains a newer stable release.
    _commit(remote, "two")
    subprocess.run(["git", "tag", "v1.1.0"], cwd=remote, check=True)

    status = stable_update.stable_update_status(clone, remote="origin")
    assert status["latest_tag"] == "v1.1.0"
    assert status["current_tag"] == "v1.0.0"
    assert status["update_available"] is True
    assert status["error"] is None


def test_up_to_date_when_on_latest(tmp_path):
    remote = tmp_path / "remote"
    _init_repo(remote)
    _commit(remote, "one")
    subprocess.run(["git", "tag", "v1.0.0"], cwd=remote, check=True)

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(remote), str(clone)], check=True
    )
    subprocess.run(["git", "checkout", "-q", "v1.0.0"], cwd=clone, check=True)

    status = stable_update.stable_update_status(clone, remote="origin")
    assert status["update_available"] is False
    assert status["error"] is None


def test_overlay_commits_on_latest_tag_not_flagged(tmp_path):
    # A customized install carries commits on top of the newest stable tag;
    # the tag is still reachable from HEAD, so it is NOT an update.
    remote = tmp_path / "remote"
    _init_repo(remote)
    _commit(remote, "one")
    subprocess.run(["git", "tag", "v1.0.0"], cwd=remote, check=True)

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(remote), str(clone)], check=True
    )
    _commit(clone, "local overlay")
    subprocess.run(["git", "tag", "v1.1.0-rc1"], cwd=clone, check=True)

    status = stable_update.stable_update_status(clone, remote="origin")
    assert status["current_tag"] == "v1.0.0"
    assert status["update_available"] is False


def test_no_remote_tags_errors_soft(tmp_path):
    remote = tmp_path / "remote"
    _init_repo(remote)
    _commit(remote, "one")

    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(remote), str(clone)], check=True
    )
    status = stable_update.stable_update_status(clone, remote="origin")
    assert status["update_available"] is False
    assert status["error"] == "no-remote-stable-tags"


def test_not_a_git_checkout(tmp_path):
    status = stable_update.stable_update_status(tmp_path / "nope")
    assert status["error"] == "not-a-git-checkout"
    assert status["update_available"] is False


# --- banner wiring: default unchanged ------------------------------------

def test_banner_default_does_not_use_stable_tags(monkeypatch):
    from hermes_cli import banner

    # No config -> _stable_tag_mode False -> legacy path chosen.
    monkeypatch.setattr(banner, "_stable_tag_mode", lambda: False)
    called = {"legacy": False, "stable": False}

    def fake_legacy(_repo):
        called["legacy"] = True
        return 3

    def fake_stable(_repo):
        called["stable"] = True
        return 0

    monkeypatch.setattr(banner, "_check_via_local_git", fake_legacy)
    monkeypatch.setattr(banner, "_check_via_stable_tag", fake_stable)
    monkeypatch.setenv("HERMES_REVISION", "")
    monkeypatch.setattr(banner.os.environ, "get", lambda *a, **k: None)

    # Force a git-checkout repo_dir and skip docker short-circuit + cache.
    monkeypatch.setattr(
        banner, "get_hermes_home", lambda: Path("/nonexistent-home-xyz")
    )
    banner.check_for_updates()
    assert called["legacy"] is True
    assert called["stable"] is False


def test_banner_stable_mode_routes_to_stable(monkeypatch):
    from hermes_cli import banner

    monkeypatch.setattr(banner, "_stable_tag_mode", lambda: True)
    called = {"legacy": False, "stable": False}

    def fake_legacy(_repo):
        called["legacy"] = True
        return 3

    def fake_stable(_repo):
        called["stable"] = True
        return banner.UPDATE_AVAILABLE_NO_COUNT

    monkeypatch.setattr(banner, "_check_via_local_git", fake_legacy)
    monkeypatch.setattr(banner, "_check_via_stable_tag", fake_stable)
    monkeypatch.setattr(banner.os.environ, "get", lambda *a, **k: None)
    monkeypatch.setattr(
        banner, "get_hermes_home", lambda: Path("/nonexistent-home-xyz")
    )
    banner.check_for_updates()
    assert called["stable"] is True
    assert called["legacy"] is False
