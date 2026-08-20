"""Tests for the update check mechanism in hermes_cli.banner."""

import json
import threading
import time
from unittest.mock import MagicMock, patch


def _fake_checkout(tmp_path):
    repo_dir = tmp_path / "hermes-agent"
    (repo_dir / "hermes_cli").mkdir(parents=True)
    (repo_dir / "hermes_cli" / "banner.py").touch()
    (repo_dir / ".git").mkdir()
    return repo_dir


def test_check_for_updates_uses_cache(tmp_path, monkeypatch):
    """When cache inputs match, check_for_updates returns cached value without fetch."""
    import hermes_cli.banner as banner
    from hermes_cli import __version__

    repo_dir = _fake_checkout(tmp_path)
    cache_rev = f"git:{repo_dir}:HEAD=head-sha:origin/main=upstream-sha"
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(
        json.dumps({"ts": time.time(), "behind": 3, "rev": cache_rev, "ver": __version__})
    )

    monkeypatch.setattr(banner, "__file__", str(repo_dir / "hermes_cli" / "banner.py"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["git", "rev-parse", "--verify"] and cmd[3] == "HEAD":
            return MagicMock(returncode=0, stdout="head-sha\n")
        if cmd[:3] == ["git", "rev-parse", "--verify"] and cmd[3] == "refs/remotes/origin/main":
            return MagicMock(returncode=0, stdout="upstream-sha\n")
        raise AssertionError(f"unexpected git command: {cmd}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run) as mock_run:
        result = banner.check_for_updates()

    assert result == 3
    assert mock_run.call_count == 2  # HEAD + origin/main cache-key probes only


def test_check_for_updates_writes_post_fetch_fingerprint_then_hits_cache(tmp_path, monkeypatch):
    """If fetch moves origin/main, cache the post-fetch ref for the next invocation."""
    import hermes_cli.banner as banner
    from hermes_cli import __version__

    repo_dir = _fake_checkout(tmp_path)
    cache_file = tmp_path / ".update_check"

    monkeypatch.setattr(banner, "__file__", str(repo_dir / "hermes_cli" / "banner.py"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    refs = {"HEAD": "head-sha", "refs/remotes/origin/main": "old-upstream-sha"}
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(tuple(cmd))
        if cmd[:3] == ["git", "rev-parse", "--verify"]:
            rev = cmd[3]
            return MagicMock(returncode=0, stdout=f"{refs[rev]}\n")
        if cmd[:2] == ["git", "remote"]:
            return MagicMock(returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n")
        if cmd[:3] == ["git", "rev-parse", "--is-shallow-repository"]:
            return MagicMock(returncode=0, stdout="false\n")
        if cmd[:3] == ["git", "fetch", "origin"]:
            refs["refs/remotes/origin/main"] = "new-upstream-sha"
            return MagicMock(returncode=0, stdout="")
        if cmd[:3] == ["git", "rev-list", "--count"]:
            return MagicMock(returncode=0, stdout="1\n")
        raise AssertionError(f"unexpected git command: {cmd}")

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        assert banner.check_for_updates() == 1

    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached["behind"] == 1
    assert cached["ver"] == __version__
    assert cached["rev"] == f"git:{repo_dir}:HEAD=head-sha:origin/main=new-upstream-sha"
    assert any(call[:3] == ("git", "fetch", "origin") for call in calls)

    calls.clear()
    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        assert banner.check_for_updates() == 1

    assert calls == [
        ("git", "rev-parse", "--verify", "HEAD"),
        ("git", "rev-parse", "--verify", "refs/remotes/origin/main"),
    ]


def test_prefetch_non_blocking():
    """prefetch_update_check() should return immediately without blocking."""
    import hermes_cli.banner as banner

    # Reset module state
    banner._update_result = None
    banner._update_check_done = threading.Event()

    with patch.object(banner, "check_for_updates", return_value=5):
        start = time.monotonic()
        banner.prefetch_update_check()
        elapsed = time.monotonic() - start

        # Should return almost immediately (well under 1 second)
        assert elapsed < 1.0

        # Wait for the background thread to finish
        banner._update_check_done.wait(timeout=5)
        assert banner._update_result == 5
