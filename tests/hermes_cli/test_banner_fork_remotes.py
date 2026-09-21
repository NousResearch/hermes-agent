"""Fork checkouts: the banner's "upstream" must be the official repo, not ``origin``.

A fork's ``origin`` is the fork, so every passive check compared HEAD to the fork's own main and
reported "up to date" (or nothing at all, since a private fork 404s the API) while the real
distance to NousResearch grew. The label also called the fork "upstream".
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

SHA_TIP = "3" * 40
SHA_MERGE_BASE = "4" * 40


def _repo_dir(tmp_path):
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    return repo_dir


def _as_fork():
    return patch("hermes_cli.banner._banner_remotes",
                 return_value={"official": "upstream/main", "is_fork": True})


# ---------------------------------------------------------------------------
# remote classification
# ---------------------------------------------------------------------------


def test_banner_remotes_names_the_official_remote_for_a_fork(tmp_path):
    from hermes_cli import banner

    urls = {
        "origin": "https://github.com/Sahil-SS9/KenseiAgent.git",
        "upstream": "https://github.com/NousResearch/hermes-agent.git",
    }

    def fake(args, *, cwd, timeout=5, network=False):
        return urls.get(args[-1]) if args[:2] == ["remote", "get-url"] else None

    with patch.object(banner, "_git_stdout", side_effect=fake):
        assert banner._banner_remotes(_repo_dir(tmp_path)) == {
            "official": "upstream/main", "is_fork": True}


def test_banner_remotes_plain_clone_is_not_a_fork(tmp_path):
    from hermes_cli import banner

    def fake(args, *, cwd, timeout=5, network=False):
        if args[:2] == ["remote", "get-url"]:
            return "https://github.com/NousResearch/hermes-agent.git"
        return None

    with patch.object(banner, "_git_stdout", side_effect=fake):
        assert banner._banner_remotes(_repo_dir(tmp_path)) == {
            "official": "origin/main", "is_fork": False}


# ---------------------------------------------------------------------------
# behind-count: boot-fresh from the API, never a fetch, never the fork's own main
# ---------------------------------------------------------------------------


def test_fork_check_is_fresh_not_the_stale_local_ref(tmp_path):
    """FAIL-BEFORE: the count came from the local ref, so it only moved on `git fetch upstream`."""
    from hermes_cli import banner

    def git_stdout(args, *, cwd, timeout=5, network=False):
        return SHA_MERGE_BASE if args[:1] == ["merge-base"] else None

    with (
        _as_fork(),
        patch.object(banner, "_git_stdout", side_effect=git_stdout),
        patch.object(banner, "_upstream_main_sha", return_value=SHA_TIP),
        patch.object(banner, "_github_compare_behind", return_value=771) as compare,
        patch.object(banner, "_behind_official", return_value=0) as stale,
    ):
        assert banner._check_via_local_git(_repo_dir(tmp_path)) == 771

    compare.assert_called_once_with(SHA_MERGE_BASE, SHA_TIP)
    stale.assert_not_called()


def test_fork_check_never_runs_git_fetch(tmp_path):
    """A pack on every start is what GitHub asked installs to stop doing."""
    from hermes_cli import banner

    calls = []
    real_run = banner.subprocess.run

    def spy(args, **kwargs):
        calls.append(list(args))
        return real_run(args, **kwargs)

    with (
        _as_fork(),
        patch.object(banner.subprocess, "run", side_effect=spy),
        patch.object(banner, "_upstream_main_sha", return_value=SHA_TIP),
        patch.object(banner, "_github_compare_behind", return_value=771),
    ):
        banner._check_via_local_git(_repo_dir(tmp_path))

    assert not [c for c in calls if "fetch" in c], f"no fetch allowed: {calls}"


def test_fork_check_falls_back_to_the_local_ref_offline(tmp_path):
    from hermes_cli import banner

    repo_dir = _repo_dir(tmp_path)
    with (
        _as_fork(),
        patch.object(banner, "_upstream_main_sha", return_value=None),
        patch.object(banner, "_behind_official", return_value=771) as stale,
    ):
        assert banner._check_via_local_git(repo_dir) == 771

    stale.assert_called_once_with("upstream/main", repo_dir)


# ---------------------------------------------------------------------------
# cache: a fork re-asks at boot, a plain clone keeps the day-long cache
# ---------------------------------------------------------------------------


def _cached_check(tmp_path, monkeypatch, *, is_fork, age_seconds):
    from hermes_cli import banner

    home = tmp_path / "home"
    home.mkdir()
    (home / ".update_check").write_text(json.dumps({
        "ts": time.time() - age_seconds, "behind": 5, "rev": None, "ver": banner.VERSION,
        "head": None, "target": None}), encoding="utf-8")
    monkeypatch.setattr(banner, "get_hermes_home", lambda: home)
    with (
        patch.object(banner, "_resolve_repo_dir", return_value=_repo_dir(tmp_path)),
        patch.object(banner, "_banner_remotes",
                     return_value={"official": "upstream/main" if is_fork else "origin/main",
                                   "is_fork": is_fork}),
        patch.object(banner, "_check_via_local_git", return_value=9),
    ):
        return banner.check_for_updates(passive=True)


def test_fork_reasks_after_the_floor(tmp_path, monkeypatch):
    """A day-old number is wrong for a fork — its upstream moves all day."""
    assert _cached_check(tmp_path, monkeypatch, is_fork=True, age_seconds=40 * 60) == 9


def test_fork_reuses_a_recent_check(tmp_path, monkeypatch):
    """A fleet restart helps itself to the recent answer instead of re-asking GitHub."""
    assert _cached_check(tmp_path, monkeypatch, is_fork=True, age_seconds=60) == 5


def test_plain_clone_keeps_the_day_long_cache(tmp_path, monkeypatch):
    assert _cached_check(tmp_path, monkeypatch, is_fork=False, age_seconds=40 * 60) == 5


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------


def test_fork_label_names_the_fork_not_upstream():
    """FAIL-BEFORE: "upstream 6c2d4c7a" — that sha is the fork."""
    from hermes_cli import banner

    state = {"upstream": "3f86ed75", "local": "175ec973", "ahead": 2, "fork": "6c2d4c7a"}
    with patch.object(banner, "get_git_banner_state", return_value=state):
        value = banner.format_banner_version_label()

    assert "· fork 6c2d4c7a" in value
    assert "upstream 6c2d4c7a" not in value
    assert value.endswith("· local 175ec973 (+2 carried commits)")


def test_fork_label_without_carried_commits_is_just_the_fork():
    from hermes_cli import banner

    state = {"upstream": "3f86ed75", "local": "6c2d4c7a", "ahead": 0, "fork": "6c2d4c7a"}
    with patch.object(banner, "get_git_banner_state", return_value=state):
        assert banner.format_banner_version_label().endswith("· fork 6c2d4c7a")


def test_plain_clone_label_is_unchanged():
    from hermes_cli import banner

    state = {"upstream": "b2f477a3", "local": "af8aad31", "ahead": 3}
    with patch.object(banner, "get_git_banner_state", return_value=state):
        value = banner.format_banner_version_label()

    assert value.endswith("· upstream b2f477a3 · local af8aad31 (+3 carried commits)")


def test_compute_git_banner_state_fork_adds_fork_key(tmp_path):
    from hermes_cli import banner

    results = {
        ("git", "rev-parse", "--short=8", "upstream/main"): "3f86ed75\n",
        ("git", "rev-parse", "--short=8", "HEAD"): "175ec973\n",
        ("git", "rev-list", "--count", "origin/main..HEAD"): "2\n",
        ("git", "rev-parse", "--short=8", "origin/main"): "6c2d4c7a\n",
    }

    def fake_run(cmd, **kwargs):
        return MagicMock(returncode=0, stdout=results.get(tuple(cmd), ""))

    with (
        _as_fork(),
        patch("hermes_cli.banner.subprocess.run", side_effect=fake_run),
    ):
        state = banner._compute_git_banner_state(_repo_dir(tmp_path))

    assert state == {"upstream": "3f86ed75", "local": "175ec973", "ahead": 2, "fork": "6c2d4c7a"}


# ---------------------------------------------------------------------------
# update line wording
# ---------------------------------------------------------------------------


def test_fork_notice_points_at_merging_upstream(monkeypatch):
    """`hermes update` pulls the fork, so it cannot close a gap to upstream main."""
    from hermes_cli import banner

    monkeypatch.setattr(banner, "_banner_remotes",
                        lambda repo_dir: {"official": "upstream/main", "is_fork": True})
    monkeypatch.setattr(banner, "_resolve_repo_dir", lambda: Path("/tmp/fork"))

    notice = banner._format_update_notice(771)

    assert "771 commits behind upstream" in notice
    assert "merge" in notice and "upstream/main" in notice


def test_plain_clone_notice_keeps_the_update_command(monkeypatch):
    from hermes_cli import banner

    monkeypatch.setattr(banner, "_banner_remotes",
                        lambda repo_dir: {"official": "origin/main", "is_fork": False})
    monkeypatch.setattr(banner, "_resolve_repo_dir", lambda: Path("/tmp/plain"))

    notice = banner._format_update_notice(3)

    assert "3 commits behind" in notice and "to update" in notice
    assert "upstream/main" not in notice
