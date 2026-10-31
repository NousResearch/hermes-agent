"""Canary policy on real Git refs; GitHub publication stays an inert boundary."""
from datetime import datetime, timezone, UTC
from types import SimpleNamespace
import subprocess

import pytest

from tests.scripts.test_release_tags import release, release_repo


def canary_args(date="20260818T103000Z", **options):
    """``release.py --canary --publish`` with its lane and green options at their defaults."""
    defaults = {"date": date, "publish": True, "no_changelog": True, "remote": "origin", "only": None,
                "with_lanes": None, "without_lanes": None, "require_green": True}
    return SimpleNamespace(**{**defaults, **options})


@pytest.fixture
def canary_repo(tmp_path, monkeypatch, release_repo):
    git = release_repo
    remote = tmp_path / "remote.git"
    git("init", "--bare", "-q", str(remote))
    git("remote", "add", "origin", "https://github.com/fixture/release")

    calls = []
    drafts = set()
    published = set()
    actual_run = subprocess.run

    def run(argv, *args, **kwargs):
        if argv[0] != "gh":
            if argv[:2] in (["git", "push"], ["git", "ls-remote"]):
                # A plain path, not a file:// URI: Git for Windows reads file:///C:/... as /C:/...
                argv = ["git", "-c", f"url.{remote.as_posix()}.insteadOf=https://github.com/fixture/release", *argv[1:]]
            return actual_run(argv, *args, **kwargs)
        calls.append(argv)
        if argv[1:3] == ["release", "create"]:
            assert git("--git-dir", str(remote), "rev-parse", argv[3] + "^{commit}") == git("rev-parse", "HEAD")
            drafts.add(argv[3])
        elif argv[1:3] == ["release", "view"]:
            if argv[3] not in drafts:
                return subprocess.CompletedProcess(argv, 1, "", "not found")
            return subprocess.CompletedProcess(
                argv, 0,
                '{"tagName":"' + argv[3] + '","isDraft":true,"isPrerelease":true}\n', "",
            )
        elif argv[1:3] == ["workflow", "run"]:
            assert any(call[1:3] in (["release", "create"], ["release", "view"])
                       for call in calls)
            published.add(next(value.removeprefix("tag=") for value in argv if value.startswith("tag=")))
        else:
            assert argv[1:3] == ["repo", "view"]
        return subprocess.CompletedProcess(argv, 0, stdout="main\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    which = release.shutil.which
    monkeypatch.setattr(release.shutil, "which", lambda name: "gh" if name == "gh" else which(name))
    monkeypatch.setattr(release, "generate_changelog", lambda *args, **kwargs: "fixture notes")
    from scripts.releases import versioning
    monkeypatch.setattr(versioning, "published_stable_identity", lambda repository: ("1.2.3", None))
    monkeypatch.setattr(
        versioning, "published_channel_identity",
        lambda _repository, _channel: (
            (release.release_version(release.get_last_canary_tag()), git("rev-parse", "HEAD"))
            if release.get_last_canary_tag() in published else None
        ),
    )
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    return git, remote, calls


def test_canary_cut_keeps_the_stable_core_and_dispatches_exactly_once(canary_repo):
    git, remote, calls = canary_repo
    git("commit", "--allow-empty", "-qm", "stable")
    git("tag", "v1.2.3")
    git("tag", "v9.9.9")
    git("commit", "--allow-empty", "-qm", "feat: next")
    tag = "canary.20260818T103000Z-v1.2.3"
    args = canary_args()

    release.cmd_canary(args)

    create = next(call for call in calls if call[1:3] == ["release", "create"])
    assert create[3] == tag
    assert {"--draft", "--prerelease", "--verify-tag"} <= set(create)
    dispatch = next(call for call in calls if call[1:3] == ["workflow", "run"])
    # A canary runs the Stable Release pipeline itself: the same job graph, every lane on.
    assert dispatch == [
        "gh", "workflow", "run", "stable-release.yml", "--ref", "main",
        "-f", f"tag={tag}", "--repo", "fixture/release",
    ]
    assert git("--git-dir", str(remote), "rev-parse", tag + "^{commit}") == git("rev-parse", "HEAD")
    # The tag message is the claim admission reads: every lane a canary can run, green required.
    from scripts.releases import lanes as lane_set, stable

    message = git("--git-dir", str(remote), "tag", "-l", tag, "--format=%(contents)")
    claim = stable.canary_claim(message, tag=tag, commit=git("rev-parse", "HEAD"))
    assert claim["lanes"] == lane_set.CANARY and claim["requireGreen"] is True

    calls.clear()
    release.cmd_canary(args)
    assert not any(call[1:3] in (["release", "create"], ["workflow", "run"])
                   for call in calls)


def test_unchanged_head_does_not_cut_another_timestamp(canary_repo):
    git, _remote, calls = canary_repo
    git("commit", "--allow-empty", "-qm", "stable")
    git("tag", "v1.2.3")
    git("commit", "--allow-empty", "-qm", "feat: next")
    release.cmd_canary(canary_args())
    calls.clear()

    release.cmd_canary(canary_args("20260818T103001Z"))

    assert not any(call[1:3] in (["release", "create"], ["workflow", "run"])
                   for call in calls)


def test_incomplete_canary_redispatches_the_existing_receipt(canary_repo, monkeypatch):
    git, _remote, calls = canary_repo
    git("commit", "--allow-empty", "-qm", "stable")
    git("tag", "v1.2.3")
    git("commit", "--allow-empty", "-qm", "feat: next")
    args = canary_args()
    release.cmd_canary(args)
    calls.clear()

    from scripts.releases import versioning
    monkeypatch.setattr(versioning, "published_channel_identity", lambda *_args: None)
    release.cmd_canary(args)

    assert [call[1:3] for call in calls].count(["workflow", "run"]) == 1
    assert not any(call[1:3] == ["release", "create"] for call in calls)


def test_the_claim_records_the_chosen_lanes_and_green_policy(canary_repo):
    git, remote, calls = canary_repo
    git("commit", "--allow-empty", "-qm", "stable")
    git("tag", "v1.2.3")
    git("commit", "--allow-empty", "-qm", "feat: next")
    from scripts.releases import lanes as lane_set, stable

    # A canary can never run the signed-package upgrade arms; asking for them cuts nothing.
    with pytest.raises(SystemExit, match="cannot run upgrade"):
        release.cmd_canary(canary_args(with_lanes=["upgrade"]))
    assert not git("tag", "--list", "canary.*") and calls == []
    tag = "canary.20260818T103000Z-v1.2.3"
    release.cmd_canary(canary_args(without_lanes=["macos-intel,termux"], require_green=False))
    message = git("--git-dir", str(remote), "tag", "-l", tag, "--format=%(contents)")
    claim = stable.canary_claim(message, tag=tag, commit=git("rev-parse", "HEAD"))
    assert claim["lanes"] == lane_set.CANARY - {"macos-intel", "termux"}
    assert claim["requireGreen"] is False


def test_tag_shape_and_prune_use_canonical_receipts(canary_repo, monkeypatch, capsys):
    git, _, calls = canary_repo
    assert release.canary_tag_for_date("0.27.4", "20260818T103000Z") == "canary.20260818T103000Z-v0.27.4"
    assert release.canary_tag_for_date("v1.4.0", "20261231T235959Z") == "canary.20261231T235959Z-v1.4.0"
    tags = [
        "canary.20260801T000000Z-v0.27.1",
        "canary.20260801T235959Z-v0.27.1",
        "canary.20260804T000000Z-v0.27.1",
        "canary.20260804T235959Z-v0.27.1",
        # The retired suffix shape ages out through the same window.
        "v0.27.0+canary.20260801T120000Z",
        "v0.27.0+canary.20260805T120000Z",
        "v0.27.1+canary.invalid",
        "v0.27.1",
        "v2026.7.20",
    ]
    for tag in tags:
        git("tag", tag)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 18, tzinfo=UTC)

    monkeypatch.setattr(release, "datetime", Clock)
    release.prune_old_canaries(SimpleNamespace(remote="origin", publish=False))
    deleted = {
        line.removeprefix("Would delete ")
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("Would delete ")
    }
    assert deleted == {*tags[:2], "v0.27.0+canary.20260801T120000Z"}
    assert calls == []
    assert set(git("tag", "--list").splitlines()) == set(tags)


def test_stable_dispatch_and_failure(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(release.shutil, "which", lambda _: "gh")
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or subprocess.CompletedProcess(command, 0, "", ""),
    )
    monkeypatch.setattr(release, "_default_branch", lambda _repo: "main")
    assert release.dispatch_desktop_build("v1.2.3", "owner/repo")
    # Stable dispatches the default-branch workflow, like canary; the tag is an input.
    assert calls == [[
        "gh", "workflow", "run", "stable-release.yml", "--ref", "main",
        "-f", "tag=v1.2.3", "--repo", "owner/repo",
    ]]
    with pytest.raises(ValueError):
        release.dispatch_desktop_build("v1.2.3/other", "owner/repo")
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "refused"),
    )
    assert release.dispatch_desktop_build("v1.2.3", "owner/repo") is False
    monkeypatch.setattr(release.shutil, "which", lambda _: None)
    assert release.dispatch_desktop_build(
        "canary.20260818T103000Z-v1.2.3", "owner/repo"
    ) is False
