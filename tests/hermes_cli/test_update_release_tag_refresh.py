"""The release tag that names a build must reach the checkout (#143361).

Every update fetch uses ``--no-tags``, and a checkout already on the release commit
fetches no tree at all, so ``refs/tags/v*`` never arrives and ``version_info()``
reports the previous release plus its distance. The post-build refresh now fetches
just the release tag instead of enumerating every published tag — that wildcard can
exceed the tag-refresh bound on a slow link and leave the tag missing.
"""

from contextlib import nullcontext
import subprocess
from pathlib import Path

from hermes_cli import gitlock
from hermes_cli import update_cmd_check
from hermes_cli.source_releases import SourceTarget

_RELEASE_SHA = "4d04303818b61bf3aa09278af48356315515965b"


class _Args:
    def __init__(self, branch=None, channel=None):
        self.branch = branch
        self.channel = channel


def _select_for(tmp_path, monkeypatch, target):
    class _Main:
        PROJECT_ROOT = tmp_path

    class _Facade:
        def _m(self):
            return _Main

        def _update_run_channel(self, args):
            return "stable"

    import hermes_cli.config
    import hermes_cli.release_channels
    import hermes_cli.source_releases
    import hermes_cli.update_channel

    monkeypatch.setattr(update_cmd_check, "_uc", lambda: _Facade())
    monkeypatch.setattr(hermes_cli.config, "require_readable_config_before_write", lambda *a, **k: {})
    monkeypatch.setattr(hermes_cli.update_channel, "channel_record", lambda *a, **k: {"channel": "stable"})
    monkeypatch.setattr(hermes_cli.update_channel, "rides_default_channel", lambda *a, **k: False)
    monkeypatch.setattr(hermes_cli.release_channels, "retrying_reads", lambda: nullcontext())
    monkeypatch.setattr(hermes_cli.source_releases, "resolve_source_target", lambda *a, **k: target)

    request = {"home": str(tmp_path)}
    update_cmd_check.select_apply_target(_Args(), "main", request, git_cmd="git", stop=lambda: None)
    return request


def test_release_tag_is_carried_for_the_forward_only_path(tmp_path, monkeypatch):
    """HEAD already contains the release; the tag is still needed for the display."""
    target = SourceTarget("stable", "stable", "NousResearch/hermes-agent",
                          commit=_RELEASE_SHA, version="0.22.4", tag="v0.22.4", ahead=True)
    request = _select_for(tmp_path, monkeypatch, target)

    assert request["release_tag"] == "v0.22.4"
    assert request["expected_sha"] == _RELEASE_SHA


def test_no_release_tag_recorded_when_the_target_has_none(tmp_path, monkeypatch):
    """A target without a published tag must not invent one."""
    target = SourceTarget("canary", "canary", "NousResearch/hermes-agent",
                          commit=_RELEASE_SHA, version="0.22.4", ahead=True)
    request = _select_for(tmp_path, monkeypatch, target)

    assert "release_tag" not in request


def test_fetch_full_commit_graph_pins_one_tag_when_the_release_is_known(monkeypatch, tmp_path):
    """The wildcard tag refresh is replaced by the one tag that names this build."""
    captured = {}

    def fake_run_git(cmd, args, **kwargs):
        captured["args"] = list(args)
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(gitlock, "run_git", fake_run_git)
    monkeypatch.setattr(gitlock, "_shallow_file_path", lambda root: None)
    monkeypatch.setattr(gitlock, "_partial_clone_filter", lambda root, **kw: None)

    gitlock.fetch_full_commit_graph(tmp_path, release_tag="v0.22.4")

    assert "refs/tags/v0.22.4:refs/tags/v0.22.4" in captured["args"]
    assert "refs/tags/v*:refs/tags/v*" not in captured["args"]


def test_fetch_full_commit_graph_keeps_the_wildcard_without_a_release_tag(monkeypatch, tmp_path):
    """No known release tag (canary, branch targets) keeps the previous behaviour."""
    captured = {}

    def fake_run_git(cmd, args, **kwargs):
        captured["args"] = list(args)
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(gitlock, "run_git", fake_run_git)
    monkeypatch.setattr(gitlock, "_shallow_file_path", lambda root: None)
    monkeypatch.setattr(gitlock, "_partial_clone_filter", lambda root, **kw: None)

    gitlock.fetch_full_commit_graph(tmp_path)

    assert "refs/tags/v*:refs/tags/v*" in captured["args"]


def _server_repo_with_releases(tmp_path):
    """Real repo with two release tags; a --no-tags clone never learns them."""
    import os

    server = tmp_path / "server"
    server.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}

    def git(*args):
        subprocess.run(["git", *args], cwd=server, env=env, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    (server / "tracked").write_text("release\n", encoding="utf-8")
    git("add", "tracked")
    git("commit", "-qm", "release")
    git("tag", "v0.21.5")
    git("commit", "-q", "--allow-empty", "-m", "after release")
    git("tag", "v0.21.6")
    return server


def test_one_tag_fetch_restores_the_version_the_build_reports(tmp_path, monkeypatch):
    """The refresh fetches the one tag naming HEAD, and version_info() then reports it (#143361)."""
    from hermes_cli import gitlock
    from hermes_cli.version_info import _git_version_info

    server = _server_repo_with_releases(tmp_path)
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "-q", "--no-tags", server.as_uri(), str(checkout)],
                   check=True, capture_output=True)

    # HEAD is the release commit, but --no-tags never delivered its tag.
    assert _git_version_info(checkout).base_version == "unknown"

    fetches = []
    real = gitlock.run_git

    def recording(git_cmd, args, **kwargs):
        if args[:1] == ["fetch"]:
            fetches.append(list(args))
        return real(git_cmd, args, **kwargs)

    monkeypatch.setattr(gitlock, "run_git", recording)

    gitlock.fetch_full_commit_graph(checkout, release_tag="v0.21.6")

    assert len(fetches) == 1, fetches  # one ref, not a wildcard sweep
    assert "refs/tags/v0.21.6:refs/tags/v0.21.6" in fetches[0]
    version = _git_version_info(checkout)
    assert version.base_version == "0.21.6"
    assert version.display_version == "0.21.6"
