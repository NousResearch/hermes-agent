"""An unchosen default channel never moves a checkout backward on a guess."""
import subprocess

import pytest

from hermes_cli import source_check, source_releases


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def history(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    shas = []
    for n in range(3):
        (origin / "f").write_text(str(n))
        _git(origin, "add", "f")
        _git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", str(n))
        shas.append(_git(origin, "rev-parse", "HEAD"))
    return origin, shas  # shas[0] is the "release", HEAD (shas[2]) is ahead of it


@pytest.mark.parametrize("release_fetched", [False, True])
@pytest.mark.parametrize("compare", [None, {"status": "ahead"}])
def test_shallow_checkout_ahead_of_release_stays_put(history, tmp_path, monkeypatch, compare, release_fetched):
    origin, shas = history
    clone = tmp_path / "shallow"
    _git(tmp_path, "clone", "-q", "--depth", "1", f"file://{origin}", str(clone))
    if release_fetched:  # present but cut off by the shallow boundary: is-ancestor exits 1
        _git(origin, "tag", "v0.0.1", shas[0])
        _git(clone, "fetch", "-q", "--depth", "1", "origin", "tag", "v0.0.1")
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: compare)
    # Shallow history hides the release commit: only GitHub can prove the relation.
    expected = "contains" if compare else None
    assert source_releases._head_relation(["git"], clone, shas[0], "o/r") == (shas[2], expected)
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: {"status": "behind"})
    assert source_releases._head_relation(["git"], clone, shas[0], "o/r") == (shas[2], "behind")


def test_full_history_behind_release_is_proof(history, tmp_path, monkeypatch):
    origin, shas = history
    clone = tmp_path / "full"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "checkout", "-q", shas[0])
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: pytest.fail("no network on proof"))
    assert source_releases._head_relation(["git"], clone, shas[2], "o/r") == (shas[0], "behind")
    assert source_releases._head_relation(["git"], clone, shas[0], "o/r") == (shas[0], "contains")


@pytest.mark.parametrize("blobless", [False, True])
@pytest.mark.parametrize("compare,relation", [
    (None, None), ({"status": "behind"}, "behind"),
    ({"status": "ahead"}, None),  # full history already proved HEAD lacks it
])
def test_full_clone_missing_a_newer_release_is_never_called_ahead(history, tmp_path, monkeypatch,
                                                                    blobless, compare, relation):
    """The release is cut after the install's last fetch, so its commit is not local yet."""
    origin, shas = history
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", *(["--filter=blob:none"] if blobless else []), f"file://{origin}", str(clone))
    (origin / "f").write_text("release")
    _git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "release")
    release = _git(origin, "rev-parse", "HEAD")
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: compare)
    assert source_releases._head_relation(["git"], clone, release, "o/r") == (shas[2], relation)


@pytest.mark.parametrize("start", ["diverged", "unknown"])
def test_unchosen_default_follows_main_instead_of_detaching_onto_the_release(history, tmp_path, monkeypatch,
                                                                             start):
    origin, shas = history
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", f"file://{origin}", str(clone))
    release = shas[2]
    if start == "diverged":  # local work on a base older than the release
        _git(clone, "checkout", "-q", "-b", "mine", shas[1])
        (clone / "f").write_text("local")
        _git(clone, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "local")
    else:  # the release is not local and GitHub cannot say how HEAD relates to it
        _git(clone, "checkout", "-q", shas[1])
        release = "0" * 39 + "1"
    monkeypatch.setattr(source_releases, "_resolve_stable", lambda repository, *_: source_releases.SourceTarget(
        "stable", "stable", repository, commit=release, version="1.0.0"))
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: None)
    target = source_releases.resolve_source_target("stable", ["git"], clone, repository="o/r", forward_only=True)
    assert (target.commit, target.branch, target.ahead) == (None, "main", False)
    assert target.main_fallback == ("diverged" if start == "diverged" else "unknown")
    # A chosen subscription still lands exactly on the release.
    chosen = source_releases.resolve_source_target("stable", ["git"], clone, repository="o/r")
    assert chosen.commit == release


@pytest.mark.parametrize("url,remote", [
    ("git@github.com:NousResearch/hermes-agent.git", "https://github.com/NousResearch/hermes-agent.git"),
    ("ssh://git@github.com/NousResearch/hermes-agent", "https://github.com/NousResearch/hermes-agent.git"),
    ("https://github.com/NousResearch/hermes-agent.git", "origin"),
    ("git@github.com:someone/hermes-agent.git", "origin"),  # a fork keeps its own auth
])
def test_official_ssh_origin_verifies_the_release_tag_over_https(tmp_path, url, remote):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "remote", "add", "origin", url)
    repository = source_releases.source_repository(["git"], tmp_path)
    assert (source_releases.official_https_remote(url, repository) or "origin") == remote


def test_apply_fetches_a_missing_release_and_decides_locally(history, tmp_path, monkeypatch):
    """select_apply_target wires the fetch: a release cut after the last fetch lands, no GitHub."""
    from types import SimpleNamespace

    from hermes_cli import main, update_cmd, update_cmd_check

    origin, shas = history
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", f"file://{origin}", str(clone))
    (origin / "f").write_text("release")
    _git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "release")
    release = _git(origin, "rev-parse", "HEAD")
    monkeypatch.setattr(main, "PROJECT_ROOT", clone)
    monkeypatch.setattr(update_cmd, "_update_run_channel", lambda args: "stable")
    monkeypatch.setattr("hermes_cli.config.require_readable_config_before_write", lambda path: {})
    monkeypatch.setattr("hermes_cli.update_channel.rides_default_channel", lambda *a: True)
    monkeypatch.setattr(source_releases, "_resolve_stable", lambda repository, *_: source_releases.SourceTarget(
        "stable", "stable", repository, commit=release, version="1.0.0"))
    monkeypatch.setattr(source_check, "_github_compare", lambda *a, **k: pytest.fail("decided locally"))
    request = {"home": str(tmp_path), "branch": "main"}
    target_ref, release_sha, target_is_head, _ = update_cmd_check.select_apply_target(
        SimpleNamespace(branch=None, channel=None), "main", request, git_cmd=["git"], stop=lambda: None)
    assert (target_ref, release_sha, target_is_head, request["expected_sha"]) == (release, release, False, release)
    assert _git(clone, "cat-file", "-t", release) == "commit"


def test_apply_fetch_reads_an_official_ssh_origin_over_https_in_the_checkout(history, tmp_path, monkeypatch):
    """No SSH use (GIT_SSH_COMMAND=false), and the fetch runs in the checkout it was given."""
    from hermes_cli import main, update_cmd_check

    origin, _ = history
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", f"file://{origin}", str(clone))
    _git(clone, "remote", "set-url", "origin", "git@github.com:NousResearch/hermes-agent.git")
    _git(clone, "config", f"url.file://{origin}.insteadOf", source_releases.OFFICIAL_HTTPS_URL)
    (origin / "f").write_text("release")
    _git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "release")
    release = _git(origin, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_SSH_COMMAND", "false")
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path / "elsewhere")
    assert update_cmd_check._fetch_commit(["git"], clone)(release)
    assert _git(clone, "cat-file", "-t", release) == "commit"
