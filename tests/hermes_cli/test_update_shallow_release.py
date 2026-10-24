"""A release update on a pristine shallow checkout fetches only the release tag and stays shallow.

The helpers behind the gate (``is_shallow_repository``, ``head_is_shallow_boundary``,
``has_merged_release_tag``) and the depth-1 tag fetch the updater issues are the contract:
a shallow install moves to a release without paying for the commit graph the detach and the
version label never read. Real local git repositories, no network.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import gitlock, update_cmd_check
from hermes_cli.version_info import _git_version_info


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)


def _must(cwd: Path, *args: str) -> str:
    result = _git(cwd, *args)
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def release_history(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """origin: c1 tagged v1.0.0, then c2, c3 tagged v1.1.0 — plus a shallow clone at v1.0.0.

    Returns ``(origin, shallow_clone, shas)``; ``shas[0]`` carries v1.0.0, ``shas[2]`` v1.1.0.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _must(origin, "init", "-q", "-b", "main")
    shas = []
    for i in range(3):
        (origin / "f.txt").write_text(f"{i}\n", encoding="utf-8")
        _must(origin, "add", "f.txt")
        _must(origin, "commit", "-qm", f"c{i}")
        shas.append(_must(origin, "rev-parse", "HEAD"))
        if i == 0:
            _must(origin, "tag", "-a", "v1.0.0", "-m", "v1.0.0")
    _must(origin, "tag", "-a", "v1.1.0", "-m", "v1.1.0")
    clone = tmp_path / "shallow"
    _must(tmp_path, "clone", "-q", "--depth", "1", "--branch", "v1.0.0",
          f"file://{origin}", str(clone))
    return origin, clone, shas


def test_head_is_a_shallow_boundary_until_a_local_commit_lands_on_top(release_history):
    _origin, clone, _shas = release_history
    assert gitlock.is_shallow_repository(clone)
    assert gitlock.head_is_shallow_boundary(clone)

    (clone / "local.txt").write_text("local\n", encoding="utf-8")
    _must(clone, "add", "local.txt")
    _must(clone, "commit", "-qm", "local work")
    assert not gitlock.head_is_shallow_boundary(clone)


def test_has_merged_release_tag_follows_git_tag_merged_semantics(release_history, tmp_path):
    _origin, _clone, _shas = release_history
    scratch = tmp_path / "scratch"
    _must(tmp_path, "clone", "-q", f"file://{_origin}", str(scratch))

    _must(scratch, "checkout", "-q", "v1.1.0")  # detached at the tagged commit
    assert gitlock.has_merged_release_tag(scratch)

    # An orphan branch has no v<digits> tag anywhere below HEAD, exactly as
    # `git tag --merged HEAD --list 'v[0-9]*'` reports it.
    _must(scratch, "checkout", "-q", "--orphan", "untagged")
    (scratch / "n.txt").write_text("n\n", encoding="utf-8")
    _must(scratch, "add", "n.txt")
    _must(scratch, "commit", "-qm", "untagged")
    assert not gitlock.has_merged_release_tag(scratch)


def test_depth1_tag_fetch_moves_a_shallow_checkout_to_the_release_and_keeps_it_shallow(release_history):
    _origin, clone, shas = release_history
    # The fetch args the updater now issues for a shallow release update, then its detach.
    _must(clone, "fetch", "--depth=1", "--no-tags", "origin",
          "refs/tags/v1.1.0:refs/tags/v1.1.0")
    _must(clone, "checkout", "-q", "--detach", "v1.1.0")

    assert gitlock.is_shallow_repository(clone)
    assert _must(clone, "rev-list", "--count", "HEAD") == "1"
    assert _must(clone, "rev-parse", "HEAD") == shas[2]
    assert gitlock.has_merged_release_tag(clone)
    # The version label — derived from the same `git tag --merged` query — resolves locally.
    assert _git_version_info(clone).base_version == "1.1.0"


def test_a_full_history_clone_is_not_a_shallow_boundary_and_the_fetch_lands_on_the_tag(
        release_history, tmp_path):
    _origin, _clone, shas = release_history
    full = tmp_path / "full"
    _must(tmp_path, "clone", "-q", f"file://{_origin}", str(full))

    assert not gitlock.is_shallow_repository(full)
    assert not gitlock.head_is_shallow_boundary(full)

    _must(full, "fetch", "--depth=1", "--no-tags", "origin",
          "refs/tags/v1.1.0:refs/tags/v1.1.0")
    _must(full, "checkout", "-q", "--detach", "v1.1.0")
    assert _must(full, "rev-parse", "HEAD") == shas[2]
    assert _git_version_info(full).base_version == "1.1.0"


def test_apply_fetch_issues_one_depth1_tag_fetch_and_reports_it(release_history, monkeypatch):
    """The apply path's shallow release update fetches only the tag, through the bounded runner."""
    _origin, clone, shas = release_history
    calls = []

    def record(git_cmd, args, **kwargs):
        calls.append((list(git_cmd), list(args), kwargs))
        return subprocess.CompletedProcess([*git_cmd, *args], 0)

    monkeypatch.setattr("hermes_cli.update_cmd._git_run", record)
    result, shallow_tag_fetch = update_cmd_check.apply_fetch(
        ["git"], clone, branch="main", target_ref="origin/main", release_sha=shas[2],
        target_is_head=False, release_version="1.1.0", release_shallow=True)

    assert shallow_tag_fetch is True and result.returncode == 0
    assert len(calls) == 1
    _git_cmd, args, kwargs = calls[0]
    assert args == ["fetch", "--depth=1", "--no-tags", "origin",
                    "refs/tags/v1.1.0:refs/tags/v1.1.0"]
    assert kwargs["cwd"] == clone and kwargs["network"] is True


def test_a_failed_tag_fetch_falls_back_to_the_sha_args_without_the_shallow_flag(monkeypatch, tmp_path):
    calls = []
    codes = iter([1, 0])  # the tag fetch fails; the sha fallback succeeds

    def record(git_cmd, args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess([*git_cmd, *args], next(codes))

    monkeypatch.setattr("hermes_cli.update_cmd._git_run", record)
    result, shallow_tag_fetch = update_cmd_check.apply_fetch(
        ["git"], tmp_path, branch="main", target_ref="origin/main", release_sha="abc123",
        target_is_head=False, release_version="1.1.0", release_shallow=True)

    assert shallow_tag_fetch is False and result.returncode == 0
    assert calls == [["fetch", "--depth=1", "--no-tags", "origin",
                      "refs/tags/v1.1.0:refs/tags/v1.1.0"],
                     ["fetch", "--no-tags", "origin", "origin/main"]]


def test_a_forward_only_release_already_at_head_fetches_nothing(monkeypatch, tmp_path):
    def record(git_cmd, args, **kwargs):
        raise AssertionError(f"no git may run when the release is already HEAD: {args}")

    monkeypatch.setattr("hermes_cli.update_cmd._git_run", record)
    result, shallow_tag_fetch = update_cmd_check.apply_fetch(
        ["git"], tmp_path, branch="main", target_ref="origin/main", release_sha="abc123",
        target_is_head=True, release_version="1.1.0", release_shallow=True)

    assert result is None and shallow_tag_fetch is False
