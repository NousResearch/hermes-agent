"""Behavioral tests for release push and remote-tag safety gates."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call


TAG_NAME = "v2026.7.16"
LOCAL_SHA = "a" * 40


def _load_release_module(monkeypatch, tmp_root: Path):
    spec = importlib.util.spec_from_file_location(
        "_release_publish_under_test",
        Path(__file__).resolve().parents[2] / "scripts" / "release.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "REPO_ROOT", tmp_root)
    return module


def _result(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_push_sends_only_the_requested_release_tag(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Release Test")
    _git(repo, "config", "user.email", "release@example.invalid")
    (repo / "README.md").write_text("initial\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    _git(tmp_path, "init", "--bare", str(remote))
    _git(repo, "remote", "add", "origin", str(remote))

    requested_tag = "v2099.1.1"
    stale_tag = "v2098.1.1"
    _git(repo, "tag", "-a", requested_tag, "-m", "current release")
    _git(repo, "tag", "-a", stale_tag, "-m", "stale local tag")

    module = _load_release_module(monkeypatch, repo)

    assert module.push_and_verify_release_tag(requested_tag) is True
    remote_tags = set(
        _git(tmp_path, "--git-dir", str(remote), "tag", "--list").splitlines()
    )
    assert requested_tag in remote_tags
    assert stale_tag not in remote_tags


def _configure_publish(module, monkeypatch):
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["release.py", "--publish", "--date", "2026.7.16"],
    )
    monkeypatch.setattr(
        module,
        "next_available_tag",
        lambda _base_tag: (TAG_NAME, "2026.7.16"),
    )
    monkeypatch.setattr(module, "get_current_version", lambda: "0.14.0")
    monkeypatch.setattr(module, "get_last_tag", lambda: "v2026.7.9")
    monkeypatch.setattr(
        module,
        "get_commits",
        lambda since_tag: [{"github_author": "@alice"}],
    )
    monkeypatch.setattr(module, "generate_changelog", lambda *args, **kwargs: "notes")

    build = MagicMock(return_value=[])
    gh_run = MagicMock(return_value=_result(stdout="https://example.test/release"))
    gh_which = MagicMock(return_value="/usr/bin/gh")
    monkeypatch.setattr(module, "build_release_artifacts", build)
    monkeypatch.setattr(module.subprocess, "run", gh_run)
    monkeypatch.setattr(module.shutil, "which", gh_which)
    return build, gh_run, gh_which


def test_push_failure_aborts_before_build_or_github_release(monkeypatch, tmp_path):
    module = _load_release_module(monkeypatch, tmp_path)
    build, gh_run, gh_which = _configure_publish(module, monkeypatch)
    git_result = MagicMock(
        side_effect=[
            _result(),
            _result(returncode=1, stderr="permission denied"),
        ]
    )
    monkeypatch.setattr(module, "git_result", git_result)

    module.main()

    assert git_result.call_args_list == [
        call(
            "tag",
            "-a",
            TAG_NAME,
            "-m",
            "Hermes Agent v0.14.0 (2026.7.16)\n\nWeekly release",
        ),
        call("push", "origin", "HEAD", f"refs/tags/{TAG_NAME}"),
    ]
    build.assert_not_called()
    gh_which.assert_not_called()
    gh_run.assert_not_called()


def test_peeled_remote_tag_mismatch_aborts_before_publish(monkeypatch, tmp_path):
    module = _load_release_module(monkeypatch, tmp_path)
    build, gh_run, gh_which = _configure_publish(module, monkeypatch)
    peeled_ref = f"refs/tags/{TAG_NAME}^{{}}"
    git_result = MagicMock(
        side_effect=[
            _result(),
            _result(),
            _result(stdout=f"{LOCAL_SHA}\n"),
            _result(stdout=f"{'b' * 40}\t{peeled_ref}\n"),
        ]
    )
    monkeypatch.setattr(module, "git_result", git_result)

    module.main()

    assert git_result.call_args_list[-2:] == [
        call("rev-parse", f"{TAG_NAME}^{{commit}}"),
        call("ls-remote", "origin", peeled_ref),
    ]
    build.assert_not_called()
    gh_which.assert_not_called()
    gh_run.assert_not_called()


def test_github_release_create_verifies_matching_remote_tag(monkeypatch, tmp_path):
    module = _load_release_module(monkeypatch, tmp_path)
    build, gh_run, _gh_which = _configure_publish(module, monkeypatch)
    peeled_ref = f"refs/tags/{TAG_NAME}^{{}}"
    git_result = MagicMock(
        side_effect=[
            _result(),
            _result(),
            _result(stdout=f"{LOCAL_SHA}\n"),
            _result(stdout=f"{LOCAL_SHA}\t{peeled_ref}\n"),
        ]
    )
    monkeypatch.setattr(module, "git_result", git_result)

    module.main()

    build.assert_called_once_with("0.14.0")
    gh_run.assert_called_once()
    gh_command = gh_run.call_args.args[0]
    assert gh_command[:5] == [
        "gh",
        "release",
        "create",
        TAG_NAME,
        "--verify-tag",
    ]
