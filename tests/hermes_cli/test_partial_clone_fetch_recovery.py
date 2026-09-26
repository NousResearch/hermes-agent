"""Regression coverage for #123324 partial-clone pack corruption recovery."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli.update_cmd_git import _recover_partial_clone_fetch_failure


def _git(cwd: Path, *args: str, check: bool = True, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )


@pytest.fixture
def partial_clone(tmp_path: Path) -> tuple[Path, Path]:
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q", "-b", "main")
    _git(seed, "config", "user.email", "fixture@example.invalid")
    _git(seed, "config", "user.name", "Fixture")
    for i in range(8):
        path = seed / f"d{i}" / "payload.txt"
        path.parent.mkdir()
        path.write_text(("payload-" + str(i) + "\n") * 200, encoding="utf-8")
        _git(seed, "add", "-A")
        _git(seed, "commit", "-qm", f"commit {i}")

    remote = tmp_path / "remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(seed), str(remote))
    _git(remote, "config", "uploadpack.allowFilter", "true")
    _git(remote, "config", "uploadpack.allowAnySHA1InWant", "true")

    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", "--no-local", "--filter=tree:0", remote.as_uri(), str(checkout))
    assert _git(checkout, "config", "--get", "remote.origin.promisor").stdout.strip() == "true"
    return checkout, remote


def _assert_no_missing_reachable_objects(checkout: Path) -> None:
    env = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
    walk = _git(checkout, "rev-list", "--objects", "--missing=print", "origin/main", env=env)
    assert not [line for line in walk.stdout.splitlines() if line.startswith("?")], walk.stdout
    fsck = _git(checkout, "fsck", "--full", "--no-dangling", env=env, check=False)
    assert fsck.returncode == 0, fsck.stdout + fsck.stderr


def test_known_promisor_assertion_recovers_with_complete_refetch(partial_clone):
    checkout, _remote = partial_clone
    for marker in (checkout / ".git" / "objects" / "pack").glob("*.promisor"):
        marker.unlink()

    tracked = checkout / "d7" / "payload.txt"
    dirty = tracked.read_text(encoding="utf-8") + "local edit\n"
    tracked.write_text(dirty, encoding="utf-8")
    failure = subprocess.CompletedProcess(
        ["git", "fetch"], 128, stdout="",
        stderr=(
            "BUG: builtin/pack-objects.c:4967: "
            "should_include_obj should only be called on existing objects\n"
            "fatal: could not finish pack-objects to repack local links\n"
            "fatal: index-pack failed\n"
        ),
    )

    result = _recover_partial_clone_fetch_failure(
        ["git"], ["fetch", "origin", "main"], checkout, failure)

    assert result.returncode == 0, result.stderr
    assert _git(checkout, "config", "--get", "remote.origin.promisor", check=False).returncode != 0
    assert _git(checkout, "config", "--get", "remote.origin.partialclonefilter", check=False).returncode != 0
    assert _git(checkout, "config", "--get", "gc.auto").stdout.strip() == "0"
    assert _git(checkout, "config", "--get", "maintenance.auto").stdout.strip() == "false"
    assert tracked.read_text(encoding="utf-8") == dirty
    _assert_no_missing_reachable_objects(checkout)


def test_failed_recovery_restores_partial_clone_contract(partial_clone, monkeypatch):
    checkout, _remote = partial_clone
    import hermes_cli.update_cmd as update_cmd

    real_run = update_cmd._git_run

    def fail_refetch(git_cmd, args, cwd=None, **kwargs):
        if "--refetch" in args:
            return subprocess.CompletedProcess(
                git_cmd + list(args), 1, stdout="", stderr="simulated recovery network failure")
        return real_run(git_cmd, args, cwd, **kwargs)

    monkeypatch.setattr(update_cmd, "_git_run", fail_refetch)
    original_filter = _git(
        checkout, "config", "--get", "remote.origin.partialclonefilter").stdout.strip()
    failure = subprocess.CompletedProcess(
        ["git", "fetch"], 128, stdout="",
        stderr="BUG: should_include_obj should only be called on existing objects\nfatal: index-pack failed\n",
    )

    result = _recover_partial_clone_fetch_failure(
        ["git"], ["fetch", "origin", "main"], checkout, failure)

    assert result.returncode != 0
    assert _git(checkout, "config", "--get", "remote.origin.promisor").stdout.strip() == "true"
    assert _git(checkout, "config", "--get", "remote.origin.partialclonefilter").stdout.strip() == original_filter
    assert _git(checkout, "config", "--get", "gc.auto", check=False).returncode != 0
    assert _git(checkout, "config", "--get", "maintenance.auto", check=False).returncode != 0


def test_unrelated_fetch_failure_does_not_mutate_partial_clone(partial_clone):
    checkout, _remote = partial_clone
    failure = subprocess.CompletedProcess(
        ["git", "fetch"], 1, stdout="", stderr="fatal: unable to access remote\n")

    result = _recover_partial_clone_fetch_failure(
        ["git"], ["fetch", "origin", "main"], checkout, failure)

    assert result is failure
    assert _git(checkout, "config", "--get", "remote.origin.promisor").stdout.strip() == "true"
    assert _git(checkout, "config", "--get", "remote.origin.partialclonefilter").stdout.strip() == "tree:0"
