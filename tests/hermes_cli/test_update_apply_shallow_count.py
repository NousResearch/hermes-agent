"""Shallow-checkout guard on the ``hermes update`` apply path (#53479, mirroring #86257).

``rev-list --count HEAD..origin/<branch>`` on a shallow install can enumerate
the entire remote ancestry ("Found 9980 new commit(s)" on a depth-1 clone).
The apply path detects shallow state, recovers the real count via the GitHub
compare API, and reports the count as unknown (-1) when that fails.

These run ``_prepare_checkout_for_update`` against real git checkouts (a
depth-1 clone and a full clone of a local origin); only the GitHub compare
API — the network boundary — is faked.
"""

from __future__ import annotations

import subprocess

import pytest

import hermes_cli.main as cli_main
from hermes_cli import source_check, update_cmd


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _commit(repo, n):
    (repo / "f.txt").write_text(f"{n}\n", encoding="utf-8")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", f"c{n}")


@pytest.fixture
def origin(tmp_path, monkeypatch):
    for key, value in {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
        "GIT_CONFIG_NOSYSTEM": "1",
    }.items():
        monkeypatch.setenv(key, value)
    repo = tmp_path / "origin"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    for n in range(1, 4):
        _commit(repo, n)
    return repo


def _checkout_behind(tmp_path, origin, monkeypatch, *, shallow):
    """Clone *origin*, then land two new upstream commits and fetch them."""
    clone = tmp_path / ("shallow" if shallow else "full")
    depth = ["--depth", "1"] if shallow else []
    _git(tmp_path, "clone", "-q", *depth, origin.as_uri(), str(clone))
    for n in (4, 5):
        _commit(origin, n)
    _git(clone, "fetch", "-q", *depth, "origin", "main")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", clone)
    return clone


def _count(monkeypatch, *, api):
    consulted = []

    def fake_compare(head_sha, target_sha):
        consulted.append((head_sha, target_sha))
        return api

    monkeypatch.setattr(source_check, "_github_compare_behind", fake_compare)
    plan = update_cmd._prepare_checkout_for_update(
        ["git"], "main", "main", is_fork=False, assume_yes=True, gateway_mode=False,
        gw_input_fn=input, switch_branch=False, _windows_gateway_resume=None)
    return plan.commit_count, consulted


def test_full_clone_keeps_exact_count_without_the_api(tmp_path, origin, monkeypatch):
    _checkout_behind(tmp_path, origin, monkeypatch, shallow=False)
    count, consulted = _count(monkeypatch, api=999)
    assert count == 2
    assert consulted == []


def test_shallow_count_is_recovered_via_compare_api(tmp_path, origin, monkeypatch):
    """FAIL-BEFORE: the shallow rev-list number was reported as the commit count."""
    clone = _checkout_behind(tmp_path, origin, monkeypatch, shallow=True)
    count, consulted = _count(monkeypatch, api=12)
    assert count == 12
    head = subprocess.run(["git", "rev-parse", "HEAD", "origin/main"], cwd=clone,
                          check=True, capture_output=True, text=True).stdout.split()
    assert consulted == [tuple(head)]


def test_shallow_count_offline_is_unknown_not_bogus(tmp_path, origin, monkeypatch):
    _checkout_behind(tmp_path, origin, monkeypatch, shallow=True)
    count, _ = _count(monkeypatch, api=None)
    assert count == -1


def test_shallow_local_ahead_is_up_to_date(tmp_path, origin, monkeypatch):
    """compare says 0 behind (local is ahead): the apply path takes the up-to-date branch."""
    _checkout_behind(tmp_path, origin, monkeypatch, shallow=True)
    count, _ = _count(monkeypatch, api=0)
    assert count == 0
