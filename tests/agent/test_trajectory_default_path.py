"""Behavior contracts for implicit trajectory export placement."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from agent import trajectory
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


SENTINEL = [
    {"from": "human", "value": "token ghp_EXAMPLENOTREAL — 日本語"},
    {"from": "gpt", "value": '<think>keep \"quotes\"</think>done'},
]


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


def _expected_path(home: Path, cwd: Path, completed: bool) -> Path:
    canonical_cwd = cwd.resolve()
    digest = hashlib.sha256(os.fsencode(canonical_cwd)).hexdigest()[:8]
    filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"
    return home / "trajectories" / f"{canonical_cwd.name}-{digest}" / filename


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_single_sample_export_uses_private_profile_bucket(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from run_agent import _save_sample_trajectory

    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    agent = SimpleNamespace(_convert_to_trajectory_format=lambda *args: SENTINEL)
    _save_sample_trajectory(agent, {"messages": [], "completed": True}, "query", "m")
    assert not list(tmp_path.glob("sample_*.json"))
    samples = list((home / "trajectories").glob("*/sample_*.json"))
    assert len(samples) == 1
    assert json.loads(samples[0].read_text(encoding="utf-8"))["conversations"] == SENTINEL
    if os.name == "posix":
        assert not samples[0].stat().st_mode & 0o077


def _assert_entry(entry: dict, *, model: str, completed: bool) -> None:
    assert set(entry) == {"conversations", "timestamp", "model", "completed"}
    assert entry["conversations"] == SENTINEL
    assert entry["model"] == model
    assert entry["completed"] is completed
    datetime.fromisoformat(entry["timestamp"])


def test_implicit_saves_append_under_context_profile_bucket(tmp_path, monkeypatch):
    process_home = tmp_path / "process-home"
    profile_home = _git_repo(tmp_path / "profile-repo") / "profile-home"
    repo = _git_repo(tmp_path / "work" / "project")
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    monkeypatch.chdir(repo)

    token = set_hermes_home_override(profile_home)
    try:
        expected_completed = _expected_path(profile_home, repo, completed=True)
        expected_failed = _expected_path(profile_home, repo, completed=False)
        assert trajectory.default_trajectory_path(True) == expected_completed
        assert trajectory.default_trajectory_path(False) == expected_failed

        trajectory.save_trajectory(SENTINEL, "model-complete", completed=True)
        trajectory.save_trajectory(SENTINEL, "model-complete", completed=True)
        trajectory.save_trajectory(SENTINEL, "model-failed", completed=False)
    finally:
        reset_hermes_home_override(token)

    assert not (repo / "trajectory_samples.jsonl").exists()
    assert not (repo / "failed_trajectories.jsonl").exists()
    assert not (process_home / "trajectories").exists()
    assert not list(profile_home.rglob(".gitignore"))

    completed_entries = _read_jsonl(expected_completed)
    failed_entries = _read_jsonl(expected_failed)
    assert len(completed_entries) == 2
    assert len(failed_entries) == 1
    for entry in completed_entries:
        _assert_entry(entry, model="model-complete", completed=True)
    _assert_entry(failed_entries[0], model="model-failed", completed=False)


def test_same_named_cwds_use_distinct_stable_buckets(tmp_path, monkeypatch):
    profile_home = tmp_path / "profile-home"
    repos = [
        _git_repo(tmp_path / "one" / "project"),
        _git_repo(tmp_path / "two" / "project"),
    ]
    paths = []
    token = set_hermes_home_override(profile_home)
    try:
        for index, repo in enumerate(repos):
            monkeypatch.chdir(repo)
            paths.append(trajectory.default_trajectory_path(True))
            trajectory.save_trajectory(SENTINEL, f"model-{index}", completed=True)
    finally:
        reset_hermes_home_override(token)

    assert paths == [_expected_path(profile_home, repo, True) for repo in repos]
    assert paths[0].parent != paths[1].parent
    assert all(path.exists() for path in paths)
    assert [_read_jsonl(path)[0]["model"] for path in paths] == ["model-0", "model-1"]
    assert all(not (repo / "trajectory_samples.jsonl").exists() for repo in repos)


def test_bucket_name_has_safe_nonempty_fallback(tmp_path, monkeypatch):
    profile_home = tmp_path / "profile-home"
    repo = _git_repo(tmp_path / "$")
    monkeypatch.chdir(repo)

    token = set_hermes_home_override(profile_home)
    try:
        path = trajectory.default_trajectory_path(True)
    finally:
        reset_hermes_home_override(token)

    digest = hashlib.sha256(os.fsencode(repo.resolve())).hexdigest()[:8]
    bucket_name = path.parent.name
    assert bucket_name.endswith(f"-{digest}")
    safe_name = bucket_name[: -len(digest) - 1]
    assert safe_name
    assert re.fullmatch(r"[A-Za-z0-9._-]+", safe_name)


def test_247_character_cwd_saves_to_bounded_bucket(tmp_path, monkeypatch):
    profile_home = tmp_path / "profile-home"
    long_name = "a" * 247
    repo = _git_repo(tmp_path / long_name)
    monkeypatch.chdir(repo)

    token = set_hermes_home_override(profile_home)
    try:
        trajectory.save_trajectory(SENTINEL, "long-cwd", completed=True)
    finally:
        reset_hermes_home_override(token)

    paths = list((profile_home / "trajectories").glob("*/trajectory_samples.jsonl"))
    assert len(paths) == 1
    bucket = paths[0].parent.name
    prefix, digest = bucket.rsplit("-", 1)
    assert prefix == long_name[:32]
    assert re.fullmatch(r"[0-9a-f]{8}", digest)
    assert len(bucket) <= 41
    _assert_entry(_read_jsonl(paths[0])[0], model="long-cwd", completed=True)


def test_canonical_cwd_is_hashed_with_filesystem_encoding(tmp_path, monkeypatch):
    profile_home = tmp_path / "profile-home"
    repo = _git_repo(tmp_path / "project")
    canonical = repo.resolve()
    encoded = b"filesystem-native-path-bytes"
    calls = []
    monkeypatch.chdir(repo)
    monkeypatch.setattr(os, "fsencode", lambda path: calls.append(path) or encoded)

    token = set_hermes_home_override(profile_home)
    try:
        path = trajectory.default_trajectory_path(True)
    finally:
        reset_hermes_home_override(token)

    digest = hashlib.sha256(encoded).hexdigest()[:8]
    assert calls == [canonical]
    assert path.parent.name == f"project-{digest}"


def test_symlink_alias_and_real_cwd_share_canonical_bucket(tmp_path, monkeypatch):
    profile_home = tmp_path / "profile-home"
    real = _git_repo(tmp_path / "real-project")
    alias = tmp_path / "alias-project"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    token = set_hermes_home_override(profile_home)
    try:
        # POSIX getcwd() commonly returns the physical path even after chdir(alias),
        # so feed Path.cwd() the real symlink spellings while keeping resolve() real.
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: real))
        real_path = trajectory.default_trajectory_path(True)
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: alias))
        alias_path = trajectory.default_trajectory_path(True)
    finally:
        reset_hermes_home_override(token)

    assert alias_path == real_path


def test_explicit_relative_and_absolute_filenames_remain_exact(tmp_path, monkeypatch):
    profile_home = tmp_path / "profile-home"
    repo = _git_repo(tmp_path / "repo")
    absolute = tmp_path / "absolute.jsonl"
    monkeypatch.chdir(repo)

    token = set_hermes_home_override(profile_home)
    try:
        trajectory.save_trajectory(SENTINEL, "relative", True, filename="explicit.jsonl")
        trajectory.save_trajectory(SENTINEL, "absolute", False, filename=str(absolute))
    finally:
        reset_hermes_home_override(token)

    relative = repo / "explicit.jsonl"
    assert _read_jsonl(relative)[0]["model"] == "relative"
    assert _read_jsonl(absolute)[0]["model"] == "absolute"
    assert not (profile_home / "trajectories").exists()


def test_deleted_cwd_resolution_failure_does_not_escape(tmp_path, monkeypatch, caplog):
    deleted = tmp_path / "deleted"
    deleted.mkdir()
    monkeypatch.chdir(deleted)
    os.rmdir(deleted)

    trajectory.save_trajectory(SENTINEL, "model", completed=True)

    assert "Failed to save trajectory" in caplog.text


def test_unwritable_home_does_not_fall_back_to_cwd(tmp_path, monkeypatch, caplog):
    repo = _git_repo(tmp_path / "repo")
    unusable_home = tmp_path / "home-is-a-file"
    unusable_home.write_text("not a directory", encoding="utf-8")
    monkeypatch.chdir(repo)

    token = set_hermes_home_override(unusable_home)
    try:
        trajectory.save_trajectory(SENTINEL, "model", completed=True)
    finally:
        reset_hermes_home_override(token)

    assert not (repo / "trajectory_samples.jsonl").exists()
    assert "Failed to save trajectory" in caplog.text
