"""Cache readers must never see an unfinished payload seed copy."""

from pathlib import Path

import pytest

import hermes_constants
from pm import packages, paths


@pytest.fixture
def seed_paths(tmp_path, monkeypatch):
    machine = tmp_path / "machine"
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: machine)
    payload = tmp_path / "payload"
    store = payload / "tools"
    store.mkdir(parents=True)
    source = payload / "uv-cache" / "entry"
    source.parent.mkdir()
    source.write_bytes(b"complete-new-payload")
    monkeypatch.setattr(paths, "store_root", lambda: store)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    return source, machine / "cache" / "uv" / "entry"


@pytest.mark.parametrize("previous", [None, b"old"])
def test_seed_publishes_only_a_complete_file(seed_paths, monkeypatch, previous):
    source, destination = seed_paths
    destination.parent.mkdir(parents=True)
    if previous is not None:
        destination.write_bytes(previous)
    real_copy = packages.shutil.copy2
    observed = []

    def observe_copy(src, dst):
        Path(dst).write_bytes(b"partial")
        observed.append(destination.read_bytes() if destination.exists() else None)
        return real_copy(src, dst)

    monkeypatch.setattr(packages.shutil, "copy2", observe_copy)
    packages.uv_cache_dir()

    assert observed == [previous]
    assert destination.read_bytes() == source.read_bytes()
    assert (destination.parent / ".seeded").is_file()
    assert {p.name for p in destination.parent.iterdir()} == {"entry", ".seeded"}


@pytest.mark.parametrize("previous", [None, b"old"])
def test_interrupted_seed_preserves_readers_and_retries(seed_paths, monkeypatch, previous):
    source, destination = seed_paths
    destination.parent.mkdir(parents=True)
    if previous is not None:
        destination.write_bytes(previous)
    real_copy = packages.shutil.copy2
    attempts = []

    def interrupted_copy(src, dst):
        attempts.append(1)
        if len(attempts) == 1:
            Path(dst).write_bytes(b"partial")
            raise OSError("synthetic copy interruption")
        return real_copy(src, dst)

    monkeypatch.setattr(packages.shutil, "copy2", interrupted_copy)
    packages.uv_cache_dir()

    assert (destination.read_bytes() if destination.exists() else None) == previous
    assert not (destination.parent / ".seeded").exists()
    assert {p.name for p in destination.parent.iterdir()} == ({"entry"} if previous else set())

    packages.uv_cache_dir()

    assert destination.read_bytes() == source.read_bytes()
    assert (destination.parent / ".seeded").is_file()
    assert {p.name for p in destination.parent.iterdir()} == {"entry", ".seeded"}
