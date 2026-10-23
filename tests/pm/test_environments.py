"""pm.environments: a git checkout is never a sealed payload (#126816)."""
from __future__ import annotations

import json

import pytest

from pm.environments import _payload_manifest, payload_venv, store_root


def _layout(tmp_path, *, git: bool):
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    if git:
        (repo / ".git").mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"repo": "hermes-agent", "venv": "payload-venv", "store": "sealed-store"})
    )
    return repo


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)


def test_checkout_never_reads_a_manifest_that_names_it(tmp_path, monkeypatch):
    repo = _layout(tmp_path, git=True)
    manifest = tmp_path / "manifest.json"
    real_read_text = type(manifest).read_text

    def guarded(self, *a, **k):
        assert self != manifest, "checkout read its sibling manifest.json"
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(type(manifest), "read_text", guarded)
    assert _payload_manifest(repo.resolve()) is None
    assert payload_venv(repo) is None
    assert store_root(repo) != (tmp_path / "sealed-store").resolve()


def test_sealed_payload_without_git_still_follows_its_manifest(tmp_path):
    repo = _layout(tmp_path, git=False)
    assert payload_venv(repo) == (tmp_path / "payload-venv").resolve()
    assert store_root(repo) == (tmp_path / "sealed-store").resolve()


def test_resident_runtime_does_not_probe_a_manifest_beside_a_checkout(tmp_path, monkeypatch):
    """pm.runtime._resident_runtime asks the same "am I a payload" question; a checkout must
    neither stat the sibling manifest nor be sent down the sealed-payload branch (which raises
    "packaged PM runtime is missing" when the sibling ``pm-runtime`` dir does not exist)."""
    from pm import paths, runtime

    repo = _layout(tmp_path, git=True)
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    manifest = tmp_path / "manifest.json"
    real_is_file = type(manifest).is_file

    def guarded(self, *a, **k):
        assert self != manifest, "checkout probed its sibling manifest.json"
        return real_is_file(self, *a, **k)

    monkeypatch.setattr(type(manifest), "is_file", guarded)
    assert runtime._resident_runtime() is None
