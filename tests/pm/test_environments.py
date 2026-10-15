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
