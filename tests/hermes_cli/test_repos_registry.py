"""Tests: the derived installed-repos registry.

The registry is a cache: absent, corrupt, or wrong-version files all read as
empty, and ``write_registry`` round-trips through an atomic replace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import repos_registry as reg


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "get_hermes_home", lambda: tmp_path)
    return tmp_path


def test_absent_file_reads_empty(home):
    data = reg.read_registry()
    assert data["version"] == reg.REGISTRY_VERSION
    assert data["installs"] == []
    assert not reg.registry_path().exists()


def test_roundtrip(home):
    reg.write_registry([{"id": "plugin:x", "surface": "plugin", "name": "x"}])
    data = reg.read_registry()
    assert [row["id"] for row in data["installs"]] == ["plugin:x"]
    assert data["synced_at"]


def test_corrupt_file_reads_empty(home):
    path = reg.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert reg.read_registry()["installs"] == []


def test_wrong_version_reads_empty(home):
    path = reg.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 999, "installs": [{"id": "x"}]}), encoding="utf-8"
    )
    assert reg.read_registry()["installs"] == []


def test_installs_must_be_a_list(home):
    path = reg.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "installs": {"id": "x"}}), encoding="utf-8"
    )
    assert reg.read_registry()["installs"] == []


def test_upsert_replaces_by_id():
    rows = [{"id": "plugin:x", "sha": "a"}]
    reg.upsert(rows, {"id": "plugin:x", "sha": "b"})
    reg.upsert(rows, {"id": "skill:y", "sha": "c"})
    assert [row["sha"] for row in rows] == ["b", "c"]


def test_write_is_atomic_and_sorted(home):
    reg.write_registry([{"id": "b"}, {"id": "a"}])
    raw = json.loads(reg.registry_path().read_text(encoding="utf-8"))
    assert raw["version"] == reg.REGISTRY_VERSION
    assert [row["id"] for row in raw["installs"]] == ["b", "a"]
