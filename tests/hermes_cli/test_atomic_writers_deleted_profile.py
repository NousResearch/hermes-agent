"""Atomic writers must not resurrect a deleted named profile home.

Issue #112592: after ``hermes profile delete <name>`` succeeds, a late background writer
(e.g. the reasoning-caps daemon through ``hermes_cli.models._write_json_cache`` ->
``utils.atomic_json_write``) re-created ``profiles/<name>/`` because the writers mkdir the
target's parent with a bare ``mkdir(parents=True)``. The #97128 tombstone guard
(``assert_named_profile_home_live`` / ``mkdir_under_hermes_home``) covered logging/state/config
but not these writers. These tests lock the same contract onto the writers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.profiles import create_profile, delete_profile
from utils import (
    atomic_json_write,
    atomic_roundtrip_yaml_save,
    atomic_roundtrip_yaml_update,
    atomic_write_bytes,
    atomic_write_text,
)


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _create_then_delete(name: str) -> Path:
    profile_dir = create_profile(name, no_alias=True, no_skills=True)
    with patch("hermes_cli.profiles._cleanup_gateway_service"), patch(
        "hermes_cli.profiles._stop_profile_backends"
    ):
        delete_profile(name, yes=True)
    assert not profile_dir.exists()
    return profile_dir


class TestAtomicWritersRefuseDeletedProfileHome:
    def test_atomic_json_write_refuses(self, profile_env):
        profile_dir = _create_then_delete("worker")
        target = profile_dir / "cache" / "reasoning_caps.json"
        with pytest.raises(FileNotFoundError, match="Named profile home does not exist"):
            atomic_json_write(target, {"m": True})
        assert not profile_dir.exists()

    def test_atomic_write_text_refuses(self, profile_env):
        profile_dir = _create_then_delete("worker")
        target = profile_dir / "cache" / "note.txt"
        with pytest.raises(FileNotFoundError, match="Named profile home does not exist"):
            atomic_write_text(target, "hello")
        assert not profile_dir.exists()

    def test_atomic_write_bytes_refuses(self, profile_env):
        profile_dir = _create_then_delete("worker")
        target = profile_dir / "cache" / "blob.bin"
        with pytest.raises(FileNotFoundError, match="Named profile home does not exist"):
            atomic_write_bytes(target, b"\x00\x01")
        assert not profile_dir.exists()

    def test_atomic_roundtrip_yaml_update_refuses(self, profile_env):
        profile_dir = _create_then_delete("worker")
        target = profile_dir / "config.yaml"
        with pytest.raises(FileNotFoundError, match="Named profile home does not exist"):
            atomic_roundtrip_yaml_update(target, "a.b", 1)
        assert not profile_dir.exists()

    def test_atomic_roundtrip_yaml_save_refuses(self, profile_env):
        profile_dir = _create_then_delete("worker")
        target = profile_dir / "config.yaml"
        with pytest.raises(FileNotFoundError, match="Named profile home does not exist"):
            atomic_roundtrip_yaml_save(target, {"a": {"b": 1}})
        assert not profile_dir.exists()

    def test_write_json_cache_caller_path_refuses_file(self, profile_env):
        """The issue's exact path: caller mkdirs, then the writer must still refuse."""
        from hermes_cli.models import _write_json_cache

        profile_dir = _create_then_delete("worker")
        target = profile_dir / "cache" / "reasoning_caps.json"
        with pytest.raises(FileNotFoundError, match="Named profile home does not exist"):
            _write_json_cache(target, {"m": True})
        # The caller's own mkdir may leave an empty shell, but no file may land.
        assert not target.exists()


class TestAtomicWritersStillWork:
    def test_live_named_profile_writes_fine(self, profile_env):
        profile_dir = create_profile("worker", no_alias=True, no_skills=True)
        target = profile_dir / "cache" / "reasoning_caps.json"
        atomic_json_write(target, {"m": True})
        assert target.exists()

    def test_non_profile_paths_unaffected(self, tmp_path):
        target = tmp_path / "sub" / "data.json"
        atomic_json_write(target, {"ok": True})
        assert target.exists()
