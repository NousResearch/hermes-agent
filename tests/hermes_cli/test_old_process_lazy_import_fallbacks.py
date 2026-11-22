"""A process started on v0.23.3 keeps its old ``hermes_constants`` / ``pm.filesystem`` /
``tools.environments.local`` / ``hermes_cli.runtime_state`` and imports newer modules from the new tree after ``hermes update``
(#158723). These load each module fresh against a sibling that lacks the newer name."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import hermes_constants
import pm.filesystem


def _fresh(name: str):
    spec = importlib.util.find_spec(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _incomplete_delete(tmp_path: Path) -> Path:
    """A delete whose rmtree removed the identity files but not the user's memories/."""
    home = tmp_path / "profiles" / "work"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("keep me", encoding="utf-8")
    tombstone = hermes_constants.profile_tombstone_path(home)
    tombstone.parent.mkdir(parents=True, exist_ok=True)
    tombstone.write_bytes(b"incomplete\n")
    return home


def test_old_constants_still_report_an_incomplete_delete(tmp_path, monkeypatch):
    home = _incomplete_delete(tmp_path)
    assert hermes_constants.named_profile_delete_incomplete(home)
    monkeypatch.delattr(hermes_constants, "named_profile_delete_incomplete")
    monkeypatch.delattr(hermes_constants, "TOMBSTONE_DELETED")
    old_process = _fresh("hermes_cli.profiles_remove")
    assert old_process.named_profile_delete_incomplete(home)
    assert "did not finish" in old_process.delete_incomplete_message("work", home)


def test_old_filesystem_lock_excludes_the_current_lock_both_ways(tmp_path, monkeypatch):
    old, new = tmp_path / ".install.lock", tmp_path / "locks" / "install.lock"
    current_lock = pm.filesystem.moved_lock
    monkeypatch.delattr(pm.filesystem, "moved_lock")
    fallback = _fresh("pm.moved_locks").moved_lock
    with fallback(old, new, timeout=0) as held:
        assert held
        with current_lock(old, new, timeout=0) as other:
            assert not other
    with current_lock(old, new, timeout=0) as held:
        assert held
        with fallback(old, new, timeout=0) as other:
            assert not other
    with fallback(old, new, timeout=0) as held:  # released on exit
        assert held


def test_old_filesystem_lock_excludes_a_v0_23_3_runtime_lock(tmp_path, monkeypatch):
    """The fallback and v0.23.3's own runtime_lock (lock_path on .install.lock) are one mutex."""
    old, new = tmp_path / ".install.lock", tmp_path / "locks" / "install.lock"
    monkeypatch.delattr(pm.filesystem, "moved_lock")
    fallback = _fresh("pm.moved_locks").moved_lock
    fd = pm.filesystem.lock_path(old, timeout=0, create_dir=True)
    try:
        with fallback(old, new, timeout=0) as held:
            assert not held
    finally:
        os.close(fd)


def test_old_constants_record_a_started_delete_as_incomplete(tmp_path, monkeypatch):
    """v0.23.3's mark has no ``incomplete``; a delete that then fails must not read as finished."""
    home = tmp_path / "profiles" / "work"
    (home / "memories").mkdir(parents=True)
    monkeypatch.delattr(hermes_constants, "named_profile_delete_incomplete")
    monkeypatch.delattr(hermes_constants, "TOMBSTONE_INCOMPLETE")
    monkeypatch.delattr(hermes_constants, "TOMBSTONE_DELETED")
    monkeypatch.setattr(hermes_constants, "mark_named_profile_deleted",
                        lambda profile_home: hermes_constants.profile_tombstone_path(profile_home).write_text("deleted\n"))
    old_process = _fresh("hermes_cli.profiles_remove")
    old_process.mark_delete_incomplete(home)
    assert old_process.named_profile_delete_incomplete(home)
    old_process.record_delete_finished(home)
    assert not old_process.named_profile_delete_incomplete(home)


def test_old_local_env_still_starts_a_gateway_with_the_deployment_stamps(tmp_path, monkeypatch):
    from tools.environments import local, local_gateway_env

    target = tmp_path / "profiles" / "work"
    target.mkdir(parents=True)
    base = {"PATH": os.environ.get("PATH", ""), "GATEWAY_RELAY_SLEEP_URL": "https://relay.invalid/s"}
    monkeypatch.delattr(local, "_profile_child_env")
    monkeypatch.setattr(local, "served_profile_child_env",
                        lambda base, *, target_home, inherit_credentials: {**base, "HERMES_HOME": str(target_home)})
    env = local_gateway_env.gateway_child_env(target, base)
    assert env["HERMES_HOME"] == str(target)
    assert env["GATEWAY_RELAY_SLEEP_URL"] == "https://relay.invalid/s"
