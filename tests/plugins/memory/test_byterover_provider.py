"""Tests for the ByteRover memory provider config gates."""

import threading

import plugins.memory.byterover as byterover
from agent import secret_scope
from plugins.memory.byterover import ByteRoverMemoryProvider


def test_auto_extract_false_skips_sync_turn(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": False})
    provider.initialize("session-1")

    monkeypatch.setattr("plugins.memory.byterover._run_brv", lambda *args, **kwargs: calls.append((args, kwargs)))

    provider.sync_turn("please remember this detail", "acknowledged")

    assert calls == []
    assert provider._sync_thread is None


# =========================================================================
# Profile-scoped child env (#108993): the brv subprocess must never inherit
# another profile's BRV_API_KEY / HERMES_HOME from the process environment.
# =========================================================================


class _FakeBrv:
    """Fake brv binary: records the env it was spawned with, exits 0 with fixed stdout."""

    def __init__(self, path):
        self.path = path
        self.seen_envs = []

    def write(self, body=None):
        body = body or (
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "json.dump(dict(os.environ), open(sys.argv[-1], 'w'))\n"
            "sys.exit(0)\n"
        )
        self.path.write_text(body, encoding="utf-8")
        self.path.chmod(0o755)


def _install_fake_brv(tmp_path, monkeypatch):
    import shutil as _shutil

    fake = _FakeBrv(tmp_path / "bin" / "brv")
    fake.path.parent.mkdir(parents=True, exist_ok=True)
    fake.write()
    monkeypatch.setattr(byterover, "_cached_brv_path", str(fake.path))
    monkeypatch.setattr(_shutil, "which", lambda name: str(fake.path) if name == "brv" else None)
    return fake


def _capture_env(fake, tmp_path, args=None):
    out = tmp_path / "env.json"
    if out.exists():
        out.unlink()
    result = byterover._run_brv((args or []) + ["--", str(out)], timeout=10, cwd=str(tmp_path / "wd"))
    assert result["success"], result
    import json

    return json.loads(out.read_text(encoding="utf-8"))


def test_multiplexed_turn_uses_scoped_key_not_default_profile_env(tmp_path, monkeypatch):
    """L1: with a profile secret scope installed, the scoped BRV_API_KEY wins and the
    default profile's value sitting in os.environ never reaches the child."""
    fake = _install_fake_brv(tmp_path, monkeypatch)
    monkeypatch.setenv("BRV_API_KEY", "DEFAULT-PROFILE-KEY")

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"BRV_API_KEY": "PROFILE-B-KEY"})
    try:
        env = _capture_env(fake, tmp_path)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)

    assert env["BRV_API_KEY"] == "PROFILE-B-KEY"


def test_scoped_miss_sends_no_key_rather_than_default_profiles(tmp_path, monkeypatch):
    """Edge 4: profile b has no key configured — the child runs keyless, never with
    the default profile's value from the process env."""
    fake = _install_fake_brv(tmp_path, monkeypatch)
    monkeypatch.setenv("BRV_API_KEY", "DEFAULT-PROFILE-KEY")

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"OTHER_KEY": "x"})  # scope without BRV_API_KEY
    try:
        env = _capture_env(fake, tmp_path)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)

    assert "BRV_API_KEY" not in env


def test_unscoped_multiplex_falls_back_to_process_env(tmp_path, monkeypatch):
    """Edge 5a: the DEFAULT profile's own turns run unscoped under multiplexing —
    os.environ IS that profile's value there, so it must still reach the child."""
    fake = _install_brv_unscoped_case(tmp_path, monkeypatch)

    secret_scope.set_multiplex_active(True)
    try:
        assert secret_scope.current_secret_scope() is None
        env = _capture_env(fake, tmp_path)
    finally:
        secret_scope.set_multiplex_active(False)

    assert env["BRV_API_KEY"] == "PROCESS-ENV-KEY"


def _install_brv_unscoped_case(tmp_path, monkeypatch):
    fake = _install_fake_brv(tmp_path, monkeypatch)
    monkeypatch.setenv("BRV_API_KEY", "PROCESS-ENV-KEY")
    return fake


def test_single_profile_no_multiplex_unchanged(tmp_path, monkeypatch):
    """Edge 5b: no multiplexing at all — the key from the process env (.env injected
    at gateway startup) still reaches the child, byte-identical behavior."""
    fake = _install_fake_brv(tmp_path, monkeypatch)
    monkeypatch.setenv("BRV_API_KEY", "SINGLE-PROFILE-KEY")

    assert not secret_scope.is_multiplex_active()
    env = _capture_env(fake, tmp_path)

    assert env["BRV_API_KEY"] == "SINGLE-PROFILE-KEY"


def test_child_env_carries_active_profile_home(tmp_path, monkeypatch):
    """Edge 6: HERMES_HOME on the child is the ACTIVE profile's home override, not
    the launch profile's process-env value."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    fake = _install_fake_brv(tmp_path, monkeypatch)
    prof_b = tmp_path / "profiles" / "b"
    prof_b.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # launch (default) home
    home_token = set_hermes_home_override(str(prof_b))
    try:
        env = _capture_env(fake, tmp_path)
    finally:
        reset_hermes_home_override(home_token)

    assert env["HERMES_HOME"] == str(prof_b)


def test_background_curate_thread_sees_scope_and_home(tmp_path, monkeypatch):
    """L3: _curate_in_background threads run in a copy of the spawner's context, so
    the scoped key and the profile home override survive into the worker."""
    fake = _install_fake_brv(tmp_path, monkeypatch)
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("BRV_API_KEY", "DEFAULT-PROFILE-KEY")
    prof_b = tmp_path / "profiles" / "b"
    prof_b.mkdir(parents=True)

    captured = {}

    real_curate = ByteRoverMemoryProvider._curate

    def _spy_curate(self, content):
        # Runs INSIDE the worker thread: record what the production path resolves.
        captured["env"] = _capture_env(fake, tmp_path)
        return {"success": True, "output": ""}

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"BRV_API_KEY": "PROFILE-B-KEY"})
    home_token = set_hermes_home_override(str(prof_b))
    try:
        provider = ByteRoverMemoryProvider({"auto_extract": True})
        provider.initialize("session-ctx")
        monkeypatch.setattr(ByteRoverMemoryProvider, "_curate", _spy_curate)
        provider.sync_turn("please remember this scoped detail for the record", "acknowledged")
        assert provider._sync_thread is not None
        provider._sync_thread.join(timeout=10)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)
        reset_hermes_home_override(home_token)
        monkeypatch.setattr(ByteRoverMemoryProvider, "_curate", real_curate)

    assert "env" in captured, "worker never ran"
    assert captured["env"]["BRV_API_KEY"] == "PROFILE-B-KEY"
    assert captured["env"]["HERMES_HOME"] == str(prof_b)


def test_background_curate_thread_records_profile_cwd(tmp_path, monkeypatch):
    """L2 invariant from a background thread: the brv context-tree cwd stays the
    active profile's home even when the worker outlives the turn."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    prof_b = tmp_path / "profiles" / "b"
    prof_b.mkdir(parents=True)

    captured = {}

    def _spy(self, content):
        captured["cwd"] = byterover._get_brv_cwd()
        return {"success": True, "cwd_recorded": True}

    real_curate = ByteRoverMemoryProvider._curate
    monkeypatch.setattr(ByteRoverMemoryProvider, "_curate", _spy)
    home_token = set_hermes_home_override(str(prof_b))
    try:
        provider = ByteRoverMemoryProvider({"auto_extract": True})
        provider.initialize("session-cwd")
        provider.sync_turn("another substantive turn worth remembering forever", "ok")
        assert provider._sync_thread is not None
        provider._sync_thread.join(timeout=10)
    finally:
        reset_hermes_home_override(home_token)
        monkeypatch.setattr(ByteRoverMemoryProvider, "_curate", real_curate)

    assert captured.get("cwd") == prof_b / "byterover"
