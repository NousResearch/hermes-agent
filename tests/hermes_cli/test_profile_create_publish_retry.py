"""Regression for #121827: profile create must survive a locked staging tree.

On Windows ``os.rename(staging, profile_dir)`` raises ``PermissionError``
(WinError 5) while another process (indexer, antivirus, editor file watcher)
holds a freshly copied skill file open. The publish step must retry briefly,
then fall back to copying the tree into place.

The failing ``os.rename`` here is a test double for that Windows kernel
behaviour (Linux renames straight through open handles), with a real open
handle held for the duration to model the watching process.
"""

import os
from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_cli.profiles import create_profile, list_profiles


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _staging_rename_double(monkeypatch, failures_before_success):
    """Fail renames of the staging dir with PermissionError, like WinError 5.

    Only the publish rename is failed (matched on the ``.staging-<pid>``
    name); every other rename delegates to the real implementation. Holds a
    real open handle is the caller's job; this double models the kernel side.
    """
    real_rename = os.rename
    calls = {"count": 0}

    def flaky(src, dst, *args, **kwargs):
        if ".staging-" in os.fspath(src):
            calls["count"] += 1
            if calls["count"] <= failures_before_success:
                raise PermissionError("[WinError 5] Access is denied: %r" % (src,))
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", flaky)
    return calls


def test_create_profile_survives_transient_publish_denied(
    profile_env, tmp_path, monkeypatch
):
    """Two denied publish renames, then success: the profile is still created."""
    calls = _staging_rename_double(monkeypatch, failures_before_success=2)
    sleeps = []
    monkeypatch.setattr(profiles.time, "sleep", lambda s: sleeps.append(s))
    # Model the watching process: a real open handle inside the profiles tree.
    profiles_root = tmp_path / ".hermes" / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)
    with open(profiles_root / "watcher.lock", "w") as watcher:
        watcher.write("x")
        watcher.flush()
        profile_dir = create_profile("locked", no_alias=True, no_skills=True)
    assert profile_dir.is_dir()
    assert "locked" in [p.name for p in list_profiles()]
    assert calls["count"] == 3
    assert sleeps, "expected backoff sleeps between publish retries"
    leftovers = [p for p in profiles_root.iterdir() if ".staging-" in p.name]
    assert leftovers == []


def test_create_profile_copies_into_place_when_publish_stays_denied(
    profile_env, tmp_path, monkeypatch
):
    """A rename denied for the whole retry budget falls back to copy-into-place."""
    calls = _staging_rename_double(monkeypatch, failures_before_success=10**9)
    monkeypatch.setattr(profiles.time, "sleep", lambda s: None)
    profiles_root = tmp_path / ".hermes" / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)
    with open(profiles_root / "watcher.lock", "w") as watcher:
        watcher.write("x")
        watcher.flush()
        profile_dir = create_profile("indexed", no_alias=True, no_skills=True)
    assert profile_dir.is_dir()
    assert (profile_dir / ".env").exists()
    assert "indexed" in [p.name for p in list_profiles()]
    assert calls["count"] >= 2, "expected rename retries before the copy fallback"
    leftovers = [p for p in profiles_root.iterdir() if ".staging-" in p.name]
    assert leftovers == []
