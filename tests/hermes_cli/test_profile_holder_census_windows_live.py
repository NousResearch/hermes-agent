"""LIVE Windows E2E for the profile-deletion holder census.

A holder that outlived its parent must block deletion: an MCP server whose Hermes parent
crashed keeps the profile's inherited ``logs/mcp-stderr.log`` handle, runs no Python, and
names the profile nowhere in its argv or cwd (PR #93508 review). Real processes against the
live system (Restart Manager on Windows); no mocked psutil.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from hermes_cli import profile_lifecycle, profiles

pytestmark = pytest.mark.platforms("windows")

_ORPHANING_PARENT = """
import subprocess, sys
log = open(sys.argv[1], "ab")
child = subprocess.Popen(["ping", "-n", "600", "127.0.0.1"], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=log, cwd=sys.argv[2])
print(child.pid, flush=True)
"""


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    root = tmp_path / "isolated-hermes"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    # No service manager, live backend or multiplexer is part of this census test.
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service",
                 "_maybe_register_gateway_service", "_stop_bot_desktop", "_notify_multiplexer"):
        monkeypatch.setattr(profiles, name, lambda *a, **k: None)
    monkeypatch.setattr(profiles, "_profile_bound_backend_pids", lambda *a, **k: [])
    monkeypatch.setattr(profile_lifecycle, "_PROFILE_DB_RELEASE_TIMEOUT_SECONDS", 0)
    return profiles.create_profile("worker", no_alias=True, no_skills=True)


def test_orphaned_holder_without_a_profile_reference_blocks_deletion(profile_home, tmp_path):
    log = profile_home / "logs" / "mcp-stderr.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    parent = subprocess.run([sys.executable, "-c", _ORPHANING_PARENT, str(log), str(elsewhere)],
                            capture_output=True, text=True, timeout=60, check=True)
    holder = psutil.Process(int(parent.stdout.strip()))
    try:
        assert not psutil.pid_exists(holder.ppid()) or (
            psutil.Process(holder.ppid()).create_time() > holder.create_time()
        ), "the holder must have outlived its parent"
        assert str(profile_home).lower() not in " ".join(holder.cmdline()).lower()
        assert not holder.cwd().lower().startswith(str(profile_home).lower())

        assert holder.pid in profile_lifecycle.external_profile_file_holders(profile_home)
        with pytest.raises(RuntimeError, match=str(holder.pid)):
            profiles.delete_profile("worker", yes=True)
        assert log.is_file() and not profiles.profile_home_is_tombstoned(profile_home)
    finally:
        holder.kill()
        holder.wait(10)

    deadline = time.monotonic() + 10
    while profile_lifecycle.external_profile_file_holders(profile_home) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert profiles.delete_profile("worker", yes=True) == profile_home
    assert not profile_home.exists()
