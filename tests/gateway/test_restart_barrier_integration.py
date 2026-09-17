"""Real-process integration tests for restart serialization and startup barriers."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from gateway.restart_barrier import barrier_path, wait_for_restart_barrier, write_restart_barrier
from hermes_cli.restart_lease import LEASE_TOKEN_ENV, RestartLeaseBusy, restart_lease


@pytest.mark.skipif(os.name != "posix", reason="process/port/lock integration is POSIX-specific")
def test_restart_lease_serializes_processes_and_allows_owned_child(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    code = (
        "import sys\n"
        "from hermes_cli.restart_lease import RestartLeaseBusy,restart_lease\n"
        "try:\n"
        "  with restart_lease('child', timeout=0): print('acquired')\n"
        "except RestartLeaseBusy:\n"
        "  sys.exit(75)\n"
    )
    with restart_lease("parent", home=tmp_path) as lease:
        contending_env = os.environ.copy()
        contending_env.pop(LEASE_TOKEN_ENV, None)
        blocked = subprocess.run(
            [sys.executable, "-c", code], env=contending_env,
            capture_output=True, text=True, timeout=10, check=False,
        )
        assert blocked.returncode == 75
        joined = subprocess.run(
            [sys.executable, "-c", code], env=lease.child_env(),
            capture_output=True, text=True, timeout=10, check=False,
        )
        assert joined.returncode == 0 and "acquired" in joined.stdout

    with restart_lease("successor", home=tmp_path, timeout=0):
        pass


@pytest.mark.skipif(os.name != "posix", reason="process/port/lock integration is POSIX-specific")
def test_successor_waits_for_real_predecessor_port_and_scoped_lock(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    lock_dir.mkdir()
    child_code = r'''
import json,os,socket,sys,time
from pathlib import Path
s=socket.socket(); s.bind(("127.0.0.1",0)); s.listen(1)
port=s.getsockname()[1]
lock=Path(os.environ["HERMES_GATEWAY_LOCK_DIR"])/"slack-app-token-integration.lock"
lock.write_text(json.dumps({"pid":os.getpid(),"platform":"slack"}))
print(json.dumps({"pid":os.getpid(),"port":port}),flush=True)
time.sleep(0.8)
s.close()
'''
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code], env=os.environ.copy(),
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    ready = json.loads(proc.stdout.readline())
    reaper = threading.Thread(target=proc.wait, daemon=True)
    reaper.start()
    write_restart_barrier(ready["pid"], home=tmp_path, ports=[ready["port"]])
    started = time.monotonic()
    ok, evidence = wait_for_restart_barrier(home=tmp_path, timeout=5, poll=0.02)
    elapsed = time.monotonic() - started
    reaper.join(timeout=5)

    assert ok is True and elapsed >= 0.5
    assert evidence["predecessor_exited"] is True
    assert evidence["scoped_locks_released"] is True
    assert evidence["ports_released"] is True
    assert not barrier_path(tmp_path).exists()


@pytest.mark.skipif(os.name != "posix", reason="process scoped-lock integration is POSIX-specific")
def test_scoped_lock_is_reacquired_after_real_owner_process_exits(monkeypatch, tmp_path):
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    child_code = r'''
import os,time
from gateway.status import acquire_scoped_lock
ok,_=acquire_scoped_lock("slack-app-token","integration-token",metadata={"platform":"slack"})
print("READY" if ok else "FAILED",flush=True)
time.sleep(0.6)
'''
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code], env=os.environ.copy(),
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None and proc.stdout.readline().strip() == "READY"
    from gateway.status import acquire_scoped_lock, release_scoped_lock
    monkeypatch.setattr("gateway.status._looks_like_gateway_process", lambda _pid: True)
    acquired, owner = acquire_scoped_lock(
        "slack-app-token", "integration-token", metadata={"platform": "slack"})
    assert acquired is False and owner is not None and int(owner["pid"]) == proc.pid
    proc.wait(timeout=5)
    acquired, _owner = acquire_scoped_lock(
        "slack-app-token", "integration-token", metadata={"platform": "slack"})
    assert acquired is True
    release_scoped_lock("slack-app-token", "integration-token")
