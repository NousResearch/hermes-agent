"""Reset forgets only proven stale identity and cannot race state publication (#117326)."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    import hermes_constants
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)
    path = supervisor.state_path()
    path.parent.mkdir(parents=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    child = subprocess.Popen([sys._base_executable, "-c", "import time; time.sleep(90)"], creationflags=flags)
    proc = psutil.Process(child.pid)
    process_type = psutil.Process
    try:
        yield path, child, proc
    finally:
        monkeypatch.setattr(psutil, "Process", process_type)
        if proc.is_running():
            proc.terminate()
        child.wait(timeout=10)


@pytest.mark.parametrize("case,allowed", [
    ("legacy_foreign", True), ("legacy_reused", True), ("dead_pid", True),
    ("modern_reused", True), ("modern_live_owner", False), ("future_birth", False),
    ("malformed", False), ("access_denied", False), ("managed_ambiguous", False),
    ("invalid_url", False), ("lock_directory", False),
])
def test_reset_identity_and_route_preserve_process_and_config(runtime, tmp_path, monkeypatch, case, allowed):
    from fastapi import HTTPException
    from hermes_cli.local_runtime import recovery
    from hermes_cli.web_routers.local_models import ServerActionBody, local_models_server

    path, child, proc = runtime
    born = proc.create_time()
    state = {"pid": child.pid, "base_url": "http://127.0.0.1:18434/v1", "api_key": "test"}
    if case.startswith("modern") or case == "future_birth":
        # An already-exited PID supplies a real dead owner, not a fabricated nonexistent PID.
        exited = subprocess.Popen([sys._base_executable, "-c", "pass"], creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        owner = psutil.Process(exited.pid)
        owner_birth = owner.create_time()
        exited.wait(timeout=10)
        state.update(create_time=born - 1, executable=proc.exe(), owner_pid=exited.pid, owner_create_time=born - 2)
        if case == "modern_live_owner":
            state.update(owner_pid=os.getpid(), owner_create_time=psutil.Process().create_time(), create_time=born)
        elif case == "future_birth":
            state.update(create_time=born + 100, owner_create_time=owner_birth)
    if case == "dead_pid":
        proc.terminate()
        child.wait(timeout=10)
    if case == "malformed":
        state["pid"] = True
    if case == "invalid_url":
        state["base_url"] = 123
    if case == "lock_directory":
        path.with_suffix(".lock").mkdir()
    path.write_text(json.dumps(state), encoding="utf-8")
    if case == "legacy_reused":
        os.utime(path, (born - 1, born - 1))
    if case == "access_denied":
        def denied(pid):
            raise psutil.AccessDenied(pid)
        monkeypatch.setattr(recovery.psutil, "Process", denied)
    if case == "managed_ambiguous":
        original = recovery.psutil.Process
        def ambiguous(pid):
            result = original(pid)
            result.exe = lambda: str(path.parent / "build" / "cpu" / "llama-server")
            return result
        monkeypatch.setattr(recovery.psutil, "Process", ambiguous)
    config = tmp_path / "config.yaml"
    config.write_text("local_runtime:\n  enabled: false\n", encoding="utf-8")
    before = path.read_bytes()
    assert recovery.stale_record_available() is (allowed or case == "lock_directory")
    if allowed:
        assert asyncio.run(local_models_server(ServerActionBody(action="reset"))) == {"ok": True, "action": "reset"}
        assert not path.exists()
        assert recovery.reset_stale_record()  # absent is idempotent
    else:
        with pytest.raises(HTTPException) as error:
            asyncio.run(local_models_server(ServerActionBody(action="reset")))
        assert error.value.status_code == 409
        assert path.read_bytes() == before
    assert config.read_text(encoding="utf-8") == "local_runtime:\n  enabled: false\n"
    assert child.poll() is None or case == "dead_pid"


def test_reset_rechecks_after_other_process_publishes(runtime, tmp_path):
    from hermes_cli.local_runtime import recovery

    path, child, proc = runtime
    path.write_text(json.dumps({"pid": child.pid, "base_url": "http://127.0.0.1:18434/v1", "api_key": "test"}), encoding="utf-8")
    ready, release, finished = tmp_path / "ready", tmp_path / "release", tmp_path / "finished"
    worker_code = '''
import json, os, sys, time
from pathlib import Path
import psutil
from hermes_cli.local_runtime import supervisor, recovery
from utils import atomic_json_write
path, ready, release, finished = map(Path, sys.argv[1:])
supervisor.state_path = lambda: path
with recovery.state_lock():
    ready.touch()
    until = time.monotonic() + 10
    while not release.exists():
        if time.monotonic() > until: raise TimeoutError("test release missing")
        time.sleep(.02)
    p = psutil.Process()
    state = dict(pid=p.pid, create_time=p.create_time(), executable=p.exe(),
                 owner_pid=p.pid, owner_create_time=p.create_time(),
                 base_url="http://127.0.0.1:18434/v1", api_key="test")
    atomic_json_write(path, state, mode=0o600)
until = time.monotonic() + 30
while not finished.exists():
    if time.monotonic() > until: raise TimeoutError("test completion missing")
    time.sleep(.02)
'''
    worker = subprocess.Popen([sys.executable, "-c", worker_code, str(path), str(ready), str(release), str(finished)], creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert worker.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.02)
        with pytest.raises(TimeoutError):
            with recovery.state_lock(timeout_s=.1):
                pytest.fail("must not acquire the other process's lock")
        with ThreadPoolExecutor() as pool:
            pending = pool.submit(recovery.reset_stale_record)
            release.touch()
            assert pending.result(timeout=5) is False
        assert json.loads(path.read_text(encoding="utf-8"))["pid"] != child.pid
    finally:
        release.touch()
        finished.touch()
        worker.wait(timeout=15)
    assert worker.returncode == 0


def test_publication_failure_reaps_new_child_without_disabling_restart(runtime, tmp_path, monkeypatch):
    from contextlib import contextmanager
    from hermes_cli.local_runtime import recovery, supervisor

    path, child, proc = runtime
    @contextmanager
    def unavailable():
        raise TimeoutError("held state lock")
        yield
    monkeypatch.setattr(recovery, "state_lock", unavailable)
    monkeypatch.setattr(supervisor, "server_binary", lambda _: Path(sys._base_executable))
    monkeypatch.setattr(supervisor, "_direct_io_args", lambda _: ())
    monkeypatch.setattr(supervisor, "spawn_server", lambda *a, **kw: (child, None))
    sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path / "models", port=18434)
    try:
        with pytest.raises(TimeoutError, match="held state lock"):
            sup._spawn()
        child.wait(timeout=10)
        assert not proc.is_running()
        assert not sup._stopping
        assert not path.exists()
    finally:
        if sup._log_handle:
            sup._log_handle.close()
