"""Recovery only targets the recorded managed router, never a process-name sweep."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
import pytest


def _wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    assert predicate(), "process transition did not complete"


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("case", [
    "modern", "legacy", "live-owner", "pid-reused", "unknown-owner", "wrong-exe",
    "legacy-key", "legacy-models", "busy", "state-replaced", "wrong-parent",
])
def test_startup_preserves_trees_and_explicit_stop_checks_owner(tmp_path, monkeypatch, case):
    from hermes_cli.local_runtime import bootstrap, supervisor

    root = tmp_path / "managed runtime"
    root.mkdir()
    monkeypatch.setattr(supervisor, "runtimes_root", lambda: root)
    monkeypatch.setattr(bootstrap, "runtimes_root", lambda: root)
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "_presets_stale", lambda: False)
    monkeypatch.setattr(bootstrap, "_detect_gpu_vendor", lambda: None)

    # A copied native interpreter stands in for the installed server; no live model is touched.
    exe = root / "test-build" / "cpu" / "llama-server.exe"
    exe.parent.mkdir(parents=True)
    shutil.copy2(sys._base_executable, exe)
    env = os.environ.copy()
    env["PYTHONHOME"] = str(Path(sys._base_executable).parent)
    env["PATH"] = str(Path(sys._base_executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    mdir = tmp_path / "models"
    mdir.mkdir()
    monkeypatch.setattr(bootstrap, "models_dir", lambda: mdir)
    # A real owner exits without cleanup, leaving a router and its model child.
    ready = tmp_path / "router.json"
    child_script = tmp_path / "router.py"
    child_script.write_text(
        "import json,os,sys,time,subprocess,psutil\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "with open(sys.argv[1], 'w') as f:\n"
        " json.dump({'router': os.getpid(), 'child': child.pid, "
        "'create_time': psutil.Process().create_time()}, f)\n"
        "time.sleep(60)\n", encoding="utf-8")
    owner = subprocess.Popen([
        sys.executable, "-c",
        "import subprocess,sys,psutil,json,os,time; "
        "p=subprocess.Popen(sys.argv[1:]); "
        "print(json.dumps({'pid':os.getpid(),'create_time':psutil.Process().create_time()}),flush=True); "
        f"time.sleep({60 if case == 'live-owner' else 1})",
        str(exe), str(child_script), str(ready),
        "--host", "127.0.0.1", "--port", "59999", "--api-key", "test-only",
        "--models-dir", str(mdir),
    ], stdout=subprocess.PIPE, text=True, env=env)
    owner_identity = json.loads(owner.stdout.readline())
    processes = []
    try:
        _wait_for(ready.exists)
        record = json.loads(ready.read_text())
        processes = [psutil.Process(record[k]) for k in ("router", "child")]
        if case != "live-owner":
            owner.wait(timeout=10)
        # Retain the identity, not a still-open Popen handle keeping a dead PID visible.
        monkeypatch.setattr(bootstrap, "staged_models", lambda: [tmp_path / "model.gguf"])
        state = {
            "pid": record["router"], "create_time": record["create_time"],
            "owner_pid": owner_identity["pid"], "owner_create_time": owner_identity["create_time"],
            "executable": processes[0].exe(),
            "base_url": "http://127.0.0.1:59999/v1", "api_key": "test-only",
        }
        if case.startswith("legacy"):
            state = {key: state[key] for key in ("pid", "base_url", "api_key")}
        if case == "pid-reused":
            state["create_time"] -= 10
        if case == "unknown-owner":
            state.pop("owner_create_time")
        if case == "wrong-exe":
            state["executable"] = str(tmp_path / "other.exe")
        if case == "legacy-key":
            state["api_key"] = "different-key"
        if case == "legacy-models":
            monkeypatch.setattr(bootstrap, "models_dir", lambda: tmp_path / "other models")
        supervisor.state_path().write_text(json.dumps(state), encoding="utf-8")
        from fastapi import HTTPException
        from hermes_cli.local_runtime import endpoint
        from hermes_cli.web_routers import local_models
        if case == "wrong-parent":
            state["owner_pid"] = os.getpid()
            supervisor.state_path().write_text(json.dumps(state))
        for _ in range(2):
            bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True}})
            assert all(p.is_running() for p in processes), f"startup killed: {case}"
            assert json.loads(supervisor.state_path().read_text()) == state
        if case == "state-replaced":
            state = {**state, "owner_pid": os.getpid(), "owner_create_time": psutil.Process().create_time()}
            supervisor.state_path().write_text(json.dumps(state))
        if case in ("modern", "legacy", "busy"):
            if case == "modern":
                from fastapi import FastAPI
                from fastapi.testclient import TestClient
                app = FastAPI()
                app.include_router(local_models.router)
                disabled = []
                monkeypatch.setattr(local_models, "_set_runtime_enabled", lambda value: disabled.append(value))
                response = TestClient(app).post("/api/local-models/server", json={"action": "stop"})
                assert response.status_code == 200, response.text
                assert disabled == [False]
            else:
                local_models._terminate_state_pid()
            _wait_for(lambda: not any(p.is_running() for p in processes))
            assert endpoint._state_endpoint() is None
        else:
            with pytest.raises(HTTPException) as exc:
                local_models._terminate_state_pid()
            assert exc.value.status_code == 409
            assert all(p.is_running() for p in processes), f"unsafe recovery: {case}"
        assert json.loads(supervisor.state_path().read_text()) == state
    finally:
        for proc in processes:
            if proc.is_running():
                proc.kill()
            try:
                proc.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        owner.stdout.close()


@pytest.mark.parametrize("failure", ["arrival", "truncated"])
def test_startup_reuses_without_activity_probe(tmp_path, monkeypatch, failure):
    import http.client
    from hermes_cli.local_runtime import bootstrap, endpoint, recovery

    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "staged_models", lambda: [tmp_path / "model.gguf"])
    monkeypatch.setattr(bootstrap, "_presets_stale", lambda: False)
    state = {"base_url": "http://127.0.0.1:59999/v1", "api_key": "test-only"}
    monkeypatch.setattr(endpoint, "_state_endpoint", lambda: state)
    calls = []
    def probe(*args, **kwargs):
        calls.append(True)
        if failure == "truncated":
            raise http.client.IncompleteRead(b"partial", 100)
        return {"data": []}  # work arrives immediately after an idle snapshot
    monkeypatch.setattr(endpoint, "managed_get_json", probe)
    monkeypatch.setattr(recovery, "read_state", lambda: state)
    from types import SimpleNamespace
    monkeypatch.setattr(recovery, "recorded_process",
                        lambda state: SimpleNamespace(is_running=lambda: False))
    monkeypatch.setattr(recovery, "_owner_is_dead", lambda state: True)
    for _ in range(2):
        assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True}}) is None
    assert not calls


def test_shutdown_during_backoff_cannot_restart_or_remove_another_server(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path, port=59998)
    sup.proc = SimpleNamespace(pid=101, poll=lambda: 1)
    other = {"pid": 202, "base_url": "http://127.0.0.1:59997/v1", "api_key": "test-only"}
    supervisor.state_path().write_text(json.dumps(other))
    spawned = []
    monkeypatch.setattr(sup, "_spawn", lambda: spawned.append(True))
    monkeypatch.setattr(sup, "_wait_health", lambda *a: None)
    monkeypatch.setattr(sup, "_reap_orphaned_children", lambda: None)
    monkeypatch.setattr(supervisor.time, "sleep", lambda *a: sup.stop())
    # A real Event.wait is independently exercised by the native lifetime tests.
    if hasattr(sup, "_stop_event"):
        monkeypatch.setattr(sup._stop_event, "wait", lambda *a: (sup.stop() or True))
    sup._watch()
    assert not spawned, "stop during restart backoff resurrected the runtime"
    assert json.loads(supervisor.state_path().read_text()) == other


@pytest.mark.platforms("windows")
def test_supervisor_reaps_owned_job_even_after_router_exit(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path, port=59998)
    sup.proc = SimpleNamespace(pid=101, poll=lambda: 1)
    closed = []
    sup._job = SimpleNamespace(close=lambda: closed.append(True))
    sup._reap_orphaned_children()
    assert closed == [True], "crashed router's model children escaped cleanup"
    assert sup._job is None
    sup._job = SimpleNamespace(close=lambda: closed.append(True))
    sup.stop()
    assert closed == [True, True]
    assert sup._job is None


def test_spawn_state_records_process_incarnations(tmp_path, monkeypatch):
    import os
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path, port=59998)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        sup.proc = proc
        sup._write_state()
        state = json.loads(supervisor.state_path().read_text())
        assert state.get("create_time") == psutil.Process(proc.pid).create_time()
        assert state.get("owner_pid") == os.getpid()
        assert state.get("owner_create_time") == psutil.Process().create_time()
        assert state.get("executable") == psutil.Process(proc.pid).exe()
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_stopped_state_is_retained_without_unlink_race(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    sup = supervisor.LlamaServerSupervisor(tmp_path, tmp_path, port=59998)
    sup.proc = SimpleNamespace(pid=101, poll=lambda: 1)
    sup._state = {"pid": 101}
    path = supervisor.state_path()
    path.write_text(json.dumps(sup._state))
    replacement = {"pid": os.getpid(), "base_url": "http://127.0.0.1:59997/v1", "api_key": "test-only"}
    unlink = Path.unlink
    def publish_before_unlink(self, *args, **kwargs):
        if self == path:
            self.write_text(json.dumps(replacement))
        return unlink(self, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", publish_before_unlink)
        sup.stop()
    assert json.loads(path.read_text()) == sup._state
    path.write_text(json.dumps(replacement))
    sup.stop()
    assert json.loads(path.read_text()) == replacement


@pytest.mark.parametrize("kind", ["psutil", "subprocess", "missing-psutil", "wait-error"])
def test_terminate_tree_escalates_and_always_cleans_children(monkeypatch, kind):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor

    child = Mock()
    child.is_running.return_value = True
    proc = Mock(pid=123)
    error = (psutil.TimeoutExpired(15) if kind == "psutil" else
             RuntimeError("wait failed") if kind == "wait-error" else
             subprocess.TimeoutExpired("router", 15))
    proc.wait.side_effect = error
    if kind == "missing-psutil":
        monkeypatch.setitem(sys.modules, "psutil", None)
    else:
        monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(children=lambda **kw: [child]))
    if kind == "wait-error":
        with pytest.raises(RuntimeError, match="wait failed"):
            LlamaServerSupervisor._terminate_tree(proc)
    else:
        LlamaServerSupervisor._terminate_tree(proc)
        proc.kill.assert_called_once()
    if kind != "missing-psutil":
        child.terminate.assert_called_once()
        child.kill.assert_called_once()


@pytest.mark.parametrize("reuse_at", ["before-walk", "during-walk", "never"])
def test_explicit_stop_preserves_verified_root_incarnation(tmp_path, monkeypatch, reuse_at):
    from unittest.mock import Mock
    from hermes_cli.local_runtime import recovery, supervisor

    state = {"pid": 123, "create_time": 1.0}
    path = tmp_path / "server.json"
    path.write_text(json.dumps(state))
    monkeypatch.setattr(supervisor, "state_path", lambda: path)
    root = Mock(spec=psutil.Process, pid=123)
    owned_child = Mock(spec=psutil.Process)
    replacement_child = Mock(spec=psutil.Process)
    replacement = Mock(spec=psutil.Process, pid=123)
    replacement.children.return_value = [replacement_child]
    replacement_child.is_running.return_value = True
    owned_child.is_running.return_value = True
    stale = False

    def final_identity_check():
        nonlocal stale
        if stale:
            return False
        # The verified root exits and its PID is reused just after this check.
        stale = reuse_at == "before-walk"
        return True

    def verified_children(*, recursive):
        nonlocal stale
        assert recursive is True
        if stale:
            raise psutil.NoSuchProcess(root.pid)
        if reuse_at == "during-walk":
            stale = True
            return [replacement_child]
        return [owned_child]

    def verified_terminate():
        if stale:
            raise psutil.NoSuchProcess(root.pid)

    root.is_running.side_effect = final_identity_check
    root.children.side_effect = verified_children
    root.terminate.side_effect = verified_terminate
    factory = Mock(side_effect=lambda pid: replacement if stale else root)
    monkeypatch.setattr(psutil, "Process", factory)
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(recovery, "recorded_process", lambda record: root)
    monkeypatch.setattr(recovery, "_owner_is_dead", lambda record: True)

    stopped = recovery.stop_recorded_orphan()

    replacement_child.terminate.assert_not_called()
    replacement_child.kill.assert_not_called()
    replacement.terminate.assert_not_called()
    replacement.kill.assert_not_called()
    root.children.assert_called_once_with(recursive=True)
    factory.assert_not_called()
    assert stopped is (reuse_at == "never")
    if reuse_at != "never":
        root.terminate.assert_not_called()
        root.kill.assert_not_called()
        owned_child.terminate.assert_not_called()
        owned_child.kill.assert_not_called()
    else:
        root.terminate.assert_called_once_with()
        owned_child.terminate.assert_called_once_with()
        owned_child.kill.assert_called_once_with()


@pytest.mark.platforms("linux")
def test_reparented_router_keeps_its_endpoint(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import endpoint, supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    read_fd, write_fd = os.pipe()
    # EOF ends the test child without signaling outside the test's subtree after reparenting.
    with os.fdopen(write_fd, "wb") as control:
        try:
            owner = subprocess.Popen([sys.executable, "-c", """
import json, os, psutil, subprocess, sys
p = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.read(1)'],
                     stdin=int(sys.argv[1]), stdout=subprocess.DEVNULL)
proc = psutil.Process(p.pid)
print(json.dumps({'pid': proc.pid, 'create_time': proc.create_time(), 'executable': proc.exe(),
                  'owner_pid': os.getpid(), 'owner_create_time': psutil.Process().create_time()}), flush=True)
""", str(read_fd)], pass_fds=(read_fd,), stdout=subprocess.PIPE, text=True)
        finally:
            os.close(read_fd)
        state = json.loads(owner.stdout.readline())
        proc = psutil.Process(state["pid"])
        try:
            owner.wait(timeout=10)
            route = {"base_url": "http://127.0.0.1:59999/v1", "api_key": "test-only"}
            supervisor.state_path().write_text(json.dumps({**state, **route}))
            assert proc.ppid() != state["owner_pid"]
            assert endpoint._state_endpoint() == route
        finally:
            control.close()
            proc.wait(timeout=10)
            owner.stdout.close()


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("damage", ["valid", "birth", "exe", "bool-pid", "bool-birth", "nan", "inf", "owner-bool", "owner-nan", "parent", "partial", "list", "invalid", "unreadable"])
def test_retained_endpoint_validates_identity(tmp_path, monkeypatch, damage):
    from hermes_cli.local_runtime import endpoint, recovery, supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        real = psutil.Process(proc.pid)
        state = {"pid": real.pid, "create_time": real.create_time(), "executable": real.exe(),
                 "owner_pid": os.getpid(), "owner_create_time": psutil.Process().create_time(),
                 "base_url": "http://127.0.0.1:59999/v1", "api_key": "test-only"}
        changes = {"birth": {"create_time": real.create_time() - 10}, "exe": {"executable": str(tmp_path / "wrong.exe")},
                   "bool-pid": {"pid": True}, "bool-birth": {"create_time": True},
                   "nan": {"create_time": float("nan")}, "inf": {"create_time": float("inf")},
                   "owner-bool": {"owner_create_time": True}, "owner-nan": {"owner_create_time": float("nan")},
                   "parent": {"owner_pid": real.pid}}
        state.update(changes.get(damage, {}))
        if damage == "partial":
            state.pop("create_time")
        path = supervisor.state_path()
        path.write_text("[]" if damage == "list" else "{" if damage == "invalid" else json.dumps(state))
        if damage == "unreadable":
            path.unlink()
            path.mkdir()
        got = endpoint._state_endpoint()
        if damage == "valid":
            assert got == {"base_url": state["base_url"], "api_key": state["api_key"]}
            proc.terminate()
            proc.wait(timeout=5)
            assert endpoint._state_endpoint() is None
            assert path.exists()
        else:
            assert got is None
            if damage not in ("list", "invalid", "unreadable"):
                assert recovery.recorded_process(state) is None
            assert real.is_running()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
