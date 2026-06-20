import json
import subprocess


class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid
        self.returncode = 0
        self._alive = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False  # simulate a process that exits promptly on SIGTERM

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, timeout=None):
        if self._alive:
            raise subprocess.TimeoutExpired(cmd="langflow", timeout=timeout)
        return 0


def test_stop_terminates_owned_process_and_waits(monkeypatch):
    from hermes_cli import workflow_backend as wb

    backend = wb.WorkflowBackend()
    proc = _FakeProc()
    backend._proc = proc
    backend._state = "ready"
    monkeypatch.setattr(wb, "is_reachable", lambda *a, **k: False)

    status = backend.stop()

    assert proc.terminated is True
    assert backend._proc is None
    assert status["state"] == "stopped"


def test_restart_respawns_owned_process_so_new_env_lands(monkeypatch, tmp_path):
    from hermes_cli import workflow_backend as wb

    backend = wb.WorkflowBackend()
    old = _FakeProc(1111)
    backend._proc = old
    backend._state = "ready"
    # Port is free once the old process is killed, so start() spawns fresh
    # instead of attaching — that is what injects the updated token env.
    monkeypatch.setattr(wb, "is_reachable", lambda *a, **k: False)
    monkeypatch.setattr(wb, "resolve_langflow_root", lambda: str(tmp_path))
    monkeypatch.setattr(wb, "build_launch", lambda root: ["echo", "x"])
    monkeypatch.setattr(wb, "build_runtime_env", lambda: {"X": "1"})
    spawned = []
    new = _FakeProc(2222)

    def fake_popen(cmd, **kwargs):
        spawned.append((cmd, kwargs))
        return new

    monkeypatch.setattr(wb.subprocess, "Popen", fake_popen)

    backend.restart()

    assert old.terminated is True
    assert len(spawned) == 1
    assert backend._proc is new


def test_restart_attaches_to_external_process_without_terminating(monkeypatch):
    from hermes_cli import workflow_backend as wb

    backend = wb.WorkflowBackend()
    backend._proc = None  # langflow owned by another launcher (e.g. Electron)
    monkeypatch.setattr(wb, "is_reachable", lambda *a, **k: True)

    def must_not_spawn(*a, **k):
        raise AssertionError("restart must not spawn when langflow is external+reachable")

    monkeypatch.setattr(wb.subprocess, "Popen", must_not_spawn)

    status = backend.restart()

    assert status["state"] == "ready"
    assert status["external"] is True


def test_auth_status_treats_unreachable_saved_kari_hub_as_logged_out(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "workflow-secrets.json").write_text(
        json.dumps({"kari": {"cloudBaseURL": "http://127.0.0.1:8900/", "token": "workspace-token"}}),
        encoding="utf-8",
    )

    from hermes_cli import workflow_backend

    monkeypatch.setattr(workflow_backend, "is_cloud_reachable", lambda url: False)

    status = workflow_backend.auth_status()

    assert status["loggedIn"] is False
    assert status["cloudBaseUrl"] == "http://127.0.0.1:8900"
    assert status["cloudReachable"] is False
    assert "Kari hub is not reachable" in status["error"]
