from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from hermes_cli import control_db as cp
from hermes_cli.control_spawn import spawn_statute_worker


def test_spawn_statute_worker_command_env_and_log(monkeypatch, tmp_path):
    calls = {}

    class FakeProc:
        pid = 777

    def fake_popen(cmd, env, stdout, stderr):
        calls["cmd"] = cmd
        calls["env"] = env
        calls["stdout_name"] = stdout.name
        calls["stderr"] = stderr
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    payload = {"repo_root": str(tmp_path / "repo"), "allowed_paths": [str(tmp_path / "repo")]}
    pid = spawn_statute_worker("disp_child", payload, tmp_path / ".hermes", "disp_parent", soft_timeout_s=600, hard_timeout_s=7200)
    assert pid == 777
    assert calls["cmd"][:6] == ["hermes", "-p", "statute-worker", "control", "worker", "run"]
    assert "--accept-hooks" not in calls["cmd"]
    assert calls["cmd"][-6:] == ["--handler", "agent", "--soft-timeout-s", "600.0", "--hard-timeout-s", "7200.0"]
    assert "disp_child" in calls["cmd"]
    assert "--root" in calls["cmd"]
    assert calls["env"]["HERMES_PROFILE_ID"] == "statute-worker"
    assert calls["env"]["HERMES_CONTROL_DISPATCH_ID"] == "disp_child"
    assert calls["env"]["HERMES_CONTROL_PARENT_DISPATCH_ID"] == "disp_parent"
    assert calls["stdout_name"].endswith("control-plane/logs/disp_child.log")


def test_spawn_statute_worker_integration_with_temp_hermes_wrapper(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    (root / "profiles" / "statute-worker").mkdir(parents=True)
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "hermes"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable!r} -m hermes_cli.main \"$@\"\n")
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{wrapper_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("HERMES_HOME", str(root))
    payload = {
        "schema": "statute_dispatch_v1",
        "silo": "statute",
        "repo_root": str(repo),
        "allowed_paths": [str(repo)],
        "task_type": "generic",
        "task_permissions": ["read", "test"],
        "parent_dispatch_id": "disp_parent",
        "instructions": "integration",
        "constraints": {"no_live_db_mutation": True, "no_push": True},
    }
    conn = cp.connect(root=root)
    try:
        cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        child = cp.create_dispatch_from_instance(conn, sender_instance_id="statutepm:bootstrap", receiver_profile="statute-worker", payload=payload)
    finally:
        conn.close()
    pid = spawn_statute_worker(child, payload, root, "disp_parent", handler="deterministic")
    assert pid > 0
    deadline = time.monotonic() + 10
    status = None
    while time.monotonic() < deadline:
        conn = cp.connect(root=root)
        try:
            row = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (child,)).fetchone()
            status = row["status"]
            if status == "completed":
                assert cp.get_latest_dispatch_result(conn, child)
                assert cp.list_artifacts(conn, child)
                break
        finally:
            conn.close()
        time.sleep(0.1)
    assert status == "completed"
