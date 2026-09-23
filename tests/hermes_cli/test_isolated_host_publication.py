"""Regression for #120165: isolated serve must not steal dashboard rendezvous."""

import os
import socket
import subprocess
import sys
import time

import psutil
import pytest

from gateway import host_rendezvous as hr
from hermes_cli import process_identity


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _await_bind(proc, port, log, *, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    pytest.fail(f"backend did not bind port {port} (exit={proc.poll()}): "
                f"{log.read_text(encoding='utf-8', errors='replace')}")


def _owned_pids(proc):
    parent = psutil.Process(proc.pid)
    return {parent.pid, *(child.pid for child in parent.children(recursive=True))}


def _await_publication(processes, dashboard_port, logs, *, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = hr.read_record(hr.ROLE_SERVE)
        entries = process_identity.ledger_entries()
        if (record is not None and record.port == dashboard_port
                and record.pid in _owned_pids(processes["dashboard"])
                and any(e["purpose"] == "dashboard" and e["pid"] == record.pid
                        for e in entries)
                and any(e["purpose"] == "serve" and e["pid"] in _owned_pids(processes["serve"])
                        for e in entries)):
            return
        if any(proc.poll() is not None for proc in processes.values()):
            break
        time.sleep(0.1)
    pytest.fail("dashboard record or independent process ledger missing: " +
                "\n".join(f"{name}: {path.read_text(encoding='utf-8', errors='replace')}"
                          for name, path in logs.items()))


@pytest.mark.parametrize("first", ["serve", "dashboard"])
def test_isolated_serve_and_dashboard_share_host_without_conflict(tmp_path, monkeypatch, first):
    home = tmp_path / "home"
    home.mkdir()
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>ready</html>", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    env = {**os.environ, "HERMES_WEB_DIST": str(dist)}
    for key in ("HERMES_DESKTOP", "HERMES_PARENT_PID", "HERMES_PARENT_START_MARKER"):
        env.pop(key, None)

    serve_port, dashboard_port = _free_port(), _free_port()
    assert serve_port != dashboard_port
    commands = {
        "serve": [sys.executable, "-m", "hermes_cli.main", "serve", "--isolated",
                  "--host", "127.0.0.1", "--port", str(serve_port)],
        "dashboard": [sys.executable, "-m", "hermes_cli.main", "dashboard",
                      "--host", "127.0.0.1", "--port", str(dashboard_port), "--no-open"],
    }
    ports = {"serve": serve_port, "dashboard": dashboard_port}
    processes = {}
    logs = {}
    try:
        for purpose in (first, "dashboard" if first == "serve" else "serve"):
            log = tmp_path / f"{purpose}.log"
            logs[purpose] = log
            with log.open("w", encoding="utf-8") as output:
                processes[purpose] = subprocess.Popen(
                    commands[purpose], env=env, stdout=output, stderr=subprocess.STDOUT,
                )
            _await_bind(processes[purpose], ports[purpose], log)

        _await_publication(processes, dashboard_port, logs)
    finally:
        for proc in processes.values():
            try:
                parent = psutil.Process(proc.pid) if proc.poll() is None else None
            except psutil.NoSuchProcess:
                parent = None
            if parent is not None:
                children = parent.children(recursive=True)
                for child in children:
                    child.terminate()
                parent.terminate()
                _, alive = psutil.wait_procs([*children, parent], timeout=5)
                for survivor in alive:
                    survivor.kill()
            proc.wait(timeout=10)