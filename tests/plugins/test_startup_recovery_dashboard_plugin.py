"""Tests for the LOCAL GSSAI startup-recovery dashboard plugin."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_plugin_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "startup-recovery" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_startup_recovery_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin_module():
    return _load_plugin_module()


@pytest.fixture
def client(tmp_path, monkeypatch, plugin_module):
    hermes_root = tmp_path / ".hermes"
    monkeypatch.setenv("STARTUP_RECOVERY_HERMES_ROOT", str(hermes_root))

    for profile in ("ivan_bb", "storm", "neo", "gssai_admin"):
        root = hermes_root / "profiles" / profile
        (root / "logs").mkdir(parents=True)
        (root / "logs" / "gateway.log").write_text(
            "normal startup line\nTraceback: simulated but line must not be returned\n"
        )
        (root / "state.db").write_bytes(b"state")

    class Completed:
        def __init__(self, returncode=0, stdout=""):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run_command(args, timeout=2.0):
        if args == ["tmux", "ls"]:
            return Completed(
                0,
                "hermes-gateway: 1 windows\n"
                "storm-gateway: 1 windows\n"
                "neo-gateway: 1 windows\n"
                "gssai-admin-gateway: 1 windows\n"
                "client-side-project: 1 windows\n",
            )
        if args[:2] == ["pgrep", "-f"]:
            return Completed(0, "123\n")
        if args == ["ss", "-tn"]:
            return Completed(0, "ESTAB 0 0 127.0.0.1:1 149.154.167.220:443\n")
        return Completed(1, "")

    monkeypatch.setattr(plugin_module, "_run_command", fake_run_command)

    app = FastAPI()
    app.include_router(plugin_module.router, prefix="/api/plugins/startup-recovery")
    return TestClient(app)


def test_status_is_read_only_secret_free_and_ok(client):
    response = client.get("/api/plugins/startup-recovery/status")
    assert response.status_code == 200
    data = response.json()
    rendered = response.text

    assert data["scope"] == "LOCAL GSSAI PROJECT ONLY"
    assert data["overall"] == "OK"
    assert data["problem_count"] == 0
    assert data["telegram_tcp_connections_detected"] == 1
    assert "Traceback: simulated" not in rendered
    assert all(profile["tmux_status"] == "OK" for profile in data["profiles"])
    assert all(profile["process_status"] == "OK" for profile in data["profiles"])


def test_sessions_returns_allowlisted_names_not_process_args(client):
    response = client.get("/api/plugins/startup-recovery/sessions")
    assert response.status_code == 200
    data = response.json()
    rendered = response.text
    assert "hermes-gateway" in data["sessions"]
    assert "client-side-project" not in data["sessions"]
    assert "hermes --profile" not in rendered


def test_status_reports_problem_count_when_gateway_missing(tmp_path, monkeypatch, plugin_module):
    hermes_root = tmp_path / ".hermes"
    monkeypatch.setenv("STARTUP_RECOVERY_HERMES_ROOT", str(hermes_root))

    class Completed:
        def __init__(self, returncode=0, stdout=""):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run_command(args, timeout=2.0):
        if args == ["tmux", "ls"]:
            return Completed(0, "storm-gateway: 1 windows\n")
        if args[:2] == ["pgrep", "-f"]:
            return Completed(1, "")
        if args == ["ss", "-tn"]:
            return Completed(1, "")
        return Completed(1, "")

    monkeypatch.setattr(plugin_module, "_run_command", fake_run_command)
    app = FastAPI()
    app.include_router(plugin_module.router, prefix="/api/plugins/startup-recovery")

    response = TestClient(app).get("/api/plugins/startup-recovery/status")
    assert response.status_code == 200
    data = response.json()
    assert data["overall"] == "CHECK"
    assert data["problem_count"] == 4
    assert any(profile["tmux_status"] == "CHECK" for profile in data["profiles"])
    assert all(profile["process_status"] == "CHECK" for profile in data["profiles"])


def test_recent_error_count_reads_bounded_tail(tmp_path, plugin_module):
    log_file = tmp_path / "gateway.log"
    log_file.write_text(("startup near start\n" * 5000) + "normal tail line\ncritical tail line\n")

    assert plugin_module._recent_error_count(log_file, max_lines=20, max_bytes=128) == 1


def test_recent_log_counts_do_not_return_log_contents(client):
    response = client.get("/api/plugins/startup-recovery/logs/ivan_bb/recent")
    assert response.status_code == 200
    data = response.json()
    assert data["recent_error_count_tail_2000"] == 1
    assert data["log_contents_returned"] is False
    assert "Traceback: simulated" not in response.text


def test_unknown_profile_log_is_404(client):
    response = client.get("/api/plugins/startup-recovery/logs/aria/recent")
    assert response.status_code == 404


def test_registered_routes_are_get_only(plugin_module):
    methods = set()
    for route in plugin_module.router.routes:
        methods.update(getattr(route, "methods", set()))
    assert methods <= {"GET", "HEAD"}
