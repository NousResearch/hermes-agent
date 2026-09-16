"""Invariants for the terminal-backend doctor lines (tools-runtime-30): each failure names the backend in
plain words and points at `hermes setup terminal`, never at raw TERMINAL_* env vars."""

import pytest

from hermes_cli import doctor_tools


@pytest.fixture
def issues():
    return []


def _joined(capsys) -> str:
    return capsys.readouterr().out


def test_docker_missing_line_names_backend_and_setup_command(monkeypatch, capsys, issues):
    monkeypatch.setattr(doctor_tools, "_safe_which", lambda name: None)
    doctor_tools._check_docker_backend("docker", False, issues)
    out = _joined(capsys)
    assert "Docker not installed" in out and "'docker' terminal backend" in out
    assert "TERMINAL_ENV" not in out
    assert issues and "hermes setup terminal" in issues[0] and "Install Docker" in issues[0]


def test_docker_daemon_down_line_says_start_docker_or_switch(monkeypatch, capsys, issues):
    monkeypatch.setattr(doctor_tools, "_safe_which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(doctor_tools, "_run_ok", lambda cmd, timeout, **kw: False)
    doctor_tools._check_docker_backend("docker", False, issues)
    out = _joined(capsys)
    assert "Docker daemon not running" in out and "'docker' terminal backend" in out
    assert issues and "Start Docker" in issues[0] and "hermes setup terminal" in issues[0]


def test_ssh_host_missing_line_points_at_setup_terminal(monkeypatch, capsys, issues):
    monkeypatch.delenv("TERMINAL_SSH_HOST", raising=False)
    doctor_tools._check_ssh_backend(issues)
    out = _joined(capsys)
    assert "SSH host not configured" in out and "'ssh' terminal backend" in out
    assert "TERMINAL_SSH_HOST" not in out
    assert issues and "hermes setup terminal" in issues[0] and "SSH host and user" in issues[0]


def test_daytona_key_missing_line_points_at_setup_terminal(monkeypatch, capsys, issues):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    doctor_tools._check_daytona_backend(issues)
    out = _joined(capsys)
    assert "Daytona API key missing" in out and "'daytona' terminal backend" in out
    assert "DAYTONA_API_KEY" not in out
    assert issues and "hermes setup terminal" in issues[0]
