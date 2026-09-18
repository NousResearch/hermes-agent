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
    monkeypatch.setattr(doctor_tools, "find_docker", lambda: None)
    doctor_tools._check_docker_backend("docker", False, issues)
    out = _joined(capsys)
    assert "Docker or Podman not installed" in out and "'docker' terminal backend" in out
    assert "TERMINAL_ENV" not in out
    assert issues and "hermes setup terminal" in issues[0] and "Install Docker or Podman" in issues[0]


def test_docker_daemon_down_line_says_start_docker_or_switch(monkeypatch, capsys, issues):
    monkeypatch.setattr(doctor_tools, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(doctor_tools, "_run_ok", lambda cmd, timeout, **kw: False)
    doctor_tools._check_docker_backend("docker", False, issues)
    out = _joined(capsys)
    assert "Docker daemon not running" in out and "'docker' terminal backend" in out
    assert issues and "Start Docker" in issues[0] and "hermes setup terminal" in issues[0]


def test_docker_backend_ready_when_only_podman_resolves(monkeypatch, capsys, issues):
    """A podman-only machine runs the 'docker' backend, so doctor reports Podman as ready."""
    monkeypatch.setattr(doctor_tools, "find_docker", lambda: "/usr/bin/podman")
    probed: list[list[str]] = []
    monkeypatch.setattr(doctor_tools, "_run_ok", lambda cmd, timeout, **kw: probed.append(cmd) or True)
    doctor_tools._check_docker_backend("docker", False, issues)
    out = _joined(capsys)
    assert "Podman (reachable)" in out
    assert not issues
    assert probed == [["/usr/bin/podman", "version"]]


def test_podman_down_line_advises_the_machine_not_a_daemon(monkeypatch, capsys, issues):
    """Podman is daemonless: the fix is `podman machine start`, never "start a daemon"."""
    monkeypatch.setattr(doctor_tools, "find_docker", lambda: "/opt/homebrew/bin/podman")
    monkeypatch.setattr(doctor_tools, "_run_ok", lambda cmd, timeout, **kw: False)
    doctor_tools._check_docker_backend("docker", False, issues)

    out = _joined(capsys)
    assert "Podman not reachable" in out
    assert issues and "podman machine start" in issues[0] and "hermes setup terminal" in issues[0]


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
