"""Invariants for the terminal-backend doctor lines (tools-runtime-30): each failure names the backend in
plain words and points at `hermes setup terminal`, never at raw TERMINAL_* env vars."""

import pytest

from hermes_cli import doctor_tools

@pytest.fixture
def issues():
    return []

def _joined(capsys) -> str:
    return capsys.readouterr().out

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
