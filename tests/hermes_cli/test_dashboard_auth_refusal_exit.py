"""Permanent auth misconfiguration must not become a supervisor restart loop."""
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import dashboard_auth, web_server


@pytest.fixture
def auth_gate(monkeypatch):
    # Keep all startup state local; no providers, background workers or sockets.
    monkeypatch.setattr(web_server.app, "state", SimpleNamespace())
    monkeypatch.setattr(web_server, "_dashboard_public_hosts", lambda: set())
    monkeypatch.setattr(web_server, "_desktop_loopback_auth_exempt", lambda *a: False)
    monkeypatch.setattr(dashboard_auth, "list_providers", lambda: [])
    monkeypatch.setattr(web_server, "load_config", lambda: {})
    return web_server


def test_public_startup_refusal_has_permanent_exit_and_actionable_stderr(auth_gate, monkeypatch, capsys):
    from hermes_cli import nous_auth_keepalive, resource_limits
    monkeypatch.setattr(nous_auth_keepalive, "start_nous_auth_keepalive", lambda: None)
    monkeypatch.setattr(resource_limits, "apply_nofile_soft_limit", lambda: None)
    built = []
    monkeypatch.setattr(auth_gate, "_build_uvicorn_server", lambda *a, **kw: built.append(True))
    with pytest.raises(SystemExit) as error:
        auth_gate.start_server(host="0.0.0.0", open_browser=False, headless=True)
    assert error.value.code == 78
    assert not built
    stderr = capsys.readouterr().err
    assert "Refusing to bind dashboard to 0.0.0.0" in stderr
    assert "auth providers" in stderr
    assert "hermes" in stderr


def test_configured_provider_and_loopback_keep_existing_gate_behavior(auth_gate, monkeypatch):
    auth_gate._configure_auth_gate("127.0.0.1", False, None, None)
    assert auth_gate.app.state.auth_required is False
    monkeypatch.setattr(dashboard_auth, "list_providers", lambda: [SimpleNamespace(name="fixture")])
    auth_gate._configure_auth_gate("0.0.0.0", False, None, None)
    assert auth_gate.app.state.auth_required is True


@pytest.mark.parametrize("enabled,code,signal,expected", [
    ("1", "78", "0", 125),
    ("true", "1", "0", 0),
    ("yes", "256", "15", 0),
    ("1", "0", "0", 0),
    ("", "1", "0", 125),
])
def test_finish_parks_config_failure_but_preserves_crash_restarts(enabled, code, signal, expected):
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("requires a POSIX shell to exercise the shipped finish script")
    script = Path(__file__).resolve().parents[2] / "docker/s6-rc.d/dashboard/finish"
    result = subprocess.run([shell, str(script), code, signal],
                            env={**os.environ, "HERMES_DASHBOARD": enabled},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == expected, result.stderr
