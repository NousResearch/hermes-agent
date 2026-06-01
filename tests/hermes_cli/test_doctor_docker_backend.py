"""Tests for Docker backend diagnostics in hermes doctor."""

import contextlib
import io
import sys
import types
from argparse import Namespace

import hermes_cli.doctor as doctor_mod


def test_docker_backend_diagnostic_reports_daemon_failure(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("terminal:\n  backend: docker\n", encoding="utf-8")

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda cmd: "/usr/bin/docker" if cmd == "docker" else None)

    def fake_run(cmd, **kwargs):
        assert cmd == ["docker", "version"]
        return types.SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?",
        )

    monkeypatch.setattr(doctor_mod.subprocess, "run", fake_run)

    diagnostic = doctor_mod._docker_backend_diagnostic()

    assert diagnostic is not None
    assert diagnostic["summary"] == "Docker backend is configured but Docker daemon is not responding"
    assert "Cannot connect to the Docker daemon" in diagnostic["detail"]


def test_run_doctor_prints_docker_backend_daemon_remediation(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "config.yaml").write_text("terminal:\n  backend: docker\nmemory: {}\n", encoding="utf-8")

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda cmd: "/usr/bin/docker" if cmd == "docker" else None)

    def fake_run(cmd, **kwargs):
        if cmd == ["docker", "version"]:
            return types.SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?",
            )
        return types.SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(doctor_mod.subprocess, "run", fake_run)
    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        ),
    )

    try:
        from hermes_cli import auth as auth_mod

        monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(auth_mod, "get_gemini_oauth_auth_status", lambda: {})
        monkeypatch.setattr(auth_mod, "get_minimax_oauth_auth_status", lambda: {})
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    out = buf.getvalue()

    assert "Docker backend is configured but Docker daemon is not responding" in out
    assert "Cannot connect to the Docker daemon" in out
    assert "Start Docker Desktop" in out
    assert "sudo systemctl start docker" in out


def test_docker_backend_diagnostic_skips_probe_when_backend_is_local(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("terminal:\n  backend: local\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda cmd: calls.append(("which", cmd)) or "/usr/bin/docker")
    monkeypatch.setattr(doctor_mod.subprocess, "run", lambda *a, **kw: calls.append(("run", a, kw)))

    assert doctor_mod._docker_backend_diagnostic() is None
    assert calls == []
