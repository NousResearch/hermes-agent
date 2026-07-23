"""Regression tests: every remaining cua-driver spawn site must sanitize the
subprocess environment.

PR #58889 fixed the CLI-fallback transport; review of that fix found four
sibling spawn sites still handing the third-party ``cua-driver`` binary the
full parent environment (provider API keys included):

- ``cua_backend._resolve_mcp_invocation`` (``cua-driver manifest``) — no
  ``env=`` at all
- ``cua_backend.cua_driver_update_check`` (``check-update --json``) —
  telemetry env but no secret sanitization
- ``doctor._drive_health_report`` (``<binary> mcp``) — telemetry env only
- ``permissions._run`` (every permission probe) — telemetry env only
"""

import json
from unittest.mock import MagicMock

SECRET = "sk-super-secret-should-not-leak"


def _fake_completed_process(stdout: str) -> MagicMock:
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = ""
    proc.returncode = 0
    return proc


def _capture_run(captured, stdout=""):
    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _fake_completed_process(stdout)
    return fake_run


def _assert_sanitized(captured):
    env = captured["env"]
    assert env is not None, "subprocess must receive an explicit env="
    assert "ANTHROPIC_API_KEY" not in env
    # Sanitization filters secrets, not everything — ordinary vars survive.
    assert env.get("PATH") == "/usr/bin:/bin"
    # Confirms the telemetry helper still ran (default: telemetry disabled).
    assert env.get("CUA_DRIVER_RS_TELEMETRY_ENABLED") == "0"


def test_resolve_mcp_invocation_sanitizes_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("HERMES_CUA_TELEMETRY", raising=False)

    from tools.computer_use import cua_backend

    captured = {}
    manifest = json.dumps({"mcp_invocation": {"command": "cua-driver", "args": ["mcp"]}})
    monkeypatch.setattr(
        cua_backend.subprocess, "run", _capture_run(captured, stdout=manifest)
    )

    cmd, args = cua_backend._resolve_mcp_invocation("cua-driver")
    assert cmd == "cua-driver"
    _assert_sanitized(captured)


def test_update_check_sanitizes_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("HERMES_CUA_TELEMETRY", raising=False)

    from tools.computer_use import cua_backend

    captured = {}
    payload = json.dumps({
        "current_version": "1.0.0",
        "latest_version": "1.0.0",
        "update_available": False,
    })
    # PATH is pinned to /usr/bin:/bin above, so the driver won't resolve;
    # pin it so the check reaches the (sanitized) subprocess spawn.
    monkeypatch.setattr(
        cua_backend, "resolve_cua_driver_cmd", lambda *a, **k: "cua-driver"
    )
    monkeypatch.setattr(
        cua_backend.subprocess, "run", _capture_run(captured, stdout=payload)
    )

    cua_backend.cua_driver_update_check(timeout=1.0)
    _assert_sanitized(captured)


def test_permissions_run_sanitizes_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("HERMES_CUA_TELEMETRY", raising=False)

    from tools.computer_use import permissions

    captured = {}
    monkeypatch.setattr(
        permissions.subprocess, "run", _capture_run(captured, stdout="{}")
    )

    permissions._run("cua-driver", "doctor", "--json", timeout=1.0)
    _assert_sanitized(captured)


def test_doctor_sanitized_env_helper(monkeypatch):
    """_drive_health_report spawns via Popen; assert the env helper it uses
    strips secrets (mocking the whole JSON-RPC handshake is not worth it)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("HERMES_CUA_TELEMETRY", raising=False)

    from tools.computer_use import doctor
    import inspect

    env = doctor._sanitized_cua_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert env.get("PATH") == "/usr/bin:/bin"
    assert env.get("CUA_DRIVER_RS_TELEMETRY_ENABLED") == "0"

    # The Popen spawn site must actually use the sanitized helper.
    src = inspect.getsource(doctor._drive_health_report)
    assert "_sanitized_cua_env()" in src
