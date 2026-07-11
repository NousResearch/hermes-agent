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
CREATE_NO_WINDOW = 0x08000000


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
        captured["creationflags"] = kwargs.get("creationflags")
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


def _patch_windows_hide_flags(monkeypatch, module):
    monkeypatch.setattr(module, "IS_WINDOWS", True, raising=False)
    monkeypatch.setattr(
        module, "windows_hide_flags", lambda: CREATE_NO_WINDOW, raising=False
    )


def test_resolve_mcp_invocation_sanitizes_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("HERMES_CUA_TELEMETRY", raising=False)

    from tools.computer_use import cua_backend

    captured = {}
    _patch_windows_hide_flags(monkeypatch, cua_backend)
    manifest = json.dumps({"mcp_invocation": {"command": "cua-driver", "args": ["mcp"]}})
    monkeypatch.setattr(
        cua_backend.subprocess, "run", _capture_run(captured, stdout=manifest)
    )

    cmd, args = cua_backend._resolve_mcp_invocation("cua-driver")
    assert cmd == "cua-driver"
    _assert_sanitized(captured)
    assert captured["creationflags"] == CREATE_NO_WINDOW


def test_update_check_sanitizes_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("HERMES_CUA_TELEMETRY", raising=False)

    from tools.computer_use import cua_backend

    captured = {}
    _patch_windows_hide_flags(monkeypatch, cua_backend)
    payload = json.dumps({
        "current_version": "1.0.0",
        "latest_version": "1.0.0",
        "update_available": False,
    })
    monkeypatch.setattr(
        cua_backend.subprocess, "run", _capture_run(captured, stdout=payload)
    )

    cua_backend.cua_driver_update_check(timeout=1.0)
    _assert_sanitized(captured)
    assert captured["creationflags"] == CREATE_NO_WINDOW


def test_cli_fallback_sanitizes_env_and_hides_console_on_windows(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("HERMES_CUA_TELEMETRY", raising=False)

    from tools.computer_use import cua_backend

    captured = {}
    _patch_windows_hide_flags(monkeypatch, cua_backend)
    monkeypatch.setattr(
        cua_backend.subprocess,
        "run",
        _capture_run(captured, stdout=json.dumps({"tree_markdown": "root"})),
    )

    session = object.__new__(cua_backend._CuaDriverSession)
    result = session._call_tool_via_cli("list_windows", {}, timeout=5.0)

    assert result["isError"] is False
    _assert_sanitized(captured)
    assert captured["creationflags"] == CREATE_NO_WINDOW


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


def test_windows_status_hides_every_reachable_subprocess(monkeypatch):
    """The Desktop status API reaches only version + doctor spawns on Windows.

    The permissions grant subprocess is intentionally excluded: its public
    entry point returns before spawning anywhere except macOS, where
    ``CREATE_NO_WINDOW`` is not applicable.
    """
    from tools.computer_use import permissions

    binary = r"C:\Program Files\cua-driver\cua-driver.exe"
    calls = []
    stdout_by_args = {
        ("--version",): "cua-driver 1.2.3\n",
        ("doctor", "--json"): json.dumps({"ok": True, "probes": []}),
    }

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _fake_completed_process(stdout_by_args[tuple(cmd[1:])])

    monkeypatch.setattr(permissions.sys, "platform", "win32")
    monkeypatch.setattr(permissions.shutil, "which", lambda command: binary)
    monkeypatch.setattr(permissions.subprocess, "run", fake_run)

    status = permissions.computer_use_status("cua-driver")

    assert status["version"] == "cua-driver 1.2.3"
    assert status["ready"] is True
    assert [cmd[1:] for cmd, _ in calls] == [
        ["--version"],
        ["doctor", "--json"],
    ]
    assert calls, "Windows status must exercise at least one subprocess boundary"
    for cmd, kwargs in calls:
        assert kwargs.get("creationflags") == CREATE_NO_WINDOW, cmd


def test_doctor_spawn_sanitizes_env_and_hides_console_on_windows(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("HERMES_CUA_TELEMETRY", raising=False)

    from tools.computer_use import doctor

    captured = {}
    _patch_windows_hide_flags(monkeypatch, doctor)
    proc = MagicMock()
    proc.stdout.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}),
        json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "structuredContent": {"schema_version": "1", "overall": "ok"}
            },
        }),
    ]

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        captured["creationflags"] = kwargs.get("creationflags")
        return proc

    monkeypatch.setattr(doctor.subprocess, "Popen", fake_popen)

    report = doctor._drive_health_report("cua-driver", timeout=1.0)

    assert report["overall"] == "ok"
    _assert_sanitized(captured)
    assert captured["creationflags"] == CREATE_NO_WINDOW
