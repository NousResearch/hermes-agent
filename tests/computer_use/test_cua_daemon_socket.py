"""Tests for an explicit Cua daemon endpoint."""

from types import SimpleNamespace
from unittest.mock import patch


def _manifest(*_args, **_kwargs):
    return {
        "mcp_invocation": {
            "command": "cua-driver",
            "args": ["mcp"],
        }
    }


def test_configured_daemon_socket_is_appended(monkeypatch):
    from tools.computer_use import cua_backend, cua_backend_driver

    endpoint = r"\\.\pipe\cua-driver"
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {"daemon_socket": endpoint})
    monkeypatch.setattr(cua_backend_driver, "_driver_json", _manifest)
    monkeypatch.setattr(cua_backend_driver, "_cua_driver_supports_no_overlay", lambda _cmd: False)

    command, args = cua_backend_driver._resolve_mcp_invocation("cua-driver")

    assert command == "cua-driver"
    assert args == ["mcp", "--socket", endpoint]


def test_configured_daemon_socket_is_used_by_cli_fallback(monkeypatch):
    from tools.computer_use import cua_backend, cua_backend_driver, cua_backend_session

    endpoint = r"\\.\pipe\cua-driver"
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {"daemon_socket": endpoint})
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda: "/resolved/cua-driver")

    captured = {}

    def fake_cli_run_json(cmd, env, name, timeout):
        captured["cmd"] = cmd
        return {"tree_markdown": "root"}

    monkeypatch.setattr(cua_backend_session, "_cli_run_json", fake_cli_run_json)
    session = object.__new__(cua_backend_session._CuaDriverSession)

    result = session._call_tool_via_cli("list_windows", {}, timeout=5.0)

    assert result["isError"] is False
    assert captured["cmd"] == [
        "/resolved/cua-driver",
        "call",
        "list_windows",
        "{}",
        "--socket",
        endpoint,
    ]


def test_manifest_socket_is_not_overridden(monkeypatch):
    from tools.computer_use import cua_backend, cua_backend_driver

    configured = r"\\.\pipe\configured"
    advertised = r"\\.\pipe\advertised"
    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {"daemon_socket": configured})
    monkeypatch.setattr(
        cua_backend_driver,
        "_driver_json",
        lambda *_args, **_kwargs: {
            "mcp_invocation": {
                "command": "cua-driver",
                "args": ["mcp", "--socket", advertised],
            }
        },
    )
    monkeypatch.setattr(cua_backend_driver, "_cua_driver_supports_no_overlay", lambda _cmd: False)

    _command, args = cua_backend_driver._resolve_mcp_invocation("cua-driver")

    assert args == ["mcp", "--socket", advertised]


def test_invalid_daemon_socket_fails_to_disabled(monkeypatch):
    from tools.computer_use import cua_backend

    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {"daemon_socket": "bad\x00pipe"})
    assert cua_backend._cua_daemon_socket() is None

    monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {"daemon_socket": ["not", "a", "string"]})
    assert cua_backend._cua_daemon_socket() is None


def test_doctor_uses_resolved_mcp_invocation():
    from tools.computer_use import cua_backend_driver, doctor

    proc = SimpleNamespace()
    with patch.object(
        cua_backend_driver,
        "_resolve_mcp_invocation",
        return_value=("resolved-cua", ["mcp", "--socket", "test-pipe"]),
    ), patch.object(doctor.subprocess, "Popen", return_value=proc) as popen:
        assert doctor._open_mcp("input-cua") is proc

    argv = popen.call_args.args[0]
    assert argv == ["resolved-cua", "mcp", "--socket", "test-pipe"]
