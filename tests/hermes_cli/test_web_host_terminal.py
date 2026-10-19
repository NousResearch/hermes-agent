"""Webapp host terminal: request policy, the /api/host-terminal gate, and argv resolution."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def web_server(monkeypatch):
    import hermes_cli.web_server as ws

    monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    return ws


@pytest.mark.parametrize(
    ("surface", "bound_host", "headers"),
    [
        ("dashboard", "testclient", {"host": "testclient"}),
        (
            "webapp",
            "0.0.0.0",
            {
                "host": "192.168.0.222:9120",
                "origin": "http://192.168.0.222:9120",
            },
        ),
    ],
)
def test_host_terminal_rejection_runs_through_websocket_route(
    web_server, monkeypatch, surface, bound_host, headers
):
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr(web_server.app.state, "ui_surface", surface, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", bound_host, raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)

    with pytest.raises(WebSocketDisconnect) as exc:
        with TestClient(web_server.app).websocket_connect(
            f"/api/host-terminal?token={web_server._SESSION_TOKEN}", headers=headers
        ):
            pass

    assert exc.value.code == 4403


def test_host_terminal_policy_rejects_dashboard_and_unauthenticated_public_bind(
    web_server, monkeypatch
):
    from hermes_cli import web_host_terminal

    app = web_server.app
    monkeypatch.setattr(app.state, "ui_surface", "dashboard", raising=False)
    monkeypatch.setattr(app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(app.state, "auth_required", False, raising=False)
    assert web_host_terminal.request_allowed() is False

    app.state.ui_surface = "webapp"
    app.state.bound_host = "0.0.0.0"
    assert web_host_terminal.request_allowed() is False

    app.state.auth_required = True
    assert web_host_terminal.request_allowed() is True


def _assert_host_terminal_resolver_uses_real_host(tmp_path):
    from hermes_cli import web_host_terminal
    from hermes_constants import get_hermes_home

    argv, cwd, env, shell_name = web_host_terminal.resolve_argv(
        home=get_hermes_home(), requested_cwd=str(tmp_path)
    )

    assert argv
    assert argv[0]
    assert cwd == str(tmp_path.resolve())
    assert shell_name
    assert env["TERM"] == "xterm-256color"
    assert env["COLORTERM"] == "truecolor"
    assert env["HERMES_DESKTOP_TERMINAL"] == "1"


def test_host_terminal_resolver_applies_selected_profile_home_and_config(
    tmp_path, monkeypatch
):
    from hermes_cli import web_host_terminal

    real_home = tmp_path / "real-home"
    current = tmp_path / "profiles" / "current"
    target = tmp_path / "profiles" / "target"
    for path in (real_home, current / "home", target / "home"):
        path.mkdir(parents=True)
    (target / "config.yaml").write_text(
        "terminal:\n  home_mode: profile\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(current / "home"))
    monkeypatch.setenv("HERMES_HOME", str(current))
    monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))
    monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
    monkeypatch.setenv("OPENAI_API_KEY", "current-profile-secret")

    _argv, _cwd, env, _shell_name = web_host_terminal.resolve_argv(home=target)

    assert env["HERMES_HOME"] == str(target)
    assert env["HOME"] == str(target / "home")
    assert env["TERMINAL_HOME_MODE"] == "profile"
    assert "OPENAI_API_KEY" not in env


@pytest.mark.platforms("linux")
def test_host_terminal_resolver_uses_real_linux_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/sh")
    _assert_host_terminal_resolver_uses_real_host(tmp_path)


@pytest.mark.platforms("posix")
def test_host_terminal_relative_shell_never_reaches_argv(tmp_path, monkeypatch):
    """The PTY chdirs into the workspace before exec, so the shell it runs is the one checked only if argv[0] is absolute."""
    from hermes_cli import web_host_terminal
    from hermes_constants import get_hermes_home

    server_cwd = tmp_path / "server"
    workspace = tmp_path / "workspace"
    (server_cwd / "bin").mkdir(parents=True)
    workspace.mkdir()
    planted = server_cwd / "bin" / "sh"
    planted.write_text("#!/bin/sh\n", encoding="utf-8")
    planted.chmod(0o755)
    monkeypatch.chdir(server_cwd)
    monkeypatch.setenv("SHELL", "bin/sh")

    argv, cwd, _env, _shell_name = web_host_terminal.resolve_argv(
        home=get_hermes_home(), requested_cwd=str(workspace)
    )

    assert cwd == str(workspace.resolve())
    assert os.path.isabs(argv[0])
    assert Path(argv[0]).resolve() != planted.resolve()


@pytest.mark.platforms("windows")
def test_host_terminal_resolver_uses_real_windows_shell(tmp_path):
    _assert_host_terminal_resolver_uses_real_host(tmp_path)
    from hermes_cli import web_host_terminal
    argv, shell_name = web_host_terminal.shell_spec()
    assert shell_name.startswith(("pwsh", "powershell", "cmd"))
    assert argv[1:] == ([] if shell_name.startswith("cmd") else ["-NoLogo"])
