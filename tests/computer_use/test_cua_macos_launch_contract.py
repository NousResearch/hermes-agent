"""LaunchServices owns the macOS app lifetime; only its MCP proxy is embedded."""

import json

import pytest

from tools.computer_use import cua_backend, cua_backend_daemon


@pytest.mark.parametrize("permission_mode", ["unrestricted", "bounded"])
def test_macos_app_launch_preserves_private_authority_without_parent_lifetime(permission_mode, monkeypatch):
    app_path = "/Applications/CuaDriver.app"
    validated = []
    monkeypatch.setattr(cua_backend_daemon, "_validate_cua_driver_app_signature", validated.append)
    serve_args = [
        "serve", "--embedded", "--socket", "/tmp/private-cua.sock",
        "--no-permissions-gate", "--permission-mode", permission_mode,
        "--capability-manifest", "/tmp/reviewed-policy.json", "--approve-capability-manifest", "--no-overlay",
    ]
    if permission_mode == "unrestricted":
        serve_args.append("--dangerously-bypass-approvals")
    original_args = list(serve_args)

    command = cua_backend_daemon._embedded_daemon_spawn_command(
        app_path + "/Contents/MacOS/cua-driver", serve_args,
        platform="darwin", app_path=app_path,
    )

    assert validated == [app_path]
    assert command[:6] == ["/usr/bin/open", "-n", "-g", "-a", app_path, "--args"]
    assert command[6:] == [arg for arg in original_args if arg != "--embedded"]
    assert serve_args == original_args


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("permission_mode", ["unrestricted", "bounded"])
def test_macos_daemon_env_drops_parent_lifetime_but_proxy_stays_guarded(permission_mode, tmp_path, monkeypatch):
    manifest = tmp_path / "reviewed-policy.json"
    manifest.write_text(json.dumps({"version": 3}), encoding="utf-8")
    inherited = {"CUA_DRIVER_EMBEDDED": "1", "PATH": "/usr/bin"}
    monkeypatch.setattr(cua_backend, "cua_driver_child_env", lambda: dict(inherited))
    monkeypatch.setattr(
        cua_backend_daemon._driver, "_mcp_args_with_overlay_flag",
        lambda args, driver_cmd: list(args),
    )
    daemon = cua_backend_daemon._EmbeddedCuaDaemon(
        "/Applications/CuaDriver.app/Contents/MacOS/cua-driver", permission_mode, str(manifest),
    )

    env = daemon.child_env()
    assert "CUA_DRIVER_EMBEDDED" not in env
    assert env["CUA_DRIVER_PERMISSION_MODE"] == permission_mode
    assert env.get("CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS") == (
        "1" if permission_mode == "unrestricted" else None
    )
    assert env["PATH"] == inherited["PATH"]
    assert inherited["CUA_DRIVER_EMBEDDED"] == "1"
    serve_args = daemon._serve_args()
    assert serve_args[serve_args.index("--socket") + 1] == daemon.socket_path
    assert serve_args[serve_args.index("--capability-manifest") + 1] == str(manifest)
    assert "--approve-capability-manifest" in serve_args
    with pytest.raises(RuntimeError, match="daemon is not running"):
        daemon.proxy_invocation()
    daemon._running = True
    command, args = daemon.proxy_invocation()
    assert command.endswith("/CuaDriver.app/Contents/MacOS/cua-driver")
    assert args == ["mcp", "--embedded", "--socket", daemon.socket_path]
