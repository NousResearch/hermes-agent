"""Regression coverage for OpenRC gateway-service support."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hermes_cli.gateway as gateway


def test_openrc_install_generates_shebang_first_and_target_identity(
    monkeypatch, tmp_path
) -> None:
    init_path = tmp_path / "init.d" / "hermes-gateway-coder"
    confd_path = tmp_path / "conf.d" / "hermes-gateway-coder"
    init_path.parent.mkdir()
    confd_path.parent.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setattr(gateway.os, "geteuid", lambda: 0)
    monkeypatch.setattr(gateway, "get_openrc_init_path", lambda: init_path)
    monkeypatch.setattr(gateway, "get_openrc_confd_path", lambda: confd_path)
    monkeypatch.setattr(gateway, "get_service_name", lambda: "hermes-gateway-coder")
    monkeypatch.setattr(
        gateway,
        "_system_service_identity",
        lambda run_as_user=None: ("alice", "staff", "/home/alice"),
    )
    monkeypatch.setattr(
        gateway,
        "_hermes_home_for_target_user",
        lambda home_dir: "/home/alice/.hermes/profiles/coder",
    )
    monkeypatch.setattr(
        gateway,
        "_profile_arg_for_target_user",
        lambda hermes_home, home_dir: "--profile coder",
    )
    monkeypatch.setattr(gateway, "get_python_path", lambda: "/root/.hermes/venv/bin/python")
    monkeypatch.setattr(gateway, "_detect_venv_dir", lambda: Path("/root/.hermes/venv"))
    monkeypatch.setattr(
        gateway,
        "_remap_path_for_user",
        lambda path, home_dir: str(path).replace("/root", home_dir),
    )
    monkeypatch.setattr(
        gateway.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )

    gateway.openrc_install(run_as_user="alice", start_now=False)

    script = init_path.read_text(encoding="utf-8")
    assert script.splitlines()[0] == "#!/sbin/openrc-run"
    assert 'command="/home/alice/.hermes/venv/bin/python"' in script
    assert 'command_args="-m hermes_cli.main --profile coder gateway run"' in script
    assert 'command_user="alice:staff"' in script
    assert 'command_chdir="${HERMES_HOME}"' in script
    assert 'HERMES_HOME="/home/alice/.hermes/profiles/coder"' in confd_path.read_text(
        encoding="utf-8"
    )
    assert calls == [["rc-update", "add", "hermes-gateway-coder", "default"]]


def test_openrc_lifecycle_uses_current_profile_service(monkeypatch, tmp_path, capsys) -> None:
    init_path = tmp_path / "hermes-gateway-coder"
    init_path.touch()
    calls: list[list[str]] = []

    monkeypatch.setattr(gateway.os, "geteuid", lambda: 0)
    monkeypatch.setattr(gateway, "supports_openrc_services", lambda: True)
    monkeypatch.setattr(gateway, "get_openrc_init_path", lambda: init_path)
    monkeypatch.setattr(gateway, "get_service_name", lambda: "hermes-gateway-coder")
    monkeypatch.setattr(
        gateway.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )

    gateway.openrc_start()
    gateway.openrc_stop()
    gateway.openrc_restart()
    gateway.openrc_status()
    gateway.openrc_uninstall()

    assert calls == [
        ["rc-service", "hermes-gateway-coder", "start"],
        ["rc-service", "hermes-gateway-coder", "stop"],
        ["rc-service", "hermes-gateway-coder", "restart"],
        ["rc-service", "hermes-gateway-coder", "status"],
        ["rc-service", "hermes-gateway-coder", "stop"],
        ["rc-update", "del", "hermes-gateway-coder", "default"],
    ]
    assert not init_path.exists()
    assert "OpenRC gateway service is running" in capsys.readouterr().out


def test_gateway_command_dispatches_openrc_start(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(gateway, "_dispatch_via_service_manager_if_s6", lambda *args: False)
    monkeypatch.setattr(gateway, "is_termux", lambda: False)
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "is_windows", lambda: False)
    monkeypatch.setattr(gateway, "is_wsl", lambda: False)
    monkeypatch.setattr(gateway, "is_container", lambda: False)
    monkeypatch.setattr(gateway, "supports_openrc_services", lambda: True)
    monkeypatch.setattr(gateway, "openrc_start", lambda: calls.append("start"))

    gateway.gateway_command(SimpleNamespace(gateway_command="start", system=True, all=False))

    assert calls == ["start"]
