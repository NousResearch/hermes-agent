"""Dashboard gateway lifecycle actions must elevate on a system-scope install.

Regression for #110820: the Restart Gateway button spawned ``hermes gateway restart`` as the
dashboard's own user, and the CLI refuses system-scope verbs below root, so the button could
never work on a systemd *system* install — the refusal only reached the action log while the
endpoint reported a started action.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def unprivileged(monkeypatch):
    """Run the predicate as a non-root user (CI containers can be root)."""
    monkeypatch.setattr("os.geteuid", lambda: 1000)


@pytest.fixture
def system_scope_install(monkeypatch, tmp_path):
    """Only the SYSTEM unit is installed, the shape that needs root."""
    system_unit = tmp_path / "etc" / "hermes-gateway.service"
    system_unit.parent.mkdir(parents=True)
    system_unit.write_text("[Service]\n")
    user_unit = tmp_path / "user" / "hermes-gateway.service"
    monkeypatch.setattr(
        "hermes_cli.gateway.get_systemd_unit_path",
        lambda system=False: system_unit if system else user_unit,
    )
    return system_unit, user_unit


def _spawn_restart(tmp_path, *, sudo_ok: bool):
    """Run ``_spawn_hermes_action(["gateway", "restart"])`` with the world stubbed out.

    Returns ``(argv, log_text)`` for the attempted spawn; Popen is the only thing faked, so
    the elevation decision runs through the production helpers.
    """
    from hermes_cli import web_server_gateway

    logs = tmp_path / "logs"
    probe = subprocess.CompletedProcess(["sudo", "-n", "true"], 0 if sudo_ok else 1)
    child = MagicMock(spec=subprocess.Popen, pid=4242)  # spec'd before Popen is patched
    with patch.object(web_server_gateway, "_ACTION_LOG_DIR", logs), patch.object(
        web_server_gateway.subprocess, "run", return_value=probe
    ), patch.object(web_server_gateway.subprocess, "Popen") as popen, patch.object(
        web_server_gateway, "_dashboard_spawn_executable", return_value="/venv/bin/python"
    ), patch.object(web_server_gateway, "PROJECT_ROOT", tmp_path, create=True), patch(
        "hermes_cli.web_server.PROJECT_ROOT", tmp_path
    ):
        popen.return_value = child
        try:
            web_server_gateway._spawn_hermes_action(["gateway", "restart"], "gateway-restart")
        finally:
            log_file = logs / "gateway-restart.log"
            log_text = log_file.read_text() if log_file.exists() else ""
    argv = popen.call_args.args[0] if popen.call_args else None
    return argv, log_text


class TestSystemScopeGatewayActionsElevate:
    def test_restart_on_a_system_install_is_spawned_under_sudo(
        self, unprivileged, system_scope_install, tmp_path
    ):
        argv, _log_text = _spawn_restart(tmp_path, sudo_ok=True)
        assert argv[:2] == ["sudo", "-n"], (
            "a system-scope restart spawned as the dashboard user can only be refused by the CLI"
        )
        assert argv[2:] == ["/venv/bin/python", "-m", "hermes_cli.main", "gateway", "restart"]

    def test_restart_without_passwordless_sudo_fails_the_request(
        self, unprivileged, system_scope_install, tmp_path
    ):
        with pytest.raises(RuntimeError, match="passwordless sudo is unavailable"):
            _spawn_restart(tmp_path, sudo_ok=False)

    def test_a_user_scope_install_is_never_elevated(self, unprivileged, system_scope_install, tmp_path):
        _system_unit, user_unit = system_scope_install
        user_unit.parent.mkdir(parents=True)
        user_unit.write_text("[Service]\n")  # both units installed -> the CLI picks user scope
        argv, _log_text = _spawn_restart(tmp_path, sudo_ok=True)
        assert argv[0] == "/venv/bin/python", "user-scope verbs need no privilege"


class TestElevationPredicateScope:
    """Only the lifecycle verbs the CLI gates on root elevate."""

    @pytest.mark.parametrize(
        "subcommand, elevated",
        [
            (["gateway", "restart"], True),
            (["-p", "default", "gateway", "start"], True),
            (["gateway", "stop"], True),
            (["gateway", "status"], False),
            (["gateway", "migrate", "--multiplex", "--yes"], False),
            (["doctor"], False),
            (["gateway"], False),
        ],
    )
    def test_verbs(self, unprivileged, system_scope_install, monkeypatch, subcommand, elevated, tmp_path):
        from hermes_cli import web_server_gateway

        monkeypatch.setattr(
            "hermes_cli.web_server_profiles._resolve_profile_dir", lambda name: Path(tmp_path) / name
        )
        assert web_server_gateway._action_targets_system_gateway(subcommand) is elevated

    def test_root_does_not_shell_out_to_sudo(self, system_scope_install, monkeypatch):
        from hermes_cli import web_server_gateway

        monkeypatch.setattr("os.geteuid", lambda: 0)
        assert web_server_gateway._action_targets_system_gateway(["gateway", "restart"]) is False
