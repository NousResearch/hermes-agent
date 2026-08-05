"""Ownership-aware updater finalization regressions.

The gateway that launched ``hermes update --gateway`` may also own the
updater process through its systemd cgroup.  Restarting that unit before the
dashboard and result marker are finalized kills the updater mid-flight.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import update_cmd, update_owner_restart


def _owner() -> tuple[str, list[str], str]:
    return (
        "user",
        ["systemctl", "--user"],
        "hermes-gateway-coding_lead",
    )


def _system_owner() -> tuple[str, list[str], str]:
    return (
        "system",
        ["systemctl"],
        "hermes-gateway-coding_lead",
    )


def test_dashboard_precedes_terminal_owner_verifier_without_early_success(
    monkeypatch, capsys
):
    events: list[str] = []

    monkeypatch.setattr(
        update_cmd,
        "_finish_dashboard_update_cleanup",
        lambda _failures: events.append("dashboard") or True,
    )
    monkeypatch.setattr(
        update_cmd,
        "_launch_updater_owner_restart_verifier",
        lambda _owner, *, final_exit_code, persist_result: events.append(
            f"verifier:{final_exit_code}:{persist_result}"
        )
        or True,
        raising=False,
    )

    ok = update_cmd._finish_update_service_finalization(
        [],
        gateway_mode=True,
        gateway_fleet_restart_incomplete=False,
        updater_owner_discovery_failed=False,
        deferred_owner=_owner(),
    )

    assert ok is False
    assert events == ["dashboard", "verifier:0:True"]
    output = capsys.readouterr().out
    assert "verification is pending" in output
    assert "✓ Update complete!" not in output


def test_partial_failure_is_staged_before_owner_verifier(monkeypatch, capsys):
    events: list[str] = []

    monkeypatch.setattr(
        update_cmd,
        "_finish_dashboard_update_cleanup",
        lambda _failures: events.append("dashboard") or False,
    )
    monkeypatch.setattr(
        update_cmd,
        "_launch_updater_owner_restart_verifier",
        lambda _owner, *, final_exit_code, persist_result: events.append(
            f"verifier:{final_exit_code}:{persist_result}"
        )
        or True,
        raising=False,
    )

    ok = update_cmd._finish_update_service_finalization(
        [],
        gateway_mode=True,
        gateway_fleet_restart_incomplete=False,
        updater_owner_discovery_failed=False,
        deferred_owner=_owner(),
    )

    assert ok is False
    assert events == ["dashboard", "verifier:1:True"]
    assert "verification is pending" in capsys.readouterr().out


def test_owner_verifier_launch_failure_persists_error_result(monkeypatch, capsys):
    events: list[str] = []

    monkeypatch.setattr(
        update_cmd, "_finish_dashboard_update_cleanup", lambda _failures: True
    )
    monkeypatch.setattr(
        update_cmd,
        "_write_gateway_update_exit_code",
        lambda _required, code: events.append(f"status:{code}") or True,
    )
    monkeypatch.setattr(
        update_cmd,
        "_launch_updater_owner_restart_verifier",
        lambda _owner, *, final_exit_code, persist_result: events.append(
            f"verifier:{final_exit_code}:{persist_result}"
        )
        or False,
        raising=False,
    )

    ok = update_cmd._finish_update_service_finalization(
        [],
        gateway_mode=True,
        gateway_fleet_restart_incomplete=False,
        updater_owner_discovery_failed=False,
        deferred_owner=_owner(),
    )

    assert ok is False
    assert events == ["verifier:0:True", "status:1"]
    assert "Could not launch" in capsys.readouterr().out


def test_non_systemd_fallback_persists_final_status_without_verifier(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        update_cmd,
        "_finish_dashboard_update_cleanup",
        lambda _failures: events.append("dashboard") or True,
    )
    monkeypatch.setattr(
        update_cmd,
        "_write_gateway_update_exit_code",
        lambda required, code: events.append(f"status:{required}:{code}") or True,
    )
    monkeypatch.setattr(
        update_cmd,
        "_launch_updater_owner_restart_verifier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-systemd fallback must not launch verifier")
        ),
        raising=False,
    )

    assert update_cmd._finish_update_service_finalization(
        [],
        gateway_mode=True,
        gateway_fleet_restart_incomplete=False,
        updater_owner_discovery_failed=False,
        deferred_owner=None,
    ) is True

    assert events == ["dashboard", "status:True:0"]


def test_owner_verifier_timeout_covers_after_turn_wait(monkeypatch):
    from hermes_cli import gateway as gateway_cli

    monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 1.0)
    monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 700.0)
    monkeypatch.setattr(
        update_cmd,
        "_service_restart_sec",
        lambda _scope_cmd, _service, *, default: 5.0,
    )

    assert update_cmd._owner_restart_verification_timeout(_owner()) == 825.0


def test_system_owner_timeout_probe_fails_without_prompt_free_privilege(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(update_cmd.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        update_cmd,
        "_service_restart_sec",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unresolved systemctl command must not run")
        ),
    )

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="password required")

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)

    with pytest.raises(PermissionError, match="prompt-free systemctl"):
        update_cmd._owner_restart_verification_timeout(_system_owner())

    assert commands == [
        ["sudo", "-n", "true"],
        [
            "sudo",
            "-n",
            "systemctl",
            "--no-ask-password",
            "show",
            "hermes-gateway-coding_lead",
            "--property=RestartUSec",
            "--value",
        ],
    ]


def test_owner_request_is_durable_before_transient_verifier_launch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(update_cmd.shutil, "which", lambda name: "/bin/systemd-run")
    monkeypatch.setattr(
        update_cmd,
        "_owner_restart_verification_timeout",
        lambda _owner: 120.0,
        raising=False,
    )
    monkeypatch.setattr(
        update_owner_restart,
        "read_systemd_service_state",
        lambda _scope, _service: {
            "active_state": "active",
            "sub_state": "running",
            "main_pid": 4321,
            "exec_start": 101,
            "active_enter": 102,
            "restart": "on-failure",
        },
    )
    monkeypatch.setattr(update_cmd.secrets, "token_hex", lambda _size: "a" * 32)
    launches: list[list[str]] = []

    def fake_run(args, *unused_args, **unused_kwargs):
        pending = json.loads(
            (tmp_path / update_owner_restart.OWNER_RESTART_PENDING_FILE).read_text()
        )
        assert pending["version"] == 2
        assert pending["old_state"]["main_pid"] == 4321
        assert pending["generation_key"] == "exec_start"
        assert not (tmp_path / ".update_exit_code").exists()
        launches.append(args)
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(update_cmd.subprocess, "run", side_effect=fake_run), patch.object(
        update_cmd.os,
        "kill",
        side_effect=AssertionError("updater cgroup must not own the terminal signal"),
    ):
        assert update_cmd._launch_updater_owner_restart_verifier(
            _owner(), final_exit_code=0, persist_result=True
        ) is True

    command = launches[0]
    assert command[0:3] == [
        "/bin/systemd-run",
        "--user",
        "--no-ask-password",
    ]
    assert "--collect" in command
    assert any(part.startswith("--unit=hermes-update-owner-verify-") for part in command)
    assert command[-6:] == [
        sys.executable,
        "-m",
        "hermes_cli.update_owner_restart",
        "--home",
        str(tmp_path),
        f"--nonce={'a' * 32}",
    ]


def test_root_system_owner_verifier_launch_never_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(update_cmd.shutil, "which", lambda name: "/bin/systemd-run")
    monkeypatch.setattr(update_cmd.os, "geteuid", lambda: 0)
    monkeypatch.setattr(update_cmd.os, "getegid", lambda: 0)
    monkeypatch.setattr(
        update_cmd,
        "_owner_restart_verification_timeout",
        lambda _owner: 120.0,
    )
    monkeypatch.setattr(
        update_owner_restart,
        "read_systemd_service_state",
        lambda _scope, _service: {
            "active_state": "active",
            "sub_state": "running",
            "main_pid": 4321,
            "exec_start": 101,
            "active_enter": 102,
            "restart": "on-failure",
        },
    )
    monkeypatch.setattr(update_cmd.secrets, "token_hex", lambda _size: "b" * 32)
    launches: list[list[str]] = []

    def fake_run(args, *unused_args, **unused_kwargs):
        launches.append(args)
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(update_cmd.subprocess, "run", side_effect=fake_run):
        assert update_cmd._launch_updater_owner_restart_verifier(
            _system_owner(), final_exit_code=0, persist_result=True
        ) is True

    assert len(launches) == 1
    assert launches[0][0:2] == ["/bin/systemd-run", "--no-ask-password"]
    assert "--user" not in launches[0]
    assert "--property=User=0" in launches[0]
    assert "--property=Group=0" in launches[0]


def test_system_owner_verifier_refuses_prompting_privilege_path(tmp_path, monkeypatch):
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(update_cmd.shutil, "which", lambda name: "/bin/systemd-run")
    monkeypatch.setattr(update_cmd.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        update_owner_restart,
        "read_systemd_service_state",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("privilege must resolve before reading owner state")
        ),
    )
    commands: list[list[str]] = []

    def fake_run(args, *unused_args, **unused_kwargs):
        commands.append(args)
        if args in (
            ["sudo", "-n", "true"],
            [
                "sudo",
                "-n",
                "/bin/systemd-run",
                "--no-ask-password",
                "--version",
            ],
        ):
            return MagicMock(returncode=1, stdout="", stderr="password required")
        raise AssertionError(f"unexpected prompting or launch command: {args}")

    with patch.object(update_cmd.subprocess, "run", side_effect=fake_run), patch.object(
        update_cmd.os,
        "kill",
        side_effect=AssertionError("owner must remain untouched"),
    ):
        assert update_cmd._launch_updater_owner_restart_verifier(
            _system_owner(), final_exit_code=0, persist_result=True
        ) is False

    assert commands == [
        ["sudo", "-n", "true"],
        [
            "sudo",
            "-n",
            "/bin/systemd-run",
            "--no-ask-password",
            "--version",
        ],
    ]
    assert not (
        tmp_path / update_owner_restart.OWNER_RESTART_PENDING_FILE
    ).exists()


def test_system_owner_verifier_uses_sudo_n_when_capability_exists(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(update_cmd.shutil, "which", lambda name: "/bin/systemd-run")
    monkeypatch.setattr(update_cmd.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(update_cmd.os, "getegid", lambda: 1000)
    monkeypatch.setattr(
        update_cmd,
        "_owner_restart_verification_timeout",
        lambda _owner: 120.0,
    )
    monkeypatch.setattr(
        update_owner_restart,
        "read_systemd_service_state",
        lambda _scope, _service: {
            "active_state": "active",
            "sub_state": "running",
            "main_pid": 4321,
            "exec_start": 101,
            "active_enter": 102,
            "restart": "on-failure",
        },
    )
    monkeypatch.setattr(update_cmd.secrets, "token_hex", lambda _size: "d" * 32)
    commands: list[list[str]] = []

    def fake_run(args, *unused_args, **unused_kwargs):
        commands.append(args)
        if args == ["sudo", "-n", "true"]:
            return MagicMock(returncode=0, stdout="", stderr="")
        assert args[0:4] == [
            "sudo",
            "-n",
            "/bin/systemd-run",
            "--no-ask-password",
        ]
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(update_cmd.subprocess, "run", side_effect=fake_run):
        assert update_cmd._launch_updater_owner_restart_verifier(
            _system_owner(), final_exit_code=0, persist_result=True
        ) is True

    assert len(commands) == 2
    launch = commands[1]
    assert "--user" not in launch
    assert "--property=User=1000" in launch
    assert "--property=Group=1000" in launch


def test_system_owner_targeted_sudo_probe_can_resolve_prompt_free_command(
    monkeypatch,
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            args=command,
            returncode=0 if command[-1] == "--version" else 1,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(update_cmd.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    command = ["/bin/systemd-run", "--no-ask-password"]

    assert update_cmd._resolve_noninteractive_systemd_command(
        "system",
        command,
        targeted_probe=[*command, "--version"],
    ) == ["sudo", "-n", *command]
    assert [call[0] for call in calls] == [
        ["sudo", "-n", "true"],
        ["sudo", "-n", *command, "--version"],
    ]
    assert all(call[1]["timeout"] == 5 for call in calls)


def test_owner_is_not_restarted_when_request_cannot_be_written(monkeypatch):
    monkeypatch.setattr(
        update_owner_restart,
        "read_systemd_service_state",
        lambda _scope, _service: {
            "active_state": "active",
            "sub_state": "running",
            "main_pid": 4321,
            "exec_start": 101,
            "active_enter": 102,
            "restart": "on-failure",
        },
    )
    monkeypatch.setattr(
        update_owner_restart,
        "prepare_owner_restart_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read-only home")),
    )
    monkeypatch.setattr(
        update_cmd,
        "_owner_restart_verification_timeout",
        lambda _owner: 120.0,
    )
    with patch.object(update_cmd.subprocess, "run") as run:
        assert update_cmd._launch_updater_owner_restart_verifier(
            _owner(), final_exit_code=0, persist_result=True
        ) is False

    run.assert_not_called()


def test_owner_detection_uses_pid_cgroup_scope(monkeypatch):
    live = SimpleNamespace(
        _get_pid_cgroup_path=lambda _pid: (
            "/user.slice/user-1000.slice/hermes-gateway-coding_lead.service"
        ),
        _get_systemd_service_for_pid=lambda _pid: (
            "hermes-gateway-coding_lead.service"
        ),
        _extract_scope_from_cgroup=lambda _path: "user",
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: live)

    owner, failed = update_cmd._detect_updater_systemd_gateway_owner()

    assert owner == _owner()
    assert failed is False


def test_nested_gateway_cgroup_resolves_owner_component(monkeypatch):
    live = SimpleNamespace(
        _get_pid_cgroup_path=lambda _pid: (
            "/user.slice/user-1000.slice/user@1000.service/app.slice/"
            "hermes-gateway-coding_lead.service/worker.scope"
        ),
        _get_systemd_service_for_pid=lambda _pid: None,
        _extract_scope_from_cgroup=lambda _path: "user",
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: live)

    assert update_cmd._detect_updater_systemd_gateway_owner() == (_owner(), False)


def test_unknown_non_systemd_ownership_is_not_a_failure(monkeypatch):
    live = SimpleNamespace(
        _get_pid_cgroup_path=lambda _pid: None,
        _get_systemd_service_for_pid=lambda _pid: None,
        _extract_scope_from_cgroup=lambda _path: None,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: live)

    assert update_cmd._detect_updater_systemd_gateway_owner() == (None, False)


def test_malformed_gateway_cgroup_that_cannot_resolve_owner_is_a_failure(monkeypatch):
    live = SimpleNamespace(
        _get_pid_cgroup_path=lambda _pid: (
            "/user.slice/user-1000.slice/hermes-gateway-coding_lead.service.extra"
        ),
        _get_systemd_service_for_pid=lambda _pid: None,
        _extract_scope_from_cgroup=lambda _path: "user",
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: live)

    assert update_cmd._detect_updater_systemd_gateway_owner() == (None, True)


def test_transient_verifier_launch_failure_clears_pending_request(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(update_cmd.shutil, "which", lambda _name: "/bin/systemd-run")
    monkeypatch.setattr(
        update_cmd,
        "_owner_restart_verification_timeout",
        lambda _owner: 120.0,
        raising=False,
    )
    monkeypatch.setattr(
        update_owner_restart,
        "read_systemd_service_state",
        lambda _scope, _service: {
            "active_state": "active",
            "sub_state": "running",
            "main_pid": 4321,
            "exec_start": 101,
            "active_enter": 102,
            "restart": "on-failure",
        },
    )
    monkeypatch.setattr(update_cmd.secrets, "token_hex", lambda _size: "c" * 32)
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *_args, **_kwargs: MagicMock(
            returncode=1, stdout="", stderr="permission denied"
        ),
    )

    assert update_cmd._launch_updater_owner_restart_verifier(
        _owner(), final_exit_code=0, persist_result=True
    ) is False
    assert not (
        tmp_path / update_owner_restart.OWNER_RESTART_PENDING_FILE
    ).exists()
