"""Contract tests for explicit account targeting in gateway restart."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from hermes_cli.subcommands.gateway import build_gateway_parser


def _gateway_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    noop = lambda _args: None
    build_gateway_parser(
        subparsers,
        cmd_gateway=noop,
        cmd_proxy=noop,
        cmd_gateway_enroll=noop,
    )
    return parser


def test_gateway_restart_accepts_explicit_user_target():
    args = _gateway_parser().parse_args(["gateway", "restart", "--user", "hermes"])

    assert args.user == "hermes"
    assert args.system is False


def test_gateway_restart_user_and_system_scopes_are_mutually_exclusive():
    with pytest.raises(SystemExit) as exited:
        _gateway_parser().parse_args(["gateway", "restart", "--system", "--user", "hermes"])

    assert exited.value.code == 2


@pytest.mark.platforms("linux")
def test_root_user_target_restart_runs_cli_as_requested_account(monkeypatch, tmp_path):
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_user_restart as target

    home = tmp_path / "account-home"
    home.mkdir()
    project = tmp_path / "source"
    project.mkdir()
    account = SimpleNamespace(pw_uid=1001, pw_gid=1002, pw_dir=str(home), pw_shell="/bin/zsh")
    calls = []
    which_calls = []

    monkeypatch.setattr(target, "_lookup_user", lambda name: account)
    monkeypatch.setattr(target, "_explicit_profile_name", lambda: "work")
    monkeypatch.setattr(target.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        target.shutil,
        "which",
        lambda name, path=None: which_calls.append((name, path)) or ("/usr/bin/setpriv" if name == "setpriv" else None),
    )
    monkeypatch.setattr(target.sys, "executable", "/opt/hermes/bin/python")
    monkeypatch.setattr(
        target.sys,
        "argv",
        ["/opt/hermes/bin/hermes", "gateway", "restart", "--user", "hermes", "--interrupt"],
    )
    monkeypatch.setattr(gw, "PROJECT_ROOT", project)
    monkeypatch.setattr(gw, "is_linux", lambda: True)
    monkeypatch.setattr(gw, "is_termux", lambda: False)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(
        target.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")

    assert target.dispatch_user_target_restart(SimpleNamespace(user="hermes")) is True

    command, kwargs = calls[0]
    assert command[:5] == [
        "/usr/bin/setpriv",
        "--reuid=1001",
        "--regid=1002",
        "--init-groups",
        "--no-new-privs",
    ]
    assert command[command.index("--") + 1 :] == [
        "/opt/hermes/bin/python",
        "-m",
        "hermes_cli.main",
        "--profile",
        "work",
        "gateway",
        "restart",
        "--user",
        "hermes",
        "--interrupt",
    ]
    assert kwargs["cwd"] == str(project)
    assert kwargs["env"]["HOME"] == str(home)
    assert kwargs["env"]["HERMES_HOME"] == str(home / ".hermes")
    assert kwargs["env"]["SHELL"] == "/bin/zsh"
    assert kwargs["env"]["PYTHONPATH"] == str(project)
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]
    assert which_calls == [("setpriv", target._SYSTEM_PATH)]
    assert kwargs["timeout"] == 3600


@pytest.mark.platforms("linux")
def test_root_target_environment_preserves_custom_xdg_config_home(monkeypatch, tmp_path):
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_user_restart as target

    xdg_config_home = tmp_path / "target-config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))

    env = target._target_environment("hermes", tmp_path / "home", "/bin/sh", gw)

    assert env["XDG_CONFIG_HOME"] == str(xdg_config_home)


def test_restart_dispatches_user_target_before_current_account_guards(monkeypatch):
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_user_restart as target

    args = SimpleNamespace(user="hermes", system=False, all=False, force=False)
    dispatched = []
    monkeypatch.setattr(target, "dispatch_user_target_restart", lambda value: dispatched.append(value) or True)
    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: pytest.fail("current-account guard ran"))

    gw._cmd_restart(args)

    assert dispatched == [args]


@pytest.mark.platforms("linux")
def test_target_user_reentry_requires_its_own_unit(monkeypatch, tmp_path, capsys):
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_user_restart as target

    account = SimpleNamespace(pw_uid=1001, pw_gid=1002, pw_dir="/home/hermes", pw_shell="/bin/sh")
    unit = tmp_path / "missing.service"
    monkeypatch.setattr(target, "_lookup_user", lambda name: account)
    monkeypatch.setattr(target.os, "geteuid", lambda: 1001)
    monkeypatch.setattr(gw, "is_linux", lambda: True)
    monkeypatch.setattr(gw, "is_termux", lambda: False)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit)

    with pytest.raises(SystemExit) as exited:
        target.dispatch_user_target_restart(SimpleNamespace(user="hermes"))

    assert exited.value.code == 1
    output = capsys.readouterr()
    assert "No user-scope gateway unit" in output.out + output.err
    assert "refusing system-scope fallback" in output.out + output.err


@pytest.mark.platforms("linux")
def test_target_user_reentry_with_its_unit_continues_normal_restart(monkeypatch, tmp_path):
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_user_restart as target

    account = SimpleNamespace(pw_uid=1001, pw_gid=1002, pw_dir="/home/hermes", pw_shell="/bin/sh")
    unit = tmp_path / "gateway.service"
    unit.touch()
    monkeypatch.setattr(target, "_lookup_user", lambda name: account)
    monkeypatch.setattr(target.os, "geteuid", lambda: 1001)
    monkeypatch.setattr(gw, "is_linux", lambda: True)
    monkeypatch.setattr(gw, "is_termux", lambda: False)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit)

    assert target.dispatch_user_target_restart(SimpleNamespace(user="hermes")) is False


@pytest.mark.platforms("linux")
def test_nonroot_cannot_target_another_account(monkeypatch, capsys):
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_user_restart as target

    account = SimpleNamespace(pw_uid=1002, pw_gid=1002, pw_dir="/home/other", pw_shell="/bin/sh")
    monkeypatch.setattr(target, "_lookup_user", lambda name: account)
    monkeypatch.setattr(target.os, "geteuid", lambda: 1001)
    monkeypatch.setattr(gw, "is_linux", lambda: True)
    monkeypatch.setattr(gw, "is_termux", lambda: False)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(target.shutil, "which", lambda name, path=None: pytest.fail("must refuse before setpriv lookup"))

    with pytest.raises(SystemExit) as exited:
        target.dispatch_user_target_restart(SimpleNamespace(user="other"))

    assert exited.value.code == 1
    assert "requires root" in capsys.readouterr().out


@pytest.mark.platforms("linux")
def test_root_fails_closed_when_setpriv_is_unavailable(monkeypatch, capsys):
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_user_restart as target

    account = SimpleNamespace(pw_uid=1002, pw_gid=1002, pw_dir="/home/other", pw_shell="/bin/sh")
    monkeypatch.setattr(target, "_lookup_user", lambda name: account)
    monkeypatch.setattr(target.os, "geteuid", lambda: 0)
    monkeypatch.setattr(gw, "is_linux", lambda: True)
    monkeypatch.setattr(gw, "is_termux", lambda: False)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(target.shutil, "which", lambda name, path=None: None)
    monkeypatch.setattr(target.subprocess, "run", lambda *a, **k: pytest.fail("must not launch without setpriv"))

    with pytest.raises(SystemExit) as exited:
        target.dispatch_user_target_restart(SimpleNamespace(user="other"))

    assert exited.value.code == 1
    assert "no service was changed" in capsys.readouterr().out


@pytest.mark.platforms("linux")
def test_child_timeout_reports_that_service_state_must_be_checked(monkeypatch, capsys):
    import subprocess

    from hermes_cli import gateway as gw
    from hermes_cli import gateway_user_restart as target

    account = SimpleNamespace(pw_uid=1002, pw_gid=1002, pw_dir="/home/other", pw_shell="/bin/sh")
    monkeypatch.setattr(target, "_lookup_user", lambda name: account)
    monkeypatch.setattr(target, "_explicit_profile_name", lambda: None)
    monkeypatch.setattr(target.os, "geteuid", lambda: 0)
    monkeypatch.setattr(target.shutil, "which", lambda name, path=None: "/usr/bin/setpriv")
    monkeypatch.setattr(gw, "is_linux", lambda: True)
    monkeypatch.setattr(gw, "is_termux", lambda: False)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)

    def timed_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(target.subprocess, "run", timed_out)

    with pytest.raises(SystemExit) as exited:
        target.dispatch_user_target_restart(SimpleNamespace(user="other"))

    assert exited.value.code == 1
    output = capsys.readouterr().out
    assert "timed out" in output
    assert "check service status" in output
