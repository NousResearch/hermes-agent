"""Deferred restart's result contract, command boundary, and lifecycle guard parity."""
import argparse
import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from hermes_cli import gateway_deferred_restart as cli
from hermes_cli.subcommands.gateway import build_gateway_parser


def parser():
    result = argparse.ArgumentParser()
    build_gateway_parser(result.add_subparsers(), cmd_gateway=Mock(), cmd_proxy=Mock(), cmd_gateway_enroll=Mock())
    return result


def ack(**changes):
    return {"accepted": True, "pid": 42, "supervisor": "external", "drain_timeout": 3.5, **changes}


@pytest.mark.parametrize("args", [[], ["--delay", "3"], ["--force"], ["--all"], ["--wait-ready"], ["--wat"], ["3"]])
def test_parser_boundary(args):
    if not args:
        assert parser().parse_args(["gateway", "defer-restart"]).func is cli.cmd_gateway_defer_restart
    else:
        with pytest.raises(SystemExit) as error:
            parser().parse_args(["gateway", "defer-restart", *args])
        assert error.value.code == 2


@pytest.mark.parametrize("changes", [
    {"pid": True}, {"pid": 0}, {"pid": 1.5}, {"drain_timeout": True}, {"drain_timeout": -1},
    {"drain_timeout": float("inf")}, {"drain_timeout": float("nan")}, {"drain_timeout": 10**1000},
    {"supervisor": "launchd"}, {"supervisor": []}, {"accepted": 1}, {"reason": None},
])
def test_malformed_positive_ack(changes):
    assert cli.classify_ack(ack(**changes), "external") == "unknown"


@pytest.mark.parametrize("result", [None, {}, {"accepted": False},
    {"accepted": False, "reason": "new_reason"},
    {"accepted": False, "reason": []},
    {"accepted": False, "reason": "already_stopping", "pid": True},
    {"accepted": False, "reason": "already_stopping", "supervisor": "manual"},
    {"accepted": False, "reason": "already_stopping", "supervisor": "systemd"},
    {"accepted": False, "reason": "already_stopping", "drain_timeout": float("inf")},
    {"accepted": False, "reason": "unsupported_drain_timeout", "drain_timeout": 1},
    {"accepted": False, "reason": "unsupported_supervisor", "supervisor": "external"},
    {"accepted": False, "reason": "already_stopping", "pausing": True}])
def test_malformed_negative_ack(result):
    assert cli.classify_ack(result, "external") == "unknown"


@pytest.mark.parametrize("reason,metadata", [
    ("unsupported_supervisor", {}), ("unsupported_supervisor", {"pid": 42}),
    ("unsupported_drain_timeout", {"pid": 42, "supervisor": "external"}),
    ("already_stopping", {}), ("already_stopping", {"pid": 42, "supervisor": "external", "drain_timeout": 0})])
def test_valid_negative_ack(reason, metadata):
    assert cli.classify_ack({"accepted": False, "reason": reason, **metadata}, "external") == "rejected"


def test_positive_ack_ignores_unrelated_fields():
    assert cli.classify_ack(ack(extra="ignored"), "external") == "accepted"


@pytest.mark.parametrize("supervisor", ["manual", "desktop", "s6", "windows", None, "mystery"])
def test_preflight_rejects_unsupported_supervisor(monkeypatch, tmp_path, supervisor):
    from gateway import control_socket
    monkeypatch.setattr(cli, "resolve_target", lambda: (tmp_path, {"supervisor": supervisor}, False))
    query = Mock()
    monkeypatch.setattr(control_socket, "query_gateway_control", query)
    assert cli.run_deferred_restart() == 1
    query.assert_not_called()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("supervisor", ["systemd", "launchd", "external"])
def test_supported_preflight_and_output(monkeypatch, tmp_path, capsys, supervisor):
    from gateway import control_socket
    monkeypatch.setattr(cli, "resolve_target", lambda: (tmp_path, {"supervisor": supervisor}, True))
    query = Mock(return_value=ack(supervisor=supervisor))
    monkeypatch.setattr(control_socket, "query_gateway_control", query)
    assert cli.run_deferred_restart() == 0
    query.assert_called_once_with(tmp_path, "defer-restart", timeout=8.0)
    output = capsys.readouterr().out
    assert "PID 42" in output and supervisor in output and "All profiles" in output
    assert "not a fixed restart delay" in output and "readiness are not confirmed" in output


def test_cron_refuses_before_resolution(monkeypatch):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    resolve = Mock(side_effect=AssertionError("must not resolve"))
    monkeypatch.setattr(cli, "resolve_target", resolve)
    assert cli.run_deferred_restart() == 1
    resolve.assert_not_called()


@pytest.mark.parametrize("result", [None, {"accepted": False}, {"accepted": False, "reason": "already_stopping"}, ack()])
@pytest.mark.platforms("posix")
def test_real_parser_dispatch_exit_status(tmp_path, result):
    code = '''
import argparse, sys
from pathlib import Path
from hermes_cli import gateway_deferred_restart as cli
from hermes_cli.subcommands.gateway import build_gateway_parser
from gateway import control_socket
cli.resolve_target = lambda: (Path(sys.argv[1]), {"supervisor": "external"}, False)
control_socket.query_gateway_control = lambda *a, **kw: RESULT
p = argparse.ArgumentParser()
build_gateway_parser(p.add_subparsers(), cmd_gateway=lambda a: None, cmd_proxy=lambda a: None, cmd_gateway_enroll=lambda a: None)
args = p.parse_args(["gateway", "defer-restart"])
args.func(args)
'''.replace("RESULT", repr(result))
    process = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=30, check=False)
    assert process.returncode == (0 if result and result.get("accepted") is True else 1), process.stderr
    if result is None or result == {"accepted": False}:
        assert "UNKNOWN" in process.stdout and "may already be processing" in process.stdout


def test_marked_cron_child_environment(tmp_path, monkeypatch):
    from gateway.session_context import _VAR_MAP
    from tools.environments.local import _inject_session_context_env
    token = _VAR_MAP["HERMES_CRON_SESSION"].set("1")
    try:
        child_env = dict(os.environ)
        _inject_session_context_env(child_env)
    finally:
        _VAR_MAP["HERMES_CRON_SESSION"].reset(token)
    child_env["HERMES_HOME"] = str(tmp_path)
    code = '''
from hermes_cli import gateway_deferred_restart as cli
cli.resolve_target = lambda: (_ for _ in ()).throw(AssertionError("must not resolve"))
cli.cmd_gateway_defer_restart(None)
'''
    process = subprocess.run([sys.executable, "-c", code], env=child_env, capture_output=True, text=True, timeout=30, check=False)
    assert process.returncode == 1, process.stderr
    assert "marked cron" in process.stdout


def test_lifecycle_guard_keeps_ordinary_verbs_blocked():
    from cron.lifecycle_guard import contains_gateway_lifecycle_command
    for verb in ("restart", "stop", "uninstall"):
        assert contains_gateway_lifecycle_command(f"hermes gateway {verb}")
    assert not contains_gateway_lifecycle_command("hermes gateway defer-restart")


@pytest.mark.parametrize("verb", ["restart", "stop", "uninstall"])
def test_cli_self_refusal_remains(monkeypatch, verb):
    from hermes_cli.gateway import _refuse_from_inside_gateway
    from tools import process_registry
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    with pytest.raises(SystemExit) as error:
        _refuse_from_inside_gateway(verb, "self shutdown")
    assert error.value.code == 1
