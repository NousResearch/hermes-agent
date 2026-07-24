"""Tests for ``hermes update --stop-services`` (refs #40449).

Covers the opt-in stop/relaunch path for this install's dashboard/serve
processes around the dependency sync:

1. ``_stop_hermes_services_for_update`` — detection (service cmdline
   patterns ∩ this-install ownership), exclusions (self/ancestors, desktop
   children, systemd-managed dashboard), stop + exit-wait, resume token.
2. ``_restart_hermes_services_after_update`` — argv/cwd replay, idempotent
   token, per-service failure isolation, PID-reuse guard.
3. ``_cmd_update_impl`` wiring — stop runs before the venv-holder guard's
   re-scan, the guard refusal restarts stopped services before exit 2,
   the ZIP finally restarts on failure, atexit registration, ``--check``
   and no-flag runs never touch the new path, and the success tail relaunch
   comes only after the stale-dashboard sweep.

Windows-specific paths are exercised via ``_is_windows`` patching so the
suite runs on any host (same approach as test_update_venv_health).
"""

from __future__ import annotations

import argparse
import inspect
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main as cli_main
from hermes_cli.subcommands.update import build_update_parser


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=lambda a: None)
    return parser


def test_parser_accepts_stop_services_flag():
    parser = _build_parser()
    assert parser.parse_args(["update", "--stop-services"]).stop_services is True
    assert parser.parse_args(["update"]).stop_services is False


def test_parser_help_renders_stop_services():
    parser = _build_parser()
    update_parser = next(
        action
        for action in parser._subparsers._group_actions
        if isinstance(action, argparse._SubParsersAction)
    ).choices["update"]
    help_text = update_parser.format_help()
    assert "--stop-services" in help_text
    assert "relaunch" in help_text
    assert "force-killed" in help_text


# ---------------------------------------------------------------------------
# _stop_hermes_services_for_update
# ---------------------------------------------------------------------------


def _proc(pid, exe, name, cmdline=None, cwd=""):
    proc = MagicMock()
    proc.info = {
        "pid": pid,
        "exe": exe,
        "name": name,
        "cmdline": cmdline or [],
        "cwd": cwd,
    }
    return proc


def _fake_psutil(procs):
    me = MagicMock()
    me.parents.return_value = []
    return types.SimpleNamespace(
        process_iter=lambda attrs: iter(procs),
        Process=lambda *a, **k: me,
    )


def _stop_args(**overrides):
    defaults = dict(stop_services=True)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_stop_flag_off_returns_none_without_scanning():
    scan = MagicMock()
    with patch.object(cli_main, "_desktop_child_pids", scan):
        assert cli_main._stop_hermes_services_for_update(SimpleNamespace()) is None
        assert (
            cli_main._stop_hermes_services_for_update(
                SimpleNamespace(stop_services=False)
            )
            is None
        )
    scan.assert_not_called()


def test_stop_no_psutil_warns_and_returns_none(capsys):
    with patch.dict(sys.modules, {"psutil": None}):
        token = cli_main._stop_hermes_services_for_update(_stop_args())
    assert token is None
    assert "psutil is unavailable" in capsys.readouterr().out


def test_stop_detects_owned_services_and_builds_token(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP_CHILD_PID", raising=False)
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    base_py = "C:\\Python311\\python.exe"
    dashboard_argv = [
        venv_py,
        "-m",
        "hermes_cli.main",
        "dashboard",
        "--host",
        "0.0.0.0",
        "--port",
        "9119",
    ]
    fake = _fake_psutil(
        [
            # this install's dashboard → stopped, argv+cwd recorded
            _proc(301, venv_py, "pythonw.exe", dashboard_argv, cwd=str(tmp_path)),
            # another install's serve (foreign cwd, exe outside venv) → left alone
            _proc(
                302,
                base_py,
                "python.exe",
                [base_py, "-m", "hermes_cli.main", "serve"],
                cwd="C:\\other-install",
            ),
            # a gateway from this venv → not a dashboard/serve pattern → left alone
            _proc(
                303,
                venv_py,
                "pythonw.exe",
                [venv_py, "-m", "hermes_cli.main", "gateway", "run"],
            ),
        ]
    )
    terminate = MagicMock(return_value=([301], []))
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake}
    ), patch.object(cli_main, "_terminate_service_pids", terminate), patch.object(
        cli_main, "_dashboard_service_main_pid", return_value=None
    ), patch("gateway.status._pid_exists", return_value=False):
        token = cli_main._stop_hermes_services_for_update(_stop_args())

    terminate.assert_called_once_with([301])
    assert token == {
        "resume_needed": True,
        "services": [
            {
                "pid": 301,
                "argv": dashboard_argv,
                "cwd": str(tmp_path),
                "label": "hermes dashboard",
            }
        ],
    }
    out = capsys.readouterr().out
    assert "--stop-services" in out
    assert "hermes dashboard (PID 301)" in out


def test_stop_excludes_desktop_children_and_managed_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP_CHILD_PID", "401, 402")
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    serve_argv = [venv_py, "-m", "hermes_cli.main", "serve"]
    fake = _fake_psutil(
        [
            _proc(401, venv_py, "python.exe", serve_argv),  # desktop child
            _proc(402, venv_py, "python.exe", serve_argv),  # desktop child
            _proc(403, venv_py, "python.exe", serve_argv),  # systemd MainPID
            _proc(404, venv_py, "python.exe", serve_argv),  # genuinely ours
        ]
    )
    terminate = MagicMock(return_value=([404], []))
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake}
    ), patch.object(cli_main, "_terminate_service_pids", terminate), patch.object(
        cli_main, "_dashboard_service_main_pid", return_value=403
    ), patch("gateway.status._pid_exists", return_value=False):
        token = cli_main._stop_hermes_services_for_update(_stop_args())

    terminate.assert_called_once_with([404])
    assert [svc["pid"] for svc in token["services"]] == [404]
    assert token["services"][0]["label"] == "hermes serve"


def test_stop_nothing_matched_returns_none(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP_CHILD_PID", raising=False)
    base_py = str(tmp_path / "elsewhere" / "python.exe")
    fake = _fake_psutil(
        [_proc(501, base_py, "python.exe", [base_py, "somescript.py"])]
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake}
    ), patch.object(cli_main, "_dashboard_service_main_pid", return_value=None):
        token = cli_main._stop_hermes_services_for_update(_stop_args())
    assert token is None
    assert "Stopping Hermes service" not in capsys.readouterr().out


def test_stop_survivor_recorded_not_in_token(tmp_path, capsys, monkeypatch):
    """A PID that will not die stays visible to the venv guard and is never
    scheduled for relaunch."""
    monkeypatch.delenv("HERMES_DESKTOP_CHILD_PID", raising=False)
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    dash = [venv_py, "-m", "hermes_cli.main", "dashboard"]
    serve = [venv_py, "-m", "hermes_cli.main", "serve"]
    fake = _fake_psutil(
        [
            _proc(601, venv_py, "python.exe", dash),
            _proc(602, venv_py, "python.exe", serve),
        ]
    )
    terminate = MagicMock(return_value=([601, 602], []))
    # 601 exits, 602 survives past the 5s deadline.
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake}
    ), patch.object(cli_main, "_terminate_service_pids", terminate), patch.object(
        cli_main, "_dashboard_service_main_pid", return_value=None
    ), patch(
        "gateway.status._pid_exists", side_effect=lambda pid: pid == 602
    ), patch(
        "time.monotonic", side_effect=[0.0, 0.1, 10.0]
    ), patch("time.sleep"):
        token = cli_main._stop_hermes_services_for_update(_stop_args())

    assert [svc["pid"] for svc in token["services"]] == [601]
    out = capsys.readouterr().out
    assert "still running" in out
    assert "venv guard" in out


def test_stop_all_survivors_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP_CHILD_PID", raising=False)
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    fake = _fake_psutil(
        [_proc(701, venv_py, "python.exe", [venv_py, "-m", "hermes_cli.main", "serve"])]
    )
    terminate = MagicMock(return_value=([], [(701, "access denied")]))
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake}
    ), patch.object(cli_main, "_terminate_service_pids", terminate), patch.object(
        cli_main, "_dashboard_service_main_pid", return_value=None
    ):
        token = cli_main._stop_hermes_services_for_update(_stop_args())
    assert token is None


# ---------------------------------------------------------------------------
# _restart_hermes_services_after_update / _spawn_detached_service
# ---------------------------------------------------------------------------


def _token(*services):
    return {"resume_needed": True, "services": list(services)}


def _svc(pid, argv, cwd=None, label="hermes dashboard"):
    return {"pid": pid, "argv": argv, "cwd": cwd, "label": label}


def test_restart_respawns_recorded_argv_and_cwd(capsys):
    spawn = MagicMock(return_value=True)
    dash = ["python.exe", "-m", "hermes_cli.main", "dashboard", "--port", "9119"]
    serve = ["python.exe", "-m", "hermes_cli.main", "serve"]
    token = _token(
        _svc(301, dash, cwd="C:\\install"),
        _svc(302, serve, label="hermes serve"),
    )
    with patch.object(cli_main, "_spawn_detached_service", spawn):
        cli_main._restart_hermes_services_after_update(token)

    assert spawn.call_args_list == [
        ((dash, "C:\\install"),),
        ((serve, None),),
    ]
    out = capsys.readouterr().out
    assert "Restarted hermes dashboard (was PID 301)" in out
    assert "Restarted hermes serve (was PID 302)" in out


def test_restart_is_idempotent(capsys):
    spawn = MagicMock(return_value=True)
    token = _token(_svc(301, ["python", "-m", "hermes_cli.main", "dashboard"]))
    with patch.object(cli_main, "_spawn_detached_service", spawn):
        cli_main._restart_hermes_services_after_update(token)
        cli_main._restart_hermes_services_after_update(token)
    assert spawn.call_count == 1
    assert token["resume_needed"] is False


def test_restart_none_and_cleared_tokens_are_noops(capsys):
    spawn = MagicMock()
    with patch.object(cli_main, "_spawn_detached_service", spawn):
        cli_main._restart_hermes_services_after_update(None)
        cli_main._restart_hermes_services_after_update({"resume_needed": False})
        cli_main._restart_hermes_services_after_update(_token())
    spawn.assert_not_called()
    assert capsys.readouterr().out == ""


def test_restart_spawn_failure_is_isolated_per_service(capsys):
    spawn = MagicMock(side_effect=[OSError("boom"), True])
    token = _token(
        _svc(301, ["python", "-m", "hermes_cli.main", "dashboard"]),
        _svc(302, ["python", "-m", "hermes_cli.main", "serve"], label="hermes serve"),
    )
    with patch.object(cli_main, "_spawn_detached_service", spawn):
        cli_main._restart_hermes_services_after_update(token)
    assert spawn.call_count == 2
    out = capsys.readouterr().out
    assert "could not restart hermes dashboard" in out
    assert "restart manually: hermes dashboard" in out
    assert "Restarted hermes serve (was PID 302)" in out


def test_restart_missing_argv_prints_manual_hint(capsys):
    spawn = MagicMock()
    token = _token(_svc(301, None))
    with patch.object(cli_main, "_spawn_detached_service", spawn):
        cli_main._restart_hermes_services_after_update(token)
    spawn.assert_not_called()
    assert "restart manually: hermes dashboard" in capsys.readouterr().out


def test_restart_refuses_unrecognized_command_line(capsys):
    """PID-reuse race guard: never replay an argv that is not a Hermes
    service command line."""
    spawn = MagicMock()
    token = _token(_svc(301, ["notepad.exe", "C:\\secrets.txt"]))
    with patch.object(cli_main, "_spawn_detached_service", spawn):
        cli_main._restart_hermes_services_after_update(token)
    spawn.assert_not_called()
    assert "unrecognized command line" in capsys.readouterr().out


def test_spawn_detached_service_posix_kwargs(tmp_path):
    argv = ["python", "-m", "hermes_cli.main", "dashboard"]
    with patch.object(cli_main, "_is_windows", return_value=False), patch.object(
        cli_main.subprocess, "Popen"
    ) as popen:
        assert cli_main._spawn_detached_service(argv, str(tmp_path)) is True
    assert popen.call_args.args[0] == argv
    assert popen.call_args.kwargs["cwd"] == str(tmp_path)
    assert popen.call_args.kwargs["start_new_session"] is True


def test_spawn_detached_service_missing_cwd_falls_back_to_project_root(tmp_path):
    argv = ["python", "-m", "hermes_cli.main", "serve"]
    with patch.object(cli_main, "_is_windows", return_value=False), patch.object(
        cli_main, "PROJECT_ROOT", tmp_path
    ), patch.object(cli_main.subprocess, "Popen") as popen:
        cli_main._spawn_detached_service(argv, str(tmp_path / "gone"))
    assert popen.call_args.kwargs["cwd"] == str(tmp_path)


# ---------------------------------------------------------------------------
# _cmd_update_impl wiring
# ---------------------------------------------------------------------------


def _update_args(**overrides):
    defaults = dict(
        gateway=False,
        check=False,
        no_backup=True,
        backup=False,
        yes=True,
        branch=None,
        force=False,
        force_venv=False,
        stop_services=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _PastGuard(Exception):
    pass


class _RootSentinel:
    def __truediv__(self, _other):
        raise _PastGuard


def _drive_update_to_guard(args, *, stop_token=None, holders=True, calls=None):
    """Drive _cmd_update_impl to the venv-holder guard with the stop/restart
    helpers mocked; returns (outcome, mocks)."""
    calls = calls if calls is not None else []

    def _record(name, ret):
        def _inner(*a, **k):
            calls.append(name)
            return ret

        return _inner

    stop = MagicMock(side_effect=_record("stop", stop_token))
    restart = MagicMock(side_effect=_record("restart", None))
    detect = MagicMock(
        side_effect=_record(
            "detect",
            [(101, "python.exe", "python.exe -m hermes_cli.main serve")]
            if holders
            else [],
        )
    )
    resume = MagicMock(side_effect=_record("resume", None))

    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(cli_main, "_run_pre_update_backup"), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update", resume
    ), patch.object(
        cli_main, "_stop_hermes_services_for_update", stop
    ), patch.object(
        cli_main, "_restart_hermes_services_after_update", restart
    ), patch.object(
        cli_main, "_detect_venv_python_processes", detect
    ), patch.object(
        cli_main, "PROJECT_ROOT", _RootSentinel()
    ):
        try:
            cli_main._cmd_update_impl(args, gateway_mode=False)
        except _PastGuard:
            outcome = "past_guard"
        except SystemExit as exc:
            outcome = f"exit_{exc.code}"
        else:
            outcome = "returned"
    return outcome, SimpleNamespace(
        stop=stop, restart=restart, detect=detect, resume=resume, calls=calls
    )


def test_guard_refusal_restarts_stopped_services_before_exit_2():
    token = {"resume_needed": True, "services": [{"pid": 1}]}
    outcome, m = _drive_update_to_guard(
        _update_args(stop_services=True), stop_token=token
    )
    assert outcome == "exit_2"
    m.restart.assert_called_once_with(token)
    m.resume.assert_called()
    # stop runs before the guard's authoritative re-scan; the restart of our
    # services precedes the gateway resume (reverse of the stop order).
    assert m.calls.index("stop") < m.calls.index("detect")
    assert m.calls.index("restart") < m.calls.index("resume")


def test_stop_helper_called_even_without_flag_but_is_gated_internally():
    """The call site is unconditional; the flag gate lives in the helper, so
    a bare SimpleNamespace without the attr must sail through (getattr
    contract)."""
    outcome, m = _drive_update_to_guard(
        SimpleNamespace(
            gateway=False,
            check=False,
            no_backup=True,
            backup=False,
            yes=True,
            branch=None,
            force=False,
            force_venv=False,
        ),
        stop_token=None,
    )
    assert outcome == "exit_2"
    m.stop.assert_called_once()
    m.restart.assert_called_once_with(None)


def test_no_flag_run_produces_no_stop_services_output(capsys):
    """Real helpers, no flag: the guard-refusal output is byte-identical to
    the pre-flag behavior."""
    args = _update_args()  # stop_services=False
    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(cli_main, "_run_pre_update_backup"), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main,
        "_detect_venv_python_processes",
        return_value=[(101, "python.exe", "python.exe -m hermes_cli.main serve")],
    ), patch.object(cli_main, "PROJECT_ROOT", _RootSentinel()):
        with pytest.raises(SystemExit) as exc_info:
            cli_main._cmd_update_impl(args, gateway_mode=False)
    assert exc_info.value.code == 2
    out = capsys.readouterr().out
    assert "Stopping Hermes service" not in out
    assert "Restarting Hermes service" not in out
    assert "--force-venv" in out


def test_force_venv_with_stop_services_stops_then_skips_guard():
    token = {"resume_needed": True, "services": [{"pid": 1}]}
    outcome, m = _drive_update_to_guard(
        _update_args(stop_services=True, force_venv=True), stop_token=token
    )
    assert outcome == "past_guard"
    m.stop.assert_called_once()
    m.detect.assert_not_called()


def test_check_never_reaches_stop_services(monkeypatch):
    stop = MagicMock()
    check = MagicMock()
    monkeypatch.setattr(cli_main, "_stop_hermes_services_for_update", stop)
    monkeypatch.setattr(cli_main, "_cmd_update_check", check)
    monkeypatch.setattr(cli_main, "_resolve_update_branch", lambda a: "main")
    import hermes_cli.config as cli_config

    monkeypatch.setattr(cli_config, "is_managed", lambda: False)
    monkeypatch.setattr(cli_config, "detect_install_method", lambda root: "git")
    cli_main.cmd_update(_update_args(check=True, stop_services=True))
    check.assert_called_once()
    stop.assert_not_called()


def test_atexit_registered_with_restart_and_token():
    token = {"resume_needed": True, "services": [{"pid": 1}]}
    registered = []
    with patch("atexit.register", side_effect=lambda fn, *a: registered.append((fn, a))):
        outcome, m = _drive_update_to_guard(
            _update_args(stop_services=True), stop_token=token
        )
    assert outcome == "exit_2"
    assert (m.restart, (token,)) in [(fn, a) for fn, a in registered]


def test_zip_path_restarts_services_in_finally(tmp_path, monkeypatch):
    """A ZIP-path failure must still trigger the services restart (and the
    gateway resume) via the finally block."""
    args = _update_args(stop_services=True)
    token = {"resume_needed": True, "services": [{"pid": 1}]}
    restart = MagicMock()
    resume = MagicMock()
    monkeypatch.setattr(sys, "platform", "win32")
    with patch.object(cli_main, "_is_windows", return_value=True), patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(cli_main, "_run_pre_update_backup"), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update", resume
    ), patch.object(
        cli_main, "_stop_hermes_services_for_update", return_value=token
    ), patch.object(
        cli_main, "_restart_hermes_services_after_update", restart
    ), patch.object(
        cli_main, "_detect_venv_python_processes", return_value=[]
    ), patch.object(
        cli_main, "PROJECT_ROOT", tmp_path  # no .git dir → ZIP path
    ), patch.object(cli_main, "_discard_lockfile_churn"), patch.object(
        cli_main, "_get_origin_url", return_value=""
    ), patch.object(cli_main, "_is_fork", return_value=False), patch.object(
        cli_main, "_update_via_zip", side_effect=RuntimeError("zip failed")
    ):
        with pytest.raises(RuntimeError, match="zip failed"):
            cli_main._cmd_update_impl(args, gateway_mode=False)
    restart.assert_called_once_with(token)
    resume.assert_called_once()


def test_success_tail_restart_comes_after_stale_dashboard_sweep():
    """Ordering contract: the success-tail relaunch of --stop-services
    services must come AFTER _kill_stale_dashboard_processes(restart_managed=
    True), or the sweep would re-kill the freshly respawned services."""
    src = inspect.getsource(cli_main._cmd_update_impl)
    sweep = src.rindex("_kill_stale_dashboard_processes(restart_managed=True)")
    relaunch = src.rindex("_restart_hermes_services_after_update(_services_resume)")
    assert sweep < relaunch, (
        "success-tail _restart_hermes_services_after_update must run after "
        "the stale-dashboard sweep"
    )
    # ...and after the Windows gateway resume for the same run.
    resume = src.rindex("_resume_windows_gateways_after_update(_windows_gateway_resume)")
    assert resume < relaunch


# ---------------------------------------------------------------------------
# Extraction seams stay behavior-compatible
# ---------------------------------------------------------------------------


def test_desktop_child_pids_parsing(monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP_CHILD_PID", " 12, x, 34,,56 ")
    assert cli_main._desktop_child_pids() == {12, 34, 56}
    monkeypatch.delenv("HERMES_DESKTOP_CHILD_PID")
    assert cli_main._desktop_child_pids() == set()


def test_dashboard_service_main_pid_windows_is_none():
    with patch.object(cli_main.sys, "platform", "win32"):
        assert cli_main._dashboard_service_main_pid() is None
