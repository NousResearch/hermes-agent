"""CLI parser and dispatch integration, isolated from every live scheduler/updater."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

import pytest

from hermes_cli import update_auto as auto
from hermes_cli import update_auto_run as runner
from hermes_cli import update_auto_schedule as scheduler
from hermes_cli import update_auto_schedule_common as schedule_common
from hermes_cli import update_auto_state as state
from hermes_cli import update_lock
from hermes_cli.subcommands.update import build_update_parser


def _manual_update(_args):
    pytest.fail("The real/manual updater must not run in dispatch tests")


def _parser():
    parser = argparse.ArgumentParser(prog="hermes")
    build_update_parser(parser.add_subparsers(dest="command"), cmd_update=_manual_update)
    return parser


def _args(*parts):
    if parts == ("run-scheduled",):
        parts = (*parts, "--scheduler-identity", state.AutoUpdateContext.current().identity)
    return _parser().parse_args(["update", "auto", *parts])


def _forbidden(*args, **kwargs):
    pytest.fail("Unexpected scheduler/updater side effect")


@pytest.fixture
def context(tmp_path, monkeypatch):
    home, install = tmp_path / "home", tmp_path / "install"
    home.mkdir()
    install.mkdir()
    (install / ".git").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_constants._get_platform_default_hermes_home", lambda: home)
    context = state.AutoUpdateContext(install, home, home / "logs" / "update_receipts")
    monkeypatch.setattr(state.AutoUpdateContext, "current", classmethod(lambda cls: context))
    monkeypatch.setattr(auto, "require_source_install", lambda ctx: None)
    monkeypatch.setattr(auto, "check_update", _forbidden)
    monkeypatch.setattr(auto, "run_update", _forbidden)
    monkeypatch.setattr(scheduler, "enable", _forbidden)
    monkeypatch.setattr(scheduler, "disable", _forbidden)
    monkeypatch.setattr(schedule_common, "run_command", _forbidden)
    return context


def _enabled_status(context):
    spec = auto._spec(context, "04:00", ["21:00"])
    info = scheduler.paths(spec)
    return {**state.default_status(context), "enabled": True, "mode": "scheduled",
            "schedule": "04:00", "planSchedule": ["21:00"],
            "schedulerType": info["backend"], "schedulerPath": str(info["path"])}


def _save(context, status):
    with state.operation_lock(context):
        state.write_status(context, status)


def test_manual_update_parser_keeps_canonical_handler():
    args = _parser().parse_args(["update", "--yes", "--require-backup", "--channel", "main"])
    assert args.func is _manual_update
    assert args.require_backup and args.yes
    assert args.channel == "main"


@pytest.mark.parametrize("subcommand", ["status", "plan", "run-now", "run-scheduled", "disable"])
def test_auto_parser_routes_to_auto_dispatch(subcommand, monkeypatch):
    calls = []
    monkeypatch.setattr(auto, "cmd_update_auto", lambda args: calls.append(args.auto_subcommand))
    args = _args(subcommand)
    args.func(args)
    assert calls == [subcommand]


def test_auto_enable_parser_collects_repeated_plan_times():
    args = _args("enable", "--time", "04:00", "--plan-time", "21:00", "--plan-time", "02:00")
    assert args.auto_subcommand == "enable"
    assert args.time == "04:00"
    assert args.plan_time == ["21:00", "02:00"]


@pytest.mark.parametrize("parts", [[], ["enable"], ["disable", "--time", "03:00"]])
def test_auto_parser_rejects_incomplete_or_wrong_options(parts):
    with pytest.raises(SystemExit) as raised:
        _args(*parts)
    assert raised.value.code == 2


@pytest.mark.parametrize("parts", [["update", "--help"], ["update", "auto", "--help"],
                                   ["update", "auto", "enable", "--help"]])
def test_help_is_available_without_action(parts, capsys):
    with pytest.raises(SystemExit) as raised:
        _parser().parse_args(parts)
    assert raised.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("parts", [["update", "--branch", "branch-name", "auto", "run-now"],
                                   ["update", "auto", "run-now", "--branch", "branch-name"]])
def test_manual_target_override_survives_both_parser_positions(parts):
    args = _parser().parse_args(parts)
    assert args.branch == "branch-name"
    auto._validate_options(args)


@pytest.mark.parametrize("subcommand", ["status", "disable", "run-scheduled"])
def test_disabled_default_does_not_write_or_call_backends(context, subcommand):
    before = list(context.home.rglob("*"))
    args = _args(subcommand)
    args.func(args)
    assert list(context.home.rglob("*")) == before
    assert not context.status_path.exists()


@pytest.mark.parametrize("flag", ["--check", "--plan", "--list-venv-holders", "--force", "--no-backup", "--no-gateway-restart"])
def test_parent_readonly_and_unsafe_flags_cannot_launch_auto_update(context, flag):
    args = _parser().parse_args(["update", flag, "auto", "run-now"])
    with pytest.raises(SystemExit) as raised:
        args.func(args)
    assert raised.value.code == 1
    assert not context.status_path.exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("field,value", [
    ("mode", "manual"), ("planSchedule", "21:00"), ("schedule", "4:00"),
    ("planSchedule", ["04:00"]), ("schedulerType", "foreign"),
    ("schedulerPath", "/foreign/owned.timer"), ("status", "recovery_required"),
    ("status", "running"),
])
def test_scheduled_dispatch_rejects_invalid_saved_activation(context, field, value):
    status = {**_enabled_status(context), field: value}
    _save(context, status)
    before = context.status_path.read_bytes()
    with pytest.raises(SystemExit) as raised:
        _args("run-scheduled").func(_args("run-scheduled"))
    assert raised.value.code == 1
    assert context.status_path.read_bytes() == before


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("action", ["plan", "run"])
def test_scheduled_dispatch_uses_saved_times_and_clears_target_overrides(context, monkeypatch, action):
    _save(context, _enabled_status(context))
    selected = []
    monkeypatch.setattr(scheduler, "scheduled_action", lambda schedule, times: action)
    monkeypatch.setattr(auto, "_plan", lambda ctx, status, args: selected.append(("plan", args.branch, args.channel)) or 0)
    monkeypatch.setattr(auto, "_run", lambda ctx, status, args: selected.append(("run", args.branch, args.channel)) or 0)
    monkeypatch.setattr(auto, "check_update", lambda *args: {"updateAvailable": True})
    _args("run-scheduled").func(_args("run-scheduled"))
    assert selected == [(action, None, None)]


@pytest.mark.platforms("posix")
def test_successful_enable_persists_exact_owned_scheduler(context, monkeypatch):
    installed = []

    def enable(spec):
        installed.append(spec)
        info = scheduler.paths(spec)
        return scheduler.SchedulerHandle(info["backend"], info["path"], lambda: {"ok": True})

    monkeypatch.setattr(scheduler, "enable", enable)
    args = _args("enable", "--time", "04:00", "--plan-time", "21:00")
    args.func(args)
    saved = state.read_status(context)
    assert saved["enabled"] is True
    assert saved["schedule"] == installed[0].schedule
    assert saved["planSchedule"] == list(installed[0].plan_times)
    assert saved["schedulerIdentity"] == installed[0].identity
    assert saved["schedulerPath"] == str(scheduler.paths(installed[0])["path"])
    assert installed[0].home == context.home


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("rollback_ok", [True, False])
def test_status_write_failure_rolls_back_scheduler_and_records_failed_recovery(context, monkeypatch, rollback_ok):
    rollbacks, writes = [], []
    receipt = {"ok": rollback_ok, "scheduler": "test", "errors": [] if rollback_ok else ["cannot restore"]}

    def enable(spec):
        info = scheduler.paths(spec)
        return scheduler.SchedulerHandle(info["backend"], info["path"], lambda: rollbacks.append(1) or receipt)

    def write(ctx, status):
        writes.append(dict(status))
        if len(writes) == 1:
            raise OSError("injected status write failure")
        state.write_status(ctx, status)

    monkeypatch.setattr(scheduler, "enable", enable)
    monkeypatch.setattr(auto, "write_status", write)
    args = _args("enable", "--time", "04:00")
    with pytest.raises(SystemExit) as raised:
        args.func(args)
    assert raised.value.code == 1
    assert rollbacks == [1]
    if rollback_ok:
        assert not context.status_path.exists()
    else:
        saved = state.read_status(context)
        assert saved["status"] == "recovery_required"
        assert saved["recoveryReceipt"] == receipt
        assert saved["enabled"] is False


@pytest.mark.parametrize("subcommand", ["enable", "plan", "run-now"])
def test_failed_enable_recovery_cannot_be_overwritten_by_reenable(context, subcommand):
    _save(context, {**state.default_status(context), "status": "recovery_required", "recoveryReceipt": {"ok": False}})
    options = ["--time", "04:00"] if subcommand == "enable" else []
    args = _args(subcommand, *options)
    with pytest.raises(SystemExit) as raised:
        args.func(args)
    assert raised.value.code == 1
    assert state.read_status(context)["status"] == "recovery_required"


def test_existing_canonical_update_blocks_auto_run_before_child(context):
    lock = update_lock.UpdateLock(install_root=context.install, path=context.home / "canonical-marker")
    assert lock.acquire()
    try:
        args = _args("run-now")
        with pytest.raises(SystemExit) as raised:
            args.func(args)
        assert raised.value.code == 1
    finally:
        lock.release()
    assert not context.status_path.exists()


def _mutex_busy(context):
    try:
        with state.operation_lock(context):
            return False
    except update_lock.MarkerBusy:
        return True


def test_parent_holds_operation_mutex_but_not_child_checkout_lock(context, monkeypatch):
    observed = []

    def run(ctx, status, args):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(_mutex_busy, ctx).result(timeout=5)
        assert not update_lock.checkout_lock_held(ctx.install)
        observed.append(args.branch)
        status["status"] = "up_to_date"
        return 0

    monkeypatch.setattr(auto, "run_update", run)
    args = _args("run-now", "--branch", "main")
    args.func(args)
    assert observed == ["main"]


@pytest.mark.parametrize("profile", ["default", "work"])
def test_scheduler_command_resists_changed_sticky_profile(context, monkeypatch, profile):
    from hermes_cli import main

    other = context.home / "profiles" / "other"
    work = context.home / "profiles" / "work"
    for home in (other, work):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n")
    (context.home / "active_profile").write_text("other")
    home = context.home if profile == "default" else work
    selected = state.AutoUpdateContext(context.install, home, context.receipt_directory)
    argv = runner.command(selected, ["update", "auto", "run-scheduled"])
    start = argv.index("--profile")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sys, "argv", ["hermes", *argv[start:]])
    main._apply_profile_override()
    assert Path(os.environ["HERMES_HOME"]) == context.home
    assert sys.argv[1:] == ["update", "auto", "run-scheduled"]


def test_parsed_run_now_forwards_safe_child_flags_and_owned_home(context, monkeypatch):
    import json

    monkeypatch.setattr(auto, "run_update", runner.run_update)
    monkeypatch.setattr(runner, "require_source_install", lambda ctx: None)
    children = []

    class Child:
        def __init__(self, argv, **kwargs):
            children.append((argv, kwargs))
            self.environment = kwargs["env"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            context.receipt_directory.mkdir(parents=True, exist_ok=True)
            receipt = {"schema": 1, "correlation_id": self.environment["HERMES_UPDATE_CORRELATION_ID"],
                       "finished_at": state.utc_now(), "outcome": "success",
                       "pre_update": {"sha": "a" * 40}, "post_update": {"sha": "b" * 40}}
            (context.receipt_directory / "update_fixture.json").write_text(json.dumps(receipt))
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    args = _args("run-now", "--channel", "preview")
    args.func(args)
    assert len(children) == 1
    argv, kwargs = children[0]
    assert argv[argv.index("--profile"):] == ["--profile", "default", "update", "--yes", "--require-backup", "--channel", "preview"]
    assert kwargs["cwd"] == context.install
    assert kwargs["env"]["HERMES_HOME"] == str(context.home)
    saved = state.read_status(context)
    assert saved["status"] == "success"
    assert saved["terminalReceipt"]["correlation_id"] == kwargs["env"]["HERMES_UPDATE_CORRELATION_ID"]


@pytest.mark.parametrize("subcommand", ["status", "disable", "run-scheduled", "enable"])
@pytest.mark.parametrize("option", ["--branch", "--channel"])
def test_scheduled_activation_rejects_parent_target_override(context, subcommand, option):
    options = ["--time", "04:00"] if subcommand == "enable" else []
    args = _parser().parse_args(["update", option, "main", "auto", subcommand, *options])
    with pytest.raises(SystemExit) as raised:
        args.func(args)
    assert raised.value.code == 1
    assert not context.status_path.exists()


@pytest.mark.parametrize("schedule,plan", [("4:00", []), ("24:00", []), ("04:00", ["04:00"]),
                                           ("04:00", ["21:60"])])
def test_bad_schedule_cannot_reach_scheduler_backend(context, schedule, plan):
    options = ["enable", "--time", schedule]
    for time in plan:
        options.extend(["--plan-time", time])
    args = _args(*options)
    with pytest.raises(SystemExit) as raised:
        args.func(args)
    assert raised.value.code == 1
    assert not context.status_path.exists()
