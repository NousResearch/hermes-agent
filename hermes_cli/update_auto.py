"""Opt-in, non-agentic scheduling around the existing update command.

Recovered from #33514 (George Andraws) and #56787 (2001Y). The current
transactional updater owns installation changes, backups and fleet recovery.
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext

from hermes_cli import update_auto_schedule as scheduler
from hermes_cli.update_auto_run import check_update, command, reconcile_run, require_source_install, run_update
from hermes_cli.update_auto_precheck import recovery_needed
from hermes_cli.update_auto_state import (
    AutoUpdateContext, append_log, operation_lock, read_status, utc_now, write_status,
)
from hermes_cli.update_lock import update_in_progress
from hermes_cli.update_auto_migrate import (
    check_legacy_channels, legacy_operation_locks, legacy_schedules, legacy_summary, migration_times,
    replace_legacy_schedulers,
)


def _spec(context: AutoUpdateContext, schedule: str, plan_times) -> scheduler.SchedulerSpec:
    return scheduler.SchedulerSpec(
        identity=context.identity, command=command(context, ["update", "auto", "run-scheduled",
                                                            "--scheduler-identity", context.identity]),
        home=context.home, schedule=schedule, plan_times=plan_times,
        log_directory=context.log_path.parent,
    )


def _configured_spec(context: AutoUpdateContext, status: dict) -> scheduler.SchedulerSpec:
    if status.get("mode") != "scheduled" or not isinstance(status.get("planSchedule"), list):
        raise ValueError("Invalid persisted scheduler mode or planSchedule")
    if status.get("status") == "recovery_required":
        raise ValueError("Scheduler recovery is required; inspect the saved recovery receipt")
    spec = _spec(context, status.get("schedule"), status["planSchedule"])
    actual = scheduler.paths(spec)
    if status.get("schedulerType") != actual["backend"] or status.get("schedulerPath") != str(actual["path"]):
        raise ValueError("Persisted scheduler does not match this installation")
    return spec


def _save_scheduler(context: AutoUpdateContext, status: dict, handle, fields: dict) -> None:
    updated = {**status, **fields}
    try:
        write_status(context, updated)
    except Exception as exc:
        receipt = handle.rollback()
        if not receipt.get("ok"):
            raise scheduler.SchedulerRecoveryError(
                f"Status write failed ({exc}); scheduler rollback also failed", receipt,
            ) from exc
        raise


def _enable(context: AutoUpdateContext, status: dict, args) -> int:
    from hermes_cli.update_auto_schedule_common import snapshot_file
    from hermes_cli.update_installation import migrate_install_channel

    require_source_install(context)
    if status["enabled"]:
        _configured_spec(context, status)
    spec = _spec(context, args.time, args.plan_time)
    legacy = legacy_schedules(context)
    check_legacy_channels(context, legacy)
    if not status["enabled"] and any(snapshot_file(path) is not None for key, path in scheduler.paths(spec).items()
                                     if key in {"path", "service_path"}):
        raise ValueError("An installation scheduler exists without its saved activation; inspect its owning data root before enabling")
    channel = migrate_install_channel(context.install, home=context.home)
    try:
        if legacy:
            replace_legacy_schedulers(context, status, legacy, spec)
        else:
            handle = scheduler.enable(spec)
            _save_scheduler(context, status, handle, {
                "enabled": True, "mode": "scheduled", "schedule": spec.schedule,
                "planSchedule": list(spec.plan_times), "schedulerType": handle.scheduler_type,
                "schedulerPath": str(handle.path), "error": None,
            })
    except (OSError, ValueError, RuntimeError) as exc:
        receipt = channel.rollback()
        if not receipt["ok"]:
            raise scheduler.SchedulerRecoveryError(f"Channel migration could not be restored: {exc}", receipt) from exc
        raise
    print(f"Auto-update enabled at {spec.schedule} local time.")
    if spec.plan_times:
        print(f"Check-only plan time(s): {', '.join(spec.plan_times)}")
    print(f"Scheduler: {scheduler.paths(spec)['path']}")
    return 0


def _disable(context: AutoUpdateContext, status: dict, _args) -> int:
    legacy = legacy_schedules(context)
    if legacy:
        replace_legacy_schedulers(context, status, legacy, None)
        print("Auto-update disabled for this installation, including its legacy profile schedules.")
        return 0
    if not status["enabled"]:
        print("Auto-update is disabled.")
        return 0
    handle = scheduler.disable(_configured_spec(context, status))
    _save_scheduler(context, status, handle, {
        "enabled": False, "mode": "manual", "schedule": None, "planSchedule": [],
        "schedulerType": None, "schedulerPath": None, "error": None,
    })
    print("Auto-update disabled.")
    return 0


def _migrate(context: AutoUpdateContext, status: dict, _args) -> int:
    from types import SimpleNamespace

    legacy = legacy_schedules(context)
    if not legacy:
        print("No legacy profile schedules need migration.")
        return 0
    time, plans = migration_times(legacy)
    if status["enabled"]:
        current = _configured_spec(context, status)
        if (current.schedule, sorted(current.plan_times)) != (time, sorted(plans)):
            raise ValueError("Legacy and installation schedules differ; choose one with hermes update auto enable --time HH:MM")
    return _enable(context, status, SimpleNamespace(time=time, plan_time=plans))


def _plan(context: AutoUpdateContext, status: dict, args) -> int:
    try:
        result = check_update(context, args)
    except Exception as exc:
        status.update(status="check_failed", lastPlanAt=utc_now(), error=str(exc))
        write_status(context, status)
        append_log(context, "plan", result="check_failed", error=str(exc))
        raise
    verdict = "planned" if result["updateAvailable"] else "up_to_date"
    status.update(status=verdict, lastPlanAt=utc_now(), plannedCheck=result, error=None)
    write_status(context, status)
    append_log(context, "plan", result=verdict, targetSha=result.get("targetSha"))
    if result["updateAvailable"]:
        print(f"Hermes update available: {result.get('currentSha')} → {result.get('targetSha')}")
        print(f"Scheduled time: {status.get('schedule') or 'not configured'}")
        print("Advisory check only; the updater resolves the selected channel again when it runs.")
    else:
        print("Hermes is up to date.")
    return 0


def _run(context: AutoUpdateContext, status: dict, args) -> int:
    code = run_update(context, status, args)
    print(f"Auto-update: {status['status']}. Log: {context.log_path}")
    if status.get("receiptPath"):
        print(f"Receipt: {status['receiptPath']}")
    if status.get("error"):
        print(status["error"], file=sys.stderr)
    return code


def _scheduled(context: AutoUpdateContext, status: dict, args) -> int:
    if getattr(args, "scheduler_identity", None) != context.identity:
        raise ValueError("This timer has no matching installation activation. Run hermes update auto migrate; "
                         "legacy timer commands cannot start the installation scheduler.")
    if legacy_schedules(context):
        raise ValueError("Legacy profile schedules still exist; run hermes update auto migrate before unattended updates")
    if not status["enabled"]:
        return 0
    spec = _configured_spec(context, status)
    if status.get("status") == "running":
        raise ValueError("An earlier auto-update has no recorded terminal result; inspect its receipt before running again")
    # Timer argv cannot select a new target or bypass the saved activation.
    from types import SimpleNamespace

    selected = SimpleNamespace(branch=None, channel=None)
    action = scheduler.scheduled_action(spec.schedule, spec.plan_times)
    if action == "plan":
        return _plan(context, status, selected)
    result = _scheduled_check(context, status, selected)
    if result["updateAvailable"] or recovery_needed(context, result.get("currentSha")):
        return _run(context, status, selected)
    checked_at = utc_now()
    status.update(status="up_to_date", lastRunAt=checked_at, finishedAt=checked_at,
                  lastCheck=result, outcomeSource="availability_check", error=None,
                  runPending=False, correlationId=None, receiptPath=None, terminalReceipt=None, exitCode=0)
    write_status(context, status)
    append_log(context, "check", result="up_to_date", currentSha=result.get("currentSha"))
    print("Hermes is up to date; no updater or backup was started.")
    return 0


def _scheduled_check(context: AutoUpdateContext, status: dict, args) -> dict:
    try:
        return check_update(context, args)
    except (OSError, ValueError, RuntimeError) as exc:
        status.update(status="check_failed", lastRunAt=utc_now(), error=str(exc), outcomeSource="availability_check")
        write_status(context, status)
        append_log(context, "check", result="check_failed", error=str(exc))
        raise


_HANDLERS = {"enable": _enable, "disable": _disable, "migrate": _migrate, "plan": _plan,
             "run-now": _run, "run-scheduled": _scheduled}


def _validate_options(args) -> None:
    unsupported = ("no_backup", "keep_stash", "force", "force_venv", "switch_branch",
                   "set_channel", "no_gateway_restart", "gateway", "check", "plan", "install_id",
                   "list_venv_holders")
    if any(getattr(args, option, False) for option in unsupported):
        raise ValueError("Auto-update does not accept manual updater override flags")
    if args.auto_subcommand not in {"plan", "run-now"} and (
        getattr(args, "branch", None) or getattr(args, "channel", None)
    ):
        raise ValueError("Use the install's saved update channel for scheduled updates; target overrides apply to plan/run-now only")


def cmd_update_auto(args) -> None:
    try:
        _validate_options(args)
        context = AutoUpdateContext.current()
        if args.auto_subcommand == "status":
            status = read_status(context)
            legacy = legacy_schedules(context)
            if legacy:
                status.update(legacySchedules=legacy_summary(legacy), migrationRequired=True)
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return
        if args.auto_subcommand in {"enable", "migrate", "plan", "run-now"}:
            from hermes_cli.update_installation_owner import ensure_installation_home

            require_source_install(context)
            home = ensure_installation_home(context.install, home=context.home)
            context = AutoUpdateContext(context.install, home, context.receipt_directory)
        initial = read_status(context)
        if initial.get("status") in {"recovery_required", "migration_running"}:
            raise ValueError("Scheduler recovery is required; inspect the saved recovery receipt before changing it")
        legacy = legacy_schedules(context)
        if args.auto_subcommand == "run-scheduled" and legacy:
            raise ValueError("Legacy profile schedules must be consolidated before another unattended update. "
                             "Run hermes update auto migrate from a terminal when the timer has finished.")
        if args.auto_subcommand in {"run-scheduled", "disable"} and not initial["enabled"] and not legacy:
            if args.auto_subcommand == "disable":
                print("Auto-update is disabled.")
            return
        with operation_lock(context):
            status = read_status(context)
            if status.get("status") in {"recovery_required", "migration_running"}:
                raise ValueError("Scheduler recovery is required; inspect the saved recovery receipt")
            if update_in_progress(context.install):
                raise ValueError("Another Hermes update is active; auto-update operation refused")
            pending = reconcile_run(context, status)
            if pending and args.auto_subcommand not in {"run-now", "disable"}:
                raise ValueError("Earlier update outcome is unverified; inspect its log/receipt, then explicitly use run-now to retry")
            try:
                legacy_lock = legacy_operation_locks(context) if args.auto_subcommand in {"enable", "disable", "migrate"} else nullcontext()
                with legacy_lock:
                    code = _HANDLERS[args.auto_subcommand](context, status, args)
            except scheduler.SchedulerRecoveryError as exc:
                status.update(status="recovery_required", error=str(exc), recoveryReceipt=exc.receipt)
                try:
                    write_status(context, status)
                except (OSError, ValueError) as write_error:
                    print(f"Could not save recovery receipt: {write_error}", file=sys.stderr)
                raise
    except scheduler.SchedulerRecoveryError as exc:
        print(f"Auto-update scheduler needs recovery: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Auto-update stopped: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if code:
        raise SystemExit(code)
