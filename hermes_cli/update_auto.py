"""Opt-in, non-agentic scheduling around the existing update command.

Recovered from #33514 (George Andraws) and #56787 (2001Y). The current
transactional updater owns installation changes, backups and fleet recovery.
"""

from __future__ import annotations

import json
import sys

from hermes_cli import update_auto_schedule as scheduler
from hermes_cli.update_auto_run import check_update, command, reconcile_run, require_source_install, run_update
from hermes_cli.update_auto_state import (
    AutoUpdateContext, append_log, operation_lock, read_status, utc_now, write_status,
)
from hermes_cli.update_lock import update_in_progress


def _spec(context: AutoUpdateContext, schedule: str, plan_times) -> scheduler.SchedulerSpec:
    return scheduler.SchedulerSpec(
        identity=context.identity, command=command(context, ["update", "auto", "run-scheduled"]),
        home=context.home, schedule=schedule, plan_times=plan_times,
    )


def _configured_spec(context: AutoUpdateContext, status: dict) -> scheduler.SchedulerSpec:
    if status.get("mode") != "scheduled" or not isinstance(status.get("planSchedule"), list):
        raise ValueError("Invalid persisted scheduler mode or planSchedule")
    if status.get("status") == "recovery_required":
        raise ValueError("Scheduler recovery is required; inspect the saved recovery receipt")
    spec = _spec(context, status.get("schedule"), status["planSchedule"])
    actual = scheduler.paths(spec)
    if status.get("schedulerType") != actual["backend"] or status.get("schedulerPath") != str(actual["path"]):
        raise ValueError("Persisted scheduler does not match this installation/profile")
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
    require_source_install(context)
    if status["enabled"]:
        _configured_spec(context, status)
    spec = _spec(context, args.time, args.plan_time)
    handle = scheduler.enable(spec)
    _save_scheduler(context, status, handle, {
        "enabled": True, "mode": "scheduled", "schedule": spec.schedule,
        "planSchedule": list(spec.plan_times), "schedulerType": handle.scheduler_type,
        "schedulerPath": str(handle.path), "error": None,
    })
    print(f"Auto-update enabled at {spec.schedule} local time.")
    if spec.plan_times:
        print(f"Check-only plan time(s): {', '.join(spec.plan_times)}")
    print(f"Scheduler: {handle.path}")
    return 0


def _disable(context: AutoUpdateContext, status: dict, _args) -> int:
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
    if not status["enabled"]:
        return 0
    spec = _configured_spec(context, status)
    if status.get("status") == "running":
        raise ValueError("An earlier auto-update has no recorded terminal result; inspect its receipt before running again")
    # Timer argv cannot select a new target or bypass the saved activation.
    from types import SimpleNamespace

    selected = SimpleNamespace(branch=None, channel=None)
    action = scheduler.scheduled_action(spec.schedule, spec.plan_times)
    return _plan(context, status, selected) if action == "plan" else _run(context, status, selected)


_HANDLERS = {"enable": _enable, "disable": _disable, "plan": _plan,
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
    context = AutoUpdateContext.current()
    try:
        _validate_options(args)
        if args.auto_subcommand == "status":
            print(json.dumps(read_status(context), indent=2, ensure_ascii=False))
            return
        initial = read_status(context)
        if initial.get("status") == "recovery_required":
            raise ValueError("Scheduler recovery is required; inspect the saved recovery receipt before changing it")
        if args.auto_subcommand in {"run-scheduled", "disable"} and not initial["enabled"]:
            if args.auto_subcommand == "disable":
                print("Auto-update is disabled.")
            return
        with operation_lock(context):
            status = read_status(context)
            if status.get("status") == "recovery_required":
                raise ValueError("Scheduler recovery is required; inspect the saved recovery receipt")
            if update_in_progress(context.install):
                raise ValueError("Another Hermes update is active; auto-update operation refused")
            pending = reconcile_run(context, status)
            if pending and args.auto_subcommand not in {"run-now", "disable"}:
                raise ValueError("Earlier update outcome is unverified; inspect its log/receipt, then explicitly use run-now to retry")
            try:
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
