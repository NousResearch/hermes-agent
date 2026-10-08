"""Fresh-process bridge to the canonical transactional updater.

The scheduler neither replaces the update lock nor diagnoses gateway health.
Only a terminal receipt from this exact invocation can certify its outcome.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from hermes_cli._launchers import installation_command
from hermes_cli.update_auto_state import AutoUpdateContext, append_log, utc_now, write_status


def command(context: AutoUpdateContext, arguments: list[str]) -> list[str]:
    # Explicit default prevents a later sticky `profile use` from retargeting a timer.
    return installation_command(context.install, ["--profile", "default", *arguments], home=context.home)


def require_source_install(context: AutoUpdateContext) -> None:
    from hermes_cli.config import is_managed
    from hermes_cli.update_contract import evaluate_update_admission

    if is_managed():
        raise ValueError("Managed installs cannot use automatic source updates")
    refusal = evaluate_update_admission(context.install)
    if refusal is not None:
        raise ValueError(refusal.message)
    if not (context.install / ".git").exists():
        raise ValueError("Automatic updates require a self-managed source checkout")


def check_update(context: AutoUpdateContext, args) -> dict:
    from hermes_cli.source_check import check_for_updates
    from hermes_cli.source_releases import resolve_source_target
    from hermes_cli.update_installation import resolve_install_channel

    require_source_install(context)
    branch = getattr(args, "branch", None)
    channel = "main" if branch else (getattr(args, "channel", None) or
                                     resolve_install_channel(context.install, home=context.home))
    if not branch:
        target = resolve_source_target(channel, ["git"], context.install)
        branch = target.branch
    # Passive UI checks follow the checked-out branch; automation follows the
    # same effective channel/branch as `hermes update` instead.
    result = check_for_updates(install_root=context.install, home=context.home,
                               branch=branch, channel=channel, force=True)
    if result.get("supported") is not True or result.get("error"):
        raise ValueError(result.get("message") or result.get("reason") or "Update check failed")
    if not isinstance(result.get("updateAvailable"), bool):
        raise ValueError("Update check returned no verified availability verdict")
    if branch is not None and result.get("branch") != branch:
        raise ValueError(f"Update check could not verify the requested branch {branch!r}; refusing a fallback target")
    return result


def find_receipt(directory: Path, correlation: str) -> tuple[Path, dict] | None:
    matches = []
    for path in directory.glob("update_*.json"):
        if path.is_symlink():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, ValueError):
            continue
        if isinstance(data, dict) and data.get("correlation_id") == correlation and data.get("finished_at"):
            matches.append((path, data))
    # More than one terminal run claiming our random id is ambiguous, never latest-wins.
    return matches[0] if len(matches) == 1 else None


def receipt_result(receipt: dict, returncode: int) -> tuple[str, int]:
    if returncode == 11:
        return "backup_failed", 11
    outcome = receipt.get("outcome")
    if outcome == "success":
        if (receipt.get("followups") or receipt.get("user_action") or returncode != 0
                or receipt.get("exit_code") not in (None, 0)):
            return "followup_required", 14
        before = (receipt.get("pre_update") or {}).get("sha")
        after = (receipt.get("post_update") or {}).get("sha")
        return ("up_to_date" if before and before == after else "success"), 0
    if outcome == "partial":
        return "followup_required", 14
    return "update_failed", returncode if returncode > 0 else 12


def reconcile_run(context: AutoUpdateContext, status: dict) -> bool:
    """Settle a parent interrupted after the updater finished; never guess from latest.json."""
    if not status.get("runPending") and status.get("status") != "running":
        return False
    correlation = status.get("correlationId")
    found = find_receipt(context.receipt_directory, correlation) if correlation else None
    if found is None:
        return True
    path, receipt = found
    exit_code = status.get("exitCode")
    if not isinstance(exit_code, int):
        exit_code = receipt.get("exit_code")
    exit_code = exit_code if isinstance(exit_code, int) else 0
    verdict, _code = receipt_result(receipt, exit_code)
    status.update(status=verdict, terminalReceipt=receipt, receiptPath=str(path),
                  runPending=False, finishedAt=receipt["finished_at"], exitCode=exit_code, error=None,
                  outcomeSource="updater_receipt")
    write_status(context, status)
    return False


def _invoke(context: AutoUpdateContext, argv: list[str], environment: dict, correlation: str) -> int:
    append_log(context, "start", correlationId=correlation)
    # Open parent-owned output before mutation. No fresh imports after the child runs.
    with context.log_path.open("a", encoding="utf-8") as log:
        # Stage watchdogs belong to the updater. A parent wall-clock kill can
        # interrupt a committed update while its completion child lives.
        with subprocess.Popen(argv, cwd=context.install, env=environment,
                              stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT) as child:
            return child.wait()  # health: allow HX006 -- updater owns stage watchdogs; a parent kill can strand committed completion


def run_update(context: AutoUpdateContext, status: dict, args) -> int:
    require_source_install(context)
    correlation = uuid.uuid4().hex
    arguments = ["update", "--yes", "--require-backup"]
    for option in ("branch", "channel"):
        value = getattr(args, option, None)
        if value:
            arguments.extend([f"--{option}", value])
    argv = command(context, arguments)
    environment = os.environ.copy()
    environment["HERMES_HOME"] = str(context.home)
    environment["HERMES_UPDATE_CORRELATION_ID"] = correlation
    status.update(status="running", lastRunAt=utc_now(), correlationId=correlation, runPending=True,
                  error=None, receiptPath=None, terminalReceipt=None, exitCode=None, finishedAt=None,
                  outcomeSource="updater")
    write_status(context, status)
    try:
        returncode = _invoke(context, argv, environment, correlation)
    except (OSError, ValueError) as exc:
        status.update(status="update_failed", error=str(exc), runPending=False, finishedAt=utc_now(),
                      outcomeSource="launch_failure")
        write_status(context, status)
        return 12
    found = find_receipt(context.receipt_directory, correlation)
    if found is None:
        status.update(status="unverified", error=f"Updater exited {returncode} without a unique terminal receipt",
                      outcomeSource="process_exit")
        code = 12
    else:
        receipt_path, receipt = found
        verdict, code = receipt_result(receipt, returncode)
        status.update(status=verdict, terminalReceipt=receipt, receiptPath=str(receipt_path), runPending=False,
                      error=receipt.get("stop_reason") if code else None, outcomeSource="updater_receipt")
    status.update(finishedAt=utc_now(), exitCode=returncode)
    write_status(context, status)
    try:
        append_log(context, "end", result=status["status"], correlationId=correlation,
                   receiptPath=status.get("receiptPath"), exitCode=returncode)
    except (OSError, ValueError) as exc:
        print(f"Could not append the final update log entry: {exc}; terminal status was saved", file=sys.stderr)
    return code
