"""Auto-update scaffolding for ``hermes update auto``.

Scheduling is intentionally a thin wrapper around ``hermes update auto
run-scheduled`` / ``run-now``. This module records stable status, writes an
append-only human log, and delegates the real update work back to the existing
``hermes update`` implementation.
"""

from __future__ import annotations

import json
import os
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hermes_constants import get_hermes_home


STATUS_FILENAME = "update-status.json"
LOG_FILENAME = "update.log"
LAUNCHD_LABEL = "com.hermes.agent.auto-update"
SYSTEMD_BASENAME = "hermes-auto-update"

STATUS_NOT_CONFIGURED = "not_configured"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_UP_TO_DATE = "up_to_date"
STATUS_PLANNED = "planned"
STATUS_CHECK_FAILED = "check_failed"
STATUS_BACKUP_FAILED = "backup_failed"
STATUS_UPDATE_FAILED = "update_failed"
STATUS_HEALTH_FAILED = "health_failed"

EXIT_CHECK_FAILED = 10
EXIT_BACKUP_FAILED = 11
EXIT_UPDATE_FAILED = 12
EXIT_HEALTH_FAILED = 13

DEFAULT_STATUS: dict[str, Any] = {
    "mode": "manual",
    "enabled": False,
    "schedule": None,
    "planSchedule": [],
    "schedulerType": None,
    "schedulerPath": None,
    "lastRunAt": None,
    "lastPlanAt": None,
    "status": STATUS_NOT_CONFIGURED,
    "previousVersion": None,
    "latestVersion": None,
    "plannedVersion": None,
    "currentVersion": None,
    "backupPath": None,
    "error": None,
    "logPath": None,
}


class AutoUpdateError(RuntimeError):
    def __init__(self, status: str, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.status = status
        self.exit_code = exit_code


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_status_path() -> Path:
    return get_hermes_home() / "state" / STATUS_FILENAME


def get_log_path() -> Path:
    return get_hermes_home() / "logs" / LOG_FILENAME


def get_stdout_log_path() -> Path:
    return get_hermes_home() / "logs" / "update-auto.out.log"


def get_stderr_log_path() -> Path:
    return get_hermes_home() / "logs" / "update-auto.err.log"


def read_status() -> dict[str, Any]:
    path = get_status_path()
    if not path.exists():
        return dict(DEFAULT_STATUS, logPath=str(get_log_path()))
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_STATUS, logPath=str(get_log_path()))
    if not isinstance(data, dict):
        return dict(DEFAULT_STATUS, logPath=str(get_log_path()))
    merged = dict(DEFAULT_STATUS)
    for key in DEFAULT_STATUS:
        if key in data:
            merged[key] = data[key]
    if not merged.get("logPath"):
        merged["logPath"] = str(get_log_path())
    return merged


def write_status(status: dict[str, Any]) -> Path:
    path = get_status_path()
    payload = dict(DEFAULT_STATUS)
    payload.update(status)
    payload["mode"] = payload.get("mode") or "manual"
    payload["enabled"] = bool(payload.get("enabled", False))
    if not isinstance(payload.get("planSchedule"), list):
        payload["planSchedule"] = []
    if not payload.get("logPath"):
        payload["logPath"] = str(get_log_path())

    from utils import atomic_json_write

    atomic_json_write(path, payload, indent=2, sort_keys=True)
    return path


def update_status_fields(**fields: Any) -> Path:
    status = read_status()
    status.update(fields)
    return write_status(status)


def append_log(
    event: str,
    *,
    result: str | None = None,
    previous_version: str | None = None,
    latest_version: str | None = None,
    current_version: str | None = None,
    backup_path: str | None = None,
    error: str | None = None,
) -> Path:
    path = get_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "event": event,
        "result": result,
        "previous": previous_version,
        "latest": latest_version,
        "current": current_version,
        "backup": backup_path,
        "error": error,
    }
    parts = [f"{key}={value}" for key, value in fields.items() if value is not None]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc_now()} {' '.join(parts)}\n")
    return path


def _current_version() -> str | None:
    from hermes_cli.main import PROJECT_ROOT
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        try:
            from hermes_cli import __version__

            return __version__
        except Exception:
            return None


def _verify_health() -> tuple[bool, str]:
    """Return a lightweight post-update health verdict.

    A full ``hermes doctor`` run can perform provider/network checks and may be
    slow or flaky in unattended contexts. Phase 1 uses the existing gateway
    runtime status file instead: it is local, non-destructive, and fails only
    when the gateway explicitly reports a failed terminal state.
    """
    try:
        from gateway.status import read_runtime_status
    except Exception as exc:
        return False, f"could not import gateway status helpers: {exc}"

    runtime = read_runtime_status()
    if not runtime:
        return True, "no gateway runtime status present"

    state = str(runtime.get("gateway_state") or "")
    if state in {"startup_failed", "failed", "stopped"}:
        reason = runtime.get("exit_reason") or "gateway reported unhealthy state"
        return False, str(reason)
    return True, f"gateway state: {state or 'unknown'}"


def _status_payload(
    *,
    status: str,
    last_run_at: str,
    previous_version: str | None = None,
    latest_version: str | None = None,
    current_version: str | None = None,
    backup_path: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    existing = read_status()
    return {
        "mode": existing.get("mode") or "manual",
        "enabled": bool(existing.get("enabled", False)),
        "schedule": existing.get("schedule"),
        "planSchedule": existing.get("planSchedule") or [],
        "schedulerType": existing.get("schedulerType"),
        "schedulerPath": existing.get("schedulerPath"),
        "lastRunAt": last_run_at,
        "lastPlanAt": existing.get("lastPlanAt"),
        "status": status,
        "previousVersion": previous_version,
        "latestVersion": latest_version,
        "plannedVersion": existing.get("plannedVersion"),
        "currentVersion": current_version,
        "backupPath": backup_path,
        "error": error,
        "logPath": str(get_log_path()),
    }


def _display(value: Any, default: str = "-") -> str:
    if value is None or value == "":
        return default
    return str(value)


def cmd_auto_status(_args) -> None:
    status = read_status()
    enabled = bool(status.get("enabled"))
    print("Hermes auto-update status")
    print("  Phase:            2 (scheduled wrapper around run-now)")
    print(f"  Mode:             {_display(status.get('mode'), 'manual')}")
    print(f"  Enabled:          {'yes' if enabled else 'no'}")
    print(f"  Schedule:         {_display(status.get('schedule'), 'not configured')}")
    plan_schedule = status.get("planSchedule") or []
    plan_display = ", ".join(str(item) for item in plan_schedule) if plan_schedule else "not configured"
    print(f"  Plan schedule:    {plan_display}")
    print(f"  Scheduler:        {_display(status.get('schedulerType'), 'not configured')}")
    print(f"  Scheduler path:   {_display(status.get('schedulerPath'))}")
    print(f"  Last run:         {_display(status.get('lastRunAt'), 'never')}")
    print(f"  Last plan:        {_display(status.get('lastPlanAt'), 'never')}")
    print(f"  Last result:      {_display(status.get('status'), 'unknown')}")
    print(f"  Previous version: {_display(status.get('previousVersion'))}")
    print(f"  Latest known:     {_display(status.get('latestVersion'))}")
    print(f"  Planned version:  {_display(status.get('plannedVersion'))}")
    print(f"  Current version:  {_display(status.get('currentVersion') or _current_version())}")
    print(f"  Backup path:      {_display(status.get('backupPath'))}")
    print(f"  Last error:       {_display(status.get('error'))}")
    print(f"  Detailed log:     {_display(status.get('logPath') or get_log_path())}")


def _run_existing_update(args, branch: str | None) -> None:
    from hermes_cli.main import cmd_update

    update_args = SimpleNamespace(
        gateway=False,
        check=False,
        no_backup=True,
        backup=False,
        yes=True,
        branch=branch,
        force=getattr(args, "force", False),
    )
    try:
        cmd_update(update_args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            raise AutoUpdateError(
                STATUS_UPDATE_FAILED,
                f"update failed with exit code {code}",
                EXIT_UPDATE_FAILED,
            ) from exc


def _parse_time(value: str) -> tuple[int, int, str]:
    raw = (value or "").strip()
    if not re.fullmatch(r"\d{2}:\d{2}", raw):
        raise ValueError("time must use HH:MM format, for example 03:00")
    hour_s, minute_s = raw.split(":", 1)
    hour = int(hour_s)
    minute = int(minute_s)
    if hour > 23 or minute > 59:
        raise ValueError("time must be a valid 24-hour HH:MM value")
    return hour, minute, f"{hour:02d}:{minute:02d}"


def _hermes_command_prefix() -> list[str]:
    argv0 = Path(sys.argv[0])
    if argv0.name and not argv0.name.startswith("python"):
        resolved = shutil.which(str(argv0)) if not argv0.is_absolute() else str(argv0)
        if resolved:
            return [resolved]

    hermes = shutil.which("hermes")
    if hermes:
        return [hermes]

    from hermes_cli.main import PROJECT_ROOT

    return [sys.executable, str(PROJECT_ROOT / "hermes_cli" / "main.py")]


def _scheduled_command() -> list[str]:
    return _hermes_command_prefix() + ["update", "auto", "run-scheduled"]


def _launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def _launchd_target() -> str:
    return f"gui/{os.getuid()}"


def _calendar_intervals(update_schedule: str, plan_schedules: list[str]) -> list[dict[str, int]]:
    intervals: list[dict[str, int]] = []
    for schedule in [*plan_schedules, update_schedule]:
        hour, minute, _normalized = _parse_time(schedule)
        interval = {"Hour": hour, "Minute": minute}
        if interval not in intervals:
            intervals.append(interval)
    return intervals


def _run_launchctl(args: list[str]) -> subprocess.CompletedProcess:
    cmd = ["launchctl"] + args
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(exc))


def _enable_launchd(hour: int, minute: int, schedule: str, plan_schedules: list[str]) -> tuple[str, Path]:
    plist_path = _launchd_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    get_log_path().parent.mkdir(parents=True, exist_ok=True)

    intervals = _calendar_intervals(schedule, plan_schedules)
    start_calendar_interval: dict[str, int] | list[dict[str, int]] = (
        intervals[0] if len(intervals) == 1 else intervals
    )
    payload = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": _scheduled_command(),
        "StartCalendarInterval": start_calendar_interval,
        "EnvironmentVariables": {"HERMES_HOME": str(get_hermes_home())},
        "StandardOutPath": str(get_stdout_log_path()),
        "StandardErrorPath": str(get_stderr_log_path()),
        "RunAtLoad": False,
    }
    with plist_path.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)

    target = _launchd_target()
    _run_launchctl(["bootout", target, str(plist_path)])
    result = _run_launchctl(["bootstrap", target, str(plist_path)])
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or "launchctl bootstrap failed")
    _run_launchctl(["enable", f"{target}/{LAUNCHD_LABEL}"])
    return "launchd", plist_path


def _disable_launchd() -> tuple[str, Path, bool]:
    plist_path = _launchd_plist_path()
    existed = plist_path.exists()
    target = _launchd_target()
    _run_launchctl(["bootout", target, str(plist_path)])
    if existed:
        plist_path.unlink()
    return "launchd", plist_path, existed


def _systemd_user_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def _systemd_paths() -> tuple[Path, Path]:
    root = _systemd_user_dir()
    return root / f"{SYSTEMD_BASENAME}.service", root / f"{SYSTEMD_BASENAME}.timer"


def _systemctl_user(args: list[str]) -> subprocess.CompletedProcess:
    cmd = ["systemctl", "--user"] + args
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(exc))


def _systemd_available() -> bool:
    if not shutil.which("systemctl"):
        return False
    try:
        result = subprocess.run(
            ["systemctl", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _enable_systemd(hour: int, minute: int, schedule: str, plan_schedules: list[str]) -> tuple[str, Path]:
    if not _systemd_available():
        raise RuntimeError("systemd user services are not available")

    service_path, timer_path = _systemd_paths()
    service_path.parent.mkdir(parents=True, exist_ok=True)
    get_log_path().parent.mkdir(parents=True, exist_ok=True)

    command = " ".join(shlex.quote(part) for part in _scheduled_command())
    on_calendar_lines = [
        f"OnCalendar=*-*-* {item['Hour']:02d}:{item['Minute']:02d}:00"
        for item in _calendar_intervals(schedule, plan_schedules)
    ]
    service_path.write_text(
        "\n".join(
            [
                "[Unit]",
                "Description=Hermes Agent auto-update",
                "",
                "[Service]",
                "Type=oneshot",
                f"Environment=HERMES_HOME={shlex.quote(str(get_hermes_home()))}",
                f"ExecStart={command}",
                f"StandardOutput=append:{get_stdout_log_path()}",
                f"StandardError=append:{get_stderr_log_path()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    timer_path.write_text(
        "\n".join(
            [
                "[Unit]",
                "Description=Run Hermes Agent auto-update",
                "",
                "[Timer]",
                *on_calendar_lines,
                "Persistent=true",
                "",
                "[Install]",
                "WantedBy=timers.target",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _systemctl_user(["daemon-reload"])
    result = _systemctl_user(["enable", "--now", timer_path.name])
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or "systemctl --user enable --now failed")
    return "systemd-user", timer_path


def _disable_systemd() -> tuple[str, Path, bool]:
    service_path, timer_path = _systemd_paths()
    existed = service_path.exists() or timer_path.exists()
    if shutil.which("systemctl"):
        _systemctl_user(["disable", "--now", timer_path.name])
    service_path.unlink(missing_ok=True)
    timer_path.unlink(missing_ok=True)
    if shutil.which("systemctl"):
        _systemctl_user(["daemon-reload"])
    return "systemd-user", timer_path, existed


def _enable_scheduler(hour: int, minute: int, schedule: str, plan_schedules: list[str]) -> tuple[str, Path]:
    if sys.platform == "darwin":
        return _enable_launchd(hour, minute, schedule, plan_schedules)
    if sys.platform.startswith("linux"):
        return _enable_systemd(hour, minute, schedule, plan_schedules)
    raise RuntimeError(f"auto-update scheduling is not supported on {sys.platform}")


def _disable_scheduler(status: dict[str, Any]) -> tuple[str | None, Path | None, bool]:
    scheduler = status.get("schedulerType")
    if scheduler == "launchd" or sys.platform == "darwin":
        kind, path, existed = _disable_launchd()
        return kind, path, existed
    if scheduler == "systemd-user" or sys.platform.startswith("linux"):
        kind, path, existed = _disable_systemd()
        return kind, path, existed
    return None, None, False


def cmd_auto_enable(args) -> None:
    try:
        hour, minute, schedule = _parse_time(getattr(args, "time", ""))
        plan_schedules: list[str] = []
        for value in getattr(args, "plan_time", []) or []:
            _plan_hour, _plan_minute, plan_schedule = _parse_time(value)
            if plan_schedule not in plan_schedules:
                plan_schedules.append(plan_schedule)
        scheduler_type, scheduler_path = _enable_scheduler(
            hour,
            minute,
            schedule,
            plan_schedules,
        )
    except ValueError as exc:
        print(f"✗ Invalid schedule time: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except Exception as exc:
        print(f"✗ Could not enable auto-update scheduler: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    update_status_fields(
        mode="scheduled",
        enabled=True,
        schedule=schedule,
        planSchedule=plan_schedules,
        schedulerType=scheduler_type,
        schedulerPath=str(scheduler_path),
        logPath=str(get_log_path()),
    )
    print("✓ Hermes auto-update scheduled.")
    print(f"  Update time: {schedule}")
    if plan_schedules:
        print(f"  Plan time:   {', '.join(plan_schedules)}")
    print(f"  Scheduler:   {scheduler_type}")
    print(f"  Path:        {scheduler_path}")
    print("  Command:     hermes update auto run-scheduled")


def cmd_auto_disable(_args) -> None:
    status = read_status()
    try:
        scheduler_type, scheduler_path, removed = _disable_scheduler(status)
    except Exception as exc:
        print(f"✗ Could not disable auto-update scheduler: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    was_enabled = bool(status.get("enabled"))

    update_status_fields(
        mode="manual",
        enabled=False,
        schedule=None,
        planSchedule=[],
        schedulerType=None,
        schedulerPath=None,
        logPath=str(get_log_path()),
    )
    if removed or was_enabled:
        print("✓ Hermes auto-update disabled.")
        if scheduler_path:
            print(f"  Removed: {scheduler_path}" if removed else f"  Scheduler path: {scheduler_path}")
    else:
        print("Hermes auto-update is already disabled.")


def _minutes_for_schedule(schedule: str) -> int:
    hour, minute, _normalized = _parse_time(schedule)
    return hour * 60 + minute


def _scheduled_action_for_now(status: dict[str, Any], now: datetime | None = None) -> str:
    """Return ``plan`` or ``run`` for the most recent configured schedule.

    launchd/systemd can fire a persistent timer a little late after sleep or
    login. Choosing the most recent configured time lets a late 21:00 plan still
    emit a plan notice, while a late 04:00 update still applies the update.
    """
    now = now or datetime.now()
    now_minutes = now.hour * 60 + now.minute
    candidates: list[tuple[int, str]] = []
    update_schedule = status.get("schedule")
    if update_schedule:
        candidates.append((_minutes_for_schedule(str(update_schedule)), "run"))
    for plan_schedule in status.get("planSchedule") or []:
        candidates.append((_minutes_for_schedule(str(plan_schedule)), "plan"))
    if not candidates:
        return "run"
    # Lower delta means the schedule happened more recently on the 24h clock.
    _delta, action = min(((now_minutes - minute) % (24 * 60), action) for minute, action in candidates)
    return action


def cmd_auto_plan(args) -> None:
    from hermes_cli.main import _get_update_check_result, _resolve_update_branch

    branch = _resolve_update_branch(args)
    planned_at = _utc_now()
    status = read_status()
    update_time = status.get("schedule") or "not configured"

    try:
        check = _get_update_check_result(
            branch=branch,
            branch_explicit=bool(getattr(args, "branch", None)),
        )
    except Exception as exc:
        update_status_fields(
            status=STATUS_CHECK_FAILED,
            lastPlanAt=planned_at,
            error=f"update check failed: {exc}",
            logPath=str(get_log_path()),
        )
        append_log("plan", result=STATUS_CHECK_FAILED, error=str(exc))
        print(f"✗ Auto-update plan check failed: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_CHECK_FAILED) from exc

    current_version = check.get("current_version") or _current_version()
    latest_version = check.get("latest_version") or None
    if not check.get("update_available"):
        update_status_fields(
            status=STATUS_UP_TO_DATE,
            lastPlanAt=planned_at,
            previousVersion=current_version,
            latestVersion=latest_version,
            plannedVersion=None,
            currentVersion=current_version,
            error=None,
            logPath=str(get_log_path()),
        )
        append_log(
            "plan",
            result=STATUS_UP_TO_DATE,
            previous_version=current_version,
            latest_version=latest_version,
            current_version=current_version,
        )
        print("✓ No Hermes update planned; already up to date.")
        return

    update_status_fields(
        status=STATUS_PLANNED,
        lastPlanAt=planned_at,
        previousVersion=current_version,
        latestVersion=latest_version,
        plannedVersion=latest_version,
        currentVersion=current_version,
        error=None,
        logPath=str(get_log_path()),
    )
    append_log(
        "plan",
        result=STATUS_PLANNED,
        previous_version=current_version,
        latest_version=latest_version,
        current_version=current_version,
    )
    print("☀ Hermes update available")
    if current_version and latest_version:
        print(f"{current_version} → {latest_version}")
    print(f"Scheduled auto-update: {update_time}")


def cmd_auto_run_scheduled(_args) -> None:
    status = read_status()
    if _scheduled_action_for_now(status) == "plan":
        return cmd_auto_plan(SimpleNamespace(branch=None))
    return cmd_auto_run_now(SimpleNamespace(branch=None, force=False))


def cmd_auto_run_now(args) -> None:
    from hermes_cli.backup import create_pre_update_backup
    from hermes_cli.main import _get_update_check_result, _resolve_update_branch

    branch = _resolve_update_branch(args)
    previous_version = _current_version()
    latest_version: str | None = None
    current_version: str | None = previous_version
    backup_path: str | None = None
    started_at = _utc_now()

    write_status(
        _status_payload(
            status=STATUS_RUNNING,
            last_run_at=started_at,
            previous_version=previous_version,
            current_version=current_version,
        )
    )
    append_log(
        "start",
        result=STATUS_RUNNING,
        previous_version=previous_version,
        current_version=current_version,
    )

    try:
        try:
            check = _get_update_check_result(
                branch=branch,
                branch_explicit=bool(getattr(args, "branch", None)),
            )
        except Exception as exc:
            raise AutoUpdateError(
                STATUS_CHECK_FAILED,
                f"update check failed: {exc}",
                EXIT_CHECK_FAILED,
            ) from exc

        latest_version = check.get("latest_version") or None
        if not check.get("update_available"):
            current_version = _current_version()
            payload = _status_payload(
                status=STATUS_UP_TO_DATE,
                last_run_at=started_at,
                previous_version=previous_version,
                latest_version=latest_version,
                current_version=current_version,
            )
            write_status(payload)
            append_log(
                "end",
                result=STATUS_UP_TO_DATE,
                previous_version=previous_version,
                latest_version=latest_version,
                current_version=current_version,
            )
            print("✓ Already up to date.")
            return

        backup = create_pre_update_backup()
        if backup is None:
            raise AutoUpdateError(
                STATUS_BACKUP_FAILED,
                "pre-update backup failed; aborting update",
                EXIT_BACKUP_FAILED,
            )
        backup_path = str(backup)

        _run_existing_update(args, getattr(args, "branch", None))

        ok, detail = _verify_health()
        if not ok:
            raise AutoUpdateError(
                STATUS_HEALTH_FAILED,
                f"post-update health check failed: {detail}",
                EXIT_HEALTH_FAILED,
            )

        current_version = _current_version()
        payload = _status_payload(
            status=STATUS_SUCCESS,
            last_run_at=started_at,
            previous_version=previous_version,
            latest_version=latest_version,
            current_version=current_version,
            backup_path=backup_path,
        )
        write_status(payload)
        append_log(
            "end",
            result=STATUS_SUCCESS,
            previous_version=previous_version,
            latest_version=latest_version,
            current_version=current_version,
            backup_path=backup_path,
        )
        print("✓ Auto-update run complete.")
    except AutoUpdateError as exc:
        current_version = _current_version()
        payload = _status_payload(
            status=exc.status,
            last_run_at=started_at,
            previous_version=previous_version,
            latest_version=latest_version,
            current_version=current_version,
            backup_path=backup_path,
            error=str(exc),
        )
        write_status(payload)
        append_log(
            "end",
            result=exc.status,
            previous_version=previous_version,
            latest_version=latest_version,
            current_version=current_version,
            backup_path=backup_path,
            error=str(exc),
        )
        print(f"✗ Auto-update run failed ({exc.status}): {exc}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
