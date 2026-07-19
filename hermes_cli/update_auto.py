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
from typing import Any, NoReturn

from hermes_constants import get_hermes_home
from hermes_cli.update_lock import (
    UpdateLockBusyError,
    UpdateLockError,
    acquire_update_lock,
)


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
    # Persisted scheduler state is an input to the fail-closed dispatcher. Do
    # not turn malformed truthy values such as ``"false"`` into an enabled
    # schedule while recording a diagnostic.
    payload["enabled"] = payload.get("enabled", False) is True
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


def _capture_gateway_runtime() -> dict[str, Any] | None:
    """Capture the gateway's pre-update intent and process identity."""
    from gateway.status import read_runtime_status

    runtime = read_runtime_status()
    return runtime if isinstance(runtime, dict) else None


def _gateway_process_identity(runtime: dict[str, Any] | None) -> tuple[int, Any] | None:
    if not isinstance(runtime, dict):
        return None
    try:
        pid = int(runtime["pid"])
    except (KeyError, TypeError, ValueError):
        return None
    return pid, runtime.get("start_time")


def _gateway_was_running(runtime: dict[str, Any] | None) -> bool:
    if not isinstance(runtime, dict):
        return False
    state = runtime.get("gateway_state")
    return state not in {None, "stopped", "startup_failed", "failed"}


def _verify_health(
    previous_runtime: dict[str, Any] | None = None,
) -> tuple[bool, str]:
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
        if _gateway_was_running(previous_runtime):
            return False, "gateway runtime status disappeared after a running gateway update"
        if isinstance(previous_runtime, dict) and previous_runtime.get("gateway_state") == "stopped":
            return True, "gateway remained stopped after update"
        return False, "could not establish the gateway state before or after update"

    state = str(runtime.get("gateway_state") or "")
    if state in {"startup_failed", "failed"}:
        reason = runtime.get("exit_reason") or "gateway reported unhealthy state"
        return False, str(reason)

    if _gateway_was_running(previous_runtime):
        if state != "running":
            return False, f"gateway did not return to running state (state: {state or 'unknown'})"
        old_identity = _gateway_process_identity(previous_runtime)
        new_identity = _gateway_process_identity(runtime)
        if old_identity is None or new_identity is None:
            return False, "could not verify the gateway process identity after restart"
        old_pid, old_start = old_identity
        new_pid, new_start = new_identity
        if old_pid == new_pid and (
            old_start is None or new_start is None or old_start == new_start
        ):
            return False, "gateway restart left the pre-update process running"
        return True, f"gateway state: running (new process {new_pid})"

    if state == "stopped":
        if isinstance(previous_runtime, dict) and previous_runtime.get("gateway_state") == "stopped":
            return True, "gateway state: stopped (already stopped before update)"
        return False, "gateway stopped after update but was running or unknown before update"
    return True, f"gateway state: {state or 'unknown'}"


def _verify_expected_sha(expected_sha: str) -> tuple[bool, str]:
    """Verify that the checked update commit is contained in the current HEAD."""
    from hermes_cli.main import PROJECT_ROOT

    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", expected_sha, "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, f"could not verify expected update {expected_sha}: {exc}"
    if result.returncode == 0:
        return True, "expected update commit is an ancestor of HEAD"
    detail = _command_detail(result)
    suffix = f": {detail}" if detail else ""
    return False, f"checked update {expected_sha} is not contained in current HEAD{suffix}"


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
        "enabled": existing.get("enabled", False) is True,
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


def _run_existing_update(
    args,
    branch: str | None,
    *,
    expected_sha: str | None = None,
) -> None:
    from hermes_cli.config import is_managed
    from hermes_cli.main import cmd_update

    if is_managed():
        raise AutoUpdateError(
            STATUS_UPDATE_FAILED,
            "automatic update is unavailable for managed installations",
            EXIT_UPDATE_FAILED,
        )

    update_args = SimpleNamespace(
        gateway=False,
        check=False,
        no_backup=True,
        backup=False,
        yes=True,
        branch=branch,
        force=getattr(args, "force", False),
        _update_lock_held=True,
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
    if expected_sha:
        ok, detail = _verify_expected_sha(expected_sha)
        if not ok:
            raise AutoUpdateError(STATUS_UPDATE_FAILED, detail, EXIT_UPDATE_FAILED)


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
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        raise RuntimeError("launchd auto-update scheduling requires a POSIX user session")
    return f"gui/{getuid()}"


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


def _command_detail(result: subprocess.CompletedProcess) -> str:
    details = []
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if stdout:
        details.append(f"stdout: {stdout}")
    if stderr:
        details.append(f"stderr: {stderr}")
    return "; ".join(details)


def _is_missing_scheduler(result: subprocess.CompletedProcess) -> bool:
    output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    if result.returncode == 127 and "errno" in output:
        return False
    return any(
        marker in output
        for marker in (
            "could not find",
            "does not exist",
            "no such file",
            "no such process",
            "not found",
            "not loaded",
            "service not loaded",
            "unit not loaded",
        )
    )


def _require_command_success(
    result: subprocess.CompletedProcess,
    operation: str,
    *,
    allow_missing: bool = False,
) -> None:
    if result.returncode == 0:
        return
    if allow_missing and _is_missing_scheduler(result):
        return
    detail = _command_detail(result)
    suffix = f": {detail}" if detail else ""
    raise RuntimeError(f"{operation} failed with exit code {result.returncode}{suffix}")


def _snapshot_file(path: Path) -> tuple[bytes, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return path.read_bytes(), stat.st_mode


def _restore_file(path: Path, snapshot: tuple[bytes, int] | None) -> None:
    if snapshot is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(snapshot[0])
    os.chmod(path, snapshot[1] & 0o7777)


def _launchd_state(target: str) -> dict[str, bool | None]:
    loaded_result = _run_launchctl(["print", f"{target}/{LAUNCHD_LABEL}"])
    if loaded_result.returncode != 0:
        _require_command_success(
            loaded_result,
            "launchctl print",
            allow_missing=True,
        )
    loaded = loaded_result.returncode == 0
    output = f"{loaded_result.stdout or ''}\n{loaded_result.stderr or ''}"
    running = loaded and bool(re.search(r"(?m)^\s*state\s*=\s*running\b", output))

    enabled_result = _run_launchctl(["print-disabled", target])
    _require_command_success(enabled_result, "launchctl print-disabled")
    match = re.search(
        rf"[\"']?{re.escape(LAUNCHD_LABEL)}[\"']?\s*=>\s*(true|false)",
        f"{enabled_result.stdout or ''}\n{enabled_result.stderr or ''}",
        re.IGNORECASE,
    )
    # print-disabled lists disabled jobs. An absent label is enabled.
    enabled = not (match and match.group(1).lower() == "true")
    return {"loaded": loaded, "enabled": enabled, "running": running}


def _restore_launchd(
    *,
    target: str,
    plist_path: Path,
    prior_file: tuple[bytes, int] | None,
    prior_state: dict[str, bool | None],
) -> None:
    errors: list[str] = []

    def run(args: list[str], operation: str, *, allow_missing: bool = False) -> None:
        try:
            _require_command_success(
                _run_launchctl(args), operation, allow_missing=allow_missing
            )
        except Exception as exc:
            errors.append(str(exc))

    # Remove the new loaded job before restoring the old file. This is also the
    # cleanup path for a newly created scheduler with no prior configuration.
    run(
        ["bootout", target, str(plist_path)],
        "launchctl rollback bootout",
        allow_missing=True,
    )
    try:
        _restore_file(plist_path, prior_file)
    except Exception as exc:
        errors.append(f"could not restore {plist_path}: {exc}")

    if prior_state.get("loaded") and prior_file is not None:
        run(
            ["bootstrap", target, str(plist_path)],
            "launchctl rollback bootstrap",
        )
    if prior_state.get("enabled") is True:
        run(
            ["enable", f"{target}/{LAUNCHD_LABEL}"],
            "launchctl rollback enable",
        )
    elif prior_state.get("enabled") is False:
        run(
            ["disable", f"{target}/{LAUNCHD_LABEL}"],
            "launchctl rollback disable",
        )
    if prior_state.get("running") and prior_state.get("loaded"):
        run(
            ["kickstart", "-k", f"{target}/{LAUNCHD_LABEL}"],
            "launchctl rollback kickstart",
        )
    if errors:
        raise RuntimeError("; ".join(errors))


def _enable_launchd(hour: int, minute: int, schedule: str, plan_schedules: list[str]) -> tuple[str, Path]:
    plist_path = _launchd_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    get_log_path().parent.mkdir(parents=True, exist_ok=True)
    target = _launchd_target()
    prior_file = _snapshot_file(plist_path)
    prior_state = _launchd_state(target) if prior_file is not None else {
        "loaded": False,
        "enabled": None,
        "running": False,
    }

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
    try:
        if prior_file is not None or prior_state.get("loaded"):
            _require_command_success(
                _run_launchctl(["bootout", target, str(plist_path)]),
                "launchctl bootout",
                allow_missing=True,
            )
        with plist_path.open("wb") as handle:
            plistlib.dump(payload, handle, sort_keys=True)

        result = _run_launchctl(["bootstrap", target, str(plist_path)])
        _require_command_success(result, "launchctl bootstrap")
        _require_command_success(
            _run_launchctl(["enable", f"{target}/{LAUNCHD_LABEL}"]),
            "launchctl enable",
        )
    except Exception as exc:
        try:
            _restore_launchd(
                target=target,
                plist_path=plist_path,
                prior_file=prior_file,
                prior_state=prior_state,
            )
        except Exception as rollback_exc:
            raise RuntimeError(f"{exc}; scheduler rollback failed: {rollback_exc}") from exc
        raise
    return "launchd", plist_path


def _disable_launchd() -> tuple[str, Path, bool]:
    plist_path = _launchd_plist_path()
    existed = plist_path.exists()
    target = _launchd_target()
    _require_command_success(
        _run_launchctl(["bootout", target, str(plist_path)]),
        "launchctl bootout",
        allow_missing=True,
    )
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


def _systemd_state(timer_name: str) -> dict[str, bool | None]:
    result = _systemctl_user(
        ["show", timer_name, "--property=LoadState,UnitFileState,ActiveState"]
    )
    if result.returncode != 0 and not _is_missing_scheduler(result):
        _require_command_success(result, "systemctl --user show")
    values: dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    missing = {"LoadState", "UnitFileState", "ActiveState"} - values.keys()
    if missing:
        raise RuntimeError(
            "systemctl --user show did not report scheduler state: "
            + ", ".join(sorted(missing))
        )
    unit_file_state = values.get("UnitFileState")
    enabled_states = {"enabled", "enabled-runtime", "linked", "linked-runtime"}
    return {
        "loaded": values.get("LoadState") == "loaded",
        "enabled": unit_file_state in enabled_states if unit_file_state is not None else None,
        "running": values.get("ActiveState") == "active",
    }


def _restore_systemd(
    *,
    service_path: Path,
    timer_path: Path,
    timer_name: str,
    prior_service: tuple[bytes, int] | None,
    prior_timer: tuple[bytes, int] | None,
    prior_state: dict[str, bool | None],
) -> None:
    errors: list[str] = []

    def run(args: list[str], operation: str, *, allow_missing: bool = False) -> None:
        try:
            _require_command_success(
                _systemctl_user(args), operation, allow_missing=allow_missing
            )
        except Exception as exc:
            errors.append(str(exc))

    run(
        ["disable", "--now", timer_name],
        "systemctl --user rollback disable --now",
        allow_missing=True,
    )
    for path, snapshot in (
        (service_path, prior_service),
        (timer_path, prior_timer),
    ):
        try:
            _restore_file(path, snapshot)
        except Exception as exc:
            errors.append(f"could not restore {path}: {exc}")
    run(["daemon-reload"], "systemctl --user rollback daemon-reload")

    if prior_state.get("enabled") is True:
        run(["enable", timer_name], "systemctl --user rollback enable")
    elif prior_state.get("enabled") is False:
        run(
            ["disable", timer_name],
            "systemctl --user rollback disable",
            allow_missing=True,
        )
    if prior_state.get("running"):
        run(["start", timer_name], "systemctl --user rollback start")
    if errors:
        raise RuntimeError("; ".join(errors))


def _enable_systemd(hour: int, minute: int, schedule: str, plan_schedules: list[str]) -> tuple[str, Path]:
    if not _systemd_available():
        raise RuntimeError("systemd user services are not available")

    service_path, timer_path = _systemd_paths()
    service_path.parent.mkdir(parents=True, exist_ok=True)
    get_log_path().parent.mkdir(parents=True, exist_ok=True)
    prior_service = _snapshot_file(service_path)
    prior_timer = _snapshot_file(timer_path)
    prior_state = _systemd_state(timer_path.name) if prior_service is not None or prior_timer is not None else {
        "loaded": False,
        "enabled": None,
        "running": False,
    }

    command = " ".join(shlex.quote(part) for part in _scheduled_command())
    on_calendar_lines = [
        f"OnCalendar=*-*-* {item['Hour']:02d}:{item['Minute']:02d}:00"
        for item in _calendar_intervals(schedule, plan_schedules)
    ]
    try:
        if prior_service is not None or prior_timer is not None:
            _require_command_success(
                _systemctl_user(["disable", "--now", timer_path.name]),
                "systemctl --user disable --now",
                allow_missing=True,
            )
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
        _require_command_success(
            _systemctl_user(["daemon-reload"]),
            "systemctl --user daemon-reload",
        )
        result = _systemctl_user(["enable", "--now", timer_path.name])
        _require_command_success(result, "systemctl --user enable --now")
    except Exception as exc:
        try:
            _restore_systemd(
                service_path=service_path,
                timer_path=timer_path,
                timer_name=timer_path.name,
                prior_service=prior_service,
                prior_timer=prior_timer,
                prior_state=prior_state,
            )
        except Exception as rollback_exc:
            raise RuntimeError(f"{exc}; scheduler rollback failed: {rollback_exc}") from exc
        raise
    return "systemd-user", timer_path


def _disable_systemd() -> tuple[str, Path, bool]:
    service_path, timer_path = _systemd_paths()
    existed = service_path.exists() or timer_path.exists()
    if not shutil.which("systemctl"):
        if existed:
            raise RuntimeError("systemctl is unavailable; scheduler files were kept")
        return "systemd-user", timer_path, False

    _require_command_success(
        _systemctl_user(["disable", "--now", timer_path.name]),
        "systemctl --user disable --now",
        allow_missing=True,
    )
    _require_command_success(
        _systemctl_user(["daemon-reload"]),
        "systemctl --user daemon-reload",
    )
    service_path.unlink(missing_ok=True)
    timer_path.unlink(missing_ok=True)
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
        if schedule in plan_schedules:
            raise ValueError(
                f"plan time {schedule} must differ from the update time"
            )
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


def _read_persisted_scheduler_status() -> dict[str, Any]:
    path = get_status_path()
    if not path.exists():
        raise ValueError("status file is missing")
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"status file cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("status file must contain an object")
    return data


def _validate_scheduler_status(status: dict[str, Any]) -> str | None:
    if status.get("enabled") is not True:
        return "enabled must be exactly true"
    if status.get("mode") != "scheduled":
        return "mode must be scheduled"

    schedule = status.get("schedule")
    if not isinstance(schedule, str):
        return "schedule must be a normalized HH:MM string"
    try:
        _hour, _minute, normalized = _parse_time(schedule)
    except (TypeError, ValueError) as exc:
        return f"schedule is invalid: {exc}"
    if schedule != normalized:
        return "schedule must use normalized HH:MM form"

    plan_schedules = status.get("planSchedule")
    if not isinstance(plan_schedules, list):
        return "planSchedule must be a list"
    seen: set[str] = set()
    for value in plan_schedules:
        if not isinstance(value, str):
            return "planSchedule entries must be normalized HH:MM strings"
        try:
            _plan_hour, _plan_minute, normalized_plan = _parse_time(value)
        except (TypeError, ValueError) as exc:
            return f"planSchedule is invalid: {exc}"
        if value != normalized_plan:
            return "planSchedule entries must use normalized HH:MM form"
        if value in seen or value == schedule:
            return "planSchedule contains a duplicate or update-time entry"
        seen.add(value)

    if sys.platform == "darwin":
        expected_type = "launchd"
        expected_path = _launchd_plist_path()
    elif sys.platform.startswith("linux"):
        expected_type = "systemd-user"
        _service_path, expected_path = _systemd_paths()
    else:
        return f"scheduler is unsupported on {sys.platform}"

    if status.get("schedulerType") != expected_type:
        return f"schedulerType must be {expected_type!r}"
    scheduler_path = status.get("schedulerPath")
    if not isinstance(scheduler_path, str) or scheduler_path != str(expected_path):
        return "schedulerPath does not match the configured scheduler"
    if not expected_path.is_file():
        return f"configured scheduler file is missing: {expected_path}"
    return None


def _record_scheduler_state_failure(message: str) -> None:
    error = f"invalid scheduler state: {message}"
    try:
        existing = read_status()
        fields: dict[str, Any] = {
            "status": STATUS_UPDATE_FAILED,
            "lastRunAt": _utc_now(),
            "error": error,
            "logPath": str(get_log_path()),
        }
        if existing.get("enabled") is not True:
            fields["enabled"] = False
        update_status_fields(**fields)
    except Exception as exc:
        print(f"✗ Could not record scheduler diagnostic: {exc}", file=sys.stderr)
    try:
        append_log("dispatch", result=STATUS_UPDATE_FAILED, error=error)
    except Exception as exc:
        print(f"✗ Could not append scheduler diagnostic: {exc}", file=sys.stderr)
    print(f"✗ Auto-update scheduler stopped: {error}", file=sys.stderr)


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
    try:
        persisted = _read_persisted_scheduler_status()
    except ValueError as exc:
        _record_scheduler_state_failure(str(exc))
        return None
    if persisted.get("enabled") is False:
        return
    validation_error = _validate_scheduler_status(persisted)
    if validation_error:
        _record_scheduler_state_failure(validation_error)
        return None
    status = read_status()
    if _scheduled_action_for_now(status) == "plan":
        return cmd_auto_plan(SimpleNamespace(branch=None))
    return cmd_auto_run_now(SimpleNamespace(branch=None, force=False))


def _record_lock_failure(message: str) -> NoReturn:
    started_at = _utc_now()
    current_version = _current_version()
    write_status(
        _status_payload(
            status=STATUS_UPDATE_FAILED,
            last_run_at=started_at,
            previous_version=current_version,
            current_version=current_version,
            error=message,
        )
    )
    append_log(
        "end",
        result=STATUS_UPDATE_FAILED,
        previous_version=current_version,
        current_version=current_version,
        error=message,
    )
    print(f"✗ Auto-update run failed ({STATUS_UPDATE_FAILED}): {message}", file=sys.stderr)
    raise SystemExit(EXIT_UPDATE_FAILED)


def cmd_auto_run_now(args) -> None:
    from hermes_cli.main import PROJECT_ROOT

    try:
        update_lock = acquire_update_lock(PROJECT_ROOT)
    except UpdateLockBusyError as exc:
        _record_lock_failure(
            f"another Hermes update is already running; auto-update is busy ({exc})"
        )
    except UpdateLockError as exc:
        _record_lock_failure(f"could not acquire the auto-update lock: {exc}")

    try:
        return _cmd_auto_run_now_locked(args)
    finally:
        update_lock.release()


def _record_unexpected_run_failure(exc: Exception) -> NoReturn:
    """Terminalize an unexpected run failure without swallowing the cause."""
    status = read_status()
    started_at = status.get("lastRunAt") or _utc_now()
    previous_version = status.get("previousVersion") or _current_version()
    latest_version = status.get("latestVersion") or None
    current_version = _current_version()
    error = f"unexpected auto-update failure: {exc}"
    try:
        write_status(
            _status_payload(
                status=STATUS_UPDATE_FAILED,
                last_run_at=started_at,
                previous_version=previous_version,
                latest_version=latest_version,
                current_version=current_version,
                backup_path=status.get("backupPath"),
                error=error,
            )
        )
    except Exception as persist_exc:
        print(
            f"✗ Could not persist terminal auto-update status: {persist_exc}",
            file=sys.stderr,
        )
    try:
        append_log(
            "end",
            result=STATUS_UPDATE_FAILED,
            previous_version=previous_version,
            latest_version=latest_version,
            current_version=current_version,
            backup_path=status.get("backupPath"),
            error=error,
        )
    except Exception as log_exc:
        print(f"✗ Could not append auto-update failure log: {log_exc}", file=sys.stderr)
    print(f"✗ Auto-update run failed ({STATUS_UPDATE_FAILED}): {error}", file=sys.stderr)
    raise SystemExit(EXIT_UPDATE_FAILED) from exc


def _cmd_auto_run_now_locked(args) -> None:
    try:
        return _cmd_auto_run_now_locked_impl(args)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        _record_unexpected_run_failure(exc)


def _cmd_auto_run_now_locked_impl(args) -> None:
    from hermes_cli.backup import create_pre_update_backup
    from hermes_cli.main import _get_update_check_result, _resolve_update_branch

    branch = _resolve_update_branch(args)
    previous_version = _current_version()
    previous_runtime = _capture_gateway_runtime()
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

        expected_sha = check.get("latest_sha")
        if check.get("install_method") == "git" and not expected_sha:
            raise AutoUpdateError(
                STATUS_UPDATE_FAILED,
                "update check reported an available git update without an expected latest SHA",
                EXIT_UPDATE_FAILED,
            )

        backup = create_pre_update_backup()
        if backup is None:
            raise AutoUpdateError(
                STATUS_BACKUP_FAILED,
                "pre-update backup failed; aborting update",
                EXIT_BACKUP_FAILED,
            )
        backup_path = str(backup)

        _run_existing_update(
            args,
            getattr(args, "branch", None),
            expected_sha=str(expected_sha) if expected_sha else None,
        )

        # ``cmd_update`` intentionally returns normally for some no-op paths,
        # including the managed-install guard.  The preflight check said an
        # update was available, so a normal return is not proof that anything
        # changed when there is no commit SHA to verify.  Require observable
        # version movement before recording success.
        current_version = _current_version()
        if not expected_sha and (
            previous_version is None
            or current_version is None
            or current_version == previous_version
        ):
            raise AutoUpdateError(
                STATUS_UPDATE_FAILED,
                "update command returned normally without applying the checked update",
                EXIT_UPDATE_FAILED,
            )

        ok, detail = _verify_health(previous_runtime)
        if not ok:
            raise AutoUpdateError(
                STATUS_HEALTH_FAILED,
                f"post-update health check failed: {detail}",
                EXIT_HEALTH_FAILED,
            )

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
