"""Validation, bounded subprocesses and exact scheduler-file transactions."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from hermes_platform.host.facts import os_family
from hermes_platform.resolver import locate_command

COMMAND_TIMEOUT = 30
FileSnapshot = tuple[bytes, int] | None


def parse_time(value: str) -> tuple[int, int, str]:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{2}:[0-9]{2}", value):
        raise ValueError("time must use HH:MM format, for example 03:00")
    hour, minute = map(int, value.split(":"))
    if hour > 23 or minute > 59:
        raise ValueError("time must be a valid 24-hour HH:MM value")
    return hour, minute, value


def validate_spec(spec) -> None:
    if not re.fullmatch(r"(?:v[12]-)?[0-9a-f]{24}", spec.identity):
        raise ValueError("scheduler identity must be a stable 24-character hexadecimal hash")
    command = tuple(spec.command)
    if not command or not all(isinstance(part, str) and "\0" not in part for part in command):
        raise ValueError("scheduler command must be a nonempty argument vector without NULs")
    if not Path(command[0]).is_absolute():
        raise ValueError("scheduler executable must use an absolute stable path")
    if any(any(ord(char) < 32 for char in part) for part in command):
        raise ValueError("scheduler arguments must not contain control characters")
    home = Path(spec.home)
    if not home.is_absolute() or any(ord(char) < 32 for char in str(home)):
        raise ValueError("scheduler home must be an absolute path without control characters")
    object.__setattr__(spec, "command", command)
    object.__setattr__(spec, "home", home)
    log_directory = Path(spec.log_directory) if spec.log_directory is not None else home / "logs"
    if not log_directory.is_absolute() or any(ord(char) < 32 for char in str(log_directory)):
        raise ValueError("scheduler log directory must be an absolute path without control characters")
    object.__setattr__(spec, "log_directory", log_directory)
    object.__setattr__(spec, "schedule", parse_time(spec.schedule)[2])
    plan_times = tuple(dict.fromkeys(parse_time(t)[2] for t in spec.plan_times))
    if spec.schedule in plan_times:
        raise ValueError("plan times must differ from the update schedule")
    object.__setattr__(spec, "plan_times", plan_times)


def backend():
    family = os_family()
    if family == "darwin":
        from hermes_cli import update_auto_schedule_launchd

        return update_auto_schedule_launchd
    if family.startswith("linux"):
        from hermes_cli import update_auto_schedule_systemd

        return update_auto_schedule_systemd
    raise RuntimeError(f"auto-update scheduling is not supported on {family}")


def identity_suffix(spec) -> str:
    return spec.identity.split("-", 1)[-1]


def calendar_intervals(spec) -> list[dict[str, int]]:
    schedules = dict.fromkeys([*spec.plan_times, spec.schedule])
    return [{"Hour": parse_time(t)[0], "Minute": parse_time(t)[1]} for t in schedules]


def action_for_time(schedule: str, plan_times: Sequence[str], now: datetime | None) -> str:
    """Dispatch the most recent daily slot, including a delayed wake after sleep.

    An overlapping plan slot wins, preserving the preview's safer tie behavior.
    """
    now = now or datetime.now()
    minute_now = now.hour * 60 + now.minute
    candidates = [(schedule, "run"), *((t, "plan") for t in plan_times)]
    slots = []
    for value, action in candidates:
        hour, minute, _ = parse_time(value)
        slots.append(((minute_now - hour * 60 - minute) % 1440, action))
    return min(slots)[1]


def run_command(executable: str, args: list[str]) -> subprocess.CompletedProcess:
    resolution = locate_command(executable)
    if not resolution.command:
        raise RuntimeError(f"{executable} is unavailable; scheduler files were kept")
    command = [*resolution.command, *args]
    try:
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{executable} timed out after {COMMAND_TIMEOUT} seconds") from exc
    except OSError as exc:
        raise RuntimeError(f"could not run {executable}: {exc}") from exc


def missing_scheduler(result: subprocess.CompletedProcess) -> bool:
    if result.returncode in {0, 126, 127}:
        return False
    output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    return any(marker in output for marker in (
        "could not find service", "could not be found", "does not exist",
        "no such process", "not loaded", "unit not found",
    ))


def require_success(result: subprocess.CompletedProcess, operation: str, *, allow_missing=False) -> None:
    if result.returncode == 0 or (allow_missing and missing_scheduler(result)):
        return
    detail = "; ".join(str(part).strip() for part in (result.stdout, result.stderr) if part)
    raise RuntimeError(f"{operation} failed with exit code {result.returncode}: {detail}")


def snapshot_file(path: Path) -> FileSnapshot:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"refusing scheduler symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"scheduler artifact is not a regular file: {path}")
    # O_NOFOLLOW also rejects a symlink swapped in after lstat.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as source:
        actual = os.fstat(source.fileno())
        if not stat.S_ISREG(actual.st_mode):
            raise RuntimeError(f"scheduler artifact is not a regular file: {path}")
        return source.read(), stat.S_IMODE(actual.st_mode)


def write_file(path: Path, data: bytes, mode: int = 0o600) -> None:
    snapshot_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
            os.fchmod(target.fileno(), mode)
        snapshot_file(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_file(path: Path) -> None:
    snapshot_file(path)
    path.unlink(missing_ok=True)


def artifact_summary(snapshot: FileSnapshot) -> dict[str, Any]:
    if snapshot is None:
        return {"exists": False}
    data, mode = snapshot
    return {"exists": True, "mode": mode, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def verify_file(path: Path, expected: FileSnapshot) -> None:
    if snapshot_file(path) != expected:
        raise RuntimeError(f"exact scheduler artifact verification failed: {path}")


def restore_file(path: Path, snapshot: FileSnapshot) -> dict[str, Any]:
    errors = []
    try:
        if snapshot is None:
            remove_file(path)
        else:
            write_file(path, *snapshot)
    except (OSError, RuntimeError) as exc:
        errors.append(f"restore failed for {path}: {exc}")
    return file_receipt(path, snapshot, errors)


def file_receipt(path: Path, snapshot: FileSnapshot, errors: list[str] | None = None) -> dict[str, Any]:
    errors = list(errors or [])
    try:
        actual = snapshot_file(path)
        if actual != snapshot:
            errors.append(f"exact artifact verification failed for {path}")
        actual_summary = artifact_summary(actual)
    except (OSError, RuntimeError) as exc:
        actual_summary = {"error": str(exc)}
        errors.append(f"could not verify artifact {path}: {exc}")
    return {
        "path": str(path), "ok": not errors, "expected": artifact_summary(snapshot),
        "actual": actual_summary, "errors": errors,
    }


def recover_or_raise(error: Exception, rollback) -> None:
    from hermes_cli.update_auto_schedule import SchedulerRecoveryError

    receipt = rollback()
    if not receipt["ok"]:
        raise SchedulerRecoveryError(f"{error}; scheduler rollback failed", receipt) from error
    raise error


def collect_command(errors: list[str], run, args: list[str], *, allow_missing=False) -> None:
    try:
        require_success(run(args), "scheduler rollback " + " ".join(args), allow_missing=allow_missing)
    except (OSError, RuntimeError) as exc:
        errors.append(str(exc))


def compare_state(errors: list[str], expected: dict, actual: dict, label: str, keys) -> None:
    for key in keys:
        if expected.get(key) != actual.get(key):
            errors.append(f"{label} {key}: expected {expected.get(key)!r}, got {actual.get(key)!r}")
