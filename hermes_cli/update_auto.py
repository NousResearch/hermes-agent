"""Auto-update scaffolding for ``hermes update auto``.

Scheduling is intentionally a thin wrapper around ``hermes update auto
run-scheduled`` / ``run-now``. This module records stable status, writes an
append-only human log, and delegates the real update work back to the existing
``hermes update`` implementation.
"""

from __future__ import annotations

import json
import hashlib
import os
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, NoReturn

from hermes_constants import get_hermes_home
from hermes_cli.update_lock import (
    UpdateLock,
    UpdateLockBusyError,
    UpdateLockError,
    acquire_update_lock,
)


STATUS_FILENAME = "update-status.json"
LOG_FILENAME = "update.log"
LAUNCHD_LABEL = "com.hermes.agent.auto-update"
SYSTEMD_BASENAME = "hermes-auto-update"
_UNSET = object()

_HEALTH_STARTUP_GRACE_SECONDS = 10.0
_HEALTH_POLL_INTERVAL_SECONDS = 0.25

STATUS_NOT_CONFIGURED = "not_configured"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_UP_TO_DATE = "up_to_date"
STATUS_PLANNED = "planned"
STATUS_CHECK_FAILED = "check_failed"
STATUS_BACKUP_FAILED = "backup_failed"
STATUS_UPDATE_FAILED = "update_failed"
STATUS_HEALTH_FAILED = "health_failed"
STATUS_RECOVERY_REQUIRED = "recovery_required"

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
    "schedulerIdentity": None,
    "runGeneration": None,
    "preUpdateGateway": None,
    "terminalReceipt": None,
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


class SchedulerRecoveryError(RuntimeError):
    """A scheduler mutation could not be restored exactly."""

    def __init__(self, message: str, receipt: dict[str, Any]) -> None:
        super().__init__(message)
        self.receipt = receipt


class StaleStatusWriteError(RuntimeError):
    """The caller no longer owns the persisted auto-update generation."""


def _persist_recovery_required(message: str, receipt: dict[str, Any]) -> None:
    """Best-effort durable marker for a scheduler needing manual recovery."""
    try:
        detail = f"{message}: {json.dumps(receipt, sort_keys=True, default=str)}"
        status = read_status()
        write_status(
            {
                **status,
                "status": STATUS_RECOVERY_REQUIRED,
                "error": detail,
                "lastRunAt": status.get("lastRunAt") or _utc_now(),
            }
        )
    except Exception as exc:
        print(
            f"✗ Could not persist scheduler recovery state: {exc}",
            file=sys.stderr,
        )


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


def _status_cas_lock() -> UpdateLock:
    """Return the profile-local lock protecting status read/compare/write."""
    return UpdateLock(get_status_path().with_name(f".{STATUS_FILENAME}.lock"))


def _write_status_locked(
    status: dict[str, Any], *, expected_run_generation: str | None | object = _UNSET
) -> Path:
    """Write status while the caller holds ``_status_cas_lock``."""
    path = get_status_path()
    existing = read_status()
    current_generation = existing.get("runGeneration")
    caller_generation = status.get("runGeneration", current_generation)
    if expected_run_generation is _UNSET:
        expected = (
            current_generation
            if current_generation is None
            else caller_generation
        )
    else:
        expected = expected_run_generation
    if expected != current_generation:
        raise StaleStatusWriteError(
            "auto-update status generation changed: "
            f"expected {expected!r}, current {current_generation!r}"
        )

    payload = dict(DEFAULT_STATUS)
    payload.update(status)
    if "runGeneration" not in status:
        payload["runGeneration"] = current_generation
    payload["mode"] = payload.get("mode") or "manual"
    # Persisted scheduler state is an input to the fail-closed dispatcher.
    # Do not turn malformed truthy values such as ``"false"`` into an
    # enabled schedule while recording a diagnostic.
    payload["enabled"] = payload.get("enabled", False) is True
    if not isinstance(payload.get("planSchedule"), list):
        payload["planSchedule"] = []
    if not payload.get("logPath"):
        payload["logPath"] = str(get_log_path())

    from utils import atomic_json_write

    atomic_json_write(path, payload, indent=2, sort_keys=True)
    return path


def write_status(
    status: dict[str, Any], *, expected_run_generation: str | None | object = _UNSET
) -> Path:
    """Persist status only if the caller still owns the current generation.

    The shared update lock serializes update work, but it cannot protect a late
    receipt from an already-running old process. This profile-local CAS lock
    makes the persisted-generation comparison and atomic replace one operation.
    """
    with _status_cas_lock():
        return _write_status_locked(
            status, expected_run_generation=expected_run_generation
        )


def update_status_fields(**fields: Any) -> Path:
    # Read, merge, compare, and replace as one transaction. In particular, a
    # scheduler enable/disable update must not write a stale same-generation
    # snapshot over a terminal auto-update receipt.
    with _status_cas_lock():
        status = read_status()
        status.update(fields)
        return _write_status_locked(
            status, expected_run_generation=status.get("runGeneration")
        )


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

    try:
        from hermes_cli.config import detect_install_method

        if detect_install_method(PROJECT_ROOT) == "pip":
            from importlib.metadata import PackageNotFoundError, version

            try:
                return version("hermes-agent")
            except PackageNotFoundError:
                pass
    except Exception:
        pass

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
            from importlib.metadata import PackageNotFoundError, version

            return version("hermes-agent")
        except PackageNotFoundError:
            pass
        except Exception:
            pass
        try:
            from hermes_cli import __version__

            return __version__
        except Exception:
            return None


def _live_process_metadata(pid: int) -> dict[str, Any] | None:
    """Read process identity from the OS, never from the status file alone."""
    try:
        from gateway.status import _pid_exists, _read_process_cmdline, get_process_start_time

        if not _pid_exists(pid):
            return None
        metadata = {
            "pid": pid,
            "start_time": get_process_start_time(pid),
            "command": _read_process_cmdline(pid),
        }
        if pid == os.getpid():
            metadata["executable"] = str(Path(sys.executable).resolve())
            metadata["installation_identity"] = _installation_identity()
            metadata["profile_home"] = str(get_hermes_home().resolve())
        return metadata
    except Exception:
        return None


def _reported_runtime_fields(runtime: dict[str, Any] | None) -> dict[str, str | None]:
    """Extract version/revision fields reported by the gateway process."""
    if not isinstance(runtime, dict):
        return {"runtime_version": None, "runtime_revision": None}
    version = next(
        (
            runtime.get(key)
            for key in ("runtime_version", "runtimeVersion", "version")
            if isinstance(runtime.get(key), str) and runtime.get(key).strip()
        ),
        None,
    )
    revision = next(
        (
            runtime.get(key)
            for key in ("runtime_revision", "runtimeRevision", "revision")
            if isinstance(runtime.get(key), str) and runtime.get(key).strip()
        ),
        None,
    )
    return {
        "runtime_version": str(version).strip() if version else None,
        "runtime_revision": str(revision).strip() if revision else None,
    }


def _version_matches(actual: str | None, expected: str | None) -> bool:
    if not actual or not expected:
        return False
    actual = str(actual).strip()
    expected = str(expected).strip()
    return bool(actual and expected and (actual == expected or actual.startswith(expected) or expected.startswith(actual)))


def _revision_matches(actual: str | None, expected: str | None) -> bool:
    """Match a runtime Git revision without consulting its semver version."""
    return _version_matches(actual, expected)


def _installation_matches(identity: dict[str, Any] | None) -> bool:
    """Require the replacement to identify the same interpreter/install."""
    if not isinstance(identity, dict):
        return False
    reported = identity.get("installation_identity")
    if reported is not None:
        if str(reported) == _installation_identity():
            return True
        try:
            return Path(str(reported)).resolve() == Path(sys.executable).resolve()
        except (OSError, RuntimeError):
            return False
    executable = identity.get("executable") or identity.get("exe")
    if not executable:
        return False
    try:
        return Path(str(executable)).resolve() == Path(sys.executable).resolve()
    except (OSError, RuntimeError):
        return False


def _live_gateway_identity(
    runtime: dict[str, Any] | None,
    *,
    allow_start_time_change: bool = False,
) -> dict[str, Any] | None:
    """Return an OS-validated gateway identity for a running status record."""
    if not isinstance(runtime, dict) or not _gateway_was_running(runtime):
        return None
    try:
        from gateway import status as gateway_status

        recorded_pid, recorded_start = _gateway_process_identity(runtime) or (None, None)
        if recorded_pid is None:
            return None
        live_pid = gateway_status.get_runtime_status_running_pid(
            runtime,
            expected_home=get_hermes_home(),
        )
        if live_pid != recorded_pid:
            return None
        live = _live_process_metadata(live_pid)
        if live is None:
            return None
        reported_profile = (
            runtime.get("profile_home")
            or runtime.get("profileHome")
            or live.get("profile_home")
        )
        if reported_profile:
            try:
                if Path(str(reported_profile)).resolve() != get_hermes_home().resolve():
                    return None
            except (OSError, RuntimeError):
                return None
        live_start = live.get("start_time")
        if (
            not allow_start_time_change
            and recorded_start is not None
            and live_start is not None
            and recorded_start != live_start
        ):
            return None
        command = live.get("command")
        if not command:
            argv = runtime.get("argv")
            command = " ".join(str(part) for part in argv) if isinstance(argv, list) else None
        if not command or not gateway_status.looks_like_gateway_runtime_command_line(command):
            return None
        return {
            "pid": live_pid,
            "start_time": live_start if live_start is not None else recorded_start,
            "generation": runtime.get("generation"),
            "command": command,
            **_reported_runtime_fields(runtime),
            "installation_identity": runtime.get("installation_identity")
            or runtime.get("installationIdentity")
            or live.get("installation_identity")
            or live.get("exe"),
            "profile_home": runtime.get("profile_home")
            or runtime.get("profileHome")
            or runtime.get("hermes_home")
            or runtime.get("hermesHome")
            or str(get_hermes_home()),
            "executable": runtime.get("executable")
            or runtime.get("executablePath")
            or live.get("executable")
            or live.get("exe"),
        }
    except Exception:
        return None


def _stopped_runtime_is_live_validated(runtime: dict[str, Any] | None) -> bool:
    """Prove that an explicitly stopped gateway has no live recorded process."""
    if not isinstance(runtime, dict) or runtime.get("gateway_state") != "stopped":
        return False
    try:
        from gateway import status as gateway_status

        if gateway_status.get_running_pid() is not None:
            return False
    except Exception:
        return False
    identity = _gateway_process_identity(runtime)
    if identity is None:
        return True
    return _live_process_metadata(identity[0]) is None


def _capture_gateway_runtime(
    expected_version: str | None = None,
    *,
    expected_revision: str | None = None,
) -> dict[str, Any] | None:
    """Capture a durable, OS-validated pre-update gateway proof."""
    try:
        from gateway import status as gateway_status

        runtime = gateway_status.read_runtime_status()
    except Exception:
        return None
    if not isinstance(runtime, dict):
        try:
            if gateway_status.get_running_pid() is not None:
                return None
        except Exception:
            return None
        return {
            "gateway_state": "stopped",
            "pid": None,
            "start_time": None,
            "_live_validated": True,
            "pre_update_proof": "no-live-gateway",
        }
    if runtime.get("gateway_state") == "stopped":
        if _stopped_runtime_is_live_validated(runtime):
            captured = dict(runtime)
            captured["_live_validated"] = True
            return captured
        return None
    if not _gateway_was_running(runtime):
        return None
    identity = _live_gateway_identity(runtime)
    if identity is None:
        return None
    if not identity.get("runtime_version") and not identity.get("runtime_revision"):
        return None
    if expected_revision and not _revision_matches(
        identity.get("runtime_revision"), expected_revision
    ):
        return None
    if expected_version:
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", expected_version):
            if not _revision_matches(identity.get("runtime_revision"), expected_version):
                return None
        elif not _version_matches(identity.get("runtime_version"), expected_version):
            return None
    if not _installation_matches(identity):
        return None
    captured = dict(runtime)
    captured.update(identity)
    captured.setdefault("profile_home", str(get_hermes_home()))
    captured["_live_validated"] = True
    return captured


def _durable_gateway_proof(runtime: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(runtime, dict):
        return None
    return {
        "gatewayState": runtime.get("gateway_state"),
        "pid": runtime.get("pid"),
        "startTime": runtime.get("start_time"),
        "generation": runtime.get("generation"),
        "runtimeVersion": runtime.get("runtime_version"),
        "runtimeRevision": runtime.get("runtime_revision"),
        "installationIdentity": runtime.get("installation_identity"),
        "profileHome": runtime.get("profile_home") or str(get_hermes_home()),
        "liveValidated": runtime.get("_live_validated") is True,
    }


def _pre_update_gateway_proof_error(runtime: dict[str, Any] | None) -> str | None:
    if not isinstance(runtime, dict) or runtime.get("_live_validated") is not True:
        return (
            "could not establish a live gateway proof with PID, process start time, "
            "and process-reported version before update; pre-update process proof "
            "is unavailable, refusing to update"
        )
    if not _gateway_was_running(runtime):
        return None
    if not runtime.get("pid") or runtime.get("start_time") is None:
        return "pre-update gateway proof is missing PID or process start time"
    if not (runtime.get("runtime_version") or runtime.get("runtime_revision")):
        return "pre-update gateway proof is missing a process-reported version or revision"
    if not runtime.get("installation_identity") and not runtime.get("executable"):
        return "pre-update gateway proof is missing the installation identity"
    profile_home = runtime.get("profile_home") or runtime.get("profileHome")
    if not profile_home:
        return "pre-update gateway proof is missing the profile identity"
    try:
        if Path(str(profile_home)).resolve() != get_hermes_home().resolve():
            return "pre-update gateway proof belongs to a different profile"
    except (OSError, RuntimeError):
        return "pre-update gateway proof has an unreadable profile identity"
    return None


def _pre_update_gateway_intent_error() -> str | None:
    """Reject ambiguous persisted gateway lifecycle states before mutation."""
    try:
        from gateway.status import read_runtime_status

        runtime = read_runtime_status()
    except Exception:
        return None
    if not isinstance(runtime, dict):
        return None
    state = runtime.get("gateway_state")
    if state in {"running", "healthy", "stopped"}:
        return None
    return (
        "could not establish explicit pre-update gateway intent: "
        f"gateway state is {state or 'unknown'}; refusing to update"
    )


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
    return state in {"running", "healthy"}


def _verify_health(
    previous_runtime: dict[str, Any] | None = None,
    expected_version: str | None = None,
    *,
    expected_revision: str | None = None,
) -> tuple[bool, str]:
    """Return a bounded, live post-update health verdict."""
    try:
        from gateway import status as gateway_status
    except Exception as exc:
        return False, f"could not import gateway status helpers: {exc}"

    was_live_validated = (
        isinstance(previous_runtime, dict) and previous_runtime.get("_live_validated") is True
    )
    was_running = was_live_validated and _gateway_was_running(previous_runtime)
    was_explicitly_stopped = (
        was_live_validated
        and isinstance(previous_runtime, dict)
        and previous_runtime.get("gateway_state") == "stopped"
    )
    if not was_running and not was_explicitly_stopped:
        return False, (
            "could not establish a live gateway state before update; "
            "pre-update process was not live-validated"
        )

    old_identity = _gateway_process_identity(previous_runtime) if was_running else None
    deadline = time.monotonic() + _HEALTH_STARTUP_GRACE_SECONDS
    last_detail = "gateway did not return to a healthy state"
    while True:
        runtime = gateway_status.read_runtime_status()
        if isinstance(runtime, dict):
            state = str(runtime.get("gateway_state") or "")
            if state in {"startup_failed", "failed"}:
                reason = runtime.get("exit_reason") or "gateway reported unhealthy state"
                return False, str(reason)

            if was_running and state in {"running", "healthy"}:
                new_identity = _live_gateway_identity(
                    runtime,
                    allow_start_time_change=True,
                )
                if new_identity is not None and old_identity is not None:
                    old_pid, old_start = old_identity
                    if isinstance(new_identity, tuple):
                        new_pid, new_start = new_identity
                    else:
                        new_pid, new_start = (
                            new_identity["pid"],
                            new_identity.get("start_time"),
                        )
                    old_install = previous_runtime.get("installation_identity")
                    new_install = new_identity.get("installation_identity")
                    old_profile = previous_runtime.get("profile_home") or str(
                        get_hermes_home()
                    )
                    new_profile = new_identity.get("profile_home")
                    try:
                        profile_matches = (
                            Path(str(old_profile)).resolve()
                            == Path(str(new_profile)).resolve()
                            == get_hermes_home().resolve()
                        )
                    except (OSError, RuntimeError, TypeError):
                        profile_matches = False
                    expected_identity = expected_revision or expected_version
                    if expected_identity and (
                        not old_install
                        or not new_install
                        or old_install != new_install
                        or not _installation_matches(new_identity)
                    ):
                        last_detail = "gateway restart reported a different installation"
                    elif expected_identity and not new_profile:
                        last_detail = "gateway restart did not report the expected profile"
                    elif expected_revision and not _revision_matches(
                        new_identity.get("runtime_revision"), expected_revision
                    ):
                        last_detail = (
                            "gateway restart did not report the expected updated Git revision "
                            f"{expected_revision}"
                        )
                    elif expected_version and not _version_matches(
                        new_identity.get("runtime_version"), expected_version
                    ):
                        last_detail = (
                            "gateway restart did not report the expected updated version "
                            f"{expected_version}"
                        )
                    elif expected_identity and not profile_matches:
                        last_detail = "gateway restart reported a different profile"
                    elif (old_pid, old_start) != (new_pid, new_start):
                        return True, f"gateway state: running (new process {new_pid})"
                    else:
                        last_detail = (
                            "gateway restart is still using the pre-update process identity"
                        )
                else:
                    last_detail = "could not verify the live gateway process identity after restart"
            elif was_running:
                last_detail = (
                    f"gateway did not return to running state (state: {state or 'unknown'})"
                )
            elif state == "stopped":
                if previous_runtime.get("_live_validated"):
                    if _stopped_runtime_is_live_validated(runtime):
                        return True, "gateway state: stopped (already stopped before update)"
                    last_detail = "gateway reported stopped but a live recorded process remains"
                else:
                    return True, "gateway state: stopped (already stopped before update)"
            else:
                last_detail = f"gateway state: {state or 'unknown'}"
        else:
            if was_explicitly_stopped:
                try:
                    if gateway_status.get_running_pid() is None:
                        return True, "gateway state: stopped (no live gateway status)"
                except Exception:
                    pass
            last_detail = "gateway runtime status disappeared after update"

        if time.monotonic() >= deadline:
            return False, last_detail
        time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)


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
    run_generation: str | None = None,
    pre_update_gateway: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = read_status()
    generation = run_generation or existing.get("runGeneration")
    receipt = existing.get("terminalReceipt")
    if status == STATUS_RUNNING:
        receipt = None
    elif generation:
        receipt = {
            "generation": generation,
            "status": status,
            "completedAt": _utc_now(),
            "error": error,
        }
    return {
        "mode": existing.get("mode") or "manual",
        "enabled": existing.get("enabled", False) is True,
        "schedule": existing.get("schedule"),
        "planSchedule": existing.get("planSchedule") or [],
        "schedulerType": existing.get("schedulerType"),
        "schedulerPath": existing.get("schedulerPath"),
        "schedulerIdentity": existing.get("schedulerIdentity"),
        "runGeneration": generation,
        "preUpdateGateway": (
            pre_update_gateway
            if pre_update_gateway is not None
            else existing.get("preUpdateGateway")
        ),
        "lastRunAt": last_run_at,
        "lastPlanAt": existing.get("lastPlanAt"),
        "status": status,
        "previousVersion": previous_version,
        "latestVersion": latest_version,
        "plannedVersion": existing.get("plannedVersion"),
        "currentVersion": current_version,
        "backupPath": backup_path,
        "error": error,
        "terminalReceipt": receipt,
        "logPath": str(get_log_path()),
    }


def _display(value: Any, default: str = "-") -> str:
    if value is None or value == "":
        return default
    return str(value)


def _print_auto_status(status: dict[str, Any], current_version: str | None) -> None:
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
    print(f"  Current version:  {_display(current_version)}")
    print(f"  Backup path:      {_display(status.get('backupPath'))}")
    print(f"  Last error:       {_display(status.get('error'))}")
    print(f"  Detailed log:     {_display(status.get('logPath') or get_log_path())}")


def cmd_auto_status(_args) -> None:
    from hermes_cli.main import PROJECT_ROOT

    try:
        update_lock = acquire_update_lock(PROJECT_ROOT)
    except (UpdateLockBusyError, UpdateLockError) as exc:
        # Status is read-only. A transiently busy updater must not probe git
        # outside the lock or write a speculative result; use the last durable
        # version snapshot instead.
        status = read_status()
        print(f"⚠ Auto-update is busy; showing the last persisted status ({exc}).", file=sys.stderr)
        _print_auto_status(status, status.get("currentVersion"))
        return

    try:
        status = read_status()
        current_version = status.get("currentVersion") or _current_version()
        _print_auto_status(status, current_version)
    finally:
        update_lock.release()


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


def _installation_identity() -> str:
    try:
        from hermes_cli.main import PROJECT_ROOT

        install_root = Path(PROJECT_ROOT).resolve()
    except Exception:
        install_root = Path(__file__).resolve().parents[1]
    executable = Path(sys.executable).resolve()
    return f"{install_root}\0{sys.prefix}\0{executable}"


def _scheduler_identity() -> str:
    profile_home = get_hermes_home().resolve()
    material = f"{_installation_identity()}\0{profile_home}"
    return "v1-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _launchd_label() -> str:
    return f"{LAUNCHD_LABEL}.{_scheduler_identity()[3:]}"


def _systemd_basename() -> str:
    return f"{SYSTEMD_BASENAME}-{_scheduler_identity()[3:]}"


@dataclass
class _SchedulerHandle:
    scheduler_type: str
    path: Path
    _rollback_fn: Callable[[], dict[str, Any]]
    removed: bool = False
    _rolled_back: bool = False

    def rollback(self) -> dict[str, Any]:
        if self._rolled_back:
            return {"ok": True, "scheduler": self.scheduler_type, "alreadyRolledBack": True}
        self._rolled_back = True
        try:
            receipt = self._rollback_fn()
        except Exception as exc:
            return {
                "ok": False,
                "scheduler": self.scheduler_type,
                "errors": [f"rollback raised unexpectedly: {exc}"],
            }
        if not isinstance(receipt, dict):
            return {
                "ok": False,
                "scheduler": self.scheduler_type,
                "errors": ["rollback returned an invalid receipt"],
            }
        return receipt


def _scheduled_command() -> list[str]:
    profile_home = get_hermes_home().resolve()
    profile_name = profile_home.name if profile_home.parent.name == "profiles" else "default"
    return _hermes_command_prefix() + [
        "--profile",
        profile_name,
        "update",
        "auto",
        "run-scheduled",
    ]


def _launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_launchd_label()}.plist"


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
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return None
    if stat.st_mode & 0o170000 == 0o120000:
        target = os.readlink(path)
        raise RuntimeError(f"refusing to follow scheduler symlink {path} -> {target}")
    if not path.is_file():
        raise RuntimeError(f"scheduler artifact is not a regular file: {path}")
    return path.read_bytes(), stat.st_mode


def _read_artifact_state(path: Path) -> dict[str, Any]:
    try:
        stat_result = path.lstat()
    except FileNotFoundError:
        return {"exists": False, "kind": "absent", "mode": None, "bytes": None}
    except Exception as exc:
        return {"exists": None, "kind": "error", "mode": None, "bytes": None, "error": str(exc)}

    mode = stat_result.st_mode & 0o7777
    if path.is_symlink():
        try:
            target = os.readlink(path)
        except Exception as exc:
            target = f"<unreadable: {exc}>"
        return {"exists": True, "kind": "symlink", "mode": mode, "target": target}
    if not path.is_file():
        return {"exists": True, "kind": "other", "mode": mode}
    try:
        return {
            "exists": True,
            "kind": "file",
            "mode": mode,
            "bytes": path.read_bytes(),
        }
    except Exception as exc:
        return {"exists": True, "kind": "file", "mode": mode, "bytes": None, "error": str(exc)}


def _restore_file(path: Path, snapshot: tuple[bytes, int] | None) -> dict[str, Any]:
    """Restore one scheduler artifact and verify the exact result, never raising."""
    expected = {
        "exists": snapshot is not None,
        "kind": "file" if snapshot is not None else "absent",
        "mode": (snapshot[1] & 0o7777) if snapshot is not None else None,
        "bytes": snapshot[0] if snapshot is not None else None,
    }
    errors: list[str] = []
    try:
        if snapshot is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            current = _read_artifact_state(path)
            if current.get("exists") and current.get("kind") == "other":
                errors.append(f"scheduler artifact is not removable regular file: {path}")
            else:
                if current.get("exists"):
                    path.unlink()
                path.write_bytes(snapshot[0])
                os.chmod(path, snapshot[1] & 0o7777)
    except Exception as exc:
        errors.append(f"restore failed for {path}: {exc}")

    actual = _read_artifact_state(path)
    matches = (
        actual.get("exists") == expected["exists"]
        and actual.get("kind") == expected["kind"]
        and actual.get("mode") == expected["mode"]
        and (expected["bytes"] is None or actual.get("bytes") == expected["bytes"])
    )
    if not matches:
        errors.append(f"exact artifact verification failed for {path}")
    return {
        "path": str(path),
        "ok": not errors,
        "expected": expected,
        "actual": actual,
        "errors": errors,
    }


def _launchd_state(target: str) -> dict[str, bool | None]:
    label = _launchd_label()
    loaded_result = _run_launchctl(["print", f"{target}/{label}"])
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
        rf"[\"']?{re.escape(label)}[\"']?\s*=>\s*(true|false)",
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
) -> dict[str, Any]:
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
        file_receipt = _restore_file(plist_path, prior_file)
    except Exception as exc:
        file_receipt = {
            "path": str(plist_path),
            "ok": False,
            "errors": [f"restore helper raised unexpectedly: {exc}"],
        }
    if not file_receipt.get("ok"):
        errors.extend(str(error) for error in file_receipt.get("errors", []))

    if prior_state.get("loaded") and prior_file is not None:
        run(
            ["bootstrap", target, str(plist_path)],
            "launchctl rollback bootstrap",
        )
    if prior_state.get("enabled") is True:
        run(
            ["enable", f"{target}/{_launchd_label()}"],
            "launchctl rollback enable",
        )
    elif prior_state.get("enabled") is False:
        run(
            ["disable", f"{target}/{_launchd_label()}"],
            "launchctl rollback disable",
        )
    if prior_state.get("running") and prior_state.get("loaded"):
        run(
            ["kickstart", "-k", f"{target}/{_launchd_label()}"],
            "launchctl rollback kickstart",
        )
    try:
        actual_state: dict[str, Any] = _launchd_state(target)
    except Exception as exc:
        actual_state = {"error": str(exc)}
        errors.append(f"could not verify launchd state: {exc}")
    for key in ("loaded", "enabled", "running"):
        if actual_state.get(key) != prior_state.get(key):
            errors.append(
                f"launchd {key} mismatch: expected {prior_state.get(key)!r}, "
                f"got {actual_state.get(key)!r}"
            )
    receipt = {
        "ok": not errors,
        "scheduler": "launchd",
        "files": [file_receipt],
        "manager": {"expected": dict(prior_state), "actual": actual_state},
        "errors": errors,
    }
    if not receipt["ok"]:
        _persist_recovery_required("launchd restoration failed", receipt)
    return receipt


def _enable_launchd(
    hour: int, minute: int, schedule: str, plan_schedules: list[str]
) -> _SchedulerHandle:
    plist_path = _launchd_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    get_log_path().parent.mkdir(parents=True, exist_ok=True)
    target = _launchd_target()
    prior_file = _snapshot_file(plist_path)
    prior_state = _launchd_state(target)
    if prior_file is None and prior_state.get("loaded"):
        raise RuntimeError(
            "launchd job is loaded without its scheduler file; refusing to unload it"
        )

    intervals = _calendar_intervals(schedule, plan_schedules)
    start_calendar_interval: dict[str, int] | list[dict[str, int]] = (
        intervals[0] if len(intervals) == 1 else intervals
    )
    payload = {
        "Label": _launchd_label(),
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
            _run_launchctl(["enable", f"{target}/{_launchd_label()}"]),
            "launchctl enable",
        )
        final_state = _launchd_state(target)
        if final_state.get("loaded") is not True or final_state.get("enabled") is not True:
            raise RuntimeError(
                "launchd enable completed without adopting the expected job: "
                f"loaded={final_state.get('loaded')!r}, "
                f"enabled={final_state.get('enabled')!r}"
            )
    except Exception as exc:
        receipt = _restore_launchd(
                target=target,
                plist_path=plist_path,
                prior_file=prior_file,
                prior_state=prior_state,
            )
        if not receipt.get("ok"):
            raise SchedulerRecoveryError(
                f"{exc}; scheduler rollback failed: {receipt.get('errors')}",
                receipt,
            ) from exc
        raise
    return _SchedulerHandle(
        "launchd",
        plist_path,
        lambda: _restore_launchd(
            target=target,
            plist_path=plist_path,
            prior_file=prior_file,
            prior_state=prior_state,
        ),
    )


def _disable_launchd() -> _SchedulerHandle:
    plist_path = _launchd_plist_path()
    prior_file = _snapshot_file(plist_path)
    existed = prior_file is not None
    target = _launchd_target()
    prior_state = _launchd_state(target)
    if prior_file is None and prior_state.get("loaded"):
        raise RuntimeError(
            "launchd job is loaded without its scheduler file; refusing to unload it"
        )
    try:
        _require_command_success(
            _run_launchctl(["bootout", target, str(plist_path)]),
            "launchctl bootout",
            allow_missing=True,
        )
        if existed:
            plist_path.unlink()
    except Exception as exc:
        receipt = _restore_launchd(
                target=target,
                plist_path=plist_path,
                prior_file=prior_file,
                prior_state=prior_state,
            )
        if not receipt.get("ok"):
            raise SchedulerRecoveryError(
                f"{exc}; scheduler rollback failed: {receipt.get('errors')}",
                receipt,
            ) from exc
        raise
    return _SchedulerHandle(
        "launchd",
        plist_path,
        lambda: _restore_launchd(
            target=target,
            plist_path=plist_path,
            prior_file=prior_file,
            prior_state=prior_state,
        ),
        removed=existed,
    )


def _systemd_user_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def _systemd_paths() -> tuple[Path, Path]:
    root = _systemd_user_dir()
    basename = _systemd_basename()
    return root / f"{basename}.service", root / f"{basename}.timer"


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


def _systemd_state(timer_name: str) -> dict[str, Any]:
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
    if not values and result.returncode == 0:
        values = {
            "LoadState": "not-found",
            "UnitFileState": "not-found",
            "ActiveState": "inactive",
        }
    elif result.returncode != 0 and _is_missing_scheduler(result):
        values.setdefault("LoadState", "not-found")
        values.setdefault("UnitFileState", "not-found")
        values.setdefault("ActiveState", "inactive")
    missing = {"LoadState", "UnitFileState", "ActiveState"} - values.keys()
    if missing:
        raise RuntimeError(
            "systemctl --user show did not report scheduler state: "
            + ", ".join(sorted(missing))
        )
    load_state = values.get("LoadState")
    unit_file_state = values.get("UnitFileState")
    active_state = values.get("ActiveState")
    enabled_states = {"enabled", "enabled-runtime", "linked", "linked-runtime"}
    return {
        "load_state": load_state,
        "unit_file_state": unit_file_state,
        "active_state": active_state,
        "loaded": load_state == "loaded",
        "enabled": unit_file_state in enabled_states if unit_file_state is not None else None,
        "running": active_state == "active",
    }


def _restore_systemd(
    *,
    service_path: Path,
    timer_path: Path,
    timer_name: str,
    prior_service: tuple[bytes, int] | None,
    prior_timer: tuple[bytes, int] | None,
    prior_state: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    timer_state = prior_state.get("timer_state", prior_state)
    service_state = prior_state.get("service_state", timer_state)
    file_receipts: list[dict[str, Any]] = []

    def run(args: list[str], operation: str, *, allow_missing: bool = False) -> None:
        try:
            _require_command_success(
                _systemctl_user(args), operation, allow_missing=allow_missing
            )
        except Exception as exc:
            errors.append(str(exc))

    # Clear both units before putting the files back. ``enable --now`` can
    # start the oneshot service while a later verification fails; stopping only
    # the timer leaves that service running across rollback.
    service_name = service_path.name
    run(["stop", service_name], "systemctl --user rollback stop service", allow_missing=True)
    run(["disable", service_name], "systemctl --user rollback disable service", allow_missing=True)
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
            receipt = _restore_file(path, snapshot)
        except Exception as exc:
            receipt = {
                "path": str(path),
                "ok": False,
                "errors": [f"restore helper raised unexpectedly: {exc}"],
            }
        file_receipts.append(receipt)
        if not receipt.get("ok"):
            errors.extend(str(error) for error in receipt.get("errors", []))
    run(["daemon-reload"], "systemctl --user rollback daemon-reload")

    def restore_unit_file_state(
        unit_name: str,
        unit_path: Path,
        state: dict[str, Any],
        label: str,
    ) -> None:
        unit_file_state = state.get("unit_file_state")
        if unit_file_state is None:
            unit_file_state = "enabled" if state.get("enabled") is True else None
        if unit_file_state == "enabled":
            run(["enable", unit_name], f"systemctl --user rollback enable {label}")
        elif unit_file_state == "enabled-runtime":
            run(
                ["enable", "--runtime", unit_name],
                f"systemctl --user rollback enable --runtime {label}",
            )
        elif unit_file_state == "linked":
            run(["link", str(unit_path)], f"systemctl --user rollback link {label}")
        elif unit_file_state == "linked-runtime":
            run(
                ["link", "--runtime", str(unit_path)],
                f"systemctl --user rollback link --runtime {label}",
            )
        elif unit_file_state == "masked":
            run(["mask", unit_name], f"systemctl --user rollback mask {label}")
        elif unit_file_state == "masked-runtime":
            run(
                ["mask", "--runtime", unit_name],
                f"systemctl --user rollback mask --runtime {label}",
            )
        elif unit_file_state in {"disabled", "indirect", "static", "generated", "transient", "alias"}:
            # The pre-restore disable already gives these states their
            # non-enabled manager state. Static/generated/transient states are
            # determined by the restored unit file and need no enable command.
            return

    restore_unit_file_state(service_name, service_path, service_state, "service")
    restore_unit_file_state(timer_name, timer_path, timer_state, "timer")

    for unit_name, state, label in (
        (service_name, service_state, "service"),
        (timer_name, timer_state, "timer"),
    ):
        if state.get("active_state") == "active" or state.get("running"):
            run(["start", unit_name], f"systemctl --user rollback start {label}")
    manager_actual: dict[str, Any] = {}
    for label, unit_name, expected in (
        ("service", service_path.name, service_state),
        ("timer", timer_name, timer_state),
    ):
        try:
            actual = _systemd_state(unit_name)
        except Exception as exc:
            actual = {"error": str(exc)}
            errors.append(f"could not verify systemd {label} state: {exc}")
        manager_actual[label] = actual
        for key in ("load_state", "unit_file_state", "active_state"):
            if actual.get(key) != expected.get(key):
                errors.append(
                    f"systemd {label} {key} mismatch: expected {expected.get(key)!r}, "
                    f"got {actual.get(key)!r}"
                )
    receipt = {
        "ok": not errors,
        "scheduler": "systemd-user",
        "files": file_receipts,
        "manager": {
            "expected": {"service": dict(service_state), "timer": dict(timer_state)},
            "actual": manager_actual,
        },
        "errors": errors,
    }
    if not receipt["ok"]:
        _persist_recovery_required("systemd restoration failed", receipt)
    return receipt


def _enable_systemd(
    hour: int, minute: int, schedule: str, plan_schedules: list[str]
) -> _SchedulerHandle:
    if not _systemd_available():
        raise RuntimeError("systemd user services are not available")

    service_path, timer_path = _systemd_paths()
    service_path.parent.mkdir(parents=True, exist_ok=True)
    get_log_path().parent.mkdir(parents=True, exist_ok=True)
    prior_service = _snapshot_file(service_path)
    prior_timer = _snapshot_file(timer_path)
    timer_state = _systemd_state(timer_path.name)
    service_state = _systemd_state(service_path.name)
    prior_state = {
        **timer_state,
        "timer_state": timer_state,
        "service_state": service_state,
    }
    if (
        (prior_timer is None and timer_state.get("loaded"))
        or (prior_service is None and service_state.get("loaded"))
    ):
        raise RuntimeError(
            "systemd service or timer is loaded without its scheduler file; "
            "refusing to unload it"
        )

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
        _verify_systemd_enabled(service_path, timer_path)
    except Exception as exc:
        receipt = _restore_systemd(
                service_path=service_path,
                timer_path=timer_path,
                timer_name=timer_path.name,
                prior_service=prior_service,
                prior_timer=prior_timer,
                prior_state=prior_state,
            )
        if not receipt.get("ok"):
            raise SchedulerRecoveryError(
                f"{exc}; scheduler rollback failed: {receipt.get('errors')}",
                receipt,
            ) from exc
        raise
    return _SchedulerHandle(
        "systemd-user",
        timer_path,
        lambda: _restore_systemd(
            service_path=service_path,
            timer_path=timer_path,
            timer_name=timer_path.name,
            prior_service=prior_service,
            prior_timer=prior_timer,
            prior_state=prior_state,
        ),
    )


def _verify_systemd_enabled(service_path: Path, timer_path: Path) -> None:
    service_state = _systemd_state(service_path.name)
    timer_state = _systemd_state(timer_path.name)
    for label, state in (("service", service_state), ("timer", timer_state)):
        if state.get("load_state") != "loaded":
            raise RuntimeError(
                f"systemd {label} did not load: expected LoadState=loaded, "
                f"got {state.get('load_state')!r}"
            )
        if state.get("unit_file_state") in {None, "not-found"}:
            raise RuntimeError(
                f"systemd {label} has no recoverable unit-file state: "
                f"got {state.get('unit_file_state')!r}"
            )
    if timer_state.get("active_state") != "active":
        raise RuntimeError(
            "systemd timer did not start: expected ActiveState=active, got "
            f"{timer_state.get('active_state')!r}"
        )
    if timer_state.get("unit_file_state") != "enabled":
        raise RuntimeError(
            "systemd timer is not persistently enabled: expected "
            "UnitFileState=enabled, got "
            f"{timer_state.get('unit_file_state')!r}"
        )
    for path in (service_path, timer_path):
        artifact = _read_artifact_state(path)
        if artifact.get("kind") != "file":
            raise RuntimeError(f"systemd scheduler artifact is not an exact file: {path}")


def _verify_systemd_disabled(
    service_path: Path, timer_path: Path
) -> None:
    for label, path in (("service", service_path), ("timer", timer_path)):
        state = _systemd_state(path.name)
        if (
            state.get("load_state") != "not-found"
            or state.get("unit_file_state") != "not-found"
            or state.get("active_state") != "inactive"
        ):
            raise RuntimeError(
                f"systemd {label} was not fully removed: expected "
                "LoadState=not-found, UnitFileState=not-found, ActiveState=inactive, "
                f"got LoadState={state.get('load_state')!r}, "
                f"UnitFileState={state.get('unit_file_state')!r}, "
                f"ActiveState={state.get('active_state')!r}"
            )
        if _read_artifact_state(path).get("exists") is not False:
            raise RuntimeError(f"systemd scheduler artifact was not removed: {path}")


def _disable_systemd() -> _SchedulerHandle:
    service_path, timer_path = _systemd_paths()
    prior_service = _snapshot_file(service_path)
    prior_timer = _snapshot_file(timer_path)
    existed = prior_service is not None or prior_timer is not None
    if not shutil.which("systemctl"):
        if existed:
            raise RuntimeError("systemctl is unavailable; scheduler files were kept")
        return _SchedulerHandle(
            "systemd-user",
            timer_path,
            lambda: {"ok": True, "scheduler": "systemd-user", "files": []},
        )

    timer_state = _systemd_state(timer_path.name)
    service_state = _systemd_state(service_path.name)
    prior_state = {
        **timer_state,
        "timer_state": timer_state,
        "service_state": service_state,
    }
    if (
        (prior_timer is None and timer_state.get("loaded"))
        or (prior_service is None and service_state.get("loaded"))
    ):
        raise RuntimeError(
            "systemd service or timer is loaded without its scheduler file; "
            "refusing to unload it"
        )
    try:
        _require_command_success(
            _systemctl_user(["disable", "--now", timer_path.name]),
            "systemctl --user disable --now",
            allow_missing=True,
        )
        service_path.unlink(missing_ok=True)
        timer_path.unlink(missing_ok=True)
        _require_command_success(
            _systemctl_user(["daemon-reload"]),
            "systemctl --user daemon-reload",
        )
        _verify_systemd_disabled(service_path, timer_path)
    except Exception as exc:
        receipt = _restore_systemd(
                service_path=service_path,
                timer_path=timer_path,
                timer_name=timer_path.name,
                prior_service=prior_service,
                prior_timer=prior_timer,
                prior_state=prior_state,
            )
        if not receipt.get("ok"):
            raise SchedulerRecoveryError(
                f"{exc}; scheduler rollback failed: {receipt.get('errors')}",
                receipt,
            ) from exc
        raise
    return _SchedulerHandle(
        "systemd-user",
        timer_path,
        lambda: _restore_systemd(
            service_path=service_path,
            timer_path=timer_path,
            timer_name=timer_path.name,
            prior_service=prior_service,
            prior_timer=prior_timer,
            prior_state=prior_state,
        ),
        removed=existed,
    )


def _enable_scheduler(
    hour: int, minute: int, schedule: str, plan_schedules: list[str]
) -> _SchedulerHandle:
    if sys.platform == "darwin":
        return _enable_launchd(hour, minute, schedule, plan_schedules)
    if sys.platform.startswith("linux"):
        return _enable_systemd(hour, minute, schedule, plan_schedules)
    raise RuntimeError(f"auto-update scheduling is not supported on {sys.platform}")


def _disable_scheduler(status: dict[str, Any]) -> _SchedulerHandle | None:
    scheduler = status.get("schedulerType")
    if status.get("enabled") is True or scheduler or status.get("schedulerPath"):
        expected_identity = _scheduler_identity()
        if status.get("schedulerIdentity") != expected_identity:
            raise RuntimeError("scheduler identity does not belong to this installation/profile")
        if sys.platform == "darwin":
            expected_type = "launchd"
            expected_path = _launchd_plist_path()
        elif sys.platform.startswith("linux"):
            expected_type = "systemd-user"
            _service_path, expected_path = _systemd_paths()
        else:
            raise RuntimeError(f"scheduler is unsupported on {sys.platform}")
        if status.get("schedulerType") != expected_type:
            raise RuntimeError(f"schedulerType must be {expected_type!r}")
        if status.get("schedulerPath") != str(expected_path):
            raise RuntimeError("schedulerPath does not match the configured scheduler")
    if scheduler == "launchd" or sys.platform == "darwin":
        return _disable_launchd()
    if scheduler == "systemd-user" or sys.platform.startswith("linux"):
        return _disable_systemd()
    return None


def _acquire_scheduler_update_lock() -> UpdateLock:
    """Serialize scheduler mutations with source-checkout update execution."""
    from hermes_cli.main import PROJECT_ROOT

    try:
        return acquire_update_lock(PROJECT_ROOT)
    except (UpdateLockBusyError, UpdateLockError) as exc:
        print(
            f"✗ Another Hermes update is already running; scheduler mutation refused ({exc}).",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_UPDATE_FAILED) from exc


def _cmd_auto_enable_locked(args) -> None:
    scheduler_handle: _SchedulerHandle | None = None
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
        scheduler_handle = _enable_scheduler(
            hour,
            minute,
            schedule,
            plan_schedules,
        )
        try:
            update_status_fields(
                mode="scheduled",
                enabled=True,
                schedule=schedule,
                planSchedule=plan_schedules,
                schedulerType=scheduler_handle.scheduler_type,
                schedulerPath=str(scheduler_handle.path),
                schedulerIdentity=_scheduler_identity(),
                logPath=str(get_log_path()),
            )
        except Exception as status_exc:
            receipt = scheduler_handle.rollback()
            if not receipt.get("ok"):
                raise RuntimeError(
                    f"could not persist scheduler status: {status_exc}; "
                    f"scheduler rollback failed: {receipt.get('errors')}"
                ) from status_exc
            raise
    except ValueError as exc:
        print(f"✗ Invalid schedule time: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except Exception as exc:
        print(f"✗ Could not enable auto-update scheduler: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("✓ Hermes auto-update scheduled.")
    print(f"  Update time: {schedule}")
    if plan_schedules:
        print(f"  Plan time:   {', '.join(plan_schedules)}")
    print(f"  Scheduler:   {scheduler_handle.scheduler_type}")
    print(f"  Path:        {scheduler_handle.path}")
    print("  Command:     hermes update auto run-scheduled")


def cmd_auto_enable(args) -> None:
    update_lock = _acquire_scheduler_update_lock()
    try:
        return _cmd_auto_enable_locked(args)
    finally:
        update_lock.release()


def _cmd_auto_disable_locked(_args) -> None:
    status = read_status()
    scheduler_handle: _SchedulerHandle | None = None
    try:
        scheduler_handle = _disable_scheduler(status)
        try:
            update_status_fields(
                mode="manual",
                enabled=False,
                schedule=None,
                planSchedule=[],
                schedulerType=None,
                schedulerPath=None,
                schedulerIdentity=None,
                logPath=str(get_log_path()),
            )
        except Exception as status_exc:
            if scheduler_handle is None:
                raise RuntimeError(
                    f"could not persist disabled scheduler status: {status_exc}"
                ) from status_exc
            receipt = scheduler_handle.rollback()
            if not receipt.get("ok"):
                raise RuntimeError(
                    f"could not persist disabled scheduler status: {status_exc}; "
                    f"scheduler rollback failed: {receipt.get('errors')}"
                ) from status_exc
            raise RuntimeError(
                f"could not persist disabled scheduler status: {status_exc}"
            ) from status_exc
    except Exception as exc:
        print(f"✗ Could not disable auto-update scheduler: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    was_enabled = bool(status.get("enabled"))

    removed = scheduler_handle.removed if scheduler_handle is not None else False
    scheduler_path = scheduler_handle.path if scheduler_handle is not None else None
    if removed or was_enabled:
        print("✓ Hermes auto-update disabled.")
        if scheduler_path:
            print(f"  Removed: {scheduler_path}" if removed else f"  Scheduler path: {scheduler_path}")
    else:
        print("Hermes auto-update is already disabled.")


def cmd_auto_disable(args) -> None:
    update_lock = _acquire_scheduler_update_lock()
    try:
        return _cmd_auto_disable_locked(args)
    finally:
        update_lock.release()


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
    if status.get("schedulerIdentity") != _scheduler_identity():
        return "scheduler identity does not belong to this installation/profile"
    scheduler_path = status.get("schedulerPath")
    if not isinstance(scheduler_path, str) or scheduler_path != str(expected_path):
        return "schedulerPath does not match the configured scheduler"
    if expected_path.is_symlink():
        return "configured scheduler file must not be a symlink"
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


def _auto_plan_ownership_error(status: dict[str, Any]) -> str | None:
    if status.get("status") == STATUS_RUNNING:
        return "another auto-update run already owns the status generation"
    if any(
        status.get(key) is not None
        for key in ("schedulerType", "schedulerPath", "schedulerIdentity")
    ):
        return _validate_scheduler_status(status)
    return None


def cmd_auto_plan(args) -> None:
    from hermes_cli.main import (
        PROJECT_ROOT,
        _get_update_check_result,
        _resolve_update_branch,
    )

    try:
        update_lock = acquire_update_lock(PROJECT_ROOT)
    except UpdateLockBusyError as exc:
        _record_lock_failure(
            f"another Hermes update is already running; auto-update is busy ({exc})"
        )
    except UpdateLockError as exc:
        _record_lock_failure(f"could not acquire the auto-update lock: {exc}")

    try:
        # Read and validate ownership only after the lock is held. This keeps a
        # scheduled plan from probing git or publishing status for a different
        # active profile or a run that has already claimed the file.
        status = read_status()
        ownership_error = _auto_plan_ownership_error(status)
        if ownership_error:
            raise RuntimeError(f"auto-update plan ownership check failed: {ownership_error}")

        branch = _resolve_update_branch(args)
        planned_at = _utc_now()
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
    finally:
        update_lock.release()


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


def _read_valid_terminal_receipt() -> dict[str, Any] | None:
    status = read_status()
    generation = status.get("runGeneration")
    receipt = status.get("terminalReceipt")
    if not isinstance(generation, str) or not generation:
        return None
    if not isinstance(receipt, dict) or receipt.get("generation") != generation:
        return None
    if receipt.get("status") != status.get("status"):
        return None
    if status.get("status") in {None, STATUS_RUNNING}:
        return None
    return receipt


def _record_lock_failure(message: str) -> NoReturn:
    """Report lock contention without mutating leader-owned shared state."""
    receipt = None
    try:
        receipt = _read_valid_terminal_receipt()
    except Exception as exc:
        print(f"⚠ Auto-update is busy; terminal receipt unreadable: {exc}", file=sys.stderr)
    if receipt is not None:
        print(
            "⚠ Auto-update is busy; the leader already recorded "
            f"{receipt.get('status')} for generation {receipt.get('generation')}",
            file=sys.stderr,
        )
    else:
        print(f"⚠ Auto-update is busy; no leader terminal receipt yet: {message}", file=sys.stderr)
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


def _record_unexpected_run_failure(
    exc: Exception, *, run_generation: str | None = None
) -> NoReturn:
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
                run_generation=run_generation,
                pre_update_gateway=status.get("preUpdateGateway"),
            ),
            expected_run_generation=run_generation,
        )
    except StaleStatusWriteError:
        print(
            "⚠ Auto-update run lost status ownership; stale failure receipt was rejected.",
            file=sys.stderr,
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
    run_context: dict[str, str] = {}
    try:
        return _cmd_auto_run_now_locked_impl(args, run_context=run_context)
    except (KeyboardInterrupt, SystemExit):
        raise
    except StaleStatusWriteError as exc:
        print(
            "⚠ Auto-update run lost status ownership; stale terminal receipt was rejected.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_UPDATE_FAILED) from exc
    except Exception as exc:
        _record_unexpected_run_failure(
            exc, run_generation=run_context.get("run_generation")
        )


def _cmd_auto_run_now_locked_impl(
    args, *, run_context: dict[str, str] | None = None
) -> None:
    from hermes_cli.backup import create_pre_update_backup
    from hermes_cli.main import _get_update_check_result, _resolve_update_branch

    branch = _resolve_update_branch(args)
    previous_version = _current_version()
    run_generation = uuid.uuid4().hex
    if run_context is not None:
        run_context["run_generation"] = run_generation
    previous_runtime = _capture_gateway_runtime(previous_version)
    latest_version: str | None = None
    current_version: str | None = previous_version
    backup_path: str | None = None
    started_at = _utc_now()

    previous_status = read_status()
    write_status(
        _status_payload(
            status=STATUS_RUNNING,
            last_run_at=started_at,
            previous_version=previous_version,
            current_version=current_version,
            run_generation=run_generation,
            pre_update_gateway=_durable_gateway_proof(previous_runtime),
        ),
        expected_run_generation=previous_status.get("runGeneration"),
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
                run_generation=run_generation,
                pre_update_gateway=_durable_gateway_proof(previous_runtime),
            )
            write_status(payload, expected_run_generation=run_generation)
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
        if check.get("install_method") == "git" or expected_sha:
            expected_runtime_revision = str(expected_sha or "").strip() or None
            expected_runtime_version = None
        else:
            expected_runtime_revision = None
            expected_runtime_version = str(latest_version or "").strip() or None

        intent_error = _pre_update_gateway_intent_error()
        if intent_error:
            raise AutoUpdateError(
                STATUS_HEALTH_FAILED,
                intent_error,
                EXIT_HEALTH_FAILED,
            )

        proof_error = _pre_update_gateway_proof_error(previous_runtime)
        if proof_error:
            raise AutoUpdateError(
                STATUS_HEALTH_FAILED,
                proof_error,
                EXIT_HEALTH_FAILED,
            )
        if _gateway_was_running(previous_runtime) and not (
            expected_runtime_version or expected_runtime_revision
        ):
            raise AutoUpdateError(
                STATUS_HEALTH_FAILED,
                "could not establish the expected updated runtime identity; refusing to update",
                EXIT_HEALTH_FAILED,
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

        if expected_runtime_revision:
            ok, detail = _verify_health(
                previous_runtime,
                expected_runtime_version,
                expected_revision=expected_runtime_revision,
            )
        else:
            ok, detail = _verify_health(previous_runtime, expected_runtime_version)
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
            run_generation=run_generation,
            pre_update_gateway=_durable_gateway_proof(previous_runtime),
        )
        write_status(payload, expected_run_generation=run_generation)
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
            run_generation=run_generation,
            pre_update_gateway=_durable_gateway_proof(previous_runtime),
        )
        write_status(payload, expected_run_generation=run_generation)
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
