"""Transactional current-user LaunchAgent scheduling for automatic updates."""

from __future__ import annotations

import os
import plistlib
import re
from pathlib import Path
from typing import Any

from hermes_cli.update_auto_schedule import SchedulerHandle, SchedulerSpec
from hermes_cli.update_auto_schedule_common import (
    calendar_intervals, collect_command, compare_state, file_receipt, identity_suffix,
    recover_or_raise, remove_file, require_success, restore_file,
    run_command, snapshot_file, verify_file, write_file,
)


def _label(spec: SchedulerSpec) -> str:
    return f"com.hermes.agent.auto-update.{identity_suffix(spec)}"


def paths(spec: SchedulerSpec) -> dict[str, Any]:
    path = Path.home() / "Library" / "LaunchAgents" / f"{_label(spec)}.plist"
    return {"backend": "launchd", "path": path}


def _target() -> str:
    # Only used by the macOS backend; never target a system LaunchDaemon.
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        raise RuntimeError("launchd scheduling requires a POSIX user session")
    return f"gui/{getuid()}"


def _run(args: list[str]):
    return run_command("launchctl", args)


def render(spec: SchedulerSpec) -> bytes:
    intervals = calendar_intervals(spec)
    return plistlib.dumps({
        "Label": _label(spec), "ProgramArguments": list(spec.command),
        "StartCalendarInterval": intervals[0] if len(intervals) == 1 else intervals,
        "EnvironmentVariables": {"HERMES_HOME": str(spec.home), "HOME": str(Path.home())},
        "StandardOutPath": str(spec.log_directory / "update-auto.out.log"),
        "StandardErrorPath": str(spec.log_directory / "update-auto.err.log"),
        "RunAtLoad": False,
    }, sort_keys=True)


def _state(target: str, label: str) -> dict[str, Any]:
    result = _run(["print", f"{target}/{label}"])
    require_success(result, "launchctl print", allow_missing=True)
    loaded = result.returncode == 0
    output = result.stdout or ""
    active_state = re.search(r"(?m)^\s*state\s*=\s*running\b", output)
    live_pid = re.search(r"(?m)^\s*pid\s*=\s*[1-9][0-9]*\b", output)
    running = loaded and bool(active_state or live_pid)
    path = re.search(r"(?m)^\s*path\s*=\s*(.+?)\s*$", output)
    disabled = _run(["print-disabled", target])
    require_success(disabled, "launchctl print-disabled")
    match = re.search(rf'["\']?{re.escape(label)}["\']?\s*=>\s*(true|false)', disabled.stdout or "", re.IGNORECASE)
    return {"loaded": loaded, "enabled": not (match and match.group(1).lower() == "true"),
            "running": running, "path": path.group(1) if path else None}


def _capture(spec: SchedulerSpec):
    path = paths(spec)["path"]
    snapshot = snapshot_file(path)
    target, label = _target(), _label(spec)
    state = _state(target, label)
    if state["loaded"] and snapshot is None:
        raise RuntimeError("launchd job is loaded without its scheduler file; refusing to unload it")
    if state["loaded"] and state["path"] != str(path):
        raise RuntimeError("launchd job is loaded from another or unverifiable path; refusing to alter it")
    _require_idle(state)
    return path, snapshot, target, label, state


def _require_idle(state: dict) -> None:
    if state["running"]:
        raise RuntimeError("scheduled updater is active; it was left running and its files were kept")


def _restore(path, snapshot, target, label, prior_state) -> dict[str, Any]:
    errors: list[str] = []
    try:
        current = _state(target, label)
        _require_idle(current)
    except (OSError, RuntimeError) as exc:
        return {
            "ok": False, "scheduler": "launchd", "errors": [str(exc)],
            "files": [file_receipt(path, snapshot)],
            "manager": {"expected": prior_state, "actual": {"error": str(exc)}},
        }
    collect_command(errors, _run, ["bootout", f"{target}/{label}"], allow_missing=True)
    receipt = restore_file(path, snapshot)
    errors.extend(receipt["errors"])
    # Re-enable before bootstrap: launchd refuses bootstrapping a disabled job.
    if receipt["ok"]:
        collect_command(errors, _run, ["enable", f"{target}/{label}"])
    if prior_state["loaded"] and snapshot is not None and receipt["ok"]:
        collect_command(errors, _run, ["bootstrap", target, str(path)])
    if not prior_state["enabled"]:
        collect_command(errors, _run, ["disable", f"{target}/{label}"])
    try:
        actual = _state(target, label)
    except (OSError, RuntimeError) as exc:
        actual = {"error": str(exc)}
        errors.append(f"could not verify launchd state: {exc}")
    compare_state(errors, prior_state, actual, label, ("loaded", "enabled", "running", "path"))
    return {"ok": not errors, "scheduler": "launchd", "files": [receipt],
            "manager": {"expected": prior_state, "actual": actual}, "errors": errors}


def enable(spec: SchedulerSpec) -> SchedulerHandle:
    path, snapshot, target, label, state = _capture(spec)
    rollback = lambda: _restore(path, snapshot, target, label, state)
    data = render(spec)
    try:
        spec.log_directory.mkdir(parents=True, exist_ok=True)
        _require_idle(_state(target, label))
        if state["loaded"]:
            require_success(_run(["bootout", f"{target}/{label}"]), "launchctl bootout", allow_missing=True)
        write_file(path, data)
        require_success(_run(["enable", f"{target}/{label}"]), "launchctl enable")
        require_success(_run(["bootstrap", target, str(path)]), "launchctl bootstrap")
        actual = _state(target, label)
        if not actual["loaded"] or not actual["enabled"] or actual["path"] != str(path):
            raise RuntimeError("launchd did not adopt the expected scheduler job")
        verify_file(path, (data, 0o600))
    except (OSError, RuntimeError) as exc:
        recover_or_raise(exc, rollback)
    return SchedulerHandle("launchd", path, rollback)


def disable(spec: SchedulerSpec) -> SchedulerHandle:
    path, snapshot, target, label, state = _capture(spec)
    rollback = lambda: _restore(path, snapshot, target, label, state)
    try:
        _require_idle(_state(target, label))
        require_success(_run(["bootout", f"{target}/{label}"]), "launchctl bootout", allow_missing=True)
        remove_file(path)
        if _state(target, label)["loaded"]:
            raise RuntimeError("launchd did not unload the expected scheduler job")
        verify_file(path, None)
    except (OSError, RuntimeError) as exc:
        recover_or_raise(exc, rollback)
    return SchedulerHandle("launchd", path, rollback, removed=snapshot is not None)
