"""Transactional systemd user timers; never system services or shell commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hermes_cli.update_auto_schedule import SchedulerHandle, SchedulerSpec
from hermes_cli.update_auto_schedule_common import (
    calendar_intervals, collect_command, compare_state, file_receipt, identity_suffix,
    missing_scheduler, recover_or_raise, remove_file, require_success,
    restore_file, run_command, snapshot_file, verify_file, write_file,
)

_STATE_KEYS = ("load_state", "unit_file_state", "active_state", "fragment_path")


def paths(spec: SchedulerSpec) -> dict[str, Any]:
    root = Path.home() / ".config" / "systemd" / "user"
    name = f"hermes-auto-update-{identity_suffix(spec)}"
    return {"backend": "systemd-user", "path": root / f"{name}.timer", "service_path": root / f"{name}.service"}


def _run(args: list[str]):
    return run_command("systemctl", ["--user", *args])


def _quote(value: str) -> str:
    if any(ord(char) < 32 for char in value):
        raise ValueError("systemd values must not contain control characters")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def render(spec: SchedulerSpec) -> tuple[bytes, bytes]:
    # ':' suppresses systemd's $VAR argument expansion. This is argv, not a shell.
    command = ":" + " ".join(_quote(part) for part in spec.command)
    stdout = "append:" + str(spec.log_directory / "update-auto.out.log")
    stderr = "append:" + str(spec.log_directory / "update-auto.err.log")
    service = "\n".join([
        "[Unit]", "Description=Hermes Agent auto-update", "", "[Service]", "Type=oneshot", "TimeoutStartSec=infinity",
        f"Environment={_quote(f'HERMES_HOME={spec.home}')}",
        f"Environment={_quote(f'HOME={Path.home()}')}", f"ExecStart={command}",
        # These directives parse the append: prefix before the path; surrounding
        # quotes are literal here and make systemd ignore the redirection.
        f"StandardOutput={stdout.replace('%', '%%')}",
        f"StandardError={stderr.replace('%', '%%')}", "",
    ])
    intervals = calendar_intervals(spec)
    timer = "\n".join([
        "[Unit]", "Description=Run Hermes Agent auto-update", "", "[Timer]",
        *[f"OnCalendar=*-*-* {item['Hour']:02d}:{item['Minute']:02d}:00" for item in intervals],
        "Persistent=true", "", "[Install]", "WantedBy=timers.target", "",
    ])
    return service.encode("utf-8"), timer.encode("utf-8")


def _state(name: str) -> dict[str, str]:
    result = _run(["show", name, "--property=LoadState,UnitFileState,ActiveState,FragmentPath"])
    missing = missing_scheduler(result)
    require_success(result, "systemctl --user show", allow_missing=True)
    values = {}
    for line in (result.stdout or "").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    if missing:
        values.setdefault("LoadState", "not-found")
        values.setdefault("ActiveState", "inactive")
    if values.get("LoadState") == "not-found":
        values["UnitFileState"] = values.get("UnitFileState") or "not-found"
        values.setdefault("FragmentPath", "")
    required = {"LoadState", "UnitFileState", "ActiveState", "FragmentPath"}
    if required - values.keys():
        raise RuntimeError("systemctl --user show did not report complete scheduler state")
    return dict(zip(_STATE_KEYS, (values[k] for k in ("LoadState", "UnitFileState", "ActiveState", "FragmentPath"))))


def _capture(spec: SchedulerSpec):
    info = paths(spec)
    files = {key: snapshot_file(info[key]) for key in ("service_path", "path")}
    states = {key: _state(info[key].name) for key in files}
    for key, state in states.items():
        path = info[key]
        if state["load_state"] == "loaded" and files[key] is None:
            raise RuntimeError("systemd unit is loaded without its scheduler file; refusing to unload it")
        if state["fragment_path"] and Path(state["fragment_path"]) != path:
            raise RuntimeError(f"systemd unit is loaded from another path; refusing to alter {path.name}")
        if state["active_state"] not in {"active", "inactive", "failed"}:
            raise RuntimeError(f"systemd unit is not in a restorable state: {path.name} {state['active_state']}")
    if states["service_path"]["active_state"] not in {"inactive", "failed"}:
        raise RuntimeError("scheduled updater is active; wait for it to finish before changing its scheduler")
    return info, files, states


def _restore_unit_file_state(path: Path, state: dict, errors: list[str]) -> None:
    commands = {
        "enabled": ["enable", path.name],
        "enabled-runtime": ["enable", "--runtime", path.name],
        "linked": ["link", str(path)],
        "linked-runtime": ["link", "--runtime", str(path)],
        "masked": ["mask", path.name],
        "masked-runtime": ["mask", "--runtime", path.name],
    }
    command = commands.get(state["unit_file_state"])
    if command:
        collect_command(errors, _run, command)


def _service_is_idle(info: dict, errors: list[str]) -> bool:
    try:
        _require_idle_service(info)
        return True
    except (OSError, RuntimeError) as exc:
        errors.append(str(exc))
        return False


def _require_idle_service(info: dict) -> None:
    if _state(info["service_path"].name)["active_state"] not in {"inactive", "failed"}:
        raise RuntimeError("scheduled updater is active; it was left running and its files were kept")


def _restore(info: dict, files: dict, states: dict) -> dict[str, Any]:
    errors: list[str] = []
    collect_command(errors, _run, ["disable", "--now", info["path"].name], allow_missing=True)
    idle = _service_is_idle(info, errors)
    if idle:
        collect_command(errors, _run, ["disable", info["service_path"].name], allow_missing=True)
    restore = restore_file if idle else file_receipt
    receipts = [restore(info[key], snapshot) for key, snapshot in files.items()]
    for receipt in receipts:
        errors.extend(receipt["errors"])
    collect_command(errors, _run, ["daemon-reload"])
    if idle and all(receipt["ok"] for receipt in receipts):
        for key, state in states.items():
            _restore_unit_file_state(info[key], state, errors)
        for key, state in states.items():
            if state["active_state"] == "active":
                collect_command(errors, _run, ["start", info[key].name])
    actual = {}
    for key, expected in states.items():
        try:
            actual[key] = _state(info[key].name)
        except (OSError, RuntimeError) as exc:
            actual[key] = {"error": str(exc)}
            errors.append(f"could not verify systemd state: {exc}")
        compare_state(errors, expected, actual[key], info[key].name, _STATE_KEYS)
    return {
        "ok": not errors, "scheduler": "systemd-user", "files": receipts,
        "manager": {"expected": states, "actual": actual}, "errors": errors,
    }


def _verify_enabled(info: dict, expected: dict) -> None:
    for key in ("service_path", "path"):
        path = info[key]
        state = _state(path.name)
        if state["load_state"] != "loaded" or state["fragment_path"] != str(path):
            raise RuntimeError(f"systemd did not adopt the expected scheduler file: {path}")
        if state["unit_file_state"] in {"", "not-found", "masked", "masked-runtime"}:
            raise RuntimeError(f"systemd scheduler has invalid unit-file state: {state}")
        if key == "path" and (state["active_state"], state["unit_file_state"]) != ("active", "enabled"):
            raise RuntimeError("systemd timer is not persistently enabled and active")
        verify_file(path, (expected[key], 0o600))


def enable(spec: SchedulerSpec) -> SchedulerHandle:
    info, files, states = _capture(spec)
    service, timer = render(spec)
    content = {"service_path": service, "path": timer}
    rollback = lambda: _restore(info, files, states)
    try:
        spec.log_directory.mkdir(parents=True, exist_ok=True)
        if any(snapshot is not None for snapshot in files.values()):
            require_success(_run(["disable", "--now", info["path"].name]), "systemctl disable timer", allow_missing=True)
        _require_idle_service(info)
        for key, data in content.items():
            write_file(info[key], data)
        require_success(_run(["daemon-reload"]), "systemctl daemon-reload")
        require_success(_run(["enable", "--now", info["path"].name]), "systemctl enable timer")
        _verify_enabled(info, content)
    except (OSError, RuntimeError) as exc:
        recover_or_raise(exc, rollback)
    return SchedulerHandle("systemd-user", info["path"], rollback)


def _verify_disabled(info: dict) -> None:
    expected = {"load_state": "not-found", "unit_file_state": "not-found", "active_state": "inactive", "fragment_path": ""}
    for key in ("service_path", "path"):
        path = info[key]
        if _state(path.name) != expected:
            raise RuntimeError(f"systemd scheduler was not fully removed: {path.name}")
        verify_file(path, None)


def disable(spec: SchedulerSpec) -> SchedulerHandle:
    info, files, states = _capture(spec)
    rollback = lambda: _restore(info, files, states)
    try:
        require_success(_run(["disable", "--now", info["path"].name]), "systemctl disable timer", allow_missing=True)
        _require_idle_service(info)
        require_success(_run(["disable", info["service_path"].name]), "systemctl disable service", allow_missing=True)
        for key in files:
            remove_file(info[key])
        require_success(_run(["daemon-reload"]), "systemctl daemon-reload")
        for key in files:
            require_success(_run(["reset-failed", info[key].name]), "systemctl reset-failed", allow_missing=True)
        _verify_disabled(info)
    except (OSError, RuntimeError) as exc:
        recover_or_raise(exc, rollback)
    return SchedulerHandle("systemd-user", info["path"], rollback, removed=any(s is not None for s in files.values()))
