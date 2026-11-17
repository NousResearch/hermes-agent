"""Explicit, transactional retirement of legacy profile-owned schedulers.

An old timer must never migrate itself: unloading a running service can kill the
updater. Its next firing reports the migration command, while a foreground
enable/migrate/disable can verify idle jobs and roll back their exact artifacts.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import base64
import hashlib
import json
from pathlib import Path
import sys

from hermes_cli import update_auto_schedule as scheduler
from hermes_cli.update_auto_schedule_common import restore_file, snapshot_file
from hermes_cli.update_auto_state import AutoUpdateContext, utc_now, write_status
from hermes_cli.update_installation import profile_homes
from hermes_cli.update_lock import marker_mutex
from utils import atomic_json_write


@dataclass(frozen=True)
class LegacySchedule:
    home: Path
    path: Path
    status: dict
    spec: scheduler.SchedulerSpec


def _legacy_identity(context: AutoUpdateContext, home: Path) -> str:
    return "v1-" + hashlib.sha256(f"{context.install}\0{home}".encode()).hexdigest()[:24]


def _refuse_orphan_artifacts(context: AutoUpdateContext, home: Path) -> None:
    from hermes_platform.host.facts import os_family

    if os_family() not in {"linux", "darwin"}:
        return
    spec = scheduler.SchedulerSpec(_legacy_identity(context, home), [sys.executable], home, "00:00")
    for key, path in scheduler.paths(spec).items():
        if key in {"path", "service_path"} and snapshot_file(path) is not None:
            raise ValueError(f"Legacy scheduler artifact has no enabled activation; inspect it before migration: {path}")


@contextmanager
def legacy_operation_locks(context: AutoUpdateContext):
    """Fence old enable/disable processes, including profiles with no active timer yet."""
    from hermes_constants import mkdir_under_hermes_home

    with ExitStack() as stack:
        seen = set()
        for home in profile_homes(context.home):
            path = home / "state" / "update-auto-operation"
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            mkdir_under_hermes_home(path.parent)
            snapshot_file(path.with_name(path.name + ".lock"))
            stack.enter_context(marker_mutex(path, wait=0))
        yield


def legacy_schedules(context: AutoUpdateContext) -> list[LegacySchedule]:
    from hermes_cli._launchers import installation_command

    schedules = []
    seen_status = set()
    for home in profile_homes(context.home):
        path = home / "state" / "update-status.json"
        snapshot = snapshot_file(path)
        if snapshot is None:
            _refuse_orphan_artifacts(context, home)
            continue
        try:
            status = json.loads(snapshot[0].decode("utf-8-sig"))
        except (UnicodeError, ValueError) as exc:
            raise ValueError(f"Cannot read legacy auto-update status: {path}") from exc
        if not isinstance(status, dict):
            raise ValueError(f"Invalid legacy auto-update status: {path}")
        # A different install sharing this data root retains its own scheduler.
        if not status.get("installationRoot"):
            raise ValueError(f"Legacy scheduler has no installation identity; inspect it before migrating: {path}")
        if status.get("installationRoot") != str(context.install):
            continue
        recorded = status.get("profileHome")
        if not isinstance(recorded, str) or not Path(recorded).is_absolute() or Path(recorded).resolve() != home.resolve():
            raise ValueError(f"Legacy scheduler home does not belong to this installation data root: {path}")
        if home != Path(recorded):
            _refuse_orphan_artifacts(context, home)
        home = Path(recorded)
        identity = _legacy_identity(context, home)
        if (status.get("schema") != 2 or status.get("schedulerIdentity") != identity
                or status.get("profileHome") != str(home) or not isinstance(status.get("enabled"), bool)):
            raise ValueError(f"Legacy scheduler identity is invalid: {path}")
        if path.resolve() in seen_status:
            continue
        seen_status.add(path.resolve())
        if status.get("status") in {"running", "recovery_required"} or status.get("runPending"):
            raise ValueError(f"Legacy auto-update outcome needs recovery before migration: {path}")
        if not status["enabled"]:
            _refuse_orphan_artifacts(context, home)
            continue
        if status.get("mode") != "scheduled" or not isinstance(status.get("planSchedule"), list):
            raise ValueError(f"Legacy scheduler activation is invalid: {path}")
        profile = home.name if home.parent.name == "profiles" else "default"
        command = installation_command(context.install, ["--profile", profile, "update", "auto", "run-scheduled"], home=home)
        spec = scheduler.SchedulerSpec(identity, command, home, status.get("schedule"), status["planSchedule"])
        actual = scheduler.paths(spec)
        if status.get("schedulerType") != actual["backend"] or status.get("schedulerPath") != str(actual["path"]):
            raise ValueError(f"Legacy scheduler path does not belong to this installation: {path}")
        schedules.append(LegacySchedule(home, path, status, spec))
    return schedules


def legacy_summary(schedules: list[LegacySchedule]) -> list[dict]:
    return [{"home": str(item.home), "schedule": item.spec.schedule,
             "planSchedule": list(item.spec.plan_times), "schedulerPath": str(scheduler.paths(item.spec)["path"])}
            for item in schedules]


def migration_times(schedules: list[LegacySchedule]) -> tuple[str, list[str]]:
    settings = {(item.spec.schedule, tuple(sorted(item.spec.plan_times))) for item in schedules}
    if len(settings) != 1:
        choices = "; ".join(f"{item.home}: {item.spec.schedule}, plan={list(item.spec.plan_times)}" for item in schedules)
        raise ValueError(f"Conflicting legacy auto-update schedules: {choices}. "
                         "Choose one installation schedule with hermes update auto enable --time HH:MM [--plan-time HH:MM].")
    time, plans = settings.pop()
    return time, list(plans)


def check_legacy_channels(context: AutoUpdateContext, schedules: list[LegacySchedule]) -> None:
    from hermes_cli.config import require_readable_config_before_write
    from hermes_cli.update_channel import channel_record, resolve_update_channel
    from hermes_cli.update_installation import read_install_channel_record

    record = read_install_channel_record(context.install, home=context.home)
    if record.get("scope") == "installation":
        return  # An explicit installation-wide selection already supersedes old profile records.
    choices = [(item.home, resolve_update_channel(
        require_readable_config_before_write(item.home / "config.yaml"), context.install)) for item in schedules]
    for home in profile_homes(context.home):
        configured = channel_record(require_readable_config_before_write(home / "config.yaml"), context.install)
        if configured.get("channel") is not None:
            choices.append((home, configured["channel"]))
    if len({channel for _, channel in choices}) > 1:
        detail = "; ".join(f"{home}: {channel}" for home, channel in choices)
        raise ValueError(f"Conflicting legacy scheduler channels: {detail}. "
                         "Choose one installation channel with hermes update --set-channel CHANNEL.")


def _rollback(context, state_snapshot, snapshots, handles) -> dict:
    receipts = [handle.rollback() for handle in reversed(handles)]
    receipts.extend(restore_file(path, snapshot) for path, snapshot in snapshots.items())
    receipts.append(restore_file(context.status_path, state_snapshot))
    return {"ok": all(item.get("ok") for item in receipts), "operations": receipts}


def _saved_artifacts(schedules: list[LegacySchedule]) -> list[dict]:
    artifacts = []
    for item in schedules:
        paths = [item.path, *(path for key, path in scheduler.paths(item.spec).items()
                              if key in {"path", "service_path"})]
        for path in paths:
            saved = snapshot_file(path)
            artifacts.append({"path": str(path), "contentBase64": base64.b64encode(saved[0]).decode() if saved else None,
                              "mode": saved[1] if saved else None})
    return artifacts


def replace_legacy_schedulers(context: AutoUpdateContext, status: dict, schedules: list[LegacySchedule],
                              spec: scheduler.SchedulerSpec | None) -> None:
    """Caller holds installation and legacy operation locks; None disables all schedules."""
    if legacy_schedules(context) != schedules:
        raise ValueError("Legacy scheduler configuration changed; retry migration")
    _replace_locked(context, status, schedules, spec)


def _replace_locked(context, status, schedules, spec) -> None:
    from hermes_cli.update_auto import _configured_spec

    state_snapshot = snapshot_file(context.status_path)
    snapshots = {item.path: snapshot_file(item.path) for item in schedules}
    handles = []
    migration = {"startedAt": utc_now(), "sources": legacy_summary(schedules),
                 "previousStatuses": [item.status for item in schedules],
                 "previousArtifacts": _saved_artifacts(schedules)}
    # A crash leaves durable evidence and refuses another unattended run. The old
    # statuses and scheduler rollback receipts remain available for recovery.
    write_status(context, {**status, "status": "migration_running", "migration": migration})
    try:
        for item in schedules:
            handles.append(scheduler.disable(item.spec))
        if spec is not None:
            handles.append(scheduler.enable(spec))
        elif status["enabled"]:
            handles.append(scheduler.disable(_configured_spec(context, status)))
        for item in schedules:
            atomic_json_write(item.path, {**item.status, "enabled": False, "mode": "manual",
                                         "status": "migrated", "migratedTo": str(context.status_path)},
                              indent=2, mode=0o600, fsync_dir=True)
        fields = {"enabled": False, "mode": "manual", "schedule": None, "planSchedule": [],
                  "schedulerType": None, "schedulerPath": None}
        if spec is not None:
            fields.update(enabled=True, mode="scheduled", schedule=spec.schedule,
                          planSchedule=list(spec.plan_times), schedulerType=handles[-1].scheduler_type,
                          schedulerPath=str(handles[-1].path))
        migration["finishedAt"] = utc_now()
        status.update(fields, status="migrated", migration=migration, error=None)
        write_status(context, status)
    except (OSError, ValueError, RuntimeError) as exc:
        receipt = _rollback(context, state_snapshot, snapshots, handles)
        if not receipt["ok"]:
            raise scheduler.SchedulerRecoveryError(f"Legacy scheduler migration failed: {exc}", receipt) from exc
        raise
