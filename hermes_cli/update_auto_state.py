"""Profile-owned state for the opt-in update scheduler.

The operation mutex is separate from the updater's checkout lock: holding the
checkout lock while waiting for a fresh updater would deadlock its admission.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from hermes_constants import get_default_hermes_root, get_hermes_home, mkdir_under_hermes_home
from hermes_cli.update_lock import marker_mutex
from utils import atomic_json_write


@dataclass(frozen=True)
class AutoUpdateContext:
    install: Path
    home: Path
    receipt_directory: Path

    @classmethod
    def current(cls) -> AutoUpdateContext:
        from hermes_cli.config import get_project_root
        from hermes_cli.update_owning_install import owning_install_root

        # Keep a symlinked named profile's lexical provenance, as the CLI does.
        home = get_hermes_home().absolute()
        root = get_project_root().resolve()
        root = owning_install_root(root) or root
        return cls(root, home,
                   get_default_hermes_root(home=home) / "logs" / "update_receipts")

    @property
    def identity(self) -> str:
        # Python versions/dependency generations change during an update.
        material = f"{self.install}\0{self.home}"
        return "v1-" + hashlib.sha256(material.encode()).hexdigest()[:24]

    @property
    def status_path(self) -> Path:
        return self.home / "state" / "update-status.json"

    @property
    def log_path(self) -> Path:
        return self.home / "logs" / "update.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_status(context: AutoUpdateContext) -> dict:
    return {
        "schema": 2, "enabled": False, "mode": "manual", "schedule": None,
        "planSchedule": [], "schedulerIdentity": context.identity,
        "installationRoot": str(context.install), "profileHome": str(context.home),
        "status": "not_configured", "logPath": str(context.log_path),
    }


def read_status(context: AutoUpdateContext) -> dict:
    path = context.status_path
    if path.is_symlink():
        raise ValueError(f"Auto-update status must not be a symlink: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default_status(context)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read auto-update status at {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != 2:
        raise ValueError("Unrecognized auto-update status schema; inspect the existing schedule before replacing it")
    expected = default_status(context)
    for key in ("schedulerIdentity", "installationRoot", "profileHome"):
        if data.get(key) != expected[key]:
            raise ValueError(f"Auto-update {key} does not belong to this installation/profile")
    if not isinstance(data.get("enabled"), bool):
        raise ValueError("Auto-update enabled must be a boolean")
    return data


def write_status(context: AutoUpdateContext, status: dict) -> None:
    # Called only while operation_lock is held, including late subprocess results.
    mkdir_under_hermes_home(context.status_path.parent)
    if context.status_path.is_symlink():
        raise ValueError("Refusing to overwrite a symlinked auto-update status")
    atomic_json_write(context.status_path, status, indent=2, mode=0o600, fsync_dir=True)


@contextmanager
def operation_lock(context: AutoUpdateContext):
    mkdir_under_hermes_home(context.status_path.parent)
    lock_path = context.status_path.with_name("update-auto-operation")
    if lock_path.with_name(lock_path.name + ".lock").is_symlink():
        raise ValueError("Refusing a symlinked auto-update operation lock")
    with marker_mutex(lock_path, wait=0):
        yield


def append_log(context: AutoUpdateContext, event: str, **fields) -> None:
    mkdir_under_hermes_home(context.log_path.parent)
    if context.log_path.is_symlink():
        raise ValueError("Refusing a symlinked auto-update log")
    with context.log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": utc_now(), "event": event, **fields}, ensure_ascii=False) + "\n")
