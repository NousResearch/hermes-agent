"""Owned resource leases for dispatcher-spawned Kanban workers.

The dispatcher gives each worker a task/run-scoped manifest and ownership labels.
Cleanup is deliberately label-scoped and best-effort: it can remove only
resources carrying both exact task and run labels, never an unlabeled or
persistent development resource.
"""

from __future__ import annotations

import json
import contextlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

_RESOURCE_SCHEMA = 1
_TASK_LABEL = "com.hermes.kanban.task_id"
_RUN_LABEL = "com.hermes.kanban.run_id"
_RESOURCE_LABEL = "com.hermes.kanban.resource"
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_KINDS = ("container", "network", "volume")

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess]


@dataclass(frozen=True)
class ResourceLease:
    task_id: str
    run_id: int
    board: str
    manifest_path: Path
    task_label: str
    run_label: str


def _validate_token(value: str, name: str) -> str:
    value = str(value).strip()
    if not value or not _TOKEN_RE.fullmatch(value):
        raise ValueError(f"invalid resource {name}")
    return value


def lease_for_run(
    *,
    board_dir: Path,
    board: str,
    task_id: str,
    run_id: int,
    workspace: Optional[str] = None,
) -> ResourceLease:
    """Create the durable lease manifest for one claimed worker run."""
    task_id = _validate_token(task_id, "task id")
    run_id = int(run_id)
    if run_id <= 0:
        raise ValueError("resource run id must be positive")
    board = _validate_token(board or "default", "board")
    root = board_dir / "resources" / task_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{run_id}.json"
    lease = ResourceLease(
        task_id=task_id,
        run_id=run_id,
        board=board,
        manifest_path=path,
        task_label=f"{_TASK_LABEL}={task_id}",
        run_label=f"{_RUN_LABEL}={run_id}",
    )
    _write_manifest(
        path,
        {
            "schema_version": _RESOURCE_SCHEMA,
            "task_id": task_id,
            "run_id": run_id,
            "board": board,
            "workspace": workspace,
            "status": "active",
            "created_at": int(time.time()),
            "labels": {
                "task": lease.task_label,
                "run": lease.run_label,
                "resource": f"{_RESOURCE_LABEL}=1",
            },
            "resources": {kind: [] for kind in _KINDS},
            "cleanup": {
                "status": "pending",
                "attempts": 0,
                "removed": {kind: [] for kind in _KINDS},
                "residual": {kind: [] for kind in _KINDS},
            },
        },
    )
    return lease


def lease_path(*, board_dir: Path, task_id: str, run_id: int) -> Path:
    """Return the deterministic manifest path without creating or rewriting it."""
    task_id = _validate_token(task_id, "task id")
    run_id = int(run_id)
    if run_id <= 0:
        raise ValueError("resource run id must be positive")
    return board_dir / "resources" / task_id / f"{run_id}.json"


def resource_environment(lease: ResourceLease) -> dict[str, str]:
    """Environment contract consumed by disposable executor wrappers."""
    return {
        "HERMES_KANBAN_RESOURCE_MANIFEST": str(lease.manifest_path),
        "HERMES_KANBAN_RESOURCE_TASK_LABEL": lease.task_label,
        "HERMES_KANBAN_RESOURCE_RUN_LABEL": lease.run_label,
        "HERMES_KANBAN_RESOURCE_LABEL": f"{_RESOURCE_LABEL}=1",
    }


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _resource_ids(
    kind: str,
    lease: ResourceLease,
    runner: CommandRunner,
) -> tuple[list[str], Optional[str]]:
    if kind == "container":
        command = ["docker", "ps", "-aq"]
    elif kind == "network":
        command = ["docker", "network", "ls", "-q"]
    elif kind == "volume":
        command = ["docker", "volume", "ls", "-q"]
    else:
        raise ValueError(f"unsupported resource kind: {kind}")
    command.extend([
        "--filter",
        f"label={lease.task_label}",
        "--filter",
        f"label={lease.run_label}",
    ])
    try:
        result = runner(command)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], str(exc)
    if result.returncode != 0:
        return [], (
            result.stderr or result.stdout or f"docker {kind} list failed"
        ).strip()
    return [line.strip() for line in result.stdout.splitlines() if line.strip()], None


def _remove_ids(
    kind: str, ids: list[str], runner: CommandRunner
) -> tuple[list[str], list[str]]:
    if not ids:
        return [], []
    command = {
        "container": ["docker", "rm", "-f"],
        "network": ["docker", "network", "rm"],
        "volume": ["docker", "volume", "rm"],
    }[kind]
    try:
        result = runner([*command, *ids])
    except (OSError, subprocess.SubprocessError):
        return [], list(ids)
    if result.returncode == 0:
        return list(ids), []
    # Docker can remove a subset before reporting a dependency/error. Readback
    # decides the true residual set; keep the attempted ids for the manifest.
    return [], list(ids)


def cleanup_owned_resources(
    lease: ResourceLease,
    *,
    runner: Optional[CommandRunner] = None,
) -> dict:
    """Reconcile only resources carrying this exact task/run ownership lease."""
    runner = runner or _default_runner
    result = {
        "status": "unavailable"
        if shutil.which("docker") is None and runner is _default_runner
        else "pending",
        "task_id": lease.task_id,
        "run_id": lease.run_id,
        "attempted_at": int(time.time()),
        "removed": {kind: [] for kind in _KINDS},
        "residual": {kind: [] for kind in _KINDS},
        "errors": [],
    }
    discovered: dict[str, list[str]] = {kind: [] for kind in _KINDS}
    for kind in _KINDS:
        ids, error = _resource_ids(kind, lease, runner)
        discovered[kind] = ids
        if error:
            result["errors"].append({"kind": kind, "error": error})
            continue
        removed, residual = _remove_ids(kind, ids, runner)
        result["removed"][kind] = removed
        remaining, read_error = _resource_ids(kind, lease, runner)
        result["residual"][kind] = remaining or residual
        if read_error:
            result["errors"].append({"kind": kind, "error": read_error})
    result["discovered"] = discovered
    if result["status"] != "unavailable":
        result["status"] = (
            "cleaned"
            if not any(result["residual"].values()) and not result["errors"]
            else "partial"
        )
    try:
        payload = json.loads(lease.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {
            "schema_version": _RESOURCE_SCHEMA,
            "task_id": lease.task_id,
            "run_id": lease.run_id,
        }
    payload["status"] = (
        "cleaned" if result["status"] == "cleaned" else "cleanup_pending"
    )
    payload["resources"] = discovered
    payload["cleanup"] = {
        "status": result["status"],
        "attempts": int(payload.get("cleanup", {}).get("attempts", 0)) + 1,
        "attempted_at": result["attempted_at"],
        "removed": result["removed"],
        "residual": result["residual"],
        "errors": result["errors"],
    }
    try:
        _write_manifest(lease.manifest_path, payload)
    except OSError as exc:
        result["errors"].append({"kind": "manifest", "error": str(exc)})
    return result


def lease_from_path(path: str | Path) -> Optional[ResourceLease]:
    """Recover a lease from a manifest path without trusting arbitrary fields."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        task_id = _validate_token(str(payload["task_id"]), "task id")
        run_id = int(payload["run_id"])
        board = _validate_token(str(payload["board"]), "board")
        if run_id <= 0:
            return None
        return ResourceLease(
            task_id=task_id,
            run_id=run_id,
            board=board,
            manifest_path=Path(path),
            task_label=f"{_TASK_LABEL}={task_id}",
            run_label=f"{_RUN_LABEL}={run_id}",
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def iter_leases(board_dir: Path) -> list[ResourceLease]:
    """Read manifests without mutating them; malformed manifests are ignored."""
    root = board_dir / "resources"
    if not root.is_dir():
        return []
    leases: list[ResourceLease] = []
    for path in sorted(root.glob("*/*.json")):
        lease = lease_from_path(path)
        if lease is not None:
            leases.append(lease)
    return leases
