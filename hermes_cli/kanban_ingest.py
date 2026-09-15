"""Fail-closed ingest pipeline for isolation-scoped Kanban tasks.

The ordering is deliberately fixed: kill -> validate -> diff -> quarantine.
Quarantine is represented by the existing ``tasks.block_kind='scope'`` state and
an auditable ``task_events`` row; there is no second quarantine column.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

# These values are task policy names, not Hermes profile names.  Unknown values
# are intentionally not treated as isolated: policy owners must add them here.
ISOLATION_ROLES = frozenset({"security-engineer", "adversarial-reviewer", "isolated-agent"})
ISOLATION_CARD_CLASSES = frozenset({"isolated", "isolation-required", "blind", "single-blind"})


@dataclass(frozen=True)
class IngestResult:
    ok: bool
    stage: str
    changed_paths: tuple[str, ...] = ()
    reason: Optional[str] = None


def isolation_required(task: Any) -> bool:
    """Return whether explicit task policy requires the isolation check."""
    role = str(getattr(task, "role", None) or "").strip().casefold()
    card_class = str(getattr(task, "card_class", None) or "").strip().casefold()
    return role in ISOLATION_ROLES or card_class in ISOLATION_CARD_CLASSES


def _manifest(task: Any) -> Optional[dict]:
    value = getattr(task, "scope_manifest", None)
    if isinstance(value, dict):
        return value
    raw = value
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def _kill(task: Any) -> None:
    """Best-effort kill stage; sandbox enforcement is not claimed here."""
    pid = getattr(task, "worker_pid", None)
    if not pid or int(pid) == os.getpid():
        return
    try:
        os.kill(int(pid), signal.SIGTERM)
        _log.info("isolation ingest kill stage: sent SIGTERM to worker pid %s", pid)
    except (ProcessLookupError, PermissionError, ValueError, OSError):
        _log.debug("isolation ingest kill stage: worker pid %s already gone", pid)


def _validate(task: Any) -> tuple[bool, str]:
    manifest = _manifest(task)
    if manifest is None:
        return False, "isolation-required task has no valid scope_manifest"
    if not isinstance(manifest.get("read", []), list) or not isinstance(manifest.get("write", []), list):
        return False, "scope_manifest must contain list-valued read and write entries"
    for key in ("read", "write"):
        for path in manifest[key]:
            if not isinstance(path, str) or not path.strip() or Path(path).is_absolute() or ".." in Path(path).parts:
                return False, f"scope_manifest.{key} contains an unsafe path"
    return True, "ok"


def _diff(task: Any) -> tuple[bool, tuple[str, ...], str]:
    workspace = getattr(task, "workspace_path", None)
    if not workspace:
        return False, (), "isolation-required task has no workspace_path"
    try:
        # ``diff HEAD`` omits untracked files, which is unsafe for ingest. Status
        # includes tracked edits, deletions, renames, and every untracked path.
        result = subprocess.run(
            ["git", "-C", str(workspace), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, (), f"unable to compute workspace diff: {type(exc).__name__}"
    if result.returncode != 0:
        return False, (), "workspace is not a readable git worktree"
    paths = []
    for line in result.stdout.splitlines():
        if not line or len(line) < 4:
            continue
        path = line[3:].strip()
        # Porcelain rename is ``old -> new``; authorize the resulting path and
        # retain old too so a scoped rename cannot hide an unauthorized source.
        if " -> " in path:
            old, new = path.split(" -> ", 1)
            paths.extend((old, new))
        else:
            paths.append(path)
    paths = tuple(sorted(set(paths)))
    writes = [str(p) for p in (_manifest(task) or {}).get("write", [])]
    if not writes:
        return False, paths, "scope_manifest.write is empty; refusing unscoped ingest"
    unauthorized = [p for p in paths if not any(fnmatch.fnmatchcase(p, pattern) for pattern in writes)]
    if unauthorized:
        return False, paths, "diff contains paths outside scope_manifest.write: " + ", ".join(unauthorized[:10])
    return True, paths, "ok"


def _quarantine(conn: Any, task_id: str, reason: str, expected_run_id: Optional[int]) -> None:
    """Persist canonical quarantine state, then append its audit event."""
    from hermes_cli import kanban_db as kb
    # block_task commits the state transition. The event follows in a tiny
    # transaction so claim/block CAS remains the single source of truth.
    if not kb.block_task(conn, task_id, reason=reason, kind="scope", expected_run_id=expected_run_id):
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "quarantine_failed", {"reason": reason})
        return
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, "quarantine", {"reason": reason, "block_kind": "scope"})


def run_ingest_pipeline(conn: Any, task: Any, *, expected_run_id: Optional[int] = None) -> IngestResult:
    """Run all stages for an isolation-implying task; failure quarantines it."""
    if not isolation_required(task):
        return IngestResult(True, "not_required")
    _log.warning("LOCAL BACKEND CAVEAT: isolation ingest is best-effort until Phase 1 bwrap sandboxing is confirmed enabled")
    _kill(task)
    valid, reason = _validate(task)
    if not valid:
        _quarantine(conn, task.id, reason, expected_run_id)
        return IngestResult(False, "validate", reason=reason)
    valid, paths, reason = _diff(task)
    if not valid:
        _quarantine(conn, task.id, reason, expected_run_id)
        return IngestResult(False, "diff", paths, reason)
    _log.info("isolation ingest passed: kill -> validate -> diff; local enforcement remains best-effort")
    return IngestResult(True, "diff", paths)
