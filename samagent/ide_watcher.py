"""Automatic External IDE File Watcher & Auto-Verifier (`samagent/ide_watcher.py`).

Detects when a developer edits and saves files in an external IDE (VS Code, Cursor, Zed, JetBrains,
Neovim) on their local machine without requiring any CLI command or manual sync button:
1. Computes SHA-256 digests of workspace source files (`app/**`, `.vscode/**`, `.samagent/spec.yaml`).
2. When a digest changes on disk, automatically runs L0–L4 + OWASP security verification.
3. Logs the external IDE change and verification result into the bi-temporal Project Ledger.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from samagent.conductor.verify import VerificationRunner
from samagent.ledger.store import ProjectLedger
from samagent.spec.models import SpecDocument


_WATCH_SNAPSHOTS: Dict[str, Dict[str, str]] = {}
_LAST_SYNC_EVENTS: Dict[str, Dict[str, Any]] = {}


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
        return h.hexdigest()
    except Exception:
        return ""


def compute_workspace_digests(project_dir: Path) -> Dict[str, str]:
    """Return relative_path -> sha256 for user-editable source and config files."""
    root = Path(project_dir).resolve()
    digests: Dict[str, str] = {}
    if not root.exists():
        return digests

    watch_Suffixes = {".py", ".html", ".css", ".js", ".ts", ".tsx", ".json", ".yaml", ".yml", ".sql"}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if (
            rel.startswith(".git/")
            or rel.startswith(".samagent/runs/")
            or rel.startswith(".samagent/releases/")
            or rel.startswith(".samagent/worktrees/")
            or "__pycache__" in rel
            or rel.endswith(".sqlite3")
            or rel.endswith(".db")
        ):
            continue
        if p.suffix.lower() in watch_Suffixes:
            digests[rel] = _hash_file(p)
    return digests


def check_and_sync_external_edits(project_dir: Path) -> Dict[str, Any]:
    """Compare on-disk file digests against the last known snapshot.

    If files were modified externally in VS Code / Cursor, automatically re-run L0–L4 verification,
    write a new deliverable snapshot, and record the IDE edit in the Project Ledger.
    """
    root = Path(project_dir).resolve()
    key = str(root)
    current = compute_workspace_digests(root)

    if key not in _WATCH_SNAPSHOTS:
        _WATCH_SNAPSHOTS[key] = current
        return {
            "changed": False,
            "changed_files": [],
            "tracked_file_count": len(current),
            "last_sync": _LAST_SYNC_EVENTS.get(key),
        }

    prev = _WATCH_SNAPSHOTS[key]
    changed_files: List[str] = []
    for rel, sha in current.items():
        if prev.get(rel) != sha:
            changed_files.append(rel)
    for rel in prev:
        if rel not in current:
            changed_files.append(rel)

    if not changed_files:
        return {
            "changed": False,
            "changed_files": [],
            "tracked_file_count": len(current),
            "last_sync": _LAST_SYNC_EVENTS.get(key),
        }

    # Update snapshot immediately so we don't loop
    _WATCH_SNAPSHOTS[key] = current

    # Run automatic L0-L4 + OWASP verification on the externally edited workspace
    spec = SpecDocument.load(root)
    verifier = VerificationRunner(root, spec)
    report = verifier.run_all()

    ts = int(time.time())
    run_id = f"run_ide_autosync_{ts}"
    runs_dir = root / ".samagent" / "runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    deliverable = {
        "run_id": run_id,
        "goal": spec.goal,
        "stack": spec.stack,
        "trigger": "external_ide_save",
        "changed_files": changed_files,
        "verification": report.to_dict(),
    }
    (runs_dir / "deliverable.json").write_text(json.dumps(deliverable, indent=2), encoding="utf-8")

    # Record in bi-temporal Project Ledger
    try:
        ledger = ProjectLedger(root)
        status_str = "PASSED L0-L4" if report.passed else f"BLOCKED ({', '.join(report.failed_levels)})"
        ledger.record_fact(
            scope="ide_sync",
            kind="external_edit",
            text=f"VS Code / IDE modified {', '.join(changed_files[:4])} -> Auto-Verify: {status_str}",
            source_ref="ide_watcher",
        )
    except Exception:
        pass

    event = {
        "timestamp": ts,
        "run_id": run_id,
        "changed_files": changed_files,
        "verification_passed": report.passed,
        "failed_levels": report.failed_levels,
    }
    _LAST_SYNC_EVENTS[key] = event
    return {
        "changed": True,
        "changed_files": changed_files,
        "tracked_file_count": len(current),
        "last_sync": event,
    }
