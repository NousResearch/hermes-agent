"""Offline, non-destructive recovery for a damaged Hermes session database.

The store admin's ``recover_transcripts`` runs it: the source is never opened by SQLite (it and its WAL/SHM/journal
sidecars are copied to a disposable work dir first), canonical rows are copied into a fresh current-schema database,
derived FTS tables and migration bookkeeping are rebuilt, not copied, and the result is never installed over the
active database.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from persistence.core.admin import SessionRecoveryError, SessionRecoverySafetyError, SessionRecoverySourceError
from persistence_assembly import open_store_admin

__all__ = [
    "SessionRecoveryError", "SessionRecoverySafetyError", "SessionRecoverySourceError", "inspect_session_database",
    "recover_session_database", "write_recovery_report",
]

ProgressCallback = Callable[[dict[str, Any]], None]


def _resolved_output_path(path: Path) -> Path:
    """Resolve a not-yet-created output path without requiring it to exist."""
    return path.expanduser().parent.resolve(strict=True) / path.name


def inspect_session_database(source_path: Path, *, work_dir: Optional[Path] = None) -> dict[str, Any]:
    """Inspect canonical table readability without opening the source itself."""
    # An inspection writes nothing, so the output argument goes unused.
    return open_store_admin(source_path).recover_transcripts(source_path, source_path, inspect_only=True,
                                                             work_dir=work_dir)


def recover_session_database(
    source_path: Path, output_path: Path, *, work_dir: Optional[Path] = None, chunk_size: int = 1_000,
    progress_cb: Optional[ProgressCallback] = None, allow_partial: bool = False,
) -> dict[str, Any]:
    """Separate-output salvage is allowed; a canonical output requires offline ownership."""
    return open_store_admin(source_path).recover_transcripts(
        source_path, output_path, work_dir=work_dir, chunk_size=chunk_size, progress_cb=progress_cb,
        allow_partial=allow_partial,
    )


def write_recovery_report(path: Path, report: dict[str, Any]) -> Path:
    """Write a JSON report without overwriting an existing file."""
    destination = _resolved_output_path(path)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination
