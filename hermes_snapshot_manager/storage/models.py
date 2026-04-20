from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SnapshotSummary:
    id: str
    created_at: str
    label: str | None
    trigger_type: str
    status: str
    total_files: int
    total_bytes: int
    is_known_good: bool


@dataclass(slots=True)
class RestoreRecord:
    id: int
    snapshot_id: str
    restored_at: str
    result: str
    pre_restore_snapshot_id: str | None
    notes: str | None
