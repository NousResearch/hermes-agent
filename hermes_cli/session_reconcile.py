"""Safe reconciliation of orphaned Telegram group sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set

from hermes_cli.active_sessions import locked_active_session_ids
from hermes_state import SessionDB


def _legacy_routed_session_ids(sessions_file: Path) -> Set[str]:
    """Read legacy sessions.json references, failing closed on malformed data."""
    try:
        data = json.loads(sessions_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    except Exception as exc:
        raise RuntimeError(
            f"legacy session route registry is unreadable: {sessions_file}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"legacy session route registry is malformed: {sessions_file}"
        )

    routed: Set[str] = set()
    for key, entry in data.items():
        if str(key).startswith("_"):
            continue
        if not isinstance(entry, dict) or not entry.get("session_id"):
            raise RuntimeError(
                f"legacy session route registry is malformed: {sessions_file}"
            )
        routed.add(str(entry["session_id"]))
    return routed


def reconcile_telegram_group_orphans(
    db: SessionDB,
    *,
    sessions_dir: Path,
    min_age_seconds: float = 3600.0,
    apply: bool = False,
) -> Dict[str, Any]:
    """Preview or safely finalize unowned Telegram group sessions.

    The cross-process lease lock stays held from protection discovery through
    backup and the transactional database update. Canonical state.db routes are
    re-read by ``SessionDB`` within the same write transaction; the legacy JSON
    mirror is also treated as a protected source. Apply always creates and
    verifies a fresh online backup first.
    """
    sessions_dir = Path(sessions_dir)
    with locked_active_session_ids() as leased_ids:
        legacy_routed_ids = _legacy_routed_session_ids(
            sessions_dir / "sessions.json"
        )
        protected_ids = set(leased_ids) | legacy_routed_ids
        preview = db.reconcile_orphaned_telegram_group_sessions(
            protected_session_ids=protected_ids,
            min_age_seconds=min_age_seconds,
            apply=False,
        )
        if not apply:
            return {**preview, "backup_path": None}
        if not preview["candidate_ids"]:
            return {
                "candidate_ids": [],
                "finalized_ids": [],
                "dry_run": False,
                "backup_path": None,
            }

        backup_path = db.create_verified_backup("telegram-orphan-reconcile")
        result = db.reconcile_orphaned_telegram_group_sessions(
            protected_session_ids=protected_ids,
            min_age_seconds=min_age_seconds,
            apply=True,
        )
        return {**result, "backup_path": str(backup_path)}
