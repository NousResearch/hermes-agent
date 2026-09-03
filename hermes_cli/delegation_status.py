"""Read-only durable observer for background delegation state."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


class DelegationStatusReadError(RuntimeError):
    """The durable delegation store could not be read reliably."""


def _read_only_uri(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    return f"{resolved.as_uri()}?mode=ro"


def query_delegation_status(
    delegation_id: str,
    *,
    db_path: str | Path,
) -> Optional[dict[str, Any]]:
    """Return one durable delegation row, ``None`` only for an absent id.

    The SQLite connection uses ``mode=ro`` and never initializes or migrates
    schema. Missing, unreadable, corrupt, or incompatible databases raise a
    typed error so callers cannot mistake uncertainty for absence.
    """
    path = Path(db_path)
    try:
        conn = sqlite3.connect(_read_only_uri(path), uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """SELECT delegation_id, origin_session, origin_ui_session_id,
                          parent_session_id, state, dispatched_at,
                          completed_at, updated_at, result_json,
                          delivery_state, delivery_attempts, delivered_at,
                          owner_pid, task_json
                   FROM async_delegations
                   WHERE delegation_id = ?""",
                (delegation_id,),
            ).fetchone()
        finally:
            conn.close()
    except (OSError, sqlite3.Error) as exc:
        raise DelegationStatusReadError(
            f"cannot read delegation state from {path}: {exc}"
        ) from exc

    if row is None:
        return None
    result: dict[str, Any] = dict(row)
    raw_result_json = result.pop("result_json", None)
    if raw_result_json:
        try:
            result["result"] = json.loads(raw_result_json)
        except (json.JSONDecodeError, TypeError):
            result["result"] = raw_result_json
    else:
        result["result"] = None
    raw_task_json = result.pop("task_json", None)
    if raw_task_json:
        try:
            result["task"] = json.loads(raw_task_json)
        except (json.JSONDecodeError, TypeError):
            result["task"] = None
    else:
        result["task"] = None
    return result
