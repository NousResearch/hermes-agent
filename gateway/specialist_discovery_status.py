"""Read-only specialist-discovery status projection for diagnostics."""

from pathlib import Path
from typing import Any


def specialist_discovery_status(
    candidate_id: str,
    *,
    db_path: Path | None = None,
    board: str | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    """Return sanitized lifecycle rows without replaying or mutating discovery."""
    from gateway.lifecycle_ledger import recover_specialist_discovery

    recovery = recover_specialist_discovery(candidate_id, db_path=db_path, board=board, now=now)
    return {
        "candidate_id": recovery.candidate_id,
        "recovery_action": recovery.recovery_action,
        "rows": [
            {
                "stage": row.stage,
                "status": row.status,
                "receipt_hash": row.receipt_hash,
                "expires_at": row.expires_at,
            }
            for row in recovery.rows
        ],
    }
