"""In-place credential status resets for explicitly scoped auth stores."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import time


def reset_credential_pool_statuses(
    provider_id: str,
    *,
    auth_file: Path | None = None,
    credential_ids: Iterable[str] | None = None,
) -> int:
    """Clear cooldown state in exactly one store; return the number of rows cleared.

    Pool reads include root fallback rows, but a scoped reset must edit each local store
    without copying credentials between profiles. Match ``CredentialPool.reset_status``:
    clear status fields, drop model cooldowns and failure reason, and stamp
    ``status_cleared_at`` so a live session cannot restore stale exhaustion (#89415).

    Without *credential_ids* only rows carrying error state are cleared; otherwise clear
    exactly the listed ids, whatever their state.
    """
    from hermes_cli import auth

    wanted = {cid for cid in credential_ids if cid} if credential_ids is not None else None
    target_path = auth_file if auth_file is not None else auth._auth_file_path()
    if not target_path.exists():
        return 0
    with auth._auth_store_lock(target_path=target_path):
        auth_store = auth._load_auth_store(target_path)
        pool = auth_store.get("credential_pool")
        entries = pool.get(provider_id) if isinstance(pool, dict) else None
        if not isinstance(entries, list):
            return 0
        cleared_at = time.time()
        count = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if wanted is not None:
                if entry.get("id") not in wanted:
                    continue
            elif not any(
                entry.get(field) for field in (
                    "last_status", "last_status_at", "last_error_code", "failure_reason", "model_cooldowns")
            ):
                continue
            for field in auth._POOL_STATUS_FIELDS:
                entry[field] = None
            entry.pop("model_cooldowns", None)
            entry.pop("failure_reason", None)
            entry["status_cleared_at"] = cleared_at
            count += 1
        if count:
            auth._save_auth_store(auth_store, target_path=target_path)
        return count
