"""Durable pool snapshots, token generations and concurrent cooldown merges."""

from __future__ import annotations
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from auth import store
from auth.persistence import sanitize_borrowed_credential_payload


def read_credential_pool(provider_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the persisted credential pool, or one provider slice.

    In profile mode the global-root ``auth.json`` is a read-only fallback applied per provider ONLY
    when the profile has zero entries for it (``hermes auth add`` in the profile shadows global)."""
    pool = store._load_auth_store().get("credential_pool")
    pool = pool if isinstance(pool, dict) else {}
    global_pool = store._load_global_auth_store().get("credential_pool")
    global_pool = global_pool if isinstance(global_pool, dict) else {}

    if provider_id is None:
        merged = dict(pool)
        for gp_key, gp_entries in global_pool.items():
            existing = merged.get(gp_key)
            if not (isinstance(gp_entries, list) and gp_entries):
                continue
            if not (
                isinstance(existing, list) and existing
            ):  # profile wins when it has ANY entries
                merged[gp_key] = list(gp_entries)
        return merged

    provider_entries = pool.get(provider_id)
    if isinstance(provider_entries, list) and provider_entries:
        return list(provider_entries)
    global_entries = global_pool.get(provider_id)
    return list(global_entries) if isinstance(global_entries, list) else []


_POOL_STATUS_FIELDS = (
    "last_status",
    "last_status_at",
    "last_error_code",
    "last_error_reason",
    "last_error_message",
    "last_error_reset_at",
    "status_cleared_at",
)


_POOL_TOKEN_GENERATION_FIELDS = (
    "access_token",
    "refresh_token",
    "expires_at",
    "expires_at_ms",
    "expires_in",
    "obtained_at",
    "last_refresh",
    "agent_key",
    "agent_key_expires_at",
    "agent_key_expires_in",
    "agent_key_id",
    "agent_key_obtained_at",
    "agent_key_reused",
    # Refresh-coupled metadata: a Nous refresh rewrites scope and the validated
    # inference route together with the new pair, so they travel with it.
    "scope",
    "inference_base_url",
)


def _credential_token_pair(row: Any) -> Tuple[Any, Any]:
    if not isinstance(row, dict):
        return (None, None)
    return row.get("access_token"), row.get("refresh_token")


def _token_pairs_by_id(rows: Iterable[Any]) -> Dict[str, Tuple[Any, Any]]:
    """Token-generation base per row id, INCLUDING ``(None, None)`` for token-less rows.

    A blank base is a known generation ("no pair when we last looked"), so a peer that
    later lands a pair on that row is kept by ``_merge_pool_row_generation`` on every
    flush alike; dropping blank bases would make the first and later flushes disagree."""
    return {
        row_id: _credential_token_pair(row) for row_id, row in _entry_ids(rows).items()
    }


def _merge_pool_row_generation(
    entry: Dict[str, Any],
    disk_entry: Optional[Dict[str, Any]],
    provider_id: str,
    *,
    base_pair: Optional[Tuple[Any, Any]] = None,
    status_cleared: bool = False,
) -> Dict[str, Any]:
    """Keep a newer on-disk token generation authoritative during stale writes.

    Only a terminal auth verdict (``last_status == dead``) is scoped to the token pair
    it was observed on; account-wide cooldowns (402 billing, 429 throttle) from the
    stale writer still apply to the rotated pair and go through the ordinary recency
    merge in ``_merge_disk_cooldown_state``."""
    from auth.credential_pool import STATUS_DEAD

    merge_disk = None if status_cleared else disk_entry
    disk_pair = _credential_token_pair(disk_entry)
    if base_pair is None or not any(disk_pair) or disk_pair == base_pair:
        return _merge_disk_cooldown_state(entry, merge_disk, provider_id)

    merged = dict(entry)

    def _take_from_disk(fields: Iterable[str]) -> None:
        # Absent-on-disk fields are popped, not set to None: a None would make the
        # UPDATE-only root merge see a changed row and force a spurious save.
        for field in fields:
            if field in disk_entry:
                merged[field] = disk_entry[field]
            else:
                merged.pop(field, None)

    _take_from_disk(_POOL_TOKEN_GENERATION_FIELDS)
    if not status_cleared and entry.get("last_status") == STATUS_DEAD:
        _take_from_disk((*_POOL_STATUS_FIELDS, "failure_reason"))
    return _merge_disk_cooldown_state(merged, merge_disk, provider_id)


def _merge_disk_cooldown_state(
    entry: Dict[str, Any],
    disk_entry: Optional[Dict[str, Any]],
    provider_id: str,
) -> Dict[str, Any]:
    """Keep a newer on-disk cooldown/quarantine over a stale in-memory one.

    ``write_credential_pool`` persists an in-memory snapshot that may predate another process
    marking the same credential exhausted/dead; without this merge the later rewrite resurrects a
    rate-limited key as healthy and both processes resume hammering it. The mirror image is a
    ``hermes auth reset`` that postdates the snapshot's cooldown (``status_cleared_at`` newer than
    its ``last_status_at``): the disk row wins there too, or a live session's next ordinary flush
    would write the reset cooldown straight back (#89415)."""
    if not isinstance(disk_entry, dict):
        return entry
    try:
        from auth.credential_pool import (
            PooledCredential,
            STATUS_DEAD,
            STATUS_EXHAUSTED,
            _exhausted_until,
            _parse_absolute_timestamp,
        )

        # Model cooldowns are independent observations: keep the latest reset per model so a
        # writer that just cooled one model cannot erase another process's cooldown for another.
        from auth.credential_pool_model_cooldowns import merge_model_cooldowns

        merged_cooldowns = merge_model_cooldowns(
            disk_entry.get("model_cooldowns"), entry.get("model_cooldowns")
        )
        merged = (
            {**entry, "model_cooldowns": merged_cooldowns}
            if merged_cooldowns
            else entry
        )
        disk_status_fields = {f: disk_entry.get(f) for f in _POOL_STATUS_FIELDS}

        mem_ts = _parse_absolute_timestamp(entry.get("last_status_at")) or 0.0
        cleared_ts = (
            _parse_absolute_timestamp(disk_entry.get("status_cleared_at")) or 0.0
        )
        if (
            entry.get("last_status") in (STATUS_DEAD, STATUS_EXHAUSTED)
            and cleared_ts > mem_ts
        ):
            return {**merged, **disk_status_fields}
        disk_status = disk_entry.get("last_status")
        if disk_status not in (STATUS_DEAD, STATUS_EXHAUSTED):
            return merged
        # A token change means the caller re-authed this entry and intentionally cleared its status:
        # never resurrect the old cooldown onto fresh credentials.
        mem_access = entry.get("access_token") or ""
        disk_access = disk_entry.get("access_token") or ""
        if mem_access and disk_access and mem_access != disk_access:
            return entry
        disk_ts = _parse_absolute_timestamp(disk_entry.get("last_status_at")) or 0.0
        if disk_ts <= mem_ts:
            return merged
        if disk_status == STATUS_EXHAUSTED:
            until = _exhausted_until(
                PooledCredential.from_dict(provider_id, disk_entry)
            )
            if until is None or until <= time.time():
                return merged
        return {**merged, **disk_status_fields}
    except Exception:  # pragma: no cover - best-effort merge
        return entry


def _entry_ids(entries: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    return {e.get("id"): e for e in entries if isinstance(e, dict) and e.get("id")}


def write_credential_pool(
    provider_id: str,
    entries: List[Dict[str, Any]],
    *,
    removed_ids: Optional[Iterable[str]] = None,
    status_cleared_ids: Optional[Iterable[str]] = None,
    token_bases: Optional[Dict[str, Tuple[Any, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Persist one provider's credential pool under auth.json.

    Final disk-boundary sanitizer for borrowed credentials (callers may pass raw dicts). Entries on
    disk but missing from *entries* (added concurrently) are merged back unless in *removed_ids*,
    so a rotation/exhaustion rewrite never drops a concurrent credential. Entries in
    *status_cleared_ids* were cleared deliberately (``hermes auth reset``) and skip the
    recency merge, which would otherwise read their cleared ``last_status_at`` (None ->
    epoch 0) as a stale snapshot and copy a still-binding cooldown back."""
    removed = {rid for rid in (removed_ids or ()) if rid}
    bases = token_bases or {}
    with store._auth_store_lock():
        auth_store = store._load_auth_store()
        pool = store._store_section(auth_store, "credential_pool")
        sanitized = [
            sanitize_borrowed_credential_payload(e, provider_id)
            if isinstance(e, dict)
            else e
            for e in entries
        ]
        existing_list = pool.get(provider_id)
        existing_list = existing_list if isinstance(existing_list, list) else []
        existing_by_id = _entry_ids(existing_list)
        new_ids = set(_entry_ids(sanitized))
        status_cleared = {cid for cid in (status_cleared_ids or ()) if cid}
        merged: List[Dict[str, Any]] = [
            _merge_pool_row_generation(
                e,
                existing_by_id.get(e.get("id")),
                provider_id,
                base_pair=bases.get(e.get("id")),
                status_cleared=e.get("id") in status_cleared,
            )
            if isinstance(e, dict)
            else e
            for e in sanitized
        ]
        for disk_entry in existing_list:
            disk_id = disk_entry.get("id") if isinstance(disk_entry, dict) else None
            if disk_id and disk_id not in new_ids and disk_id not in removed:
                merged.append(
                    sanitize_borrowed_credential_payload(disk_entry, provider_id)
                )
        pool[provider_id] = merged
        store._save_auth_store(auth_store)
        return merged


def migrate_legacy_custom_pool_key(provider: str, legacy_key: str) -> bool:
    """Move a keyed provider's old ``custom:`` pool into its runtime slug."""
    with store._auth_store_lock():
        auth_store = store._load_auth_store()
        credential_pool = auth_store.get("credential_pool")
        if not isinstance(credential_pool, dict):
            return False
        legacy_entries = credential_pool.get(legacy_key)
        if not isinstance(legacy_entries, list) or not legacy_entries:
            return False
        current_entries = credential_pool.get(provider)
        merged = list(current_entries) if isinstance(current_entries, list) else []
        known_ids = {e.get("id") for e in merged if isinstance(e, dict) and e.get("id")}
        for entry in legacy_entries:
            entry_id = entry.get("id") if isinstance(entry, dict) else None
            if not entry_id or entry_id not in known_ids:
                merged.append(entry)
                if entry_id:
                    known_ids.add(entry_id)
        credential_pool[provider] = merged
        del credential_pool[legacy_key]
        store._save_auth_store(auth_store)
        return True
