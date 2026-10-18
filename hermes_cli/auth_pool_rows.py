"""Merging one writer's credential-pool rows into the rows already on disk.

Several Hermes processes share one ``auth.json``. Each holds an in-memory snapshot of the pool and
rewrites whole rows when it persists, so every field that another process may have changed since
the snapshot needs a rule for which side wins.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional


_POOL_STATUS_FIELDS = (
    "last_status", "last_status_at", "last_error_code", "last_error_reason", "last_error_message",
    "last_error_reset_at", "status_cleared_at")
_POOL_TOKEN_GENERATION_FIELDS = (
    "access_token", "refresh_token", "expires_at", "expires_at_ms", "expires_in", "obtained_at",
    "last_refresh", "agent_key", "agent_key_expires_at", "agent_key_expires_in", "agent_key_id",
    "agent_key_obtained_at", "agent_key_reused",
    # Refresh-coupled metadata: a Nous refresh rewrites scope and the validated
    # inference route together with the new pair, so they travel with it.
    "scope", "inference_base_url",
)


def _credential_token_pair(row: Any) -> tuple[Any, Any]:
    if not isinstance(row, dict):
        return (None, None)
    return row.get("access_token"), row.get("refresh_token")


def _token_pairs_by_id(rows: Iterable[Any]) -> dict[str, tuple[Any, Any]]:
    """Token-generation base per row id, INCLUDING ``(None, None)`` for token-less rows.

    A blank base is a known generation ("no pair when we last looked"), so a peer that
    later lands a pair on that row is kept by ``_merge_pool_row_generation`` on every
    flush alike; dropping blank bases would make the first and later flushes disagree."""
    return {row_id: _credential_token_pair(row) for row_id, row in _entry_ids(rows).items()}


def _merge_pool_row_generation(
    entry: dict[str, Any],
    disk_entry: Optional[dict[str, Any]],
    provider_id: str,
    *,
    base_pair: Optional[tuple[Any, Any]] = None,
    status_cleared: bool = False,
) -> dict[str, Any]:
    """Keep a newer on-disk token generation authoritative during stale writes.

    Only a terminal auth verdict (``last_status == dead``) is scoped to the token pair
    it was observed on; account-wide cooldowns (402 billing, 429 throttle) from the
    stale writer still apply to the rotated pair and go through the ordinary recency
    merge in ``_merge_disk_cooldown_state``."""
    from agent.credential_pool import STATUS_DEAD

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
    entry: dict[str, Any], disk_entry: Optional[dict[str, Any]], provider_id: str,
) -> dict[str, Any]:
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
        from agent.credential_pool import (
            PooledCredential, STATUS_DEAD, STATUS_EXHAUSTED, _exhausted_until, _parse_absolute_timestamp,
        )

        # Model cooldowns are independent observations: keep the latest reset per model so a
        # writer that just cooled one model cannot erase another process's cooldown for another.
        from agent.credential_pool_model_cooldowns import merge_model_cooldowns
        merged_cooldowns = merge_model_cooldowns(disk_entry.get("model_cooldowns"), entry.get("model_cooldowns"))
        merged = {**entry, "model_cooldowns": merged_cooldowns} if merged_cooldowns else entry
        disk_status_fields = {f: disk_entry.get(f) for f in _POOL_STATUS_FIELDS}

        mem_ts = _parse_absolute_timestamp(entry.get("last_status_at")) or 0.0
        cleared_ts = _parse_absolute_timestamp(disk_entry.get("status_cleared_at")) or 0.0
        if entry.get("last_status") in (STATUS_DEAD, STATUS_EXHAUSTED) and cleared_ts > mem_ts:
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
            until = _exhausted_until(PooledCredential.from_dict(provider_id, disk_entry))
            if until is None or until <= time.time():
                return merged
        return {**merged, **disk_status_fields}
    except Exception:  # pragma: no cover - best-effort merge
        return entry


def _entry_ids(entries: Iterable[Any]) -> dict[str, dict[str, Any]]:
    return {e.get("id"): e for e in entries if isinstance(e, dict) and e.get("id")}
