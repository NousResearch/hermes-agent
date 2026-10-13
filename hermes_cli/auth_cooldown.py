"""Model-cooldown merge and secret-rotation helpers for the credential-pool disk boundary.

Extracted from ``hermes_cli.auth`` so that module stays under its code-health
FILE_LINES cap. The timestamp parser lives in ``agent.credential_pool_cooldowns``
and is imported from there directly.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from agent.credential_pool_cooldowns import _parse_absolute_timestamp


def _merge_model_cooldown_state(
    disk_entry: dict[str, Any], entry: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """Merge concurrent model cooldowns, never resurrecting one an explicit reset cleared (#128995).

    Returns the merged row and the disk-side clear timestamp. Model cooldowns are
    independent observations, so concurrent model failures merge, but a snapshot older
    than an explicit reset must not bring the map back.
    """
    from agent.credential_pool_model_cooldowns import (
        MODEL_COOLDOWN_OBSERVED_AT_KEY,
        MODEL_ENTITLEMENT_KEY,
        bound_rehydrated_model_cooldowns,
        merge_model_cooldown_observations,
        merge_model_cooldowns,
        model_cooldowns_after_clear,
    )

    disk_cleared_ts = _parse_absolute_timestamp(disk_entry.get("status_cleared_at")) or 0.0
    mem_cleared_ts = _parse_absolute_timestamp(entry.get("status_cleared_at")) or 0.0
    latest_clear = max(disk_cleared_ts, mem_cleared_ts)

    # The disk side gets the same stale-sentinel bound as a load: the merge keeps the later
    # value per model, so an unbounded legacy year-long bench on disk would otherwise be
    # written straight back and the bound would never reach the store.
    disk_cooldowns, disk_observations = model_cooldowns_after_clear(
        bound_rehydrated_model_cooldowns(disk_entry.get("model_cooldowns")),
        disk_entry.get(MODEL_COOLDOWN_OBSERVED_AT_KEY),
        latest_clear,
    )
    mem_cooldowns, mem_observations = model_cooldowns_after_clear(
        entry.get("model_cooldowns"),
        entry.get(MODEL_COOLDOWN_OBSERVED_AT_KEY),
        latest_clear,
    )
    merged_cooldowns = merge_model_cooldowns(disk_cooldowns, mem_cooldowns)
    merged_observations = merge_model_cooldown_observations(
        disk_observations, mem_observations)
    merged = dict(entry)
    if merged_cooldowns:
        merged["model_cooldowns"] = merged_cooldowns
    else:
        merged.pop("model_cooldowns", None)
    if merged_observations:
        merged[MODEL_COOLDOWN_OBSERVED_AT_KEY] = merged_observations
    else:
        merged.pop(MODEL_COOLDOWN_OBSERVED_AT_KEY, None)
    marks = {**(disk_entry.get(MODEL_ENTITLEMENT_KEY) or {}), **(entry.get(MODEL_ENTITLEMENT_KEY) or {})}
    marks = {m: u for m, u in marks.items() if m in merged_cooldowns}
    if marks:
        merged[MODEL_ENTITLEMENT_KEY] = marks
    else:
        merged.pop(MODEL_ENTITLEMENT_KEY, None)
    # The clear marker itself is sticky even for a healthy model-benched row. Without this,
    # the first stale ordinary flush could erase the tombstone and a later flush could revive
    # the exact cooldown that was reset.
    if disk_cleared_ts > mem_cleared_ts:
        merged["status_cleared_at"] = disk_entry.get("status_cleared_at")
    return merged, disk_cleared_ts


def _secret_fingerprint_changed(entry: dict[str, Any], disk_entry: dict[str, Any]) -> bool:
    """Whether an env-backed/borrowed row's secret changed between memory and disk.

    Those rows are persisted without their secret, so both ``access_token`` sides are
    empty and the only rotation signal is ``secret_fingerprint``
    (agent/credential_persistence.py writes it for exactly this comparison). A changed
    secret is a new credential, so a stale cooldown must not be resurrected onto it.
    """
    mem_access = entry.get("access_token") or ""
    disk_access = disk_entry.get("access_token") or ""
    if mem_access or disk_access:
        return False
    mem_fp = entry.get("secret_fingerprint") or (entry.get("extra") or {}).get("secret_fingerprint")
    disk_fp = disk_entry.get("secret_fingerprint") or (disk_entry.get("extra") or {}).get("secret_fingerprint")
    return bool(mem_fp and disk_fp and mem_fp != disk_fp)


def _pool_status_fields() -> tuple[str, ...]:
    from hermes_cli.auth import _POOL_STATUS_FIELDS
    return _POOL_STATUS_FIELDS


def _merge_disk_cooldown_state(
    entry: dict[str, Any], disk_entry: Optional[dict[str, Any]], provider_id: str,
    *, sole_credential: bool = False,
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
        from agent.credential_pool import PooledCredential, STATUS_DEAD, STATUS_EXHAUSTED
        from agent.credential_pool_cooldowns import _exhausted_until, _parse_absolute_timestamp

        # Model cooldowns are independent observations, merged in _merge_model_cooldown_state so a
        # snapshot older than an explicit reset never resurrects the map (#128995).
        merged, disk_cleared_ts = _merge_model_cooldown_state(disk_entry, entry)
        disk_status_fields = {f: disk_entry.get(f) for f in _pool_status_fields()}
        mem_ts = _parse_absolute_timestamp(entry.get("last_status_at")) or 0.0
        if entry.get("last_status") in (STATUS_DEAD, STATUS_EXHAUSTED) and disk_cleared_ts > mem_ts:
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
        # Env-backed rows persist without their secret; secret_fingerprint is their rotation signal.
        if _secret_fingerprint_changed(entry, disk_entry):
            return entry
        disk_ts = _parse_absolute_timestamp(disk_entry.get("last_status_at")) or 0.0
        if disk_ts <= mem_ts:
            return merged
        if disk_status == STATUS_EXHAUSTED:
            until = _exhausted_until(
                PooledCredential.from_dict(provider_id, disk_entry), sole_credential=sole_credential)
            if until is None or until <= time.time():
                return merged
        return {**merged, **disk_status_fields}
    except Exception:  # pragma: no cover - best-effort merge
        return entry
