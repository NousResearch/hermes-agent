"""Model-cooldown merge and secret-rotation helpers for the credential-pool disk boundary.

Extracted from ``hermes_cli.auth`` so that module stays under its code-health
FILE_LINES cap. The timestamp parser lives in ``agent.credential_pool_cooldowns``
and is imported from there directly.
"""

from __future__ import annotations

from typing import Any, Dict

from agent.credential_pool_cooldowns import _parse_absolute_timestamp


def _merge_model_cooldown_state(
    disk_entry: Dict[str, Any], entry: Dict[str, Any],
) -> tuple[Dict[str, Any], float]:
    """Merge concurrent model cooldowns, never resurrecting one an explicit reset cleared (#128995).

    Returns the merged row and the disk-side clear timestamp. Model cooldowns are
    independent observations, so concurrent model failures merge, but a snapshot older
    than an explicit reset must not bring the map back.
    """
    from agent.credential_pool_model_cooldowns import (
        MODEL_COOLDOWN_OBSERVED_AT_KEY,
        merge_model_cooldown_observations,
        merge_model_cooldowns,
        model_cooldowns_after_clear,
    )

    disk_cleared_ts = _parse_absolute_timestamp(disk_entry.get("status_cleared_at")) or 0.0
    mem_cleared_ts = _parse_absolute_timestamp(entry.get("status_cleared_at")) or 0.0
    latest_clear = max(disk_cleared_ts, mem_cleared_ts)

    disk_cooldowns, disk_observations = model_cooldowns_after_clear(
        disk_entry.get("model_cooldowns"),
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
    # The clear marker itself is sticky even for a healthy model-benched row. Without this,
    # the first stale ordinary flush could erase the tombstone and a later flush could revive
    # the exact cooldown that was reset.
    if disk_cleared_ts > mem_cleared_ts:
        merged["status_cleared_at"] = disk_entry.get("status_cleared_at")
    return merged, disk_cleared_ts


def _secret_fingerprint_changed(entry: Dict[str, Any], disk_entry: Dict[str, Any]) -> bool:
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
