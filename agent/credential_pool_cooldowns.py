"""Cooldown / TTL helpers for the credential pool, split out of agent.credential_pool.

Extracted so ``agent.credential_pool`` stays under its code-health FILE_LINES cap.
These helpers read the pool's constants and ``PooledCredential`` from the facade,
which lazy-imports the helpers back for its own internal callers; external importers
point here directly.
"""

from __future__ import annotations

from datetime import datetime
import logging
import time
from typing import Any, Optional

from agent.credential_pool import (
    EXHAUSTED_TTL_401_SECONDS,
    EXHAUSTED_TTL_429_SECONDS,
    EXHAUSTED_TTL_DEFAULT_SECONDS,
    EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS,
    FAILURE_REASON_BILLING,
    FAILURE_REASON_BILLING_UNVERIFIED,
    SOURCE_MANUAL,
    STATUS_EXHAUSTED,
    PooledCredential,
)


logger = logging.getLogger(__name__)


def _is_manual_source(source: str) -> bool:
    normalized = (source or "").strip().lower()
    return normalized == SOURCE_MANUAL or normalized.startswith(f"{SOURCE_MANUAL}:")


def _is_billing_failure(error_code: Optional[int], failure_reason: Optional[str]) -> bool:
    """Confirmed billing: a quick retry cannot help, so the full bench stands.

    One home for the rule so the TTL bench and the absolute-reset clamp cannot
    drift apart. ``billing_unverified`` is deliberately NOT billing here — the
    same 400 also covers a content-filter rejection on a healthy credential
    (#82154), and only a true 402 outranks the status.
    """
    return error_code == 402 or failure_reason == FAILURE_REASON_BILLING


def _exhausted_ttl(
    error_code: Optional[int],
    *,
    sole_credential: bool = False,
    failure_reason: Optional[str] = None,
) -> int:
    """Return cooldown seconds based on the HTTP status that caused exhaustion.

    *sole_credential*: the pool has nothing to rotate to, so transient
    throttles (429 and the catch-all default covering 403/5xx/unknown) are
    capped to a brief cooldown; 401 keeps its own already-short TTL.

    *failure_reason* is the classifier verdict: an OpenRouter ``key limit
    exceeded`` and an xAI spending block both arrive as 403 but are billing,
    and a 60s retry on a spent account just re-fails. Billing keeps the full
    bench regardless of status; 402 is billing by definition.
    Unverified billing (#82154) gets the short cooldown regardless of pool
    size (the credential may be healthy), unless the status is a true 402.
    """
    if error_code == 401:
        return EXHAUSTED_TTL_401_SECONDS
    base = EXHAUSTED_TTL_429_SECONDS if error_code == 429 else EXHAUSTED_TTL_DEFAULT_SECONDS
    if failure_reason == FAILURE_REASON_BILLING_UNVERIFIED and error_code != 402:
        return min(base, EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS)
    if sole_credential and not _is_billing_failure(error_code, failure_reason):
        return min(base, EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS)
    return base


def _ttl_bench_until(entry: PooledCredential, *, sole_credential: bool = False) -> Optional[float]:
    """Epoch of the TTL bench for an exhausted entry, or ``None`` without a status stamp."""
    if not entry.last_status_at:
        return None
    return entry.last_status_at + _exhausted_ttl(
        entry.last_error_code,
        sole_credential=sole_credential,
        failure_reason=entry.failure_reason,
    )


def _parse_absolute_timestamp(value: Any) -> Optional[float]:
    """Best-effort parse of epoch seconds / epoch ms / ISO-8601 into epoch seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric <= 0:
            return None
        return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            numeric = float(raw)
            return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _exhausted_until(entry: PooledCredential, *, sole_credential: bool = False) -> Optional[float]:
    """Epoch when an exhausted entry may re-enter rotation, else ``None``.

    Clamp rule: a sole non-billing credential's persisted absolute
    ``last_error_reset_at`` is capped at the TTL bench, so a subscription-period
    429 (monthly/weekly window) cannot bench it for the whole window (#119163).
    Confirmed billing and pools with siblings keep the provider-stated reset.
    When the status stamp is missing (stale row), the bench is measured from now
    rather than letting the provider reset stand unclamped.
    """
    if entry.last_status != STATUS_EXHAUSTED:
        return None
    reset_at = _parse_absolute_timestamp(entry.last_error_reset_at)
    bench_until = _ttl_bench_until(entry, sole_credential=sole_credential)
    if reset_at is None:
        return bench_until
    if sole_credential and not _is_billing_failure(entry.last_error_code, entry.failure_reason):
        if bench_until is None:
            # No status stamp (stale persisted row): bench from now so a missing
            # stamp cannot resurrect the whole-window provider reset (#119163).
            bench_until = time.time() + _exhausted_ttl(
                entry.last_error_code,
                sole_credential=True,
                failure_reason=entry.failure_reason,
            )
        return min(reset_at, bench_until)
    return reset_at


def read_pool_rows_by_id(provider: str) -> dict:
    """Persisted *provider* rows keyed by entry id — one store read for a whole selection pass."""
    from hermes_cli.auth import read_credential_pool
    try:
        return {row["id"]: row for row in read_credential_pool(provider)
                if isinstance(row, dict) and row.get("id")}
    except Exception:
        logger.debug("Pool %s: could not read disk rows", provider, exc_info=True)
        return {}


def reset_cleared_after(
    provider: str, entry: PooledCredential, disk_rows: Optional[dict] = None,
) -> Optional[float]:
    """Epoch of a ``hermes auth reset`` persisted by another process AFTER *entry*'s status, else None.

    *disk_rows* (from :func:`read_pool_rows_by_id`) skips the store read when the caller has it.
    """
    if disk_rows is None:
        from hermes_cli.auth import read_credential_pool
        try:
            row = next((p for p in read_credential_pool(provider)
                        if isinstance(p, dict) and p.get("id") == entry.id), None)
        except Exception as exc:
            logger.debug("Pool entry %s: could not read reset marker: %s", entry.id, exc, exc_info=True)
            return None
    else:
        row = disk_rows.get(entry.id)
    cleared = _parse_absolute_timestamp((row or {}).get("status_cleared_at"))
    return cleared if cleared and cleared > (entry.last_status_at or 0.0) else None
