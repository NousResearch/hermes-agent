"""Shared fallback decision helpers."""

from __future__ import annotations


def pool_may_recover_from_rate_limit(
    pool, *, provider: str | None = None, base_url: str | None = None
) -> bool:
    """Decide whether credential-pool rotation can recover a rate limit.

    Rotation is only useful when the pool exists, has an available credential,
    and contains more than one entry. Single-entry pools retry the same
    exhausted credential. CloudCode/Gemini CLI quotas are account-wide, so
    rotating entries does not escape the same throttle window.
    """
    if pool is None:
        return False
    if not pool.has_available():
        return False
    if provider == "google-gemini-cli" or str(base_url or "").startswith("cloudcode-pa://"):
        return False
    return len(pool.entries()) > 1
