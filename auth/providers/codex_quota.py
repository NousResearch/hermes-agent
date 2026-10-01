"""codex quota protocol/lifecycle responsibilities."""

from __future__ import annotations
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
from auth.token_validation import _decode_jwt_claims
from auth.errors import AuthError
from auth.constants import CODEX_RATE_LIMITED_CODE, _codex_err


def _codex_quota_exhausted_error(retry_after: Optional[int]) -> AuthError:
    message = (
        f"Codex provider quota exhausted (429); retry after {retry_after}s. "
        "Credentials are still valid."
        if retry_after is not None
        else "Codex provider quota exhausted (429). Credentials are still valid; "
        "retry after the usage limit resets."
    )
    return _codex_err(message, CODEX_RATE_LIMITED_CODE, relogin=False)


def _is_codex_rate_limit_shaped(code: Any, reason: Any, message: Any) -> bool:
    """True when persisted pool-entry error metadata describes a 429/quota stop."""
    reason_l, message_l = str(reason or "").lower(), str(message or "").lower()
    return (
        code == 429
        or any(k in reason_l for k in ("rate_limit", "usage_limit", "quota"))
        or any(k in message_l for k in ("rate limit", "usage limit", "quota"))
    )


CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS = 300

_codex_quota_probe_cache: Dict[str, Tuple[float, Optional[bool]]] = {}


def _codex_usage_probe_url(base_url: Optional[str]) -> str:
    """Resolve the Codex usage endpoint for a probe.

    Mirrors the Codex CLI's PathStyle split: base URLs containing ``/backend-api`` use the ChatGPT
    ``/wham/usage`` path, everything else ``/api/codex/usage``. Kept local so this low-level auth
    module does not import the auxiliary account-usage module.
    """
    from auth.providers.codex import _codex_base_url, _stripped

    normalized = _stripped(base_url).rstrip("/") or _codex_base_url()
    if normalized.endswith("/codex"):
        normalized = normalized[: -len("/codex")]
    prefix = normalized + ("/wham" if "/backend-api" in normalized else "/api/codex")
    return prefix + "/usage"


def _probe_codex_quota_restored(
    access_token: Any,
    *,
    base_url: Optional[str] = None,
    min_interval_seconds: float = CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS,
) -> Optional[bool]:
    """Ask the Codex usage endpoint whether this account's quota is usable again.

    Probes are throttled per access token (module-local cache) so the hot selection path can fire
    this freely.
    """
    from auth.providers.codex import (
        _codex_quota_probe_cache_key,
        _codex_quota_probe_lock,
        _stripped,
        logger,
    )
    from auth.providers.codex_http import _codex_http_client
    from auth.providers.codex_quota import _codex_quota_probe_cache

    token = _stripped(access_token)
    # Real Codex access tokens are JWTs. Refusing to probe non-JWT tokens avoids pointless
    # network calls for corrupt/placeholder entries (and keeps hermetic test fixtures offline).
    if not token or not _decode_jwt_claims(token):
        return None
    cache_key = _codex_quota_probe_cache_key(token)
    now = time.monotonic()
    with _codex_quota_probe_lock:
        cached = _codex_quota_probe_cache.get(cache_key)
        if cached is not None and (now - cached[0]) < min_interval_seconds:
            return cached[1]
        # Reserve the slot immediately so concurrent selectors don't stampede the endpoint.
        _codex_quota_probe_cache[cache_key] = (now, None)
    result: Optional[bool] = None
    try:
        # Account/residency headers from the JWT (required for some account shapes).
        from agent.codex_headers import codex_account_headers

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "codex-cli",
            **codex_account_headers(token),
        }
        with _codex_http_client(timeout=10.0) as client:
            response = client.get(_codex_usage_probe_url(base_url), headers=headers)
        if response.status_code == 200:
            payload = response.json() or {}
            # A model-scoped allowance (``additional_rate_limits``) at 100% still 429s that
            # model, so it counts against "restored" like the account-wide windows (#97315).
            windows = [payload.get("rate_limit") or {}] + [
                extra.get("rate_limit") or {}
                for extra in (payload.get("additional_rate_limits") or [])
                if isinstance(extra, dict)
            ]
            worst_used: Optional[float] = None
            for rate_limit in windows:
                for key in ("primary_window", "secondary_window"):
                    used = (rate_limit.get(key) or {}).get("used_percent")
                    if isinstance(used, (int, float)):
                        worst_used = max(worst_used or 0.0, float(used))
            if worst_used is not None:
                result = worst_used < 100.0
        elif response.status_code == 429:
            result = False
    except Exception:
        logger.debug("Codex quota probe failed", exc_info=True)
        result = None
    with _codex_quota_probe_lock:
        _codex_quota_probe_cache[cache_key] = (now, result)
    return result


def _refresh_expired_codex_probe_token(
    access_token: Any,
    refresh_token: Any,
    *,
    min_interval_seconds: float = CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS,
    environment,
) -> Optional[Dict[str, Any]]:
    """Refresh an EXPIRED stored access token so the quota probe can get a real answer.

    Exhausted pool entries are skipped by the proactive refresh chain (#44799), so by the time
    anything probes with the stored token it has expired; the usage endpoint answers
    ``401 token_expired``, the probe returns None, and the cooldown is kept until
    ``last_error_reset_at`` no matter what happened upstream (top-up, plan upgrade) — #89415.
    Returns the rotated token pair (callers MUST persist it: refresh tokens are single-use) or
    None when no refresh was needed/possible. The cooldown itself is left untouched.

    Shares the probe's per-token throttle: a refresh that keeps failing (revoked grant, network
    down) would otherwise POST to the token endpoint on every credential selection, while the
    probe itself is capped at one call per ``min_interval_seconds``. A failed attempt reserves
    the stale token's probe slot, so neither the refresh nor the doomed 401 probe fire again
    until the interval has elapsed.
    """
    from auth.providers.codex import (
        _codex_access_token_is_expiring,
        _codex_quota_probe_cache_key,
        _codex_quota_probe_lock,
        _stripped,
        logger,
        refresh_codex_oauth_pure,
    )

    environment.require_current_scope()
    from auth.providers.codex_quota import _codex_quota_probe_cache

    token, refresh = _stripped(access_token), _stripped(refresh_token)
    if not token or not refresh or not _codex_access_token_is_expiring(token, 0):
        return None
    cache_key = _codex_quota_probe_cache_key(token)
    now = time.monotonic()
    with _codex_quota_probe_lock:
        cached = _codex_quota_probe_cache.get(cache_key)
        if cached is not None and (now - cached[0]) < min_interval_seconds:
            return None
    try:
        return refresh_codex_oauth_pure(token, refresh, environment=environment)
    except Exception:
        logger.debug("Codex pre-probe token refresh failed", exc_info=True)
        with _codex_quota_probe_lock:
            _codex_quota_probe_cache[cache_key] = (now, None)
        return None


def _probe_codex_pool_entry_quota_restored(
    entry: Dict[str, Any], *, environment
) -> Optional[bool]:
    """``_probe_codex_quota_restored`` for a persisted pool entry, refreshing an expired token first."""
    from auth.providers.codex import _pool_entries, _stripped, logger

    environment.require_current_scope()
    from auth.store import _auth_store_lock, _load_auth_store, _save_auth_store

    token = _stripped(entry.get("access_token"))
    fresh = _refresh_expired_codex_probe_token(
        token, entry.get("refresh_token"), environment=environment
    )
    if fresh:
        token = fresh["access_token"]
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                for disk_entry in _codex_pool_dicts(
                    _pool_entries(auth_store, "openai-codex")
                ):
                    if disk_entry.get("id") == entry.get("id"):
                        disk_entry.update(fresh)
                        _save_auth_store(auth_store)
                        break
        except Exception:
            logger.debug("Failed to persist refreshed Codex pool tokens", exc_info=True)
    if not token:
        return None
    # The row keeps the canonical URL; a gateway key belongs to its route host (#121486).
    return _probe_codex_quota_restored(
        token,
        base_url=environment.provider_hooks("openai-codex").route_base_url(
            entry.get("base_url")
        ),
    )


def clear_codex_pool_quota_cooldowns(access_token: Optional[str] = None) -> int:
    """Clear rate-limit cooldowns on persisted openai-codex pool entries.

    Called after the upstream quota is KNOWN to be restored (a ``/usage reset`` redemption or a
    positive live probe) so auth.json stops freezing credentials behind a stale
    ``last_error_reset_at``. With *access_token* only the matching entry clears; otherwise every
    rate-limited entry does (a redeemed banked reset restores the whole account; a still-exhausted
    entry just re-freezes with fresh metadata on its next 429).
    """
    from auth.providers.codex import (
        _clear_pool_entry_status,
        _entry_is_rate_limit_exhausted,
        _pool_entries,
        logger,
    )
    from auth.credential_pool import (
        _borrowed_single_use_pool_root,
        _profile_owns_pool_provider,
    )
    from auth.store import _auth_store_lock, _load_auth_store, _save_auth_store

    cleared = 0
    try:
        # Same owner rule as ``persist_pool_entries``: a profile with no Codex rows of its own
        # borrows the global-root pool, so the cooldown must clear where the rows actually live.
        target = (
            None
            if _profile_owns_pool_provider("openai-codex")
            else _borrowed_single_use_pool_root()
        )
        with _auth_store_lock(target_path=target):
            auth_store = _load_auth_store(target)
            for entry in _codex_pool_dicts(_pool_entries(auth_store, "openai-codex")):
                if (
                    access_token
                    and str(entry.get("access_token") or "") != access_token
                ):
                    continue
                if _entry_is_rate_limit_exhausted(entry):
                    _clear_pool_entry_status(entry)
                    cleared += 1
            if cleared:
                _save_auth_store(auth_store, target_path=target)
    except Exception:
        logger.debug("Failed to clear Codex pool quota cooldowns", exc_info=True)
    return cleared


def _codex_pool_dicts(entries: Optional[List[Any]]) -> Iterator[Dict[str, Any]]:
    for entry in entries or ():
        if isinstance(entry, dict):
            yield entry


def _codex_pool_rate_limit_status() -> Optional[Dict[str, Any]]:
    """Return metadata for a pool-only Codex credential in quota cooldown.

    Reads through ``read_credential_pool`` so a named profile with no Codex rows of its own sees
    the global-root pool (the per-provider fallback every other pool read uses)."""
    from auth.providers.codex import _entry_is_rate_limit_exhausted, logger
    from auth.token_validation import _nonempty_str
    from auth.pool_persistence import read_credential_pool
    from auth.credential_pool import _parse_absolute_timestamp

    try:
        now = time.time()
        for entry in _codex_pool_dicts(read_credential_pool("openai-codex")):
            token = entry.get("access_token")
            if not _nonempty_str(token) or not _entry_is_rate_limit_exhausted(entry):
                continue
            reset_at = _parse_absolute_timestamp(entry.get("last_error_reset_at"))
            if reset_at is None or reset_at > now:
                return {
                    "label": entry.get("label"),
                    "last_refresh": entry.get("last_refresh"),
                    "reset_at": reset_at,
                    "reason": entry.get("last_error_reason"),
                    "message": entry.get("last_error_message"),
                    "access_token": token.strip(),
                    "refresh_token": entry.get("refresh_token"),
                    "id": entry.get("id"),
                    "base_url": entry.get("base_url"),
                }
    except Exception:
        logger.debug("Codex pool rate-limit lookup failed", exc_info=True)
    return None


def _pool_codex_credential() -> Tuple[str, str]:
    """``(access_token, row base_url)`` of the first pool entry with a non-empty access_token that is
    not in an exhaustion cooldown window, so the caller routes the token to the host that row belongs
    to; ``("", "")`` when none is usable.

    Fallback for ``resolve_codex_runtime_credentials`` when the singleton has no creds; reads
    through ``read_credential_pool`` so a profile inherits the global-root pool (#34143)."""
    from auth.providers.codex import _stripped, logger
    from auth.credential_pool import _parse_absolute_timestamp
    from auth.token_validation import _nonempty_str
    from auth.pool_persistence import read_credential_pool

    try:
        for entry in _codex_pool_dicts(read_credential_pool("openai-codex")):
            token = entry.get("access_token")
            # Same normaliser as ``_codex_pool_rate_limit_status``: a millisecond epoch compared
            # raw reads as far-future here and as elapsed there, hiding a usable entry (#103349).
            reset_at = _parse_absolute_timestamp(entry.get("last_error_reset_at"))
            in_cooldown = reset_at is not None and reset_at > time.time()
            if _nonempty_str(token) and not in_cooldown:
                return token.strip(), _stripped(entry.get("base_url"))
    except Exception:
        logger.debug("Codex pool fallback lookup failed", exc_info=True)
    return "", ""
