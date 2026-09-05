"""Background keepalive for long-lived Nous Portal sessions."""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from hermes_cli.auth import (
    ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
    AuthError,
    _agent_key_is_usable,
    _is_expiring,
    get_provider_auth_state,
    resolve_nous_runtime_credentials,
)

logger = logging.getLogger(__name__)

# Two things must line up for the keepalive to keep anything alive:
# 1. The tick must be frequent enough to see the credential before it dies. Lifetimes vary by
#    account (~3594s and ~899s observed), so the tick derives from the lifetime the server issued,
#    capped by the configured interval and floored so a pathological lifetime can't spin the thread.
# 2. The refresh must fire while the tick can still act on it: refresh triggers only within a skew
#    window of expiry, so the keepalive widens that window to "will this credential outlive my
#    next tick?" instead of the request path's 120s. Ticking faster alone never closes the gap.
NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS = 15 * 60
NOUS_AUTH_KEEPALIVE_MIN_INTERVAL_SECONDS = 60
# Ticks per credential lifetime: four keeps refresh comfortably ahead of expiry without chatter.
NOUS_AUTH_KEEPALIVE_TICKS_PER_LIFETIME = 4
NOUS_AUTH_KEEPALIVE_INITIAL_DELAY_SECONDS = 60
NOUS_AUTH_KEEPALIVE_INTERVAL_CONFIG_KEY = "keepalive_interval_seconds"

_keepalive_lock = threading.Lock()
_keepalive_stop = threading.Event()
_keepalive_thread: Optional[threading.Thread] = None


def _timeout_seconds(value: Optional[float]) -> float:
    if value is not None:
        return float(value)
    try:
        return float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15"))
    except (TypeError, ValueError):
        return 15.0


def _nous_config() -> dict:
    """The ``nous:`` section of config.yaml, or {} on any failure (config loader imported lazily)."""
    try:
        from hermes_cli.config import load_config

        section = load_config().get("nous")
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _interval_seconds(value: Optional[int]) -> int:
    """Tick interval: explicit argument, then ``nous.keepalive_interval_seconds`` in config.yaml,
    then the module default. Non-positive disables the keepalive thread (the documented way off).
    """
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS
    raw = _nous_config().get(NOUS_AUTH_KEEPALIVE_INTERVAL_CONFIG_KEY)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid nous.%s=%r; using %ds",
            NOUS_AUTH_KEEPALIVE_INTERVAL_CONFIG_KEY, raw, NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS,
        )
        return NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS


def _observed_lifetime_seconds() -> Optional[int]:
    """Server-issued lifetime (seconds) of the current Nous credentials; the shorter of the access
    token and the invoke agent key governs. None when nothing usable is stored.
    """
    state = get_provider_auth_state("nous") or {}
    lifetimes = []
    for key in ("expires_in", "agent_key_expires_in"):
        try:
            value = int(float(state.get(key)))
        except (TypeError, ValueError):
            continue
        if value > 0:
            lifetimes.append(value)
    return min(lifetimes, default=None)


def _tick_seconds(configured_interval: int, lifetime: Optional[int]) -> int:
    """Tick fast enough to refresh several times per credential lifetime."""
    if not lifetime or lifetime <= 0:
        return configured_interval
    derived = lifetime // NOUS_AUTH_KEEPALIVE_TICKS_PER_LIFETIME
    return max(NOUS_AUTH_KEEPALIVE_MIN_INTERVAL_SECONDS, min(configured_interval, derived))


def _refresh_horizon_seconds(tick_seconds: int, floor_seconds: int) -> int:
    """Life a credential needs to be left alone this tick: it must survive until the next tick
    (nothing looks at it again before then), hence tick + skew rather than the bare skew.
    """
    return max(floor_seconds, tick_seconds + ACCESS_TOKEN_REFRESH_SKEW_SECONDS)


def _entry_state(entry: object) -> dict:
    return {k: getattr(entry, k, None) for k in ("agent_key", "agent_key_expires_at", "scope")}


def _refresh_selected_pool_entry(*, min_key_ttl_seconds: int, min_access_ttl_seconds: Optional[int] = None) -> Optional[bool]:
    """Refresh the current pool entry when stale. True = usable/refreshed; False = pool exists but
    no usable entry; None = no Nous pool.
    """
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("nous")
    except Exception as exc:
        logger.debug("Nous auth keepalive: credential pool unavailable: %s", exc)
        return None
    if not pool or not pool.has_credentials():
        return None
    try:
        entry = pool.select()
    except Exception as exc:
        logger.debug("Nous auth keepalive: credential pool selection failed: %s", exc)
        return False
    if entry is None:
        return False
    if min_access_ttl_seconds is None:
        min_access_ttl_seconds = ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    access_expiring = _is_expiring(getattr(entry, "expires_at", None), min_access_ttl_seconds)
    key_usable = _agent_key_is_usable(_entry_state(entry), min_key_ttl_seconds)
    if access_expiring or not key_usable:
        if pool.try_refresh_current() is None:
            return False
        logger.debug("Nous auth keepalive: refreshed credential pool entry")
    return True


def refresh_nous_auth_keepalive_once(
    *, min_key_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
    min_access_ttl_seconds: Optional[int] = None, timeout_seconds: Optional[float] = None,
) -> bool:
    """Refresh Nous auth once if credentials are configured (pool entry first, then singleton state)."""
    pool_result = _refresh_selected_pool_entry(
        min_key_ttl_seconds=max(60, int(min_key_ttl_seconds)), min_access_ttl_seconds=min_access_ttl_seconds
    )
    if pool_result is not None:
        return pool_result
    if not get_provider_auth_state("nous"):
        return False
    try:
        resolve_nous_runtime_credentials(timeout_seconds=_timeout_seconds(timeout_seconds))
        logger.debug("Nous auth keepalive: refreshed singleton auth state")
        return True
    except Exception as exc:
        if isinstance(exc, AuthError) and exc.relogin_required:
            logger.info("Nous auth keepalive requires re-login: %s", exc)
        else:
            logger.debug("Nous auth keepalive failed: %s", exc)
        return False


def _keepalive_loop(
    stop_event: threading.Event, *, interval_seconds: int, initial_delay_seconds: int,
    min_key_ttl_seconds: int, timeout_seconds: Optional[float],
) -> None:
    if initial_delay_seconds > 0 and stop_event.wait(initial_delay_seconds):
        return
    while not stop_event.is_set():
        # Re-read each pass: the lifetime changes with account/plan/policy; caching it would go
        # stale in exactly the case the keepalive exists to cover.
        tick = _tick_seconds(interval_seconds, _observed_lifetime_seconds())
        horizon = _refresh_horizon_seconds(tick, min_key_ttl_seconds)
        refresh_nous_auth_keepalive_once(
            min_key_ttl_seconds=horizon, min_access_ttl_seconds=horizon, timeout_seconds=timeout_seconds
        )
        stop_event.wait(tick)


def start_nous_auth_keepalive(
    *, interval_seconds: Optional[int] = None,
    initial_delay_seconds: int = NOUS_AUTH_KEEPALIVE_INITIAL_DELAY_SECONDS,
    min_key_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS, timeout_seconds: Optional[float] = None,
) -> Optional[threading.Thread]:
    """Start the process-wide Nous auth keepalive thread (idempotent; None when disabled)."""
    interval_seconds = _interval_seconds(interval_seconds)
    if interval_seconds <= 0:
        return None
    global _keepalive_thread
    with _keepalive_lock:
        if _keepalive_thread is not None and _keepalive_thread.is_alive():
            return _keepalive_thread
        _keepalive_stop.clear()
        _keepalive_thread = threading.Thread(
            target=_keepalive_loop, args=(_keepalive_stop,), daemon=True, name="nous-auth-keepalive",
            kwargs={
                "interval_seconds": int(interval_seconds),
                "initial_delay_seconds": max(0, int(initial_delay_seconds)),
                "min_key_ttl_seconds": max(60, int(min_key_ttl_seconds)),
                "timeout_seconds": timeout_seconds,
            },
        )
        _keepalive_thread.start()
        logger.debug("Nous auth keepalive started")
        return _keepalive_thread


def stop_nous_auth_keepalive(timeout: float = 5.0) -> None:
    """Stop the keepalive thread. Intended for graceful shutdown/tests."""
    global _keepalive_thread
    with _keepalive_lock:
        thread = _keepalive_thread
        _keepalive_stop.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
    with _keepalive_lock:
        if _keepalive_thread is thread:
            _keepalive_thread = None
