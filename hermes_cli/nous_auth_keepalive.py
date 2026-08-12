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

# Two things have to line up for the keepalive to actually keep anything alive.
#
# 1. The tick has to be frequent enough to see the credential before it dies.
#    Nous credential lifetimes are not fixed: this varies by account, and
#    installs have been observed at both ~3594s and ~899s. The tick therefore
#    derives from the lifetime the server actually issued rather than assuming
#    an hour, capped by the configured interval and floored so a pathological
#    lifetime cannot spin the thread.
#
# 2. The refresh has to fire while the tick can still act on it. The refresh
#    only triggers once a credential is within a skew window of expiry, so a
#    tick spaced wider than that window steps straight over it. The keepalive
#    widens the window to "will this credential outlive my next tick?" instead
#    of the request-path default of 120s. Without this, ticking faster only
#    narrows the gap; it never closes it.
#
# The original 6-hour tick against a one-hour credential failed both tests, so
# in practice every hour expired reactively into a 401 plus a re-auth retry.
NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS = 15 * 60
NOUS_AUTH_KEEPALIVE_MIN_INTERVAL_SECONDS = 60
# Ticks per credential lifetime. Four keeps the refresh comfortably ahead of
# expiry without making the thread chatty.
NOUS_AUTH_KEEPALIVE_TICKS_PER_LIFETIME = 4
NOUS_AUTH_KEEPALIVE_INITIAL_DELAY_SECONDS = 60
NOUS_AUTH_KEEPALIVE_INTERVAL_ENV = "HERMES_NOUS_KEEPALIVE_INTERVAL_SECONDS"

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


def _interval_seconds(value: Optional[int]) -> int:
    """Resolve the keepalive tick interval.

    Explicit argument wins, then the environment override, then the module
    default. A non-positive result disables the keepalive thread entirely,
    which is the documented way to turn it off.
    """
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS

    raw = os.getenv(NOUS_AUTH_KEEPALIVE_INTERVAL_ENV)
    if raw is None or not raw.strip():
        return NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid %s=%r; using %ds",
            NOUS_AUTH_KEEPALIVE_INTERVAL_ENV,
            raw,
            NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS,
        )
        return NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS


def _observed_lifetime_seconds() -> Optional[int]:
    """Lifetime the server issued for the current Nous credentials, in seconds.

    Both the access token and the invoke agent key carry their own lifetime and
    they are not always equal, so the shorter one governs. Returns None when no
    usable value is stored, in which case the caller keeps its configured tick.
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
    return min(lifetimes) if lifetimes else None


def _tick_seconds(configured_interval: int, lifetime: Optional[int]) -> int:
    """Tick fast enough to refresh several times per credential lifetime."""
    if not lifetime or lifetime <= 0:
        return configured_interval
    derived = lifetime // NOUS_AUTH_KEEPALIVE_TICKS_PER_LIFETIME
    return max(
        NOUS_AUTH_KEEPALIVE_MIN_INTERVAL_SECONDS,
        min(configured_interval, derived),
    )


def _refresh_horizon_seconds(tick_seconds: int, floor_seconds: int) -> int:
    """How much life a credential needs to be left alone this tick.

    A credential that will not survive until the next tick has to be refreshed
    now, because nothing will look at it again before it expires. Hence
    tick + skew rather than the request path's bare skew.
    """
    return max(floor_seconds, tick_seconds + ACCESS_TOKEN_REFRESH_SKEW_SECONDS)


def _entry_state(entry: object) -> dict:
    return {
        "agent_key": getattr(entry, "agent_key", None),
        "agent_key_expires_at": getattr(entry, "agent_key_expires_at", None),
        "scope": getattr(entry, "scope", None),
    }


def _refresh_selected_pool_entry(
    *,
    min_key_ttl_seconds: int,
    min_access_ttl_seconds: Optional[int] = None,
) -> Optional[bool]:
    """Refresh the current Nous credential pool entry when it is stale.

    Returns True when a pool entry exists and is usable/refreshed, False when a
    pool exists but no entry can be used, and None when no Nous pool exists.
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
    access_expiring = _is_expiring(
        getattr(entry, "expires_at", None),
        min_access_ttl_seconds,
    )
    key_usable = _agent_key_is_usable(_entry_state(entry), min_key_ttl_seconds)
    if access_expiring or not key_usable:
        refreshed = pool.try_refresh_current()
        if refreshed is None:
            return False
        logger.debug("Nous auth keepalive: refreshed credential pool entry")
        return True

    return True


def refresh_nous_auth_keepalive_once(
    *,
    min_key_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
    min_access_ttl_seconds: Optional[int] = None,
    timeout_seconds: Optional[float] = None,
) -> bool:
    """Refresh Nous auth once if credentials are configured."""
    min_key_ttl_seconds = max(60, int(min_key_ttl_seconds))

    pool_result = _refresh_selected_pool_entry(
        min_key_ttl_seconds=min_key_ttl_seconds,
        min_access_ttl_seconds=min_access_ttl_seconds,
    )
    if pool_result is not None:
        return pool_result

    state = get_provider_auth_state("nous")
    if not state:
        return False

    try:
        resolve_nous_runtime_credentials(
            timeout_seconds=_timeout_seconds(timeout_seconds),
        )
        logger.debug("Nous auth keepalive: refreshed singleton auth state")
        return True
    except AuthError as exc:
        if exc.relogin_required:
            logger.info("Nous auth keepalive requires re-login: %s", exc)
        else:
            logger.debug("Nous auth keepalive failed: %s", exc)
        return False
    except Exception as exc:
        logger.debug("Nous auth keepalive failed: %s", exc)
        return False


def _keepalive_loop(
    stop_event: threading.Event,
    *,
    interval_seconds: int,
    initial_delay_seconds: int,
    min_key_ttl_seconds: int,
    timeout_seconds: Optional[float],
) -> None:
    if initial_delay_seconds > 0 and stop_event.wait(initial_delay_seconds):
        return

    while not stop_event.is_set():
        # Re-read each pass: the lifetime can change when the account, plan, or
        # server-side policy does, and a keepalive that caches it would go stale
        # in exactly the case it exists to cover.
        tick = _tick_seconds(interval_seconds, _observed_lifetime_seconds())
        horizon = _refresh_horizon_seconds(tick, min_key_ttl_seconds)
        refresh_nous_auth_keepalive_once(
            min_key_ttl_seconds=horizon,
            min_access_ttl_seconds=horizon,
            timeout_seconds=timeout_seconds,
        )
        stop_event.wait(tick)


def start_nous_auth_keepalive(
    *,
    interval_seconds: Optional[int] = None,
    initial_delay_seconds: int = NOUS_AUTH_KEEPALIVE_INITIAL_DELAY_SECONDS,
    min_key_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
    timeout_seconds: Optional[float] = None,
) -> Optional[threading.Thread]:
    """Start the process-wide Nous auth keepalive thread."""
    interval_seconds = _interval_seconds(interval_seconds)
    if interval_seconds <= 0:
        return None

    global _keepalive_thread
    with _keepalive_lock:
        if _keepalive_thread is not None and _keepalive_thread.is_alive():
            return _keepalive_thread

        _keepalive_stop.clear()
        _keepalive_thread = threading.Thread(
            target=_keepalive_loop,
            args=(_keepalive_stop,),
            kwargs={
                "interval_seconds": int(interval_seconds),
                "initial_delay_seconds": max(0, int(initial_delay_seconds)),
                "min_key_ttl_seconds": max(60, int(min_key_ttl_seconds)),
                "timeout_seconds": timeout_seconds,
            },
            daemon=True,
            name="nous-auth-keepalive",
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
