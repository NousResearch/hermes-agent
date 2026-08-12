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

# Nous Portal access tokens carry a one-hour lifetime, and the refresh only
# fires once a token is within ACCESS_TOKEN_REFRESH_SKEW_SECONDS (120s) of
# expiry. A tick interval must therefore be comfortably below
# 3600 - 120 = 3480s, or the hour rolls over untouched between ticks and the
# next inference call pays a 401 plus a re-auth round trip. The previous
# 6-hour interval could only ever land inside the 2-minute refresh window by
# coincidence, so in practice every hour expired reactively.
NOUS_AUTH_KEEPALIVE_INTERVAL_SECONDS = 15 * 60
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


def _entry_state(entry: object) -> dict:
    return {
        "agent_key": getattr(entry, "agent_key", None),
        "agent_key_expires_at": getattr(entry, "agent_key_expires_at", None),
        "scope": getattr(entry, "scope", None),
    }


def _refresh_selected_pool_entry(
    *,
    min_key_ttl_seconds: int,
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

    access_expiring = _is_expiring(
        getattr(entry, "expires_at", None),
        ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
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
    timeout_seconds: Optional[float] = None,
) -> bool:
    """Refresh Nous auth once if credentials are configured."""
    min_key_ttl_seconds = max(60, int(min_key_ttl_seconds))

    pool_result = _refresh_selected_pool_entry(
        min_key_ttl_seconds=min_key_ttl_seconds,
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
        refresh_nous_auth_keepalive_once(
            min_key_ttl_seconds=min_key_ttl_seconds,
            timeout_seconds=timeout_seconds,
        )
        stop_event.wait(interval_seconds)


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
