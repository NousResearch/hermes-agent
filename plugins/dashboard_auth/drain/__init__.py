"""DrainSecretProvider — shared-bearer-secret auth for the drain-control endpoint.

Non-interactive token capability of the ``DashboardAuthProvider`` ABC (``verify_token`` +
the ``token_auth`` middleware seam): ``nous-account-service`` provisions a per-agent unique
secret (``HERMES_DASHBOARD_DRAIN_SECRET``, env-only — it is a credential); an inbound bearer
is compared constant-time and vouched for as the ``drain-control`` principal. Fail-CLOSED
entropy gate at registration (length, distinct chars, Shannon bits); interactive ABC methods
raise. Knobs ``scope`` / ``min_secret_chars`` live under ``dashboard.drain_auth``.
"""
from __future__ import annotations

import hmac
import logging
import math
import os
from collections import Counter
from typing import Optional

from hermes_cli.dashboard_auth import DashboardAuthProvider, Session, TokenPrincipal
from plugins.dashboard_auth._shared import NonInteractiveMixin, SkipRegistration, load_config_section, register_provider

logger = logging.getLogger(__name__)
_TAG = "dashboard-auth-drain"

# token_urlsafe(32) produces exactly 43 chars, so a correctly-provisioned
# secret clears the default bar exactly.
_DEFAULT_MIN_SECRET_CHARS = 43
# Rejects degenerate values like "aaaa..." that are long but trivially low-entropy.
_MIN_DISTINCT_CHARS = 16
# Distribution-aware second guard on top of length + distinct-count.
_MIN_SHANNON_BITS = 128.0

# Kept here (not imported from web_server) to avoid a heavy import at plugin load.
DRAIN_ROUTE_PATH = "/api/gateway/drain"

LAST_SKIP_REASON: str = ""


def _shannon_bits(value: str) -> float:
    """Total Shannon entropy (bits) of ``value`` over its character distribution."""
    if not value:
        return 0.0
    n = len(value)
    per_char = -sum((c / n) * math.log2(c / n) for c in Counter(value).values())
    return per_char * n


def assess_secret_strength(secret: str, *, min_chars: int = _DEFAULT_MIN_SECRET_CHARS) -> Optional[str]:
    """Human-readable rejection reason if ``secret`` is too weak, else ``None``. Checks, in
    order: length >= ``min_chars``, distinct chars >= ``_MIN_DISTINCT_CHARS``, Shannon
    entropy >= ``_MIN_SHANNON_BITS``."""
    if not secret:
        return "secret is empty"
    if len(secret) < min_chars:
        return (
            f"secret too short: {len(secret)} chars (need >= {min_chars}; "
            "use a >=256-bit value, e.g. `python -c \"import secrets; "
            "print(secrets.token_urlsafe(32))\"`)")
    distinct = len(set(secret))
    if distinct < _MIN_DISTINCT_CHARS:
        return f"secret has only {distinct} distinct characters (need >= {_MIN_DISTINCT_CHARS}); looks structured/low-entropy"
    bits = _shannon_bits(secret)
    if bits < _MIN_SHANNON_BITS:
        return f"secret entropy too low: {bits:.0f} bits (need >= {_MIN_SHANNON_BITS:.0f}); looks structured/repeated"
    return None


class DrainSecretProvider(NonInteractiveMixin, DashboardAuthProvider):
    """Non-interactive shared-bearer-secret provider for drain control."""

    name = "drain-secret"
    display_name = "Drain Control (service credential)"
    supports_token = True
    supports_session = False
    _NOT_INTERACTIVE = "DrainSecretProvider is a non-interactive service credential."
    _NO_START_LOGIN = "DrainSecretProvider is a non-interactive service credential; there is no login flow."

    def __init__(self, *, secret: str, scope: str = "drain") -> None:
        # Defence in depth: construction enforces the entropy bar too, so a
        # caller bypassing register() still can't build a weak provider.
        reason = assess_secret_strength(secret)
        if reason is not None:
            raise ValueError(f"drain secret rejected: {reason}")
        self._secret = secret
        self._scope = scope or "drain"

    # ---- token capability (the only thing this provider implements) --------

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        """Constant-time compare; ``drain-control`` principal on match, else
        ``None`` so the generic seam falls through / fails closed."""
        if token and hmac.compare_digest(token.encode("utf-8"), self._secret.encode("utf-8")):
            return TokenPrincipal(principal="drain-control", provider=self.name, scopes=(self._scope,))
        return None

    # ---- interactive methods: unsupported (service credential only) --------

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        # Never mints a Session, so never recognises a cookie. Return None (don't raise)
        # so it stacks harmlessly in the cookie-verify loop.
        return None

    def refresh_session(self, *, refresh_token: str) -> Session:
        raise NotImplementedError(self._NOT_INTERACTIVE)

    def revoke_session(self, *, refresh_token: str) -> None:
        return None


# ---- Plugin entry point ----

def _load_config_drain_auth_section() -> dict:
    return load_config_section(logger, _TAG, "dashboard", "drain_auth")


def _settings() -> dict:
    """Resolve DrainSecretProvider kwargs from env/config; raises ``SkipRegistration``."""
    secret = os.environ.get("HERMES_DASHBOARD_DRAIN_SECRET", "").strip()
    if not secret:
        raise SkipRegistration(
            "HERMES_DASHBOARD_DRAIN_SECRET is not set. Set a per-agent >=256-bit secret "
            "(e.g. `python -c \"import secrets; print(secrets.token_urlsafe(32))\"`) to enable "
            "NAS-driven drain coordination; leave it unset to disable the drain endpoint.")
    section = _load_config_drain_auth_section()
    scope = str(section.get("scope", "drain") or "drain").strip() or "drain"
    try:
        min_chars = int(section.get("min_secret_chars", _DEFAULT_MIN_SECRET_CHARS))
    except (TypeError, ValueError):
        min_chars = _DEFAULT_MIN_SECRET_CHARS
    reason = assess_secret_strength(secret, min_chars=min_chars)
    if reason is not None:
        raise SkipRegistration(
            f"HERMES_DASHBOARD_DRAIN_SECRET rejected — {reason}. The drain endpoint stays disabled (fail-closed).",
            level="warning")
    return {"secret": secret, "scope": scope}


def register(ctx) -> None:
    """Register ``DrainSecretProvider`` when a strong secret is set; no-op (records a skip
    reason) when ``HERMES_DASHBOARD_DRAIN_SECRET`` is unset or fails the entropy gate. On
    success also registers the drain route as token-authable via the generic seam."""
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""
    kwargs, LAST_SKIP_REASON = register_provider(ctx, logger, _TAG, DrainSecretProvider, _settings)
    if kwargs is None:
        return
    # Opt the drain endpoint into the token-auth seam so the interactive cookie gate
    # doesn't bounce NAS's bearer call.
    try:
        from hermes_cli.dashboard_auth.token_auth import register_token_route

        register_token_route(DRAIN_ROUTE_PATH)
    except Exception as exc:  # noqa: BLE001 — seam import must not crash plugin load
        logger.warning("dashboard-auth-drain: could not register token route %s: %s", DRAIN_ROUTE_PATH, exc)
    logger.info(
        "dashboard-auth-drain: registered drain service-credential provider (scope=%s, route=%s)",
        kwargs["scope"], DRAIN_ROUTE_PATH)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'LoginStart': ('hermes_cli.dashboard_auth', 'LoginStart'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
