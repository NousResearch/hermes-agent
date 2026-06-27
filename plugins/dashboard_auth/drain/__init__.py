"""DrainSecretProvider — shared-bearer-secret auth for the drain-control endpoint.

Task 2.0b of the safe-shutdown plan, and the FIRST consumer of the generic
non-interactive token-auth capability added in Task 2.0a
(``supports_token`` / ``verify_token`` on the ``DashboardAuthProvider`` ABC +
the route-agnostic ``token_auth`` middleware seam).

What it is
----------
A service-to-service auth provider. ``nous-account-service`` (NAS) provisions a
**per-agent unique** shared secret into each deployed agent's environment; this
provider verifies an inbound ``Authorization`` bearer token against that secret
with a constant-time compare and, on a match, vouches for the caller as the
``drain-control`` principal. It is NOT an interactive identity provider — there
is no login, cookie, session, or refresh. It implements ONLY the token
capability (``supports_token = True`` + ``verify_token``); the five interactive
ABC methods raise ``NotImplementedError``.

Why a plugin (not an ad-hoc header check on the drain route)
------------------------------------------------------------
Decisions.md Q-A: the drain credential MUST be a real auth plugin in the
dashboard auth framework, not a bolt-on. Q-C: the framework widening that
hosts it is generic (Task 2.0a) and this plugin is merely its first consumer.

Security properties (decisions.md Q-A)
--------------------------------------
* **Per-agent unique secret** — each agent gets a distinct secret; a leak's
  blast radius is one agent.
* **Entropy gate at registration** — a weak/short/low-entropy secret fails
  CLOSED at load (the plugin declines to register and records a skip reason);
  it is never silently accepted. Bar: >= 256 bits of entropy / >= 43
  url-safe-base64 chars, and the value must not be obviously structured
  (all-one-character, too few distinct characters).
* **Constant-time compare** — ``hmac.compare_digest`` on the request path, so
  the endpoint is not a timing oracle.

Configuration
-------------
The secret is a CREDENTIAL, so it is carried via an env var (the ``.env``-is-
for-secrets-only rule), provisioned by NAS at deploy time (Phase 3):

    HERMES_DASHBOARD_DRAIN_SECRET   # the per-agent shared secret (>=43 url-safe-b64 chars)

Behavioural knobs live in config.yaml (canonical surface):

    dashboard:
      drain_auth:
        scope: drain            # capability label attached to the principal
        min_secret_chars: 43    # entropy bar (optional; default 43 ~= 256 bits)

When ``HERMES_DASHBOARD_DRAIN_SECRET`` is unset, the plugin is a no-op (records
a skip reason) — agents that don't want NAS-driven drain just don't set it.
"""
from __future__ import annotations

import hmac
import logging
import math
import os
from collections import Counter
from typing import Optional

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    LoginStart,
    Session,
    TokenPrincipal,
)

logger = logging.getLogger(__name__)

# Default entropy bar: 43 url-safe-base64 chars ~= 256 bits. token_urlsafe(32)
# produces 43 chars, so a correctly-provisioned secret clears this exactly.
_DEFAULT_MIN_SECRET_CHARS = 43
# A secret must contain at least this many DISTINCT characters — rejects
# degenerate values like "aaaa..." that are long but trivially low-entropy.
_MIN_DISTINCT_CHARS = 16
# Shannon entropy floor (bits) over the secret's characters — a second,
# distribution-aware guard on top of the length + distinct-count checks.
_MIN_SHANNON_BITS = 128.0

# The path the begin/cancel-drain endpoint lives on. Registered as a
# token-authable route by ``register()`` so the generic seam guards it. Kept
# here (not imported from web_server) to avoid a heavy import at plugin load.
DRAIN_ROUTE_PATH = "/api/gateway/drain"

LAST_SKIP_REASON: str = ""


def _shannon_bits(value: str) -> float:
    """Total Shannon entropy (bits) of ``value`` over its character distribution.

    H = len * sum(-p_i * log2(p_i)). A long string drawn from a wide alphabet
    scores high; a long run of one character scores ~0.
    """
    if not value:
        return 0.0
    counts = Counter(value)
    n = len(value)
    per_char = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return per_char * n


def assess_secret_strength(
    secret: str, *, min_chars: int = _DEFAULT_MIN_SECRET_CHARS
) -> Optional[str]:
    """Return a rejection reason if ``secret`` is too weak, else ``None``.

    Fail-closed entropy gate (decisions.md Q-A). Checks, in order:
      * length >= ``min_chars`` (default 43 url-safe-b64 chars ~= 256 bits),
      * at least ``_MIN_DISTINCT_CHARS`` distinct characters,
      * Shannon entropy >= ``_MIN_SHANNON_BITS`` bits.

    A ``None`` return means the secret passes. Any string return is a
    human-readable reason the caller logs + records as the skip reason.
    """
    if not secret:
        return "secret is empty"
    if len(secret) < min_chars:
        return (
            f"secret too short: {len(secret)} chars (need >= {min_chars}; "
            "use a >=256-bit value, e.g. `python -c \"import secrets; "
            "print(secrets.token_urlsafe(32))\"`)"
        )
    distinct = len(set(secret))
    if distinct < _MIN_DISTINCT_CHARS:
        return (
            f"secret has only {distinct} distinct characters (need >= "
            f"{_MIN_DISTINCT_CHARS}); looks structured/low-entropy"
        )
    bits = _shannon_bits(secret)
    if bits < _MIN_SHANNON_BITS:
        return (
            f"secret entropy too low: {bits:.0f} bits (need >= "
            f"{_MIN_SHANNON_BITS:.0f}); looks structured/repeated"
        )
    return None


class DrainSecretProvider(DashboardAuthProvider):
    """Non-interactive shared-bearer-secret provider for drain control."""

    name = "drain-secret"
    display_name = "Drain Control (service credential)"
    supports_token = True
    supports_session = False

    def __init__(self, *, secret: str, scope: str = "drain") -> None:
        # Defence in depth: construction also enforces the entropy bar, so a
        # caller that bypasses register()'s check still can't build a weak
        # provider. register() does the friendly skip-reason path; this raises.
        reason = assess_secret_strength(secret)
        if reason is not None:
            raise ValueError(f"drain secret rejected: {reason}")
        self._secret = secret
        self._scope = scope or "drain"

    # ---- token capability (the only thing this provider implements) --------

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        """Constant-time compare against the per-agent shared secret.

        Returns a ``drain-control`` principal on an exact match, else ``None``
        (the generic seam falls through / fails closed). Uses
        ``hmac.compare_digest`` so a wrong token can't be recovered by timing.
        """
        if not token:
            return None
        if hmac.compare_digest(token.encode("utf-8"), self._secret.encode("utf-8")):
            return TokenPrincipal(
                principal="drain-control",
                provider=self.name,
                scopes=(self._scope,),
            )
        return None

    # ---- interactive methods: unsupported (service credential only) --------

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        raise NotImplementedError(
            "DrainSecretProvider is a non-interactive service credential; "
            "there is no login flow."
        )

    def complete_login(
        self, *, code: str, state: str, code_verifier: str, redirect_uri: str
    ) -> Session:
        raise NotImplementedError(
            "DrainSecretProvider is a non-interactive service credential."
        )

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        # Not a cookie-session provider — it never mints a Session, so it can
        # never recognise a session cookie. Return None (don't raise) so it
        # stacks harmlessly in the cookie-verify loop.
        return None

    def refresh_session(self, *, refresh_token: str) -> Session:
        raise NotImplementedError(
            "DrainSecretProvider is a non-interactive service credential."
        )

    def revoke_session(self, *, refresh_token: str) -> None:
        return None


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def _load_config_drain_auth_section() -> dict:
    """Return ``dashboard.drain_auth`` from config.yaml, or ``{}``."""
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional
        logger.debug(
            "dashboard-auth-drain: load_config() raised %s; "
            "falling back to env-only configuration",
            exc,
        )
        return {}
    section = cfg_get(cfg, "dashboard", "drain_auth", default=None)
    return section if isinstance(section, dict) else {}


def register(ctx) -> None:
    """Plugin entry — registers DrainSecretProvider when a strong secret is set.

    No-op (records a skip reason) when ``HERMES_DASHBOARD_DRAIN_SECRET`` is
    unset or fails the entropy gate. On success, also registers the
    begin/cancel-drain route as token-authable via the generic seam.
    """
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""

    secret = os.environ.get("HERMES_DASHBOARD_DRAIN_SECRET", "").strip()
    if not secret:
        LAST_SKIP_REASON = (
            "HERMES_DASHBOARD_DRAIN_SECRET is not set. Set a per-agent "
            ">=256-bit secret (e.g. `python -c \"import secrets; "
            "print(secrets.token_urlsafe(32))\"`) to enable NAS-driven drain "
            "coordination; leave it unset to disable the drain endpoint."
        )
        logger.debug("dashboard-auth-drain: %s", LAST_SKIP_REASON)
        return

    section = _load_config_drain_auth_section()
    scope = str(section.get("scope", "drain") or "drain").strip() or "drain"
    try:
        min_chars = int(section.get("min_secret_chars", _DEFAULT_MIN_SECRET_CHARS))
    except (TypeError, ValueError):
        min_chars = _DEFAULT_MIN_SECRET_CHARS

    reason = assess_secret_strength(secret, min_chars=min_chars)
    if reason is not None:
        LAST_SKIP_REASON = (
            f"HERMES_DASHBOARD_DRAIN_SECRET rejected — {reason}. "
            "The drain endpoint stays disabled (fail-closed)."
        )
        logger.warning("dashboard-auth-drain: %s", LAST_SKIP_REASON)
        return

    try:
        provider = DrainSecretProvider(secret=secret, scope=scope)
    except ValueError as exc:
        LAST_SKIP_REASON = f"DrainSecretProvider construction failed: {exc}"
        logger.warning("dashboard-auth-drain: %s", LAST_SKIP_REASON)
        return

    ctx.register_dashboard_auth_provider(provider)

    # Opt the begin/cancel-drain endpoint into the generic token-auth seam so
    # the dashboard's interactive cookie gate doesn't bounce NAS's bearer call.
    try:
        from hermes_cli.dashboard_auth.token_auth import register_token_route

        register_token_route(DRAIN_ROUTE_PATH)
    except Exception as exc:  # noqa: BLE001 — seam import must not crash plugin load
        logger.warning(
            "dashboard-auth-drain: could not register token route %s: %s",
            DRAIN_ROUTE_PATH, exc,
        )

    logger.info(
        "dashboard-auth-drain: registered drain service-credential provider "
        "(scope=%s, route=%s)",
        scope, DRAIN_ROUTE_PATH,
    )
