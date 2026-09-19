"""KanbanApiSecretProvider — shared-bearer-secret auth for the kanban REST API.

The second consumer of the generic non-interactive token-auth capability
(``supports_token`` / ``verify_token`` on the ``DashboardAuthProvider`` ABC +
the route-agnostic ``token_auth`` middleware seam), after the drain-control
plugin it mirrors.

What it is
----------
A service-to-service auth provider for the sanitized external kanban REST
adapter (``hermes_cli.kanban_api``, mounted at ``/api/plugins/kanban``). An
external control plane presents ``Authorization: Bearer <secret>`` on each
request; this provider verifies it against the operator-provisioned secret
with a constant-time compare and, on a match, vouches for the caller as the
``kanban-api`` principal scoped to ``kanban``. It is NOT an interactive
identity provider — there is no login, cookie, session, or refresh.

How the routes are guarded
--------------------------
``register()`` opts the whole external surface into the token seam with a
prefix registration (the adapter's paths are parameterised, so exact-path
registration cannot cover them) while EXCLUDING the interactive operator
subtree ``/api/plugins/kanban/dashboard`` — that richer first-party surface
stays on the dashboard's cookie/session gate. The registration demands the
``kanban`` scope, so another stacked service credential (e.g. the drain
secret) cannot open these routes, and vice versa.

Consequence worth knowing: once the secret is set, the external surface is
token-only on EVERY bind, including loopback — the seam fully owns a
registered route (fail-closed), so the dashboard session token no longer
authenticates there. Leave the env var unset to keep the pre-plugin
behaviour (loopback: session token; gated: cookie session).

Security properties (mirrors the drain plugin)
----------------------------------------------
* **Entropy gate at registration** — a weak/short/low-entropy secret fails
  CLOSED at load (the plugin declines to register and records a skip reason);
  it is never silently accepted. Bar: >= 256 bits of entropy / >= 43
  url-safe-base64 chars (shared ``dashboard_auth.secret_strength`` gate).
* **Constant-time compare** — ``hmac.compare_digest`` on the request path, so
  the endpoint is not a timing oracle.
* **Scoped principal** — the credential opens only the kanban API surface.

Configuration
-------------
The secret is a CREDENTIAL, so it is carried via an env var (the ``.env``-is-
for-secrets-only rule):

    HERMES_KANBAN_API_SECRET   # the shared secret (>=43 url-safe-b64 chars)

Behavioural knobs live in config.yaml (canonical surface):

    dashboard:
      kanban_api_auth:
        scope: kanban           # capability label attached to the principal
        min_secret_chars: 43    # entropy bar (optional; default 43 ~= 256 bits)

When ``HERMES_KANBAN_API_SECRET`` is unset, the plugin is a no-op (records a
skip reason) — deployments without an external kanban controller just don't
set it.
"""
from __future__ import annotations

import hmac
import logging
import os
from typing import Optional

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    LoginStart,
    Session,
    TokenPrincipal,
)
from hermes_cli.dashboard_auth.secret_strength import (
    DEFAULT_MIN_SECRET_CHARS as _DEFAULT_MIN_SECRET_CHARS,
    assess_secret_strength,
)

logger = logging.getLogger(__name__)

# The external adapter's mount point. Registered as a token-authable prefix
# by ``register()``. Kept here (not imported from web_server) to avoid a
# heavy import at plugin load.
KANBAN_API_PREFIX = "/api/plugins/kanban/"
# The interactive operator dashboard subtree under the same mount — MUST stay
# out of the token seam so the first-party UI keeps cookie/session auth.
KANBAN_DASHBOARD_SUBTREE = "/api/plugins/kanban/dashboard"

LAST_SKIP_REASON: str = ""


class KanbanApiSecretProvider(DashboardAuthProvider):
    """Non-interactive shared-bearer-secret provider for the kanban REST API."""

    name = "kanban-api-secret"
    display_name = "Kanban REST API (service credential)"
    supports_token = True
    supports_session = False

    def __init__(self, *, secret: str, scope: str = "kanban") -> None:
        # Defence in depth: construction also enforces the entropy bar, so a
        # caller that bypasses register()'s check still can't build a weak
        # provider. register() does the friendly skip-reason path; this raises.
        reason = assess_secret_strength(secret)
        if reason is not None:
            raise ValueError(f"kanban API secret rejected: {reason}")
        self._secret = secret
        self._scope = scope or "kanban"

    # ---- token capability (the only thing this provider implements) --------

    def verify_token(self, *, token: str) -> Optional[TokenPrincipal]:
        """Constant-time compare against the provisioned shared secret.

        Returns a ``kanban-api`` principal on an exact match, else ``None``
        (the generic seam falls through / fails closed). Uses
        ``hmac.compare_digest`` so a wrong token can't be recovered by timing.
        """
        if not token:
            return None
        if hmac.compare_digest(token.encode("utf-8"), self._secret.encode("utf-8")):
            return TokenPrincipal(
                principal="kanban-api",
                provider=self.name,
                scopes=(self._scope,),
            )
        return None

    # ---- interactive methods: unsupported (service credential only) --------

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        raise NotImplementedError(
            "KanbanApiSecretProvider is a non-interactive service credential; "
            "there is no login flow."
        )

    def complete_login(
        self, *, code: str, state: str, code_verifier: str, redirect_uri: str
    ) -> Session:
        raise NotImplementedError(
            "KanbanApiSecretProvider is a non-interactive service credential."
        )

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        # Not a cookie-session provider — it never mints a Session, so it can
        # never recognise a session cookie. Return None (don't raise) so it
        # stacks harmlessly in the cookie-verify loop.
        return None

    def refresh_session(self, *, refresh_token: str) -> Session:
        raise NotImplementedError(
            "KanbanApiSecretProvider is a non-interactive service credential."
        )

    def revoke_session(self, *, refresh_token: str) -> None:
        return None


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def _load_config_kanban_api_auth_section() -> dict:
    """Return ``dashboard.kanban_api_auth`` from config.yaml, or ``{}``."""
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional
        logger.debug(
            "dashboard-auth-kanban-api: load_config() raised %s; "
            "falling back to env-only configuration",
            exc,
        )
        return {}
    section = cfg_get(cfg, "dashboard", "kanban_api_auth", default=None)
    return section if isinstance(section, dict) else {}


def register(ctx) -> None:
    """Plugin entry — registers KanbanApiSecretProvider when a strong secret is set.

    No-op (records a skip reason) when ``HERMES_KANBAN_API_SECRET`` is unset
    or fails the entropy gate. On success, also registers the external kanban
    API prefix as token-authable via the generic seam (excluding the
    interactive dashboard subtree).
    """
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""

    secret = os.environ.get("HERMES_KANBAN_API_SECRET", "").strip()
    if not secret:
        LAST_SKIP_REASON = (
            "HERMES_KANBAN_API_SECRET is not set. Set a >=256-bit secret "
            "(e.g. `python -c \"import secrets; "
            "print(secrets.token_urlsafe(32))\"`) to let external control "
            "planes call the kanban REST API with a bearer credential; leave "
            "it unset to keep the API on dashboard-session auth only."
        )
        logger.debug("dashboard-auth-kanban-api: %s", LAST_SKIP_REASON)
        return

    section = _load_config_kanban_api_auth_section()
    scope = str(section.get("scope", "kanban") or "kanban").strip() or "kanban"
    try:
        min_chars = int(section.get("min_secret_chars", _DEFAULT_MIN_SECRET_CHARS))
    except (TypeError, ValueError):
        min_chars = _DEFAULT_MIN_SECRET_CHARS

    reason = assess_secret_strength(secret, min_chars=min_chars)
    if reason is not None:
        LAST_SKIP_REASON = (
            f"HERMES_KANBAN_API_SECRET rejected — {reason}. "
            "The kanban API service credential stays disabled (fail-closed)."
        )
        logger.warning("dashboard-auth-kanban-api: %s", LAST_SKIP_REASON)
        return

    try:
        provider = KanbanApiSecretProvider(secret=secret, scope=scope)
    except ValueError as exc:
        LAST_SKIP_REASON = f"KanbanApiSecretProvider construction failed: {exc}"
        logger.warning("dashboard-auth-kanban-api: %s", LAST_SKIP_REASON)
        return

    ctx.register_dashboard_auth_provider(provider)

    # Opt the external kanban surface into the generic token-auth seam so the
    # dashboard's interactive cookie gate doesn't bounce the controller's
    # bearer call. The operator dashboard subtree is excluded (cookie/session
    # auth), and the scope requirement keeps other stacked service
    # credentials (e.g. the drain secret) off this surface.
    try:
        from hermes_cli.dashboard_auth.token_auth import register_token_route_prefix

        register_token_route_prefix(
            KANBAN_API_PREFIX,
            scope=scope,
            exclude=(KANBAN_DASHBOARD_SUBTREE,),
        )
    except Exception as exc:  # noqa: BLE001 — seam import must not crash plugin load
        logger.warning(
            "dashboard-auth-kanban-api: could not register token route prefix %s: %s",
            KANBAN_API_PREFIX, exc,
        )

    logger.info(
        "dashboard-auth-kanban-api: registered kanban API service-credential "
        "provider (scope=%s, prefix=%s, excluded=%s)",
        scope, KANBAN_API_PREFIX, KANBAN_DASHBOARD_SUBTREE,
    )
