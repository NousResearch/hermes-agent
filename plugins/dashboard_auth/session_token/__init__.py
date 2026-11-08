"""SessionTokenProvider — the deployment session token as a first-class credential.

Gated mode (non-loopback bind) demands a registered ``DashboardAuthProvider`` and verifies
every bearer / WS ``?token=`` through ``verify_session``. The nix/agenix deployment story
(``backend.sessionTokenFile`` → ``HERMES_DASHBOARD_SESSION_TOKEN=file:<path>``) provisions
exactly one shared secret — the same token the Desktop's "Session token" connection field
presents — but until now nothing vouched for it in gated mode, so a token-only deployment
was forced through the interactive provider prompt into basic-auth credentials it never
asked for. This provider closes that gap: it registers whenever the session token is
provisioned (and not explicitly disabled), and ``verify_session`` constant-time-compares
the presented credential against it. Both auth legs the Desktop uses — the REST
``Authorization: Bearer`` path and the ``/api/ws`` ``?token=`` upgrade credential — run
through that seam, so one provisioned token authorizes both.

It is deliberately NOT an interactive login: no login-page button, no cookies, no refresh.
A browser visiting a token-gated dashboard sees a note pointing at the deployment's
credential channel instead of a dead sign-in link. Disable with
``dashboard.session_token_auth.enabled: false`` in config.yaml (the env var itself is the
opt-in; it is a credential, so there is deliberately no new env var for the toggle).
"""
from __future__ import annotations

import hmac
import logging
import os
import time
from typing import Optional

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider, RefreshExpiredError, Session)
from plugins.dashboard_auth._shared import (
    NonInteractiveMixin, SkipRegistration, load_config_section, register_provider)

logger = logging.getLogger(__name__)
_TAG = "dashboard-auth-session-token"

# The verifier re-derives an expiry on every verify (the credential itself never
# expires — rotation is replacing the env var / secret file), so this only bounds
# how long anything that ever minted a cookie from a verified session would keep it.
_SESSION_HORIZON_SECONDS = 24 * 60 * 60

LAST_SKIP_REASON: str = ""


def _resolve_provisioned_token() -> str:
    """The exact token ``web_server`` serves under: env value, or the contents of a
    ``file:``-prefixed path (BOM-tolerant, same reader). Raises ``SystemExit`` for an
    unreadable/empty token file — ``register`` maps that to a skip so plugin discovery
    never dies for a sibling provider's misconfiguration."""
    from hermes_cli.web_server_session_token import resolve_session_token

    return resolve_session_token()


def _load_session_token_auth_section() -> dict:
    return load_config_section(logger, _TAG, "dashboard", "session_token_auth")


def _opt_out() -> bool:
    """``dashboard.session_token_auth.enabled: false`` disables the provider. There is no
    env toggle: the enabling side of this switch IS ``HERMES_DASHBOARD_SESSION_TOKEN``, a
    credential; behaviour belongs in config.yaml."""
    section = _load_session_token_auth_section()
    return str(section.get("enabled", True)).strip().lower() in ("0", "false", "no", "off")


def _settings() -> dict:
    """Resolve SessionTokenProvider kwargs; raises ``SkipRegistration`` when the token is
    not provisioned, the provider is disabled, or the token file is unreadable (the file
    resolver's ``SystemExit`` — a deployment misconfiguration must not crash plugin
    discovery for every sibling plugin)."""
    if not os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN", "").strip():
        raise SkipRegistration(
            "HERMES_DASHBOARD_SESSION_TOKEN is not set. Provision it (a literal token, or "
            "'file:<path>' pointing at a secret-manager-decrypted token file) to let the "
            "deployment session token authorize the gated dashboard.")
    if _opt_out():
        raise SkipRegistration(
            "dashboard.session_token_auth.enabled is false; the deployment session token "
            "does not authenticate the gated dashboard.")
    try:
        token = _resolve_provisioned_token()
    except SystemExit as exc:
        raise SkipRegistration(str(exc), level="error") from exc
    if len(token) < 32:
        logger.warning(
            "%s: the provisioned session token is short (%d chars); use a "
            "`secrets.token_urlsafe(32)`-shaped value for a deployment credential.",
            _TAG, len(token))
    return {"token": token}


class SessionTokenProvider(NonInteractiveMixin, DashboardAuthProvider):
    """Vouches for the deployment session token (``HERMES_DASHBOARD_SESSION_TOKEN``).

    ``supports_session`` is True — the presented bearer/``?token=`` IS the session
    credential the middleware seam verifies — while ``interactive_login`` is False: there
    is no browser sign-in flow, so login surfaces must not offer it.
    """

    name = "session-token"
    display_name = "Deployment Session Token"
    supports_password = False
    supports_session = True
    interactive_login = False
    _NOT_INTERACTIVE = (
        "SessionTokenProvider has no login flow: clients present the provisioned session "
        "token directly (Authorization: Bearer, or ?token= on the /api/ws upgrade).")
    _NO_START_LOGIN = _NOT_INTERACTIVE

    def __init__(self, *, token: str) -> None:
        if not token:
            raise ValueError("token must be non-empty")
        self._token = token

    # ---- session verification (the one capability) --------------------------

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        if not access_token or not hmac.compare_digest(
                access_token.encode("utf-8"), self._token.encode("utf-8")):
            return None
        return Session(
            user_id="deployment-token", email="", display_name="Deployment session token",
            org_id="", provider=self.name,
            expires_at=int(time.time()) + _SESSION_HORIZON_SECONDS,
            access_token=access_token, refresh_token="")

    def refresh_session(self, *, refresh_token: str) -> Session:
        # The credential never rotates server-side mid-process; there is nothing to
        # refresh into. Any refresh attempt is a foreign cookie's — rejected.
        raise RefreshExpiredError("the deployment session token has no refresh token")

    def revoke_session(self, *, refresh_token: str) -> None:
        # Stateless shared credential — revocation is rotation (redeploy the secret).
        # Must not raise.
        return None


# ---- Plugin entry point ----

def register(ctx) -> None:
    """Register ``SessionTokenProvider`` when the deployment session token is provisioned
    and not disabled; a no-op otherwise (so the gate's fail-closed message keeps steering
    interactive deployments toward basic-auth / OAuth)."""
    global LAST_SKIP_REASON
    LAST_SKIP_REASON = ""
    kwargs, LAST_SKIP_REASON = register_provider(ctx, logger, _TAG, SessionTokenProvider, _settings)
    if kwargs is not None:
        logger.info(
            "%s: registered deployment session-token provider — the provisioned "
            "HERMES_DASHBOARD_SESSION_TOKEN authorizes the gated dashboard "
            "(bearer REST + /api/ws ?token=).", _TAG)
