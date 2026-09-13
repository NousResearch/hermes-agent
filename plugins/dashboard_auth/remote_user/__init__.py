"""RemoteUserAuthProvider — trust an authenticated reverse proxy (X-Remote-User).

The reference implementation of the dashboard's ``supports_request_auth``
capability. It lets an upstream proxy (Apache, nginx, Caddy) do the REAL
authentication (pwauth, htpasswd, SSO front-end) and then vouch for the user
to Hermes via a request header — the ``X-Remote-User`` / ``REMOTE_USER``
pattern — so the user logs in ONCE at the proxy and the dashboard trusts it
without a second login and without any session cookie.

Unlike the naive "trust whatever header is there" version, this provider only
honors ``X-Remote-User`` when BOTH hold:

  1. The request's socket peer is one of the configured trusted proxies
     (the IP/CIDR allowlist). This is the primary, non-forgeable control:
     the peer IP is the TCP source address, which an off-proxy client cannot
     fake. If we can't confirm the peer is a proxy we've been told to trust,
     we decline (return None) rather than honor the header.
  2. (Optional, defense in depth) The ``X-Remote-User-Secret`` header matches
     the configured shared secret (constant-time compare). Set this if you
     want a second factor even for a client that tunnels through the trusted
     proxy's network path.

Configuration (env wins over config.yaml when non-empty):

  Trusted proxies (REQUIRED — the provider fails closed without them):
    config:      dashboard.remote_user.trusted_proxies: ["192.168.0.2"]
    env:         HERMES_DASHBOARD_REMOTE_USER_TRUSTED_PROXIES=192.168.0.2

  Header name (optional; default "X-Remote-User"):
    config:      dashboard.remote_user.header: "X-Remote-User"
    env:         HERMES_DASHBOARD_REMOTE_USER_HEADER=X-Remote-User

  Shared secret header + value (optional):
    config:      dashboard.remote_user.secret && .secret_header
    env:         HERMES_DASHBOARD_REMOTE_USER_SECRET / ..._SECRET_HEADER

  Session TTL advertised on the minted Session (optional; default 12h):
    config:      dashboard.remote_user.ttl_seconds
    env:         HERMES_DASHBOARD_REMOTE_USER_TTL_SECONDS

The provider registers only when at least one trusted proxy is configured —
otherwise it is a no-op and the gate behaves exactly as before.

Operator obligations: bind the dashboard to an interface the configured
proxies can reach, configure an allowlist of ONLY those proxy addresses, and
terminate TLS at the proxy. This provider is not appropriate for a dashboard
that any client can reach over a network outside those trusted proxies.

Proxy best practices (see the web-dashboard docs for Apache + nginx examples):

  * The header name is configurable (``HERMES_DASHBOARD_REMOTE_USER_HEADER``,
    default ``X-Remote-User``). The nginx ecosystem often calls it
    ``X-Forwarded-User`` — set it to whatever your proxy emits.
  * NEVER derive identity from ``X-Forwarded-For``: it is user-controllable
    until it reaches a trusted proxy. This provider therefore decides trust
    from the actual socket peer (``request.client.host``), not from any
    forwarded header.
  * The proxy must OVERWRITE (not append) the identity header — strip any
    inbound value a client could have set, then set it from the authenticated
    user (Apache: ``RequestHeader unset X-Remote-User`` then ``set ...`` from
    ``expr=%{REMOTE_USER}``; nginx: ``proxy_set_header X-Remote-User
    $remote_user``).
  * ``remote_user`` mirrors the app-level trusted-proxy validation used by
    projects such as Keycloak (``--proxy-trusted-addresses=``).
  * The proxy delivers only a *username*; the provider synthesizes ``email``
    as ``<user>@proxy.local`` (there is no real mailbox). Any surface that
    displays ``email`` verbatim will show this synthetic value.
"""
from __future__ import annotations

import hmac
import ipaddress
import logging
import os
import time
from typing import Any, Optional

from hermes_cli.dashboard_auth import (
    DashboardAuthProvider,
    InvalidCodeError,
    InvalidCredentialsError,
    LoginStart,
    RefreshExpiredError,
    Session,
)

logger = logging.getLogger(__name__)

_DEFAULT_HEADER = "X-Remote-User"
_DEFAULT_SECRET_HEADER = "X-Remote-User-Secret"
_DEFAULT_TTL_SECONDS = 12 * 60 * 60


def _parse_ip_list(value: str) -> list[str]:
    """Parse a comma/space separated list of IPs or CIDRs."""
    return [part.strip() for part in value.replace(",", " ").split() if part.strip()]


def _peer_is_trusted(peer: str, trusted: list[str]) -> bool:
    """True if ``peer`` falls inside any trusted IP or CIDR."""
    if not peer:
        return False
    try:
        addr = ipaddress.ip_address(peer)
    except ValueError:
        return False
    for entry in trusted:
        try:
            if addr in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


def _clean_username(raw: str) -> str:
    """Validate/normalize a username into an identifier, or '' if unusable."""
    username = raw.strip()
    if not username:
        return ""
    if len(username) > 255:
        return ""
    # Reject control chars / newlines that could break logging or headers.
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in username):
        return ""
    return username


class RemoteUserAuthProvider(DashboardAuthProvider):
    """Vouch for a proxy-authenticated user delivered via a request header."""

    name = "remote-user"
    display_name = "Authenticated Reverse Proxy (Remote User)"

    supports_session: bool = False
    supports_password: bool = False
    supports_token: bool = False
    supports_request_auth: bool = True

    def __init__(
        self,
        *,
        trusted_proxies: list[str],
        header: str = _DEFAULT_HEADER,
        secret: str = "",
        secret_header: str = _DEFAULT_SECRET_HEADER,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        if not trusted_proxies:
            raise ValueError(
                "remote-user auth requires at least one trusted proxy "
                "(dashboard.remote_user.trusted_proxies or "
                "HERMES_DASHBOARD_REMOTE_USER_TRUSTED_PROXIES)"
            )
        self._trusted_proxies = list(trusted_proxies)
        self._header = header.strip() or _DEFAULT_HEADER
        self._secret = secret or ""
        self._secret_header = secret_header.strip() or _DEFAULT_SECRET_HEADER
        if ":" in self._secret:
            raise ValueError("remote-user auth secret must not contain ':'")
        self._ttl_seconds = int(ttl_seconds)

    # ---- the actual trusted-request auth --------------------------------

    def verify_request_auth(self, *, request) -> Optional[Session]:
        """Honor ``X-Remote-User`` only from a trusted proxy peer."""
        client = getattr(request, "client", None)
        peer = client.host if client else ""
        if not _peer_is_trusted(peer, self._trusted_proxies):
            # Peer isn't a proxy we trust — do NOT honor the header (even if
            # present): that's the forged-header case.
            return None
        username = _clean_username(str(request.headers.get(self._header, "")))
        if not username:
            return None
        if self._secret:
            presented = str(request.headers.get(self._secret_header, ""))
            # Constant-time compare on BYTES: hmac.compare_digest(str, str)
            # raises TypeError when either side contains non-ASCII, and the
            # presented value is attacker-controlled via the header — comparing
            # the utf-8-encoded bytes keeps an arbitrary header from turning an
            # authentication attempt into an unhandled 500.
            if not hmac.compare_digest(
                presented.encode("utf-8"), self._secret.encode("utf-8")
            ):
                return None
        return Session(
            user_id=username,
            email=f"{username}@proxy.local",
            display_name=username,
            org_id="",
            provider=self.name,
            expires_at=int(time.time()) + self._ttl_seconds,
            access_token="",
            refresh_token="",
        )

    # ---- dead-end surfaces (not an interactive/session provider) --------

    def verify_session(self, *, access_token: str) -> Optional[Session]:
        return None  # decline: no cookie/session tokens in this scheme

    def refresh_session(self, *, refresh_token: str) -> Session:
        raise RefreshExpiredError("remote-user auth does not refresh")

    def revoke_session(self, *, refresh_token: str) -> None:
        return None

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        raise InvalidCodeError("remote-user auth has no OAuth flow")

    def complete_login(
        self,
        *,
        code: str,
        state: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> Session:
        raise InvalidCodeError("remote-user auth has no OAuth flow")

    def complete_password_login(self, *, username: str, password: str) -> Session:
        raise InvalidCredentialsError("proxy-only authentication")


def _cfg_section() -> dict:
    try:
        from hermes_cli.config import load_config
        return dict((load_config().get("dashboard") or {}).get("remote_user", {}) or {})
    except Exception:  # pragma: no cover - best-effort
        return {}


def _resolve(key_env: str, cfg_key: str, default: Any = "") -> Any:
    env = os.environ.get(key_env, "").strip()
    if env:
        return env
    return _cfg_section().get(cfg_key, default)


def register(ctx) -> None:
    """Plugin entry — registers RemoteUserAuthProvider when configured."""
    proxies_raw = _resolve(
        "HERMES_DASHBOARD_REMOTE_USER_TRUSTED_PROXIES", "trusted_proxies", []
    )
    if isinstance(proxies_raw, str):
        trusted_proxies = _parse_ip_list(proxies_raw)
    elif isinstance(proxies_raw, list):
        trusted_proxies = [str(p).strip() for p in proxies_raw if str(p).strip()]
    else:
        trusted_proxies = []

    if not trusted_proxies:
        reason = (
            "remote-user dashboard auth is not configured: no trusted proxy "
            "allowed. Set dashboard.remote_user.trusted_proxies in config.yaml "
            "(or HERMES_DASHBOARD_REMOTE_USER_TRUSTED_PROXIES in .env) to the "
            "IP/CIDR of the reverse proxy that performs auth, then restart."
        )
        logger.debug("dashboard-auth-remote-user: %s", reason)
        return

    try:
        provider = RemoteUserAuthProvider(
            trusted_proxies=trusted_proxies,
            header=str(_resolve("HERMES_DASHBOARD_REMOTE_USER_HEADER", "header", _DEFAULT_HEADER)),
            secret=str(_resolve("HERMES_DASHBOARD_REMOTE_USER_SECRET", "secret", "")),
            secret_header=str(
                _resolve("HERMES_DASHBOARD_REMOTE_USER_SECRET_HEADER", "secret_header", _DEFAULT_SECRET_HEADER)
            ),
            ttl_seconds=int(
                _resolve("HERMES_DASHBOARD_REMOTE_USER_TTL_SECONDS", "ttl_seconds", _DEFAULT_TTL_SECONDS)
            ) or _DEFAULT_TTL_SECONDS,
        )
    except (ValueError, TypeError) as exc:
        logger.warning(
            "dashboard-auth-remote-user: RemoteUserAuthProvider construction failed: %s",
            exc,
        )
        return

    ctx.register_dashboard_auth_provider(provider)
    logger.info(
        "dashboard-auth-remote-user: registered upstream-proxy auth provider "
        "(trusted_proxies=%r, header=%s, secret_header=%s)",
        trusted_proxies, provider._header, provider._secret_header,
    )
