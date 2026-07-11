"""Abstract base + dataclasses + exceptions for dashboard auth providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Session:
    """A verified identity. Returned by ``complete_login`` and ``verify_session``.

    All fields are mandatory. Providers that don't have a concept of orgs
    should set ``org_id`` to an empty string. ``access_token`` and
    ``refresh_token`` are opaque to Hermes — provider-specific.
    """

    user_id: str
    email: str
    display_name: str
    org_id: str
    provider: str
    expires_at: int  # unix seconds; the access_token's exp claim
    access_token: str
    refresh_token: str


@dataclass(frozen=True)
class TokenPrincipal:
    """A verified non-interactive (service-to-service) caller.

    The token analog of :class:`Session`. Where a ``Session`` represents an
    interactive human identity behind a session cookie, a ``TokenPrincipal``
    represents a machine/service caller that authenticated by presenting a
    bearer token in the ``Authorization`` request header on a single
    request — no login, no cookie, no refresh.

    Returned by :meth:`DashboardAuthProvider.verify_token` and attached to
    ``request.state.token_principal`` by the token-auth middleware seam so a
    route handler can see *who* called it.

    Fields:
      * ``principal`` — stable identifier for the caller (e.g. the provider
        name, a service account id, or an agent id). Opaque to the seam.
      * ``provider`` — the ``name`` of the provider that verified the token.
      * ``scopes`` — capability strings this principal is authorised for.
        Empty tuple means "unscoped" (the provider vouches for the caller but
        attaches no capability list); a route MAY enforce a required scope.
    """

    principal: str
    provider: str
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoginStart:
    """First leg of the OAuth round trip.

    ``redirect_url`` is the URL the browser must navigate to (e.g. the
    Portal's ``/oauth/authorize``). ``cookie_payload`` is a dict of cookie
    name → serialised value that the auth route will ``Set-Cookie`` on the
    response. Used for PKCE state, CSRF nonces, etc. Cookies set here MUST
    be HttpOnly + Secure (when over HTTPS) + SameSite=Lax with a TTL ≤ 10
    minutes (the login lifetime).
    """

    redirect_url: str
    cookie_payload: dict[str, str]


class ProviderError(Exception):
    """IDP unreachable, network error, or other transient failure.

    Middleware translates this to HTTP 503.
    """


class InvalidCodeError(Exception):
    """The OAuth callback ``code`` / ``state`` failed validation.

    Middleware translates this to HTTP 400.
    """


class InvalidCredentialsError(Exception):
    """A username/password pair was rejected by a password provider.

    Raised by :meth:`DashboardAuthProvider.complete_password_login`. The
    ``/auth/password-login`` route translates this to HTTP 401 with a
    deliberately generic detail (never distinguishing "unknown user" from
    "wrong password") so the endpoint can't be used as a username oracle.
    """


class RefreshExpiredError(Exception):
    """The refresh token is dead.

    Middleware clears cookies and forces re-login (302 → ``/login``).
    """


class DashboardAuthProvider(ABC):
    """Protocol every dashboard-auth provider plugin implements.

    Lifecycle:
      1. ``start_login`` — user clicks "Log in with X" on the login page.
         Provider returns a redirect URL and any PKCE/CSRF state to stash
         in short-lived cookies.
      2. Browser bounces through the OAuth IDP and lands at /auth/callback.
      3. ``complete_login`` — exchange the code + verifier for a Session.
      4. ``verify_session`` — called on every request to validate the
         access token in the cookie. Returns ``None`` if the token is
         expired or invalid (middleware then triggers refresh or logout).
      5. ``refresh_session`` — called when the access token is near expiry.
         Returns a new Session with rotated tokens.
      6. ``revoke_session`` — called on /auth/logout. Best-effort.

    Failure semantics:
      * ``start_login`` may raise ``ProviderError`` if the IDP is
        unreachable.
      * ``complete_login`` raises ``InvalidCodeError`` on bad code/state;
        ``ProviderError`` if the IDP is unreachable.
      * ``verify_session`` returns ``None`` on expiry / unknown token;
        raises ``ProviderError`` if the IDP is unreachable. Middleware
        treats expiry and unreachable differently (expiry → refresh;
        unreachable → 503).
      * ``refresh_session`` raises ``RefreshExpiredError`` when the
        refresh token is also invalid; middleware then forces re-login.
        Raises ``ProviderError`` on network failure.
      * ``revoke_session`` is best-effort and must not raise.

    Subclasses MUST set ``name`` (lowercase identifier, stable forever)
    and ``display_name`` (user-facing label on the login page).

    Password (non-redirect) providers:
      A provider that authenticates with a username + password instead of
      an OAuth redirect sets ``supports_password = True`` and implements
      ``complete_password_login``. The login page then renders a
      credential form (POSTing to ``/auth/password-login``) instead of a
      "Log in with X" redirect button. Everything downstream of login —
      ``verify_session`` / ``refresh_session`` / ``revoke_session``, the
      session cookies, the WS-ticket mint — is identical to the OAuth
      path, because a password session is just a :class:`Session` with
      provider-minted opaque tokens. The OAuth methods (``start_login`` /
      ``complete_login``) remain abstract; a pure-password provider that
      will never be reached via the redirect flow may implement them as
      stubs that raise ``NotImplementedError``.
    """

    name: str = ""
    display_name: str = ""

    # When True, this provider authenticates via username + password
    # (``complete_password_login``) rather than (or in addition to) the
    # OAuth redirect flow. The login page renders a credential form for
    # such providers; the ``/auth/password-login`` route dispatches to
    # ``complete_password_login``. OAuth-only providers leave this False
    # and are completely unaffected.
    supports_password: bool = False

    # When True, this provider can verify a non-interactive bearer token
    # (``verify_token``) presented on a single request by a service-to-service
    # caller — no login, no cookie, no refresh. This is the generic
    # API-token capability flag, mirroring ``supports_password``: a route
    # opts into token auth (see ``token_auth`` middleware seam) and the
    # gate consults every ``supports_token`` provider in turn until one
    # recognises the token. OAuth/password providers leave this False and
    # are completely unaffected. The drain bearer-secret plugin is the
    # first consumer, but the capability is deliberately generic so any
    # future machine-credential provider drops in without core changes.
    supports_token: bool = False

    # When True, this provider does the interactive cookie-session flow (login,
    # verify, refresh). The login page, /auth/login, and the gate's
    # verify/refresh loops consult only supports_session providers, so a
    # token-only credential (e.g. drain) is never offered a login. Mirrors
    # supports_token.
    supports_session: bool = True

    # When True, this provider implements the interactive OAuth redirect flow
    # (``start_login`` returns a real ``LoginStart`` / ``complete_login`` runs
    # the auth-code exchange). The gate's silent auto-SSO optimization
    # (``_auto_sso_response``) only fires for such providers; a password-only
    # provider leaves this False so the gate falls through to the ``/login``
    # credential form instead of bouncing the browser to ``/auth/login``
    # (whose ``start_login`` would raise NotImplementedError → HTTP 500).
    # Mirrors supports_password / supports_token / supports_session.
    supports_oauth: bool = True

    @abstractmethod
    def start_login(self, *, redirect_uri: str) -> LoginStart: ...

    @abstractmethod
    def complete_login(
        self,
        *,
        code: str,
        state: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> Session: ...

    @abstractmethod
    def verify_session(self, *, access_token: str) -> Optional[Session]: ...

    @abstractmethod
    def refresh_session(self, *, refresh_token: str) -> Session: ...

    @abstractmethod
    def revoke_session(self, *, refresh_token: str) -> None: ...

    def complete_password_login(
        self, *, username: str, password: str
    ) -> "Session":
        """Verify a username/password pair and mint a :class:`Session`.

        Only called when ``supports_password`` is True (the
        ``/auth/password-login`` route guards on the flag). The default
        raises ``NotImplementedError`` so an OAuth-only provider that
        forgets to set the flag fails loudly rather than silently
        accepting credentials.

        The returned ``Session`` carries provider-minted opaque
        ``access_token`` / ``refresh_token`` exactly like the OAuth path,
        so all downstream session handling (cookies, verify, refresh,
        ws-tickets, logout) is identical.

        Failure semantics:
          * ``InvalidCredentialsError`` — username/password rejected. The
            route surfaces a generic 401 (no user-vs-password
            distinction). Implementations SHOULD spend constant time on
            unknown users (dummy hash verify) to avoid a timing oracle.
          * ``ProviderError`` — the backing credential store is
            unreachable (LDAP/DB down); the route surfaces 503.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support password login "
            "(set supports_password = True and override "
            "complete_password_login)"
        )

    def verify_token(self, *, token: str) -> "Optional[TokenPrincipal]":
        """Verify a non-interactive bearer token; return its principal.

        The token analog of ``verify_session``. Only consulted when
        ``supports_token`` is True. Called by the ``token_auth`` middleware
        seam for every request to a token-authable route, in registration
        order, until one provider returns a non-None principal.

        Contract (mirrors ``verify_session`` stacking semantics):
          * Return a :class:`TokenPrincipal` if this provider recognises and
            accepts the token.
          * Return ``None`` for a token this provider does NOT recognise —
            never raise, so the seam can fall through to the next provider.
            A malformed/expired/wrong token is "not recognised" → ``None``.
          * Raise ``ProviderError`` ONLY for a genuine backing-store outage
            (the provider can neither confirm nor deny). The seam treats this
            like ``verify_session``: remember it, keep trying other providers,
            and surface 503 only if NO provider accepts the token AND at least
            one was unreachable.

        Implementations MUST use a constant-time comparison
        (``hmac.compare_digest``) when matching a shared secret so the
        endpoint isn't a timing oracle.

        The default raises ``NotImplementedError`` so a provider that sets
        ``supports_token`` but forgets to implement this fails loudly rather
        than silently accepting every caller.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support token auth "
            "(set supports_token = True and override verify_token)"
        )


def assert_protocol_compliance(cls: type) -> None:
    """Raise ``TypeError`` if ``cls`` doesn't fully implement the provider protocol.

    Call this in every provider plugin's unit tests::

        def test_protocol_compliance():
            assert_protocol_compliance(MyProvider)

    Returns ``None`` on success so callers can assert it explicitly.
    """
    required_methods = (
        "start_login",
        "complete_login",
        "verify_session",
        "refresh_session",
        "revoke_session",
    )
    required_attrs = ("name", "display_name")

    for attr in required_attrs:
        val = getattr(cls, attr, "")
        if not val:
            raise TypeError(
                f"{cls.__name__} missing or empty attribute: {attr!r}"
            )
    for method in required_methods:
        if not callable(getattr(cls, method, None)):
            raise TypeError(f"{cls.__name__} missing method: {method}")
    # Also catch the ABC-not-overridden case.
    if getattr(cls, "__abstractmethods__", None):
        raise TypeError(
            f"{cls.__name__} has unimplemented abstract methods: "
            f"{sorted(cls.__abstractmethods__)}"
        )
