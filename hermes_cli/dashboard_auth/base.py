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
    """

    name: str = ""
    display_name: str = ""

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
