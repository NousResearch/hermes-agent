"""Abstract base + dataclasses + exceptions for dashboard auth providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Session:
    """A verified interactive identity (from ``complete_login`` / ``verify_session``). All fields
    mandatory; providers without orgs set ``org_id=""``. The tokens are opaque to Hermes."""
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
    """A verified non-interactive (service-to-service) caller — the token analog of
    :class:`Session`: one bearer token on one request, no login/cookie/refresh. ``principal`` is
    an opaque stable caller id; ``scopes`` empty means "unscoped" (a route MAY enforce one)."""
    principal: str
    provider: str
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoginStart:
    """First leg of the OAuth round trip: ``redirect_url`` is the IDP's authorize endpoint;
    ``cookie_payload`` maps cookie name -> serialised PKCE/CSRF state that the auth route sets
    (HttpOnly, Secure + ``SameSite=None`` over HTTPS, TTL <= 10 min; see ``set_pkce_cookie``)."""
    redirect_url: str
    cookie_payload: dict[str, str]


class ProviderError(Exception):
    """IDP unreachable / transient failure. Middleware -> HTTP 503."""


class InvalidCodeError(Exception):
    """OAuth callback ``code``/``state`` failed validation. Middleware -> HTTP 400."""


class InvalidCredentialsError(Exception):
    """Username/password rejected. The route answers a generic 401 (no username oracle)."""


class RefreshExpiredError(Exception):
    """This provider rejects the refresh token. Not proof of ownership in a multi-provider
    deployment: middleware tries the rest and forces re-login only after every reachable one
    rejects it."""


def classify_jwks_lookup_error(exc: BaseException) -> Exception:
    """Map a ``PyJWKClient.get_signing_key_from_jwt`` failure to the protocol. Only a genuine
    transport failure (``PyJWKClientConnectionError``, or an unexpected JWKS shape) is a
    :class:`ProviderError` (503, never forces logout). A non-JWT bearer (``DecodeError``), a JWKS
    with no key for this ``kid`` (``PyJWKSetError``) or any other invalid token is simply not
    verifiable by this provider -> :class:`InvalidCodeError` (``verify_session`` returns ``None``).
    Folding "cannot parse" into "cannot reach" once made every opaque bearer a fast 503.

    * ``jwt.DecodeError`` — the bearer is not a JWT at all (an opaque peer key, a legacy session token,
    garbage). #94558: hosted agents answered every non-JWT bearer with a fast 503 ``Auth provider 'nous'
    unreachable`` even though Portal was healthy, because "cannot parse" and "cannot reach" were folded into
    one branch. * ``jwt.PyJWKSetError`` — the JWKS was fetched fine but holds no key for this token's
    ``kid`` (rotated/foreign key).
    """
    try:
        import jwt
    except Exception:  # pragma: no cover - jwt is a hard dep of these providers
        return ProviderError(f"JWKS lookup failed: {exc!r}")
    # Order matters: DecodeError/PyJWKSetError before their PyJWKClientError/InvalidTokenError
    # parents.
    if isinstance(exc, jwt.PyJWKClientConnectionError):
        return ProviderError(f"JWKS lookup failed: {exc}")
    if isinstance(exc, (jwt.DecodeError, jwt.PyJWKSetError)):
        return InvalidCodeError(f"token not verifiable by this provider: {exc}")
    if isinstance(exc, jwt.PyJWKClientError):
        return ProviderError(f"JWKS lookup failed: {exc}")
    if isinstance(exc, jwt.InvalidTokenError):
        return InvalidCodeError(f"token not verifiable by this provider: {exc}")
    return ProviderError(f"JWKS lookup failed: {exc!r}")


class DashboardAuthProvider(ABC):
    """Protocol every dashboard-auth provider plugin implements.

    Lifecycle: ``start_login`` (redirect URL + PKCE state) -> IDP -> ``complete_login`` (code +
    verifier -> Session) -> ``verify_session`` per request -> ``refresh_session`` near expiry ->
    ``revoke_session`` on logout (best-effort, must not raise). Failure semantics: ``start_login``
    / ``complete_login`` raise ``ProviderError`` when the IDP is unreachable, ``complete_login``
    ``InvalidCodeError`` on a bad code/state; ``verify_session`` returns ``None`` for
    expired/unknown tokens (middleware refreshes) and raises ``ProviderError`` when unreachable
    (503); ``refresh_session`` raises ``RefreshExpiredError`` when the token is invalid for that
    provider (a foreign opaque token looks expired, so middleware tries the rest) and
    ``ProviderError`` on network failure (503, cookies kept).

    Subclasses MUST set ``name`` (stable lowercase id) and ``display_name``. Capability flags:
    ``supports_password`` (credential form + ``complete_password_login``; OAuth methods may be
    ``NotImplementedError`` stubs), ``supports_token`` (``verify_token`` for the token-auth seam),
    ``supports_session`` (False for token-only credentials such as drain, never offered a login).
    """
    name: str = ""
    display_name: str = ""
    supports_password: bool = False
    supports_token: bool = False
    supports_session: bool = True

    @abstractmethod
    def start_login(self, *, redirect_uri: str) -> LoginStart: ...

    @abstractmethod
    def complete_login(
        self, *, code: str, state: str, code_verifier: str, redirect_uri: str) -> Session: ...

    @abstractmethod
    def verify_session(self, *, access_token: str) -> Optional[Session]: ...

    @abstractmethod
    def refresh_session(self, *, refresh_token: str) -> Session: ...

    @abstractmethod
    def revoke_session(self, *, refresh_token: str) -> None: ...

    def complete_password_login(self, *, username: str, password: str) -> "Session":
        """Verify a username/password pair and mint a :class:`Session` (only called when
        ``supports_password``). Raise ``InvalidCredentialsError`` on rejection (SHOULD be constant
        time for unknown users — no timing oracle) and ``ProviderError`` when the store is
        unreachable. The default raises so a mis-flagged provider fails loudly."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support password login "
            "(set supports_password = True and override complete_password_login)")

    def verify_token(self, *, token: str) -> "Optional[TokenPrincipal]":
        """Verify a non-interactive bearer token; return its principal. Mirrors ``verify_session``:
        return ``None`` (never raise) for an unrecognised token so the seam falls through; raise
        ``ProviderError`` ONLY for a genuine backing-store outage. Shared secrets MUST be compared
        with ``hmac.compare_digest``. The default raises so a mis-flagged provider fails loudly."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support token auth "
            "(set supports_token = True and override verify_token)")


def assert_protocol_compliance(cls: type) -> None:
    """Raise ``TypeError`` if ``cls`` doesn't fully implement the protocol (call it from every
    provider plugin's unit tests)."""
    for attr in ("name", "display_name"):
        if not getattr(cls, attr, ""):
            raise TypeError(f"{cls.__name__} missing or empty attribute: {attr!r}")
    for method in ("start_login", "complete_login", "verify_session", "refresh_session",
                   "revoke_session"):
        if not callable(getattr(cls, method, None)):
            raise TypeError(f"{cls.__name__} missing method: {method}")
    if getattr(cls, "__abstractmethods__", None):
        raise TypeError(
            f"{cls.__name__} has unimplemented abstract methods: {sorted(cls.__abstractmethods__)}")
