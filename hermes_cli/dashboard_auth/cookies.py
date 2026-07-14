"""Cookie helpers for dashboard auth.

Three cookies in play:
  - hermes_session_at:   the OAuth access token
                         (HttpOnly, lifetime = token TTL, ~15 min)
  - hermes_session_rt:   the OAuth refresh token
                         (HttpOnly, lifetime = 24h, ROTATING + reuse-detected)
                         Nous Portal issues a rotating refresh token for the
                         dashboard auth-code grant (Portal NAS #293 / hermes
                         #37247). ``set_session_cookies`` writes this cookie
                         whenever the provider returns a non-empty
                         ``refresh_token``; the middleware uses it to rotate a
                         fresh access token transparently on AT expiry. A
                         provider that omits the refresh token (empty string)
                         degrades gracefully to access-token-only sessions —
                         the RT cookie is simply not written.
  - hermes_session_pkce: short-lived PKCE state + CSRF nonce + provider
                         hint (HttpOnly, lifetime = 10 minutes)

All three are ``SameSite=Lax`` (browser will send on cross-site GET
top-level navigation, which we need for the IDP redirect back to
``/auth/callback``) and live under the prefix's Path. ``Secure`` is set
ONLY when the dashboard was reached over HTTPS — detected via the
request URL scheme, which honours ``X-Forwarded-Proto`` upstream of
Fly's TLS terminator when uvicorn is configured with
``proxy_headers=True``. Loopback dev traffic is always HTTP so
``Secure`` would lock the cookies out of the browser.

Cookie prefix selection (browser hardening per
https://datatracker.ietf.org/doc/html/draft-west-cookie-prefixes):

  * Loopback HTTP — bare name. ``__Host-`` / ``__Secure-`` require
    ``Secure``, which is incompatible with HTTP.
  * Gated HTTPS, direct deploy (Path=/) — ``__Host-`` prefix. Binds the
    cookie to the exact origin (no Domain attribute) — strongest spec
    guarantee.
  * Gated HTTPS, behind a reverse-proxy prefix (Path=/hermes) —
    ``__Secure-`` prefix. ``__Host-`` is disallowed when Path != "/";
    ``__Secure-`` keeps the Secure-required hardening without the
    Path constraint, and the explicit ``Path=/hermes`` covers
    same-origin app isolation.

The setters and readers BOTH consult the active prefix because the
cookie *name* changes — a reader that looked up the bare name when the
setter wrote ``__Secure-hermes_session_at`` would never find the value.

Refresh-token handling:
   ``set_session_cookies`` accepts ``refresh_token=""`` (provider omitted
   it) and silently skips writing the RT cookie in that case, so a
   refresh-token-less provider degrades to access-token-only sessions.
   ``clear_session_cookies`` always emits a Max-Age=0 deletion for the RT
   cookie on logout / session expiry so a stale cookie from an earlier
   deployment gets cleared. The transparent rotation flow ("expired AT +
   live RT → rotate server-side, else 401 → /login") lives in
   ``middleware._attempt_refresh``.
"""
from __future__ import annotations

from typing import Optional, Tuple

from fastapi import Request
from fastapi.responses import Response

# Bare cookie names — the request-scoped ``_resolved_name`` helper
# decides whether to prepend ``__Host-`` / ``__Secure-`` based on the
# request's HTTPS + prefix combination.
SESSION_AT_COOKIE = "hermes_session_at"
SESSION_RT_COOKIE = "hermes_session_rt"
PKCE_COOKIE = "hermes_session_pkce"
# One-shot loop-guard marker for the auto-SSO redirect (Phase 1,
# cloud-auto-discovery). Set when the gate auto-initiates the portal OAuth
# redirect on an unauthenticated document load; its mere PRESENCE on the next
# unauthenticated load tells the gate "we already bounced once" so a genuinely
# absent portal session degrades to the /login page instead of ping-ponging.
# Carries no secret — it's a boolean breadcrumb — but is set HttpOnly/Lax/Secure
# like the others for consistency. Short TTL so a user who returns later gets a
# fresh silent attempt rather than a permanently-disabled one.
SSO_ATTEMPT_COOKIE = "hermes_sso_attempt"

# Possible name variants we may have to read back. Sorted so most-strict
# wins on iteration when both happen to be present (shouldn't happen in
# practice — a single request emits exactly one variant).
_NAME_VARIANTS = ("__Host-", "__Secure-", "")

# RT cookie Max-Age. Kept at 30 days as a generous upper bound on the cookie's
# browser lifetime; Portal's actual refresh-token TTL (24h, rotating) is the
# real authority — once the RT itself expires/rotates out, a refresh attempt
# returns 400 → RefreshExpiredError → clean re-login, regardless of how long
# the cookie lingers. (Not tightened to 24h here to avoid coupling the cookie
# lifetime to a server-side TTL that can change independently; revisit if the
# stale-cookie refresh churn ever matters.)
_RT_MAX_AGE = 30 * 24 * 60 * 60
_PKCE_MAX_AGE = 10 * 60
# Auto-SSO loop-guard marker TTL. Just long enough to cover one redirect
# round trip to the portal and back (a few seconds in practice); kept at 60s
# so a slow portal hop or a manual back-button still trips the guard, while a
# user returning minutes later gets a fresh silent attempt rather than being
# stuck on /login forever. The marker is also cleared explicitly on a
# successful callback and whenever the gate falls back to /login.
_SSO_ATTEMPT_MAX_AGE = 60


def _resolved_name(bare: str, *, use_https: bool, prefix: str) -> str:
    """Pick the cookie-prefix variant for the active request shape.

    See module docstring for the prefix selection rules. Mismatch
    between setter and reader would silently break sessions, so this
    function is the single source of truth for naming.
    """
    if not use_https:
        return bare
    if prefix:
        # Path != "/" forbids __Host-; fall back to __Secure-.
        return f"__Secure-{bare}"
    return f"__Host-{bare}"


def _cookie_path(prefix: str) -> str:
    """Cookie ``Path`` attribute for the active deploy shape.

    Under ``X-Forwarded-Prefix: /hermes`` we want ``Path=/hermes`` so:
      a) the browser sends the cookie back on requests under the prefix
         (browsers omit the cookie if request path doesn't start with
         Path);
      b) the cookie doesn't leak to other apps on the same origin
         (``mission-control.tilos.com/billing/...``).

    Direct-deploy (no proxy prefix) gets ``Path=/``.
    """
    return prefix if prefix else "/"


def _common_attrs(*, use_https: bool, prefix: str) -> dict:
    attrs: dict = {
        "httponly": True,
        "samesite": "lax",
        "path": _cookie_path(prefix),
    }
    if use_https:
        attrs["secure"] = True
    return attrs


def set_session_cookies(
    response: Response,
    *,
    access_token: str,
    refresh_token: str,
    access_token_expires_in: int,
    use_https: bool,
    prefix: str = "",
) -> None:
    """Set the session cookies on the response.

    ``access_token_expires_in`` is in seconds. Use the provider's reported
    TTL for the access token.

    ``refresh_token`` is written as the RT cookie when non-empty. Nous Portal
    issues a 24h rotating refresh token (hermes #37247); a provider that
    omits it returns ``Session.refresh_token == ""`` and we simply don't
    persist the RT cookie — the session then behaves as access-token-only
    until the AT expires. No other branch changes between the two cases.

    ``prefix`` is the normalised X-Forwarded-Prefix value (e.g. ``/hermes``)
    or ``""`` for a direct deploy. It influences both the cookie name
    (``__Host-`` vs ``__Secure-`` vs bare) and the ``Path`` attribute.
    """
    response.set_cookie(
        _resolved_name(SESSION_AT_COOKIE, use_https=use_https, prefix=prefix),
        access_token,
        max_age=access_token_expires_in,
        **_common_attrs(use_https=use_https, prefix=prefix),
    )
    # Contract v1: empty refresh token means "don't persist RT cookie".
    # Keeping a literal empty-value cookie around would be dead state at
    # best, attack surface at worst.
    if refresh_token:
        response.set_cookie(
            _resolved_name(SESSION_RT_COOKIE, use_https=use_https, prefix=prefix),
            refresh_token,
            max_age=_RT_MAX_AGE,
            **_common_attrs(use_https=use_https, prefix=prefix),
        )


def _emit_deletion(response: Response, bare: str, variant: str, prefix: str) -> None:
    """Emit one Max-Age=0 deletion for a single cookie-name variant.

    The cookie-prefix rules that gate the *setters* gate deletions too: a
    browser silently rejects a ``Set-Cookie`` for a ``__Secure-`` / ``__Host-``
    name that lacks ``Secure`` (and ``__Host-`` additionally requires
    ``Path=/``). Emitting the prefixed deletions as bare HTTP cookies — as we
    used to — means the browser drops them on the floor and keeps replaying the
    real, still-live session cookie after logout/expiry. So the deletion must
    mirror the attributes the setter would have used for that variant.
    """
    secure = variant != ""
    # __Host- is pinned to Path=/ by spec; a deletion under any other path is
    # rejected, so override the path for that one variant.
    path = "/" if variant == "__Host-" else _cookie_path(prefix)
    response.set_cookie(
        f"{variant}{bare}", "", max_age=0,
        path=path, httponly=True, samesite="lax", secure=secure,
    )


def clear_session_cookies(response: Response, *, prefix: str = "") -> None:
    """Emit Max-Age=0 deletions for both session cookies.

    To delete a cookie reliably the deletion's ``Path`` must match the
    set path AND the cookie name must match the variant the setter used.
    We don't know which variant was originally set (cookie prefix
    depends on the request that set it), so we emit deletions for every
    plausible variant under the active path — each carrying the ``Secure``/
    ``Path`` attributes that variant's prefix requires (see
    :func:`_emit_deletion`).
    """
    for variant in _NAME_VARIANTS:
        _emit_deletion(response, SESSION_AT_COOKIE, variant, prefix)
        _emit_deletion(response, SESSION_RT_COOKIE, variant, prefix)


def set_pkce_cookie(
    response: Response, *, payload: str, use_https: bool, prefix: str = "",
) -> None:
    response.set_cookie(
        _resolved_name(PKCE_COOKIE, use_https=use_https, prefix=prefix),
        payload,
        max_age=_PKCE_MAX_AGE,
        **_common_attrs(use_https=use_https, prefix=prefix),
    )


def clear_pkce_cookie(response: Response, *, prefix: str = "") -> None:
    for variant in _NAME_VARIANTS:
        _emit_deletion(response, PKCE_COOKIE, variant, prefix)


def _read_with_fallback(
    request: Request, bare_name: str,
) -> Optional[str]:
    """Read a cookie by checking every prefix variant in order.

    The setter chooses one variant based on the active request shape;
    the reader doesn't know which one fired (the request that READS
    the cookie may not be the same shape as the request that SET it
    in pathological cases). Trying all three guarantees we find it.
    """
    for variant in _NAME_VARIANTS:
        value = request.cookies.get(f"{variant}{bare_name}")
        if value is not None:
            return value
    return None


def read_session_cookies(request: Request) -> Tuple[Optional[str], Optional[str]]:
    """Returns (access_token, refresh_token), either may be None."""
    at = _read_with_fallback(request, SESSION_AT_COOKIE)
    rt = _read_with_fallback(request, SESSION_RT_COOKIE)
    return at, rt


def read_pkce_cookie(request: Request) -> Optional[str]:
    return _read_with_fallback(request, PKCE_COOKIE)


def set_sso_attempt_cookie(
    response: Response, *, use_https: bool, prefix: str = "",
) -> None:
    """Set the one-shot auto-SSO loop-guard marker (Phase 1).

    Written by the gate the moment it auto-initiates the portal OAuth
    redirect on an unauthenticated document load. The value is a constant
    (``"1"``) — only its presence matters. Short Max-Age so a stale marker
    can't permanently suppress a future silent attempt.
    """
    response.set_cookie(
        _resolved_name(SSO_ATTEMPT_COOKIE, use_https=use_https, prefix=prefix),
        "1",
        max_age=_SSO_ATTEMPT_MAX_AGE,
        **_common_attrs(use_https=use_https, prefix=prefix),
    )


def read_sso_attempt_cookie(request: Request) -> Optional[str]:
    """Return the auto-SSO marker value if present (any variant), else None."""
    return _read_with_fallback(request, SSO_ATTEMPT_COOKIE)


def clear_sso_attempt_cookie(response: Response, *, prefix: str = "") -> None:
    """Emit Max-Age=0 deletions for the auto-SSO marker, every name variant.

    Called on a successful callback and whenever the gate falls back to
    /login, so the marker never lingers to suppress a later silent attempt.
    """
    path = _cookie_path(prefix)
    for variant in _NAME_VARIANTS:
        response.set_cookie(
            f"{variant}{SSO_ATTEMPT_COOKIE}", "", max_age=0,
            path=path, httponly=True, samesite="lax",
        )


def detect_https(request: Request) -> bool:
    """Decide whether to set the ``Secure`` cookie flag.

    Reads ``request.url.scheme`` — under uvicorn's ``proxy_headers=True``
    (which start_server enables when the gate is active), this honours
    ``X-Forwarded-Proto`` from Fly's TLS terminator. Loopback traffic is
    always HTTP so this returns False there.
    """
    return request.url.scheme == "https"
