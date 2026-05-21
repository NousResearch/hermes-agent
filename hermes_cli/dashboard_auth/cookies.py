"""Cookie helpers for dashboard auth.

Three cookies in play:
  - hermes_session_at:   the OAuth access token
                         (HttpOnly, lifetime = token TTL)
  - hermes_session_rt:   the OAuth refresh token
                         (HttpOnly, lifetime = 30 days)
                         **DEPRECATED in OAuth contract v1** — Nous Portal
                         does not issue refresh tokens; we keep the cookie
                         name and clear semantics for forward compatibility
                         and to flush stale cookies from old browsers.
  - hermes_session_pkce: short-lived PKCE state + CSRF nonce + provider
                         hint (HttpOnly, lifetime = 10 minutes)

All three are ``SameSite=Lax`` (browser will send on cross-site GET
top-level navigation, which we need for the IDP redirect back to
``/auth/callback``) and ``Path=/``. ``Secure`` is set ONLY when the
dashboard was reached over HTTPS — detected via the request URL scheme,
which honours ``X-Forwarded-Proto`` upstream of Fly's TLS terminator
when uvicorn is configured with ``proxy_headers=True``. Loopback dev
traffic is always HTTP so ``Secure`` would lock the cookies out of
the browser.

.. deprecated:: contract v1
   ``set_session_cookies`` accepts ``refresh_token=""`` (the contract-v1
   default) and silently skips writing ``hermes_session_rt`` in that case.
   ``clear_session_cookies`` still emits a Max-Age=0 deletion for the RT
   cookie so users carrying a stale cookie from an earlier deployment get
   it cleared on logout / session expiry. The full refresh-flow machinery
   was rewritten as "401 → redirect to /login" in Phase 6.
"""
from __future__ import annotations

from typing import Optional, Tuple

from fastapi import Request
from fastapi.responses import Response

SESSION_AT_COOKIE = "hermes_session_at"
SESSION_RT_COOKIE = "hermes_session_rt"
PKCE_COOKIE = "hermes_session_pkce"

# 30 days — matches Portal's REFRESH_TOKEN_TTL_SECONDS
_RT_MAX_AGE = 30 * 24 * 60 * 60
_PKCE_MAX_AGE = 10 * 60


def _common_attrs(use_https: bool) -> dict:
    attrs: dict = {
        "httponly": True,
        "samesite": "lax",
        "path": "/",
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
) -> None:
    """Set the session cookies on the response.

    ``access_token_expires_in`` is in seconds. Use the provider's reported
    TTL for the access token.

    ``refresh_token`` is accepted for backward / forward compatibility but
    SKIPPED when empty — Nous Portal contract v1 issues no refresh tokens
    so a ``Session.refresh_token == ""`` from the provider means we don't
    persist anything. If a future contract revision starts emitting refresh
    tokens, this helper will write the RT cookie again with no other change.
    """
    response.set_cookie(
        SESSION_AT_COOKIE, access_token,
        max_age=access_token_expires_in,
        **_common_attrs(use_https),
    )
    # Contract v1: empty refresh token means "don't persist RT cookie".
    # Keeping a literal empty-value cookie around would be dead state at
    # best, attack surface at worst.
    if refresh_token:
        response.set_cookie(
            SESSION_RT_COOKIE, refresh_token,
            max_age=_RT_MAX_AGE,
            **_common_attrs(use_https),
        )


def clear_session_cookies(response: Response) -> None:
    """Emit Max-Age=0 deletions for both session cookies."""
    # Path must match the set-path for the delete to apply.
    response.set_cookie(
        SESSION_AT_COOKIE, "", max_age=0,
        path="/", httponly=True, samesite="lax",
    )
    response.set_cookie(
        SESSION_RT_COOKIE, "", max_age=0,
        path="/", httponly=True, samesite="lax",
    )


def set_pkce_cookie(response: Response, *, payload: str, use_https: bool) -> None:
    response.set_cookie(
        PKCE_COOKIE, payload,
        max_age=_PKCE_MAX_AGE,
        **_common_attrs(use_https),
    )


def clear_pkce_cookie(response: Response) -> None:
    response.set_cookie(
        PKCE_COOKIE, "", max_age=0,
        path="/", httponly=True, samesite="lax",
    )


def read_session_cookies(request: Request) -> Tuple[Optional[str], Optional[str]]:
    """Returns (access_token, refresh_token), either may be None."""
    at = request.cookies.get(SESSION_AT_COOKIE)
    rt = request.cookies.get(SESSION_RT_COOKIE)
    return at, rt


def read_pkce_cookie(request: Request) -> Optional[str]:
    return request.cookies.get(PKCE_COOKIE)


def detect_https(request: Request) -> bool:
    """Decide whether to set the ``Secure`` cookie flag.

    Reads ``request.url.scheme`` — under uvicorn's ``proxy_headers=True``
    (which start_server enables when the gate is active), this honours
    ``X-Forwarded-Proto`` from Fly's TLS terminator. Loopback traffic is
    always HTTP so this returns False there.
    """
    return request.url.scheme == "https"
