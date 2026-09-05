"""Cookie helpers for dashboard auth.

All HttpOnly, ``SameSite=Lax`` unless noted, Path = proxy prefix or /: ``hermes_session_at``
(access token; Max-Age = token TTL), ``hermes_session_rt`` (rotating refresh token; written only
when the provider returned one, always cleared on logout/expiry), ``hermes_session_provider``
(non-secret routing hint so an RT is not handed to the wrong provider), ``hermes_session_pkce``
(PKCE state + CSRF nonce + provider hint, 10 min; ``SameSite=None; Secure`` over HTTPS because it
is set on the /auth/login 302 and must survive the cross-site redirect chain — Chromium drops Lax
cookies set on such a 302, crbug 40508226), ``hermes_sso_attempt`` (auto-SSO loop guard, 60 s).
``Secure`` only when ``request.url.scheme`` is https. Cookie-prefix hardening per
draft-west-cookie-prefixes: bare name over HTTP; ``__Host-`` on gated HTTPS with Path=/;
``__Secure-`` behind a proxy prefix (``__Host-`` forbids Path != /). Setters and readers BOTH
resolve the name via :func:`_resolved_name` — a mismatch silently breaks sessions.
"""
from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Literal, Optional, Tuple
from urllib.parse import unquote

from fastapi import Request
from fastapi.responses import Response

SESSION_AT_COOKIE = "hermes_session_at"
SESSION_RT_COOKIE = "hermes_session_rt"
SESSION_PROVIDER_COOKIE = "hermes_session_provider"
PKCE_COOKIE = "hermes_session_pkce"
SSO_ATTEMPT_COOKIE = "hermes_sso_attempt"

# Name variants a reader may have to try; most strict first.
_NAME_VARIANTS = ("__Host-", "__Secure-", "")

# RT cookie lifetime is a generous browser-side upper bound; the provider's own RT TTL is the
# real authority (an expired RT -> RefreshExpiredError -> re-login).
_RT_MAX_AGE = 30 * 24 * 60 * 60
_PKCE_MAX_AGE = 10 * 60
# Long enough for one portal round trip / back-button; short enough that a user returning later
# gets a fresh silent attempt rather than a stuck /login.
_SSO_ATTEMPT_MAX_AGE = 60
# Cheap pre-filter: legacy wire forms always contain ``%`` or ``;`` (outside base64url).
_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+={0,2}$")


def _resolved_name(bare: str, *, use_https: bool, prefix: str) -> str:
    """Cookie-prefix variant for the request shape (see module docstring)."""
    if not use_https:
        return bare
    return f"__Secure-{bare}" if prefix else f"__Host-{bare}"


def _cookie_path(prefix: str) -> str:
    """``Path=/hermes`` under a proxy prefix (no leak to sibling apps), else ``/``."""
    return prefix if prefix else "/"


def _common_attrs(*, use_https: bool, prefix: str) -> dict:
    attrs: dict = {"httponly": True, "samesite": "lax", "path": _cookie_path(prefix)}
    if use_https:
        attrs["secure"] = True
    return attrs


def _pkce_attrs(*, use_https: bool, prefix: str) -> dict:
    """Attributes shared by the PKCE set AND clear paths (a shape mismatch
    means the browser silently keeps the stale cookie)."""
    attrs = _common_attrs(use_https=use_https, prefix=prefix)
    if use_https:
        attrs["samesite"] = "none"
    return attrs


def _set(response: Response, bare: str, value: str, *, max_age: int,
         use_https: bool, prefix: str, attrs: dict | None = None) -> None:
    response.set_cookie(
        _resolved_name(bare, use_https=use_https, prefix=prefix), value, max_age=max_age,
        **(attrs if attrs is not None else _common_attrs(use_https=use_https, prefix=prefix)))


def set_session_provider_cookie(
    response: Response, *, provider: str, use_https: bool, prefix: str = "") -> None:
    """Persist the non-secret provider routing hint for token refresh."""
    if provider:
        _set(response, SESSION_PROVIDER_COOKIE, provider, max_age=_RT_MAX_AGE,
             use_https=use_https, prefix=prefix)


def set_session_cookies(
    response: Response, *, access_token: str, refresh_token: str, access_token_expires_in: int,
    use_https: bool, prefix: str = "", provider: str = "") -> None:
    """``access_token_expires_in`` is seconds (the provider's reported TTL). An empty
    ``refresh_token`` means "don't persist the RT cookie" — a literal empty cookie would be dead
    state at best, attack surface at worst.

    Nous Portal issues a 24h rotating refresh token (hermes #37247); a provider that omits it returns
    ``Session.refresh_token == ""`` and we simply don't persist the RT cookie — the session then behaves as
    access-token-only until the AT expires. No other branch changes between the two cases.
    """
    _set(response, SESSION_AT_COOKIE, access_token, max_age=access_token_expires_in,
         use_https=use_https, prefix=prefix)
    if refresh_token:
        _set(response, SESSION_RT_COOKIE, refresh_token, max_age=_RT_MAX_AGE,
             use_https=use_https, prefix=prefix)
    set_session_provider_cookie(response, provider=provider, use_https=use_https, prefix=prefix)


def _clear_cookie_variants(
    response: Response, bare_name: str, *, prefix: str,
    https_samesite: Literal["lax", "strict", "none"], bare_attrs: dict) -> None:
    """Emit Max-Age=0 deletions for every plausible name variant (the setting request's shape is
    unknown). Prefixed names are rejected by the browser unless they carry ``Secure`` (``__Host-``
    additionally ``Path=/``), so those deletions always do; the bare deletion mirrors the setter's
    shape (``bare_attrs``), which works on both HTTP and HTTPS origins."""
    for variant, path in (("__Host-", "/"), ("__Secure-", _cookie_path(prefix))):
        response.set_cookie(
            f"{variant}{bare_name}", "", max_age=0, path=path, httponly=True,
            samesite=https_samesite, secure=True)
    response.set_cookie(bare_name, "", max_age=0, **bare_attrs)


def clear_session_cookies(response: Response, *, prefix: str = "") -> None:
    """Delete the AT, RT and provider cookies (every name variant, active path)."""
    bare_attrs = _common_attrs(use_https=False, prefix=prefix)
    for name in (SESSION_AT_COOKIE, SESSION_RT_COOKIE, SESSION_PROVIDER_COOKIE):
        _clear_cookie_variants(
            response, name, prefix=prefix, https_samesite="lax", bare_attrs=bare_attrs)


def encode_pkce_payload(parts: dict[str, str]) -> str:
    """Wire value ``base64url(JSON)``, no padding. The urlsafe alphabet is a strict subset of RFC
    6265 cookie-octets, so http.cookies never quotes it (strict proxies such as Go net/http reject
    the quoted form) and no value can collide with a delimiter; padding ``=`` would trigger
    quoting, the parser restores it."""
    raw = json.dumps(parts, separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def set_pkce_cookie(
    response: Response, *, payload: dict[str, str], use_https: bool, prefix: str = "") -> None:
    """``payload`` is the segment dict; see module docstring for the SameSite=None rationale."""
    _set(response, PKCE_COOKIE, encode_pkce_payload(payload), max_age=_PKCE_MAX_AGE,
         use_https=use_https, prefix=prefix, attrs=_pkce_attrs(use_https=use_https, prefix=prefix))


def clear_pkce_cookie(response: Response, *, use_https: bool, prefix: str = "") -> None:
    """Delete every PKCE cookie variant (prefixed ones carry ``Secure; SameSite=None``)."""
    _clear_cookie_variants(
        response, PKCE_COOKIE, prefix=prefix, https_samesite="none",
        bare_attrs=_pkce_attrs(use_https=use_https, prefix=prefix))


def _read_with_fallback(request: Request, bare_name: str) -> Optional[str]:
    """Try every prefix variant (the reading request may not match the setting request's shape)."""
    return next((v for v in (request.cookies.get(f"{p}{bare_name}") for p in _NAME_VARIANTS)
                 if v is not None), None)


def read_session_cookies(request: Request) -> Tuple[Optional[str], Optional[str]]:
    """Returns (access_token, refresh_token), either may be None."""
    return (
        _read_with_fallback(request, SESSION_AT_COOKIE),
        _read_with_fallback(request, SESSION_RT_COOKIE))


def read_session_provider(request: Request) -> Optional[str]:
    """Return the provider routing hint associated with the session cookies."""
    return _read_with_fallback(request, SESSION_PROVIDER_COOKIE)


def read_pkce_cookie(request: Request) -> Optional[str]:
    return _read_with_fallback(request, PKCE_COOKIE)


def parse_pkce_payload(raw: str) -> dict[str, str]:
    """Inverse of :func:`encode_pkce_payload`. EVERY reader must go through this — reading the raw
    wire value parses zero segments and silently disables the check it feeds (provider dispatch,
    CSRF state, broker binding). Compatibility ladder for cookies minted by an older server
    mid-upgrade: 1. base64url(JSON); 2. flat form with raw ``;`` delimiters, split WITHOUT
    unquoting (the ``next`` segment carries its own URL-encoding); 3. URL-encoded flat form,
    unquote once then split. A NEW cookie hitting an OLD server fails the state check.

    1. **base64url(JSON)** (current): the wire value is pure urlsafe base64 that decodes to a JSON object.
    Legacy forms can never match — they always contain ``%`` (URL-encoded, #99176) or a raw ``;`` (oldest
    flat form), both outside the base64url alphabet. Split as-is WITHOUT unquoting the payload — the
    ``next`` segment carries its own single URL-encoding, and unquoting here would turn a ``%3B`` inside it
    into a bogus delimiter and truncate the post-login target. Neither newer format can contain a raw ``;``.
    """
    if _B64URL_RE.match(raw):
        try:
            padded = raw + "=" * (-len(raw) % 4)
            decoded = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        except (binascii.Error, ValueError, UnicodeDecodeError):
            decoded = None
        if isinstance(decoded, dict):
            return {str(k): str(v) for k, v in decoded.items()}
    flat = raw if ";" in raw else unquote(raw)
    return dict(seg.split("=", 1) for seg in flat.split(";") if "=" in seg)


def set_sso_attempt_cookie(response: Response, *, use_https: bool, prefix: str = "") -> None:
    """Set the auto-SSO loop-guard marker; only its presence matters."""
    _set(response, SSO_ATTEMPT_COOKIE, "1", max_age=_SSO_ATTEMPT_MAX_AGE,
         use_https=use_https, prefix=prefix)


def read_sso_attempt_cookie(request: Request) -> Optional[str]:
    """Return the auto-SSO marker value if present (any variant), else None."""
    return _read_with_fallback(request, SSO_ATTEMPT_COOKIE)


def clear_sso_attempt_cookie(response: Response, *, prefix: str = "") -> None:
    """Delete the auto-SSO marker (every variant) so it never suppresses a later silent attempt."""
    _clear_cookie_variants(
        response, SSO_ATTEMPT_COOKIE, prefix=prefix, https_samesite="lax",
        bare_attrs=_common_attrs(use_https=False, prefix=prefix))


def detect_https(request: Request) -> bool:
    """``Secure`` flag decision (honours ``X-Forwarded-Proto`` under uvicorn ``proxy_headers``)."""
    return request.url.scheme == "https"
