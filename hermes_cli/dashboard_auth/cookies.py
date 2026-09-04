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

The two session cookies are ``SameSite=Lax`` and live under the prefix's
Path. The PKCE cookie is the exception: ``SameSite=None`` over HTTPS,
falling back to ``Lax`` on plain HTTP (where ``SameSite=None`` is invalid
without ``Secure``). It is set on the ``/auth/login`` 302 and must survive
the cross-site redirect chain out to the IDP and back to
``/auth/callback``; Chromium intermittently drops ``Lax`` cookies set on a
302 in such a chain (crbug 40508226), which surfaces as "Missing PKCE
state cookie". ``Secure`` is set ONLY when the dashboard was reached over
HTTPS — detected via the request URL scheme, which honours
``X-Forwarded-Proto`` upstream of Fly's TLS terminator when uvicorn is
configured with ``proxy_headers=True``. Loopback dev traffic is always
HTTP so ``Secure`` would lock the cookies out of the browser.

NOTE: uvicorn only honours ``X-Forwarded-Proto`` from a peer inside its
``forwarded_allow_ips`` (default: ``127.0.0.1``). A TLS terminator that
reaches the dashboard from a non-loopback address — e.g. a reverse proxy
in its own container — is not trusted, so the request still looks like
HTTP here and these cookies are written in their HTTP shape.

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


def _clear_cookie_variants(
    response: Response,
    bare_name: str,
    *,
    prefix: str,
    https_samesite: Literal["lax", "strict", "none"],
    bare_attrs: dict,
) -> None:
    """Emit Max-Age=0 deletions for every plausible name variant of a cookie.

    Cookie-prefix rules make the deletion shape load-bearing: a Set-Cookie
    for a ``__Host-``/``__Secure-`` name is rejected outright by the
    browser unless it carries ``Secure`` (and ``__Host-`` additionally
    requires ``Path=/``), so those deletions always carry the attributes
    their name demands. The bare-name deletion mirrors the shape the
    setter uses (``bare_attrs``) — under RFC 6265bis a deletion sent from
    a secure origin may omit ``Secure`` and still delete a Secure cookie,
    while a ``Secure`` deletion on a plain-HTTP origin can be ignored, so
    matching the setter is the shape that works on both origins.
    """
    for variant in _NAME_VARIANTS:
        if variant == "__Host-":
            # __Host- demands Secure AND Path=/ or the header is invalid.
            response.set_cookie(
                f"{variant}{bare_name}", "", max_age=0,
                path="/", httponly=True, samesite=https_samesite,
                secure=True,
            )
        elif variant == "__Secure-":
            response.set_cookie(
                f"{variant}{bare_name}", "", max_age=0,
                path=_cookie_path(prefix), httponly=True,
                samesite=https_samesite, secure=True,
            )
        else:
            response.set_cookie(
                bare_name, "", max_age=0, **bare_attrs,
            )


def clear_session_cookies(response: Response, *, prefix: str = "") -> None:
    """Delete the AT, RT and provider cookies (every name variant, active path)."""
    bare_attrs = _common_attrs(use_https=False, prefix=prefix)
    for name in (SESSION_AT_COOKIE, SESSION_RT_COOKIE, SESSION_PROVIDER_COOKIE):
        _clear_cookie_variants(
            response, name, prefix=prefix, https_samesite="lax", bare_attrs=bare_attrs)

    To delete a cookie reliably the deletion's ``Path`` must match the
    set path AND the cookie name must match the variant the setter used.
    We don't know which variant was originally set (cookie prefix
    depends on the request that set it), so we emit deletions for every
    plausible variant under the active path.
    """
    bare_attrs = {
        "path": _cookie_path(prefix), "httponly": True, "samesite": "lax",
    }
    for name in (SESSION_AT_COOKIE, SESSION_RT_COOKIE, SESSION_PROVIDER_COOKIE):
        _clear_cookie_variants(
            response, name,
            prefix=prefix, https_samesite="lax", bare_attrs=bare_attrs,
        )


def _pkce_attrs(*, use_https: bool, prefix: str) -> dict:
    """Cookie attributes for the PKCE cookie's set AND clear paths.

    Single source of truth so a deletion always matches the shape the
    setter emitted for the same origin — a shape mismatch means the
    browser silently keeps the stale cookie.
    """
    attrs = _common_attrs(use_https=use_https, prefix=prefix)
    if use_https:
        attrs["samesite"] = "none"
    return attrs


def encode_pkce_payload(parts: dict[str, str]) -> str:
    """Serialise PKCE segments to the wire value: ``base64url(JSON)``.

    The urlsafe base64 alphabet (``A-Za-z0-9-_``, padding stripped) is a
    strict subset of the RFC 6265 cookie-octet set — no ``;`` (attribute
    terminator), no ``"`` and no ``\\`` (the chars that make Python's
    http.cookies emit the quoted ``\\073`` form, which strict cookie-aware
    proxy hops such as Go's net/http reject outright). The ``=`` padding
    is stripped because http.cookies treats ``=`` as outside its legal
    unquoted set and would re-wrap the value in the quoted form this
    codec exists to avoid; the parser restores the padding. JSON carries
    the segments, so no delimiter can ever collide with segment values —
    the delimiter/quoting bug class this codec replaces (see
    :func:`parse_pkce_payload` for the two legacy formats it superseded).
    """
    raw = json.dumps(parts, separators=(",", ":"), sort_keys=True)
    return (
        base64.urlsafe_b64encode(raw.encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )


def set_pkce_cookie(
    response: Response,
    *,
    payload: dict[str, str],
    use_https: bool,
    prefix: str = "",
) -> None:
    # SameSite=None when HTTPS: the PKCE cookie is set on the /auth/login
    # 302 response (redirecting to the IDP) and must survive the cross-site
    # redirect chain (same-site → IDP → same-site callback). Chromium has a
    # long-standing bug (crbug 40508226) where SameSite=Lax cookies set on a
    # 302 in a cross-site redirect chain are intermittently dropped, causing
    # "Missing PKCE state cookie" on the callback. SameSite=None + Secure
    # sidesteps the bug — these cookies are explicitly designed for cross-site
    # delivery and Chromium processes them reliably during redirects.
    # Loopback HTTP degrades to Lax (SameSite=None requires Secure).
    #
    # Value encoding: ``payload`` is the segment dict
    # (``{"provider": …, "state": …, "verifier": …, "next": …}``) and goes
    # on the wire as base64url(JSON) via encode_pkce_payload() — plain
    # RFC 6265 cookie-octets end to end, so every cookie-aware hop
    # (browsers, Go net/http proxies, Python parsers) passes the value
    # through untouched. Readers decode via parse_pkce_payload(), which
    # also keeps a compatibility ladder for cookies minted by the two
    # earlier wire formats during a rolling upgrade.
    response.set_cookie(
        _resolved_name(PKCE_COOKIE, use_https=use_https, prefix=prefix),
        encode_pkce_payload(payload),
        max_age=_PKCE_MAX_AGE,
        **_pkce_attrs(use_https=use_https, prefix=prefix),
    )


def clear_pkce_cookie(
    response: Response, *, use_https: bool, prefix: str = "",
) -> None:
    """Emit Max-Age=0 deletions for every plausible PKCE cookie variant.

    A deletion is only honoured when its shape is acceptable to the
    browser on the current origin: a ``Secure`` deletion can be dropped
    on a plain-HTTP origin, while the ``__Host-``/``__Secure-`` name
    variants REQUIRE ``Secure`` to be valid at all. So the bare-name
    deletion mirrors the setter's shape for the active origin (Lax
    without ``Secure`` over HTTP; ``SameSite=None; Secure`` over HTTPS,
    matching :func:`set_pkce_cookie`), and the prefixed variants — which
    can only ever have been set on an HTTPS origin — always carry
    ``Secure; SameSite=None``.
    """
    _clear_cookie_variants(
        response, PKCE_COOKIE,
        prefix=prefix, https_samesite="none",
        bare_attrs=_pkce_attrs(use_https=use_https, prefix=prefix),
    )


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


# base64url wire values are exactly the urlsafe alphabet (padding is
# stripped by the encoder; the decoder restores it). Used as a cheap
# pre-filter before attempting the JSON decode so legacy wire forms
# (which always contain ``%`` or ``;``) never even reach the base64
# decoder.
_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+={0,2}$")


def parse_pkce_payload(raw: str) -> dict[str, str]:
    """Decode + parse a PKCE cookie value into its segment dict.

    Single inverse of :func:`set_pkce_cookie` /
    :func:`encode_pkce_payload`. EVERY reader of the PKCE cookie must go
    through this helper — a reader that interprets the raw wire value
    itself parses zero segments and silently disables whatever check it
    was feeding (provider dispatch, CSRF state, native-flow broker
    binding).

    Compatibility ladder — the PKCE cookie has a 10-minute TTL and is
    opaque + server-set, so during a rolling upgrade a cookie minted by
    one server version can arrive at another. Three formats, tried in
    order; each rung is unambiguous:

    1. **base64url(JSON)** (current): the wire value is pure urlsafe
       base64 that decodes to a JSON object. Legacy forms can never
       match — they always contain ``%`` (URL-encoded, #99176) or a raw
       ``;`` (oldest flat form), both outside the base64url alphabet.
    2. **Oldest flat form** (pre-#99176): raw ``;`` between segments
       (``provider=…;state=…;verifier=…``). Split as-is WITHOUT
       unquoting the payload — the ``next`` segment carries its own
       single URL-encoding, and unquoting here would turn a ``%3B``
       inside it into a bogus delimiter and truncate the post-login
       target. Neither newer format can contain a raw ``;``.
    3. **URL-encoded flat form** (#99176): the whole flat payload passed
       through ``quote(payload, safe="")`` — no raw ``;`` possible
       (it is ``%3B``); unquote once, then split.

    Rollout directions: OLD cookie → NEW server is handled here (rungs
    2 and 3 parse both legacy forms correctly). NEW cookie → OLD server
    (a rollback, or a mixed fleet routing the callback to a not-yet-
    upgraded instance) fails the OAuth state check — the old reader
    can't find a ``state`` segment in the base64url blob — and the user
    simply retries login against the now-consistent fleet; no data loss,
    nothing minted.
    """
    if _B64URL_RE.match(raw):
        try:
            padded = raw + "=" * (-len(raw) % 4)
            decoded = json.loads(
                base64.urlsafe_b64decode(padded.encode("ascii"))
            )
        except (binascii.Error, ValueError, UnicodeDecodeError):
            decoded = None
        if isinstance(decoded, dict):
            return {str(k): str(v) for k, v in decoded.items()}
    if ";" in raw:
        # Oldest flat form: already flat, split as-is (no unquote).
        return dict(
            seg.split("=", 1) for seg in raw.split(";") if "=" in seg
        )
    # #99176 URL-encoded flat form: unquote once, then split.
    return dict(
        seg.split("=", 1) for seg in unquote(raw).split(";") if "=" in seg
    )


def set_sso_attempt_cookie(
    response: Response, *, use_https: bool, prefix: str = "",
) -> None:
    """Set the one-shot auto-SSO loop-guard marker (Phase 1).

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
    """Emit Max-Age=0 deletions for the auto-SSO marker, every name variant.

    Called on a successful callback and whenever the gate falls back to
    /login, so the marker never lingers to suppress a later silent attempt.
    """
    _clear_cookie_variants(
        response, SSO_ATTEMPT_COOKIE,
        prefix=prefix, https_samesite="lax",
        bare_attrs={
            "path": _cookie_path(prefix), "httponly": True, "samesite": "lax",
        },
    )


def detect_https(request: Request) -> bool:
    """``Secure`` flag decision (honours ``X-Forwarded-Proto`` under uvicorn ``proxy_headers``)."""
    return request.url.scheme == "https"
