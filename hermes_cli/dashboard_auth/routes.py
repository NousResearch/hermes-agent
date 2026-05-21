"""HTTP routes for the dashboard-auth OAuth round trip.

Mounted at root (no prefix) by ``web_server.py``. The router does not
auto-gate; gating is performed by ``gated_auth_middleware``, which
allowlists everything under ``/auth/*`` and ``/api/auth/providers``.

The routes:

  GET  /login              → server-rendered login page
  GET  /auth/login?provider=N → 302 to IDP, sets PKCE cookie
  GET  /auth/callback?code,state → completes login, sets session cookies
  POST /auth/logout        → clears cookies, best-effort revoke
  GET  /api/auth/providers → list registered providers (login bootstrap)
  GET  /api/auth/me        → current Session as JSON (auth-required)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from hermes_cli.dashboard_auth import (
    get_provider,
    list_providers,
)
from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_cli.dashboard_auth.base import (
    InvalidCodeError,
    ProviderError,
)
from hermes_cli.dashboard_auth.cookies import (
    clear_pkce_cookie,
    clear_session_cookies,
    detect_https,
    read_pkce_cookie,
    read_session_cookies,
    set_pkce_cookie,
    set_session_cookies,
)
from hermes_cli.dashboard_auth.login_page import render_login_html

_log = logging.getLogger(__name__)

router = APIRouter()


def _redirect_uri(request: Request) -> str:
    """Reconstruct the absolute callback URL the IDP redirects back to.

    Reads from the request URL — under uvicorn's ``proxy_headers=True``
    this picks up the public https URL from ``X-Forwarded-Host`` plus
    ``X-Forwarded-Proto``.
    """
    return str(request.url_for("auth_callback"))


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


# ---------------------------------------------------------------------------
# Public: login page (server-rendered HTML, no SPA bundle)
# ---------------------------------------------------------------------------


@router.get("/login", name="login_page")
async def login_page(request: Request) -> HTMLResponse:
    return HTMLResponse(
        render_login_html(),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


# ---------------------------------------------------------------------------
# Public: provider list for the login-page bootstrap
# ---------------------------------------------------------------------------


@router.get("/api/auth/providers", name="auth_providers")
async def api_auth_providers() -> Any:
    providers = list_providers()
    if not providers:
        # Q13: fail-closed when zero providers are registered.
        return JSONResponse(
            {"detail": "no auth providers registered"},
            status_code=503,
        )
    return {
        "providers": [
            {"name": p.name, "display_name": p.display_name}
            for p in providers
        ],
    }


# ---------------------------------------------------------------------------
# Public: OAuth round trip
# ---------------------------------------------------------------------------


@router.get("/auth/login", name="auth_login")
async def auth_login(request: Request, provider: str):
    p = get_provider(provider)
    if p is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown provider: {provider!r}",
        )

    try:
        ls = p.start_login(redirect_uri=_redirect_uri(request))
    except ProviderError as e:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=provider,
            reason="provider_unreachable",
            ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=503,
            detail=f"Provider unreachable: {e}",
        )

    audit_log(
        AuditEvent.LOGIN_START,
        provider=provider,
        ip=_client_ip(request),
    )

    resp = RedirectResponse(url=ls.redirect_url, status_code=302)
    # Pack the provider name into the PKCE cookie so the callback can
    # find it without a separate cookie. Provider may or may not have
    # already included a ``provider=`` segment.
    pkce = ls.cookie_payload.get("hermes_session_pkce", "")
    if "provider=" not in pkce:
        pkce = f"provider={provider};{pkce}" if pkce else f"provider={provider}"
    set_pkce_cookie(resp, payload=pkce, use_https=detect_https(request))
    return resp


@router.get("/auth/callback", name="auth_callback")
async def auth_callback(
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
    error_description: str = "",
    next: str = "",
):
    pkce_raw = read_pkce_cookie(request)
    if not pkce_raw:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            reason="missing_pkce_cookie",
            ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=400,
            detail="Missing PKCE state cookie",
        )

    # Parse ``provider=...;state=...;verifier=...``
    parts = dict(
        seg.split("=", 1) for seg in pkce_raw.split(";") if "=" in seg
    )
    provider_name = parts.get("provider", "")
    expected_state = parts.get("state", "")
    verifier = parts.get("verifier", "")

    p = get_provider(provider_name)
    if p is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider in cookie: {provider_name!r}",
        )

    if error:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=provider_name,
            reason="idp_error",
            error=error,
            ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=400,
            detail=f"OAuth error from provider: {error} ({error_description})",
        )

    if not state or state != expected_state:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=provider_name,
            reason="state_mismatch",
            ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=400,
            detail="OAuth state mismatch (CSRF check failed)",
        )

    try:
        session = p.complete_login(
            code=code,
            state=state,
            code_verifier=verifier,
            redirect_uri=_redirect_uri(request),
        )
    except InvalidCodeError as e:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=provider_name,
            reason="invalid_code",
            ip=_client_ip(request),
        )
        raise HTTPException(status_code=400, detail=f"Invalid code: {e}")
    except ProviderError as e:
        audit_log(
            AuditEvent.LOGIN_FAILURE,
            provider=provider_name,
            reason="provider_unreachable",
            ip=_client_ip(request),
        )
        raise HTTPException(
            status_code=503,
            detail=f"Provider unreachable: {e}",
        )

    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider=provider_name,
        user_id=session.user_id,
        email=session.email,
        org_id=session.org_id,
        ip=_client_ip(request),
    )

    expires_in = max(60, session.expires_at - int(time.time()))
    # Honour the ``next=`` query param the gate's _unauth_response set in
    # the redirect URL. Validated against the same same-origin rules as
    # the gate's _safe_next_target — any absolute URL / protocol-relative
    # path / loop back to /login is dropped in favour of ``/``.
    landing = _validate_post_login_target(next) or "/"
    resp = RedirectResponse(url=landing, status_code=302)
    set_session_cookies(
        resp,
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        access_token_expires_in=expires_in,
        use_https=detect_https(request),
    )
    clear_pkce_cookie(resp)
    return resp


def _validate_post_login_target(raw: str) -> str:
    """Return ``raw`` if it's a safe same-origin path, else empty string.

    The ``next`` query param survives a full OAuth round trip — the gate
    encodes it into the /login redirect, the login page emits it back into
    /auth/login, and the IDP preserves it across /authorize/callback. We
    have to re-validate here because the value came back in via the
    URL (an attacker could craft a /auth/callback URL with their own
    ``next=https://evil.example``).
    """
    if not raw:
        return ""
    from urllib.parse import unquote
    decoded = unquote(raw)
    if not decoded.startswith("/") or decoded.startswith("//"):
        return ""
    # Don't loop back to login pages or auth flow.
    if any(
        decoded == p or decoded.startswith(p)
        for p in ("/login", "/auth/", "/api/auth/")
    ):
        return ""
    return decoded


@router.post("/auth/logout", name="auth_logout")
async def auth_logout(request: Request):
    _at, rt = read_session_cookies(request)
    if rt:
        # Best-effort revoke. Try every provider so a session minted by
        # any registered provider is revoked correctly. Failures are
        # logged but never raised.
        for provider in list_providers():
            try:
                provider.revoke_session(refresh_token=rt)
            except Exception as e:  # noqa: BLE001 — best-effort
                _log.warning(
                    "dashboard-auth: revoke on %r failed: %s",
                    provider.name, e,
                )

    sess = getattr(request.state, "session", None)
    audit_log(
        AuditEvent.LOGOUT,
        provider=(sess.provider if sess else "unknown"),
        user_id=(sess.user_id if sess else ""),
        ip=_client_ip(request),
    )

    resp = RedirectResponse(url="/login", status_code=302)
    clear_session_cookies(resp)
    clear_pkce_cookie(resp)
    return resp


# ---------------------------------------------------------------------------
# Auth-required: identity probe for the SPA
# ---------------------------------------------------------------------------


@router.get("/api/auth/me", name="auth_me")
async def api_auth_me(request: Request):
    """Return the verified session as JSON. Auth-required (gate enforces)."""
    sess = getattr(request.state, "session", None)
    if sess is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {
        "user_id": sess.user_id,
        "email": sess.email,
        "display_name": sess.display_name,
        "org_id": sess.org_id,
        "provider": sess.provider,
        "expires_at": sess.expires_at,
    }


# ---------------------------------------------------------------------------
# Auth-required: WS upgrade ticket (Phase 5)
# ---------------------------------------------------------------------------


@router.post("/api/auth/ws-ticket", name="auth_ws_ticket")
async def api_auth_ws_ticket(request: Request):
    """Mint a short-lived single-use ticket for the authenticated session.

    Browsers cannot set ``Authorization`` on a WebSocket upgrade, so in
    gated mode the SPA POSTs this endpoint to get a ``?ticket=`` value to
    append to ``/api/pty``, ``/api/ws``, ``/api/pub``, or ``/api/events``.

    The ticket has a 30-second TTL and is single-use. Calling this endpoint
    multiple times in quick succession (e.g. one ticket per WS) is the
    expected pattern.
    """
    sess = getattr(request.state, "session", None)
    if sess is None:
        # Middleware should already have rejected, but check defensively.
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Import here so the routes module stays usable in test contexts that
    # don't load the ticket store.
    from hermes_cli.dashboard_auth.ws_tickets import TTL_SECONDS, mint_ticket

    ticket = mint_ticket(user_id=sess.user_id, provider=sess.provider)
    audit_log(
        AuditEvent.WS_TICKET_MINTED,
        provider=sess.provider,
        user_id=sess.user_id,
        ip=_client_ip(request),
    )
    return {"ticket": ticket, "ttl_seconds": TTL_SECONDS}
