"""Gateway relay routes (``hermes_cli.webapp_relay``): pairing and grants, on both sides.

Spoke (another gateway asks to relay here):
  ``GET/POST /webapp/relay/authorize``  the consent page a signed-in user approves or denies
  ``POST /api/webapp/relay/token``      the hub's code exchange (public: PKCE verifier + key proof)
  ``GET /api/webapp/relay/grants``      the grants the signed-in user approved
  ``DELETE /api/webapp/relay/grants/<id>``  revoke one (that user, or the grant itself)

Hub (this gateway relays to others):
  ``GET/POST /api/webapp/relay/gateways``    list, or start pairing with one
  ``GET /webapp/relay/callback``             where the spoke returns the browser with a code
  ``DELETE /api/webapp/relay/gateways/<id>`` forget one and revoke its grant there
  ``POST /api/webapp/relay/gateways/<id>/test``  a signed read through the relay

Every route 404s unless the relay is on (``webapp_relay.relay_enabled``).
"""
from __future__ import annotations

import html
import logging
import secrets
import urllib.parse

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from hermes_cli import webapp_relay as relay
from hermes_cli import webapp_relay_gateways as hub
from hermes_cli import webapp_relay_grants as grants
from hermes_cli.dashboard_auth.registry import get_provider

_log = logging.getLogger(__name__)
router = APIRouter()

_NO_STORE = {"Cache-Control": "no-store"}


def _require_relay(request: Request) -> None:
    if not relay.relay_enabled(request.app.state):
        raise HTTPException(404, "Not Found")


def _identity(request: Request) -> dict[str, str]:
    session = getattr(request.state, "session", None)
    if session is None:
        raise HTTPException(401, "Unauthorized")
    return grants.identity_of(session)


def _own_login(request: Request) -> dict[str, str]:
    """The signed-in user of THIS gateway: a relayed request never manages the relay."""
    if getattr(request.state, "relay_grant", None) is not None:
        raise HTTPException(403, "Not available through a gateway relay")
    return _identity(request)


def _home(request: Request):
    return request.app.state.webapp_home


def _public_base(request: Request) -> str:
    """This listener's address as browsers reach it: ``webapp.public_url`` when set, else the
    request's own origin, plus any reverse-proxy prefix."""
    from hermes_cli.dashboard_auth.prefix import prefix_from_request, resolve_public_url
    own = resolve_public_url(request.app.state)
    if own:
        return own
    return f"{request.url.scheme}://{request.url.netloc}{prefix_from_request(request)}"


# ---- Spoke: consent ---------------------------------------------------------------------------

_CONSENT_STYLE = """
body{font:15px/1.5 system-ui,sans-serif;max-width:34rem;margin:3rem auto;padding:0 1.25rem;color:#1d1d1f}
h1{font-size:1.35rem;margin:0 0 .75rem}code{font-size:.9em;word-break:break-all}
.card{border:1px solid #d2d2d7;border-radius:12px;padding:1rem 1.25rem;margin:1rem 0}
label{display:flex;gap:.6rem;align-items:flex-start;margin:.45rem 0}.muted{color:#6e6e73;font-size:.9em}
.gap{margin-top:.75rem}.error{color:#d70015}
.row{display:flex;gap:.75rem;margin-top:1.25rem}button{font:inherit;padding:.55rem 1.2rem;border-radius:9px;
border:1px solid #d2d2d7;background:#fff;cursor:pointer}button.allow{background:#0a66ff;border-color:#0a66ff;color:#fff}
@media (prefers-color-scheme:dark){body{background:#111;color:#f5f5f7}.card{border-color:#3a3a3c}
button{background:#1c1c1e;color:#f5f5f7;border-color:#3a3a3c}.muted{color:#a1a1a6}}
"""


def _consent_page(request: Request, pairing: dict, error: str = "") -> HTMLResponse:
    session = request.state.session
    nonce = secrets.token_urlsafe(16)
    esc = html.escape
    account = session.email or session.display_name or session.user_id
    provider = get_provider(session.provider)
    signed_in_with = (provider.display_name if provider else "") or session.provider
    hidden = "".join(f'<input type="hidden" name="{esc(k)}" value="{esc(str(v))}">' for k, v in (
        ("redirect_uri", pairing["redirect_uri"]), ("public_key", pairing["public_key"]),
        ("code_challenge", pairing["code_challenge"]), ("code_challenge_method", "S256"),
        ("state", pairing["state"]), ("hub_name", pairing["hub_name"]),
        ("scope", " ".join(pairing["scopes"]))))
    scopes = "".join(
        f'<label><input type="checkbox" name="grant_scope" value="{esc(s)}" checked> {esc(grants.SCOPES[s])}</label>'
        for s in pairing["scopes"])
    body = (
        f"<h1>Allow {esc(pairing['hub_name'])} to use this gateway?</h1>"
        f"<p><strong>{esc(pairing['hub_name'])}</strong> at <code>{esc(pairing['hub_origin'])}</code> wants to "
        "relay your Webapp to this gateway, so you can open its profiles, chats and settings from there.</p>"
        # One form around the scopes and the buttons: a checkbox outside it is never submitted.
        f'<form method="post">{hidden}'
        f'<div class="card"><div class="muted">It will act as this account, signed in with {esc(signed_in_with)}</div>'
        f"<code>{esc(account)}</code>"
        '<div class="muted gap">Always included: profiles, sessions, chat, settings, '
        "scheduled jobs and skills.</div>"
        f"{scopes}</div>"
        '<p class="muted">Anyone signed in to that gateway as this account can then run agents, tools and '
        "commands here as the user running Hermes. Revoke it any time in Settings → Gateways, or sign out "
        "every browser with <code>hermes webapp --sign-out</code>.</p>"
        + (f'<p class="error">{esc(error)}</p>' if error else "")
        + '<div class="row">'
        '<button class="allow" name="decision" value="allow" type="submit">Allow</button>'
        '<button name="decision" value="deny" type="submit">Deny</button></div></form>')
    csp = (f"default-src 'none'; style-src 'nonce-{nonce}'; form-action 'self' "
           f"{pairing['hub_origin']}; frame-ancestors 'none'; base-uri 'none'")
    return HTMLResponse(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<meta name="referrer" content="same-origin"><title>Allow gateway relay</title>'
        f'<style nonce="{nonce}">{_CONSENT_STYLE}</style>{body}',
        # same-origin, not no-referrer: under no-referrer the form posts with `Origin: null`, which the
        # sign-in gate refuses as a cross-origin write. Nothing crosses to the hub either way.
        headers={**_NO_STORE, "Referrer-Policy": "same-origin", "Content-Security-Policy": csp,
                 "X-Frame-Options": "DENY"})


def _bad_request_page(message: str) -> HTMLResponse:
    return HTMLResponse(
        f'<!doctype html><meta charset="utf-8"><title>Gateway relay</title><p>{html.escape(message)}</p>',
        status_code=400, headers={**_NO_STORE, "Content-Security-Policy": "default-src 'none'"})


@router.get("/webapp/relay/authorize")
async def relay_authorize_page(request: Request):
    _require_relay(request)
    _own_login(request)
    try:
        pairing = grants.parse_authorize(request.query_params)
    except grants.AuthorizeError as exc:
        return _bad_request_page(str(exc))
    return _consent_page(request, pairing)


@router.post("/webapp/relay/authorize")
async def relay_authorize_decision(request: Request):
    _require_relay(request)
    _own_login(request)
    form = await request.form()
    try:
        pairing = grants.parse_authorize(form)
    except grants.AuthorizeError as exc:
        return _bad_request_page(str(exc))
    if form.get("decision") != "allow":
        query = urllib.parse.urlencode({"error": "access_denied", "state": pairing["state"]})
        return RedirectResponse(f"{pairing['redirect_uri']}?{query}", status_code=303, headers=_NO_STORE)
    try:
        code = grants.issue_code(request, pairing, [str(s) for s in form.getlist("grant_scope")])
    except grants.AuthorizeError as exc:
        return _consent_page(request, pairing, str(exc))
    query = urllib.parse.urlencode({"code": code, "state": pairing["state"]})
    return RedirectResponse(f"{pairing['redirect_uri']}?{query}", status_code=303, headers=_NO_STORE)


@router.post("/api/webapp/relay/token")
async def relay_token(request: Request):
    _require_relay(request)
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        body = None
    if (not isinstance(body, dict)
            or not all(isinstance(body.get(k), str) for k in ("code", "code_verifier", "redirect_uri", "proof"))
            or not isinstance(body.get("audience"), str) or not 0 < len(body["audience"]) <= 2048):
        return JSONResponse({"error": "invalid_request"}, status_code=400, headers=_NO_STORE)
    try:
        grant = grants.exchange_code(_home(request), code=body["code"], verifier=body["code_verifier"],
                                     redirect_uri=body["redirect_uri"], audience=body["audience"],
                                     proof=body["proof"])
    except PermissionError as exc:
        return JSONResponse({"error": "invalid_grant", "detail": str(exc)}, status_code=400, headers=_NO_STORE)
    except relay.StoreUnreadable as exc:
        return JSONResponse({"error": "temporarily_unavailable", "detail": str(exc)}, status_code=503,
                            headers=_NO_STORE)
    return JSONResponse(grant, headers=_NO_STORE)


@router.get("/api/webapp/relay/grants")
async def relay_grants(request: Request):
    _require_relay(request)
    return JSONResponse({"grants": grants.list_grants(_home(request), _own_login(request))}, headers=_NO_STORE)


@router.delete("/api/webapp/relay/grants/{grant_id}")
async def relay_revoke_grant(request: Request, grant_id: str):
    _require_relay(request)
    relayed = getattr(request.state, "relay_grant", None)
    if relayed is not None:
        if relayed["id"] != grant_id:  # the gate already refuses this; kept explicit
            raise HTTPException(403, "A relay grant may revoke only itself")
    else:
        identity = _identity(request)
        if not any(row["id"] == grant_id for row in grants.list_grants(_home(request), identity)):
            raise HTTPException(404, "No such grant")
    try:
        grants.revoke(_home(request), grant_id)
    except relay.StoreUnreadable as exc:
        raise HTTPException(503, f"Nothing was revoked: {exc}") from None
    return JSONResponse({"revoked": True}, headers=_NO_STORE)


# ---- Hub: paired gateways ---------------------------------------------------------------------

def _client(request: Request):
    from hermes_cli.web_routers.webapp_relay_proxy import relay_client
    return relay_client(request.app.state)


@router.get("/api/webapp/relay/gateways")
async def relay_gateways(request: Request):
    _require_relay(request)
    rows = hub.gateways_for(_home(request), _own_login(request))
    return JSONResponse({"gateways": [hub.public_row(row) for row in rows]}, headers=_NO_STORE)


@router.post("/api/webapp/relay/gateways")
async def relay_start_pairing(request: Request):
    _require_relay(request)
    identity = _own_login(request)
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        body = None
    if not isinstance(body, dict) or not isinstance(body.get("url"), str):
        raise HTTPException(400, "Send the gateway's url")
    base = _public_base(request)
    hub_name = str(body.get("hub_name") or "") or urllib.parse.urlsplit(base).hostname or "Hermes"
    try:
        url = hub.start_pairing(_home(request), identity=identity, url=body["url"],
                                label=str(body.get("label") or ""), hub_name=hub_name,
                                callback_url=f"{base}/webapp/relay/callback")
    except hub.PairingError as exc:
        raise HTTPException(400, str(exc)) from None
    except relay.StoreUnreadable as exc:
        raise HTTPException(503, f"Nothing was started: {exc}") from None
    return JSONResponse({"authorize_url": url}, headers=_NO_STORE)


def _back_to_settings(request: Request, outcome: str, detail: str = "") -> RedirectResponse:
    from hermes_cli.dashboard_auth.prefix import prefix_from_request
    query = urllib.parse.urlencode({"tab": "gateway", "relay": outcome, **({"relay_detail": detail} if detail else {})})
    return RedirectResponse(f"{prefix_from_request(request)}/#/settings?{query}", status_code=303, headers=_NO_STORE)


@router.get("/webapp/relay/callback")
async def relay_callback(request: Request, state: str = "", code: str = "", error: str = ""):
    _require_relay(request)
    identity = _own_login(request)
    if error or not code:
        hub.cancel_pairing(state)
        return _back_to_settings(request, "denied" if error == "access_denied" else "failed",
                                 "The other gateway did not approve the pairing.")
    try:
        row = await hub.finish_pairing(_home(request), identity=identity, state=state, code=code,
                                       client=_client(request))
    except hub.PairingError as exc:
        return _back_to_settings(request, "failed", str(exc))
    return _back_to_settings(request, "connected", row["label"])


@router.delete("/api/webapp/relay/gateways/{gateway_id}")
async def relay_unpair(request: Request, gateway_id: str, forget: bool = False):
    """Remove a gateway once it confirms it revoked this hub's grant; ``?forget=1`` removes it here
    only. An unconfirmed revocation keeps the gateway and answers ``removed: false`` with why."""
    _require_relay(request)
    try:
        removed = await hub.unpair(_home(request), _own_login(request), gateway_id, _client(request), forget=forget)
    except hub.RevokeUnconfirmed as exc:
        return JSONResponse({"removed": False, "detail": str(exc)}, headers=_NO_STORE)
    except relay.StoreUnreadable as exc:
        raise HTTPException(503, f"The gateway list here was not changed: {exc}") from None
    if not removed:
        raise HTTPException(404, "No such gateway")
    from hermes_cli.web_routers.webapp_relay_proxy import close_gateway_sockets
    close_gateway_sockets(gateway_id)
    return JSONResponse({"removed": True, "revoked": not forget}, headers=_NO_STORE)


@router.post("/api/webapp/relay/gateways/{gateway_id}/test")
async def relay_test(request: Request, gateway_id: str):
    _require_relay(request)
    row = hub.gateway_for(_home(request), _own_login(request), gateway_id)
    if row is None:
        raise HTTPException(404, "No such gateway")
    from hermes_cli.web_routers.webapp_relay_proxy import signed_get
    try:
        response = await signed_get(request.app.state, row, "/api/profiles")
    except httpx.HTTPError as exc:  # transport failure: report it, the row stays
        return JSONResponse({"ok": False, "reachable": False, "error": type(exc).__name__}, headers=_NO_STORE)
    ok = response.status_code == 200
    profiles = []
    if ok:
        try:
            profiles = [str(p.get("name")) for p in response.json().get("profiles", []) if isinstance(p, dict)]
        except (ValueError, AttributeError):
            ok = False
    detail = "" if ok else (response.text[:200] if response.status_code != 401 else "The gateway revoked this relay.")
    return JSONResponse({"ok": ok, "reachable": True, "status": response.status_code, "profiles": profiles,
                         "error": detail}, headers=_NO_STORE)
