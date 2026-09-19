"""Mobile web screen takeover for the local Bot Desktop.

The invitation token is only a short-lived locator.  The page must also receive the
short code from the user's private Telegram/Discord message before it receives a
web-session cookie or a display ticket.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import secrets
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from hermes_cli.dashboard_auth.prefix import resolve_public_url

router = APIRouter()
_COOKIE = "hermes_screen_session"


def _store(profile_home: str | None = None):
    from gateway.screen_handoff import ScreenHandoffStore
    return ScreenHandoffStore(profile_home)


def _candidate_homes() -> list[str]:
    from hermes_constants import get_hermes_home
    homes = [str(get_hermes_home())]
    try:
        from hermes_cli.profiles import _get_profiles_root
        root = _get_profiles_root()
        if root.is_dir():
            homes.extend(str(p) for p in root.iterdir() if p.is_dir())
    except Exception:
        pass
    return list(dict.fromkeys(homes))


def _find(token: str, *, mark_opened: bool = False):
    for home in _candidate_homes():
        handoff = _store(home).by_token(token, mark_opened=mark_opened)
        if handoff is not None:
            return _store(home), handoff
    return None, None


def _find_session(cookie: str):
    for home in _candidate_homes():
        handoff = _store(home).web_session(cookie)
        if handoff is not None:
            return _store(home), handoff
    return None, None


def _find_any_session(cookie: str):
    for home in _candidate_homes():
        handoff = _store(home).any_web_session(cookie)
        if handoff is not None:
            return _store(home), handoff
    return None, None


def _cookie(request: Request) -> str:
    return str(request.cookies.get(_COOKIE) or "")


def _viewer_id(cookie: str) -> str:
    return "screen-" + hashlib.sha256(cookie.encode("utf-8")).hexdigest()[:32]


def _json_error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"success": False, "error": message}, status_code=status,
                        headers={"Cache-Control": "no-store"})


def _cookie_response(response: Response, value: str, request: Request) -> Response:
    public = resolve_public_url()
    secure = request.url.scheme == "https" or public.startswith("https://")
    response.set_cookie(
        _COOKIE, value, max_age=30 * 60, httponly=True, secure=secure,
        samesite="strict", path="/screen-handoff",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def _novnc_root() -> Path | None:
    """Locate the pinned noVNC package shipped by the candidate image or Desktop checkout."""
    candidates = (
        Path("/opt/hermes-screen/node_modules/@novnc/novnc"),
        Path(__file__).resolve().parents[2] / "apps" / "desktop" / "node_modules" / "@novnc" / "novnc",
    )
    for root in candidates:
        if (root / "core" / "rfb.js").is_file():
            return root
    return None


def _page(token: str) -> str:
    safe = html.escape(token, quote=True)
    # The candidate image serves the same pinned noVNC package as Desktop. The page is deliberately
    # a tiny client: it gets one display ticket from the server and cannot call gateway RPC methods.
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hermes secure browser screen</title>
<style>
body{{margin:0;background:#111827;color:#f9fafb;font:16px system-ui,sans-serif}}
main{{max-width:900px;margin:auto;padding:18px}} h1{{font-size:1.25rem}}
button,input{{font:inherit;border-radius:8px;padding:10px 12px}} button{{border:0;background:#38bdf8;color:#082f49;font-weight:700}}
button.secondary{{background:#334155;color:#f8fafc}} input{{width:9rem;background:#fff;border:0;letter-spacing:.2em;text-transform:uppercase}}
#screen{{margin-top:14px;background:#000;min-height:260px;display:grid;place-items:center;overflow:auto}}
#screen canvas{{max-width:100%;height:auto}} .muted{{color:#cbd5e1}} .error{{color:#fda4af}} .ok{{color:#86efac}}
</style></head><body><main>
<h1>Reprendre le navigateur Hermes</h1>
<p id="status" class="muted">Ce lien ne prend pas le contrôle. Confirmez le code reçu dans votre conversation privée.</p>
<section id="confirm"><input id="code" inputmode="text" autocomplete="one-time-code" maxlength="6" placeholder="CODE"><button id="authorize">Confirmer</button> <button id="refuse" class="secondary">Refuser</button></section>
<section id="controls" hidden><button id="takeover">Prendre la main</button> <button id="return" class="secondary" hidden>Rendre la main et continuer</button></section>
<div id="screen"><span class="muted">Aucun écran partagé</span></div>
<p class="muted">Saisissez vos identifiants uniquement dans le navigateur affiché. Hermes ne reçoit ni mot de passe ni frappe.</p>
</main><script type="module">
const token = "{safe}"; const status = document.querySelector('#status'); const screen = document.querySelector('#screen');
let rfb = null;
const setStatus = (text, cls='muted') => {{ status.textContent=text; status.className=cls; }};
async function post(path, body={{}}) {{ const res=await fetch('/screen-handoff/'+encodeURIComponent(token)+path, {{method:'POST', headers:{{'content-type':'application/json'}}, credentials:'same-origin', body:JSON.stringify(body)}}); let data={{}}; try{{data=await res.json()}}catch{{}}; if(!res.ok) throw new Error(data.error||'Request failed'); return data; }}
document.querySelector('#authorize').onclick = async () => {{ try {{ await post('/authorize', {{code:document.querySelector('#code').value}}); document.querySelector('#confirm').hidden=true; document.querySelector('#controls').hidden=false; setStatus('Autorisation confirmée. Prenez la main quand vous êtes prêt.','ok'); }} catch(e) {{ setStatus(e.message,'error'); }} }};
document.querySelector('#refuse').onclick = async () => {{ try {{ await post('/refuse'); document.querySelector('#confirm').hidden=true; setStatus('Demande refusée.','ok'); }} catch(e) {{ setStatus(e.message,'error'); }} }};
fetch('/screen-handoff/'+encodeURIComponent(token)+'/status', {{credentials:'same-origin'}}).then(r=>r.ok?r.json():null).then(data=>{{ if(data && ['authorized','human'].includes(data.state)){{ document.querySelector('#confirm').hidden=true; document.querySelector('#controls').hidden=false; setStatus(data.state==='human'?'Contrôle déjà réservé à ce navigateur.':'Autorisation confirmée. Prenez la main quand vous êtes prêt.','ok'); }} }}).catch(()=>{{}});
document.querySelector('#takeover').onclick = async () => {{ try {{ const data=await post('/takeover'); const RFB=(await import('/screen-handoff/assets/novnc/core/rfb.js')).default; const ws=(location.protocol==='https:'?'wss://':'ws://')+location.host+'/api/display/ws?display_ticket='+encodeURIComponent(data.display_ticket); rfb=new RFB(screen,ws); rfb.scaleViewport=true; rfb.resizeSession=true; rfb.viewOnly=false; rfb.addEventListener('connect',()=>{{setStatus('Contrôle acquis. Vous pouvez vous connecter dans le navigateur.','ok'); document.querySelector('#takeover').hidden=true; document.querySelector('#return').hidden=false;}}); rfb.addEventListener('disconnect',()=>{{if(!document.querySelector('#return').hidden) setStatus('Connexion interrompue. Le contrôle reste réservé jusqu’à une nouvelle autorisation.','error');}}); }} catch(e) {{ setStatus(e.message,'error'); }} }};
document.querySelector('#return').onclick = async () => {{ try {{ await post('/return'); if(rfb){{rfb.viewOnly=true; rfb.disconnect();}} document.querySelector('#return').hidden=true; setStatus('Contrôle rendu. Hermes va réobserver le navigateur avant de poursuivre.','ok'); }} catch(e) {{ setStatus(e.message,'error'); }} }};
</script></body></html>"""


@router.get("/screen-handoff/assets/novnc/{asset_path:path}")
async def screen_handoff_novnc_asset(asset_path: str) -> Response:
    """Serve only JavaScript from the pinned noVNC package, never package metadata."""
    root = _novnc_root()
    if root is None or not asset_path or not asset_path.endswith(".js"):
        return Response(status_code=404, headers={"Cache-Control": "no-store"})
    candidate = (root / asset_path).resolve()
    if root.resolve() not in candidate.parents or not candidate.is_file():
        return Response(status_code=404, headers={"Cache-Control": "no-store"})
    return FileResponse(candidate, media_type="text/javascript", headers={"Cache-Control": "public, max-age=3600"})


@router.get("/screen-handoff/{token}", response_class=HTMLResponse)
async def screen_handoff_page(token: str, request: Request) -> Response:
    _store_for_token, handoff = _find(token, mark_opened=True)
    if handoff is None:
        return HTMLResponse("This screen invitation is expired or invalid.", status_code=410,
                            headers={"Cache-Control": "no-store"})
    return HTMLResponse(_page(token), headers={"Cache-Control": "no-store"})


@router.post("/screen-handoff/{token}/authorize")
async def screen_handoff_authorize(token: str, request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        body = {}
    store, _opened = _find(token)
    handoff = store.authorize(token, str((body or {}).get("code") or "")) if store else None
    if handoff is None:
        return _json_error(403, "invalid or expired confirmation code")
    cookie = handoff.web_session_token
    if not cookie:
        return _json_error(503, "could not establish screen authorization")
    return _cookie_response(JSONResponse({"success": True, "state": handoff.state}), cookie, request)


@router.post("/screen-handoff/{token}/refuse")
async def screen_handoff_refuse(token: str) -> Response:
    store, handoff = _find(token)
    if store is None or handoff is None:
        return _json_error(410, "screen invitation is expired or invalid")
    store.refuse(token)
    return JSONResponse({"success": True, "state": "revoked"}, headers={"Cache-Control": "no-store"})


@router.get("/screen-handoff/{token}/status")
async def screen_handoff_status(token: str, request: Request) -> Response:
    _store_for_token, invitation = _find(token)
    cookie = _cookie(request)
    if invitation is None or not cookie:
        return JSONResponse({"success": True, "state": invitation.state if invitation else "unknown"},
                            headers={"Cache-Control": "no-store"})
    store, session = _find_session(cookie)
    if store is None or session is None or session.request_id != invitation.request_id:
        return JSONResponse({"success": True, "state": invitation.state}, headers={"Cache-Control": "no-store"})
    return JSONResponse({"success": True, "state": session.state}, headers={"Cache-Control": "no-store"})


@router.post("/screen-handoff/{token}/takeover")
async def screen_handoff_takeover(token: str, request: Request) -> Response:
    cookie = _cookie(request)
    store, handoff = _find_session(cookie)
    if handoff is None:
        return _json_error(401, "authorization expired")
    viewer_id = _viewer_id(cookie)
    try:
        from tools.bot_desktop import lease
        lease.acquire(viewer_id, profile_key=handoff.profile_home, reason=handoff.reason)
        active = store.take_over(cookie, viewer_id)
        if active is None:
            lease.release(viewer_id, profile_key=handoff.profile_home)
            return _json_error(409, "handoff is no longer available")
        from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
        ticket = mint_ticket(
            user_id=viewer_id, provider="bot-desktop-handoff",
            extra={"hermes_home": handoff.profile_home, "viewer_id": viewer_id,
                   "handoff_id": handoff.request_id, "retain_on_disconnect": True},
        )
        return JSONResponse({"success": True, "state": "human", "display_ticket": ticket},
                            headers={"Cache-Control": "no-store"})
    except Exception:
        logging.getLogger(__name__).exception("screen handoff takeover failed")
        return _json_error(503, "screen control is unavailable")


@router.post("/screen-handoff/{token}/return")
async def screen_handoff_return(token: str, request: Request) -> Response:
    cookie = _cookie(request)
    store, handoff = _find_any_session(cookie)
    if handoff is None:
        return _json_error(401, "authorization expired")
    viewer_id = _viewer_id(cookie)
    from tools.bot_desktop import lease
    if handoff.state == "returned":
        return JSONResponse({"success": True, "state": "returned"}, headers={"Cache-Control": "no-store"})
    if not lease.viewer_may_send_input(viewer_id, profile_key=handoff.profile_home):
        return _json_error(409, "this browser does not hold screen control")
    lease.release(viewer_id, profile_key=handoff.profile_home)
    returned = store.return_to_agent(cookie, viewer_id)
    if returned is None:
        return _json_error(409, "handoff was already returned")
    return JSONResponse({"success": True, "state": "returned"}, headers={"Cache-Control": "no-store"})


@router.post("/screen-handoff/{token}/revoke")
async def screen_handoff_revoke(token: str, request: Request) -> Response:
    cookie = _cookie(request)
    store, handoff = _find_any_session(cookie)
    if handoff is None:
        return _json_error(401, "authorization expired")
    viewer_id = _viewer_id(cookie)
    from tools.bot_desktop import lease
    if lease.viewer_may_send_input(viewer_id, profile_key=handoff.profile_home):
        lease.release(viewer_id, profile_key=handoff.profile_home)
    store.revoke(handoff.request_id)
    return JSONResponse({"success": True, "state": "revoked"}, headers={"Cache-Control": "no-store"})
