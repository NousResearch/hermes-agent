"""HTTP routes for WebAuthn (passkey) ceremonies.

Mounted at root (no prefix) next to ``routes.py`` (facade + siblings); the gated
middleware allowlists ``/auth/webauthn/``. Ceremonies:

  GET  /auth/webauthn/options           public login ceremony start (challenge store)
  POST /auth/webauthn/verify            public login completion (shared login rate limit)
  GET  /auth/webauthn/register/options  enrolment start — VERIFIED session or
                                        one-shot CLI enrolment token
  POST /auth/webauthn/register/verify   enrolment completion (same auth)
  GET  /auth/webauthn/enrol             CLI one-shot enrolment landing page

The provider is consulted via ``webauthn_offered_for_request`` so every ceremony
answers 404 (no oracle) when the request origin cannot present a configured
origin — the origin-gated offering from the provider plugin.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from hermes_cli.dashboard_auth import get_provider
from hermes_cli.dashboard_auth.base import ProviderError
from hermes_cli.dashboard_auth.cookies import (
    clear_pkce_cookie, clear_webauthn_challenge_cookie, detect_https, parse_pkce_payload,
    read_pkce_cookie, read_session_cookies, read_session_provider,
    read_webauthn_challenge_cookie, set_webauthn_challenge_cookie)
from hermes_cli.dashboard_auth.prefix import prefix_from_request
from hermes_cli.dashboard_auth.request_utils import client_ip, scan_session_providers
from hermes_cli.dashboard_auth.routes import (
    _NO_STORE, _client_ip, _complete_login, _login_failure, _login_rate_limited, _prefix,
    _set_session, _validate_post_login_target, _reset_login_rate_limit)
from hermes_cli.dashboard_auth.webauthn_state import ChallengeError, ChallengeNotFound

_log = logging.getLogger(__name__)

router = APIRouter()

PROVIDER_NAME = "webauthn"


def _http(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def _provider(request: Request):
    """The webauthn provider, or ``None`` when unknown to this deployment
    (unconfigured / plugin absent / disabled) or when the request origin cannot
    present a configured origin — both answer 404 further up (no oracle)."""
    provider = get_provider(PROVIDER_NAME)
    if provider is None or not getattr(provider, "supports_webauthn", False):
        return None
    if not provider.webauthn_offered_for_request(request):
        return None
    return provider


def _set_challenge_cookie(response, request: Request, handle: str, provider) -> None:
    set_webauthn_challenge_cookie(
        response, handle=handle,
        max_age=max(30, int(getattr(provider, "challenge_ttl_seconds", 120))),
        use_https=detect_https(request), prefix=_prefix(request))


def _clear_challenge(response, request: Request) -> None:
    clear_webauthn_challenge_cookie(response, prefix=_prefix(request))


def _require_enrol_session_or_token(request: Request, provider, token: str):
    """Enrolment is never anonymous (design §2.2): the middleware-verified
    session, a manually verified session cookie (this route family is PUBLIC),
    or an unconsumed one-shot CLI enrolment token. The token is consumed only on
    registration success — a ceremony that never completes leaves the token
    usable until its TTL, bounded by the same store."""
    sess = getattr(request.state, "session", None)
    if sess is not None:
        return sess
    try:
        if token and provider.peek_enrolment_token(token):
            return None  # authorised by the pending one-shot enrolment token
    except Exception:  # noqa: BLE001 - store I/O failure means "not authorised"
        pass
    access, _rt = read_session_cookies(request)
    if not access:
        raise _http(401, "Invalid or expired enrolment authorisation.")
    session = scan_session_providers(
        read_session_provider(request),
        lambda p: p.verify_session(access_token=access),
        phase="webauthn enrolment", log=_log)
    if session is None:
        raise _http(401, "Invalid or expired enrolment authorisation.")
    return session


# --- Public: login ceremony --------------------------------------------------


@router.get("/auth/webauthn/options", name="webauthn_options")
async def webauthn_options(request: Request):
    provider = _provider(request)
    if provider is None:
        raise _http(404, "Not Found")
    try:
        options, handle = provider.webauthn_login_options(client_ip=client_ip(request))
    except ProviderError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=503)
    except ChallengeError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=503)
    response = JSONResponse(options, headers=_NO_STORE)
    _set_challenge_cookie(response, request, handle, provider)
    return response


class _WebauthnVerifyBody(BaseModel):
    credential: dict = Field(default_factory=dict)
    next: str = Field(default="", max_length=2048)


@router.post("/auth/webauthn/verify", name="webauthn_verify")
async def webauthn_verify(request: Request, body: _WebauthnVerifyBody):
    """Complete a passkey login: consume the parked challenge, verify the
    assertion (origin/RP-ID/challenge/UP/UV/signature via fido2), mint a session
    through the SHARED basic seam. Failure map mirrors /auth/password-login:
    unregistered/unoffered 404, invalid 401 (generic — no credential oracle),
    rate limited 429, provider outage 503."""
    provider = _provider(request)
    if provider is None:
        raise _http(404, "Not Found")
    if _login_rate_limited(_client_ip(request)):
        _login_failure(request, PROVIDER_NAME, "rate_limited")
        raise _http(429, "Too many login attempts. Try again shortly.")
    try:
        session = provider.complete_webauthn_login(
            handle=read_webauthn_challenge_cookie(request) or "",
            credential=body.credential)
    except ChallengeNotFound:
        _login_failure(request, PROVIDER_NAME, "challenge_invalid")
        raise _http(401, "Invalid or expired passkey ceremony.")
    except ProviderError as e:
        _login_failure(request, PROVIDER_NAME, "provider_unreachable")
        raise _http(503, f"Provider unreachable: {e}")
    except Exception:  # noqa: BLE001 - any credential rejection is the generic 401
        _login_failure(request, PROVIDER_NAME, "invalid_credentials")
        raise _http(401, "Invalid or expired passkey ceremony.")
    # Broker tail parity with /auth/password-login: the desktop's native flow
    # completes the SAME way (one-shot loopback redirect, NO session cookies).
    pkce_raw = read_pkce_cookie(request)
    pkce = parse_pkce_payload(pkce_raw) if pkce_raw else {}
    broker_state = pkce.get("broker", "")
    if broker_state and pkce.get("provider", "") not in ("", PROVIDER_NAME):
        _login_failure(request, PROVIDER_NAME, "provider_mismatch")
        raise _http(400, "This native sign-in was started for a different provider; "
                         "use that provider's form or restart sign-in.")
    target, native = _complete_login(
        request, PROVIDER_NAME, session, broker_state=broker_state,
        next_raw=_validate_post_login_target(body.next))
    response = JSONResponse({"ok": True, "next": target})
    if native:
        clear_pkce_cookie(response, use_https=detect_https(request), prefix=_prefix(request))
    else:
        _set_session(response, request, session)
    _clear_challenge(response, request)
    return response


# --- Enrolment (authenticated session or one-shot CLI token) ------------------


@router.get("/auth/webauthn/register/options", name="webauthn_register_options")
async def webauthn_register_options(request: Request, enrolment: str = ""):
    provider = _provider(request)
    if provider is None:
        raise _http(404, "Not Found")
    session = _require_enrol_session_or_token(request, provider, enrolment)
    if session is not None and getattr(session, "provider", "") == "loopback":
        raise _http(401, "Invalid or expired enrolment authorisation.")
    try:
        options, handle = provider.webauthn_register_options(client_ip=client_ip(request))
    except ChallengeError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=503)
    response = JSONResponse(options, headers=_NO_STORE)
    _set_challenge_cookie(response, request, handle, provider)
    return response


class _WebauthnRegisterBody(BaseModel):
    credential: dict = Field(default_factory=dict)
    label: str = Field(default="", max_length=80)
    enrolment: str = Field(default="", max_length=256)


@router.post("/auth/webauthn/register/verify", name="webauthn_register_verify")
async def webauthn_register_verify(request: Request, body: _WebauthnRegisterBody):
    provider = _provider(request)
    if provider is None:
        raise _http(404, "Not Found")
    if _login_rate_limited(_client_ip(request)):
        _login_failure(request, PROVIDER_NAME, "rate_limited")
        raise _http(429, "Too many attempts. Try again shortly.")
    session = _require_enrol_session_or_token(request, provider, body.enrolment)
    token_used = bool(body.enrolment) and session is None
    try:
        record = provider.complete_webauthn_registration(
            handle=read_webauthn_challenge_cookie(request) or "",
            credential=body.credential, label=body.label)
    except ChallengeNotFound:
        raise _http(401, "Invalid or expired passkey ceremony.")
    except ProviderError as exc:
        raise _http(503, f"Provider unreachable: {exc}")
    except Exception:  # noqa: BLE001 - credential rejection mapping
        _login_failure(request, PROVIDER_NAME, "registration_rejected")
        raise _http(401, "Invalid or expired passkey ceremony.")
    if token_used:
        try:
            provider.consume_enrolment_token(body.enrolment)
        except Exception as exc:  # noqa: BLE001 - best effort; registration succeeded
            _log.warning("dashboard-auth: enrolment token consume failed: %s", exc)
    response = JSONResponse({"ok": True, "credential": record}, headers=_NO_STORE)
    _clear_challenge(response, request)
    return response


@router.get("/auth/webauthn/enrol", name="webauthn_enrol_page")
async def webauthn_enrol_page(request: Request, token: str = "", label: str = ""):
    """One-shot CLI enrolment landing page: ``hermes dashboard webauthn enrol``
    prints ``<origin>/auth/webauthn/enrol?token=…``; the operator opens it on the
    configured (TLS/localhost) origin and runs the create ceremony there."""
    provider = get_provider(PROVIDER_NAME)
    if (
        provider is None
        or not getattr(provider, "supports_webauthn", False)
        or not provider.webauthn_offered_for_request(request)
        or not token
    ):
        raise _http(404, "Not Found")
    try:
        valid = provider.peek_enrolment_token(token)
    except Exception:  # noqa: BLE001 - unreadable store = link no longer trusted
        valid = False
    if not valid:
        return HTMLResponse(_ENROL_PAGE_EXPIRED, headers=_NO_STORE)
    return HTMLResponse(_ENROL_PAGE, headers=_NO_STORE)


# --- Enrolment page (self-contained HTML; plain string like login_page's) ----

_ENROL_PAGE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Enrol a passkey — Hermes Agent</title>
<style>
  body { background: #170d02; color: #fff; font-family: system-ui, sans-serif;
         display: grid; place-items: center; min-height: 100vh; margin: 0; }
  main { max-width: 26rem; padding: 2rem; }
  h1 { font-size: 1.4rem; letter-spacing: .05em; text-transform: uppercase; }
  button { width: 100%; padding: .95rem 1rem; border: 0; background: #ffac02;
           color: #170d02; font-weight: 700; letter-spacing: .2em;
           text-transform: uppercase; cursor: pointer; margin-top: 1rem; }
  button:disabled { opacity: .6; cursor: default; }
  p.msg { color: #ff6b6b; white-space: pre-wrap; }
  p.ok { color: #7dd87d; }
</style>
</head>
<body>
<main>
<h1>Enrol a passkey</h1>
<p>This one-shot page adds a passkey to this Hermes dashboard. The enrolment
token is single-use and expires 10 minutes after issuance (see the CLI output).</p>
<p class="msg" id="err" hidden></p>
<p class="ok" id="ok" hidden>Passkey enrolled — this browser can now sign in
with it. Enrol a SECOND passkey now (another device or a hardware key) so one
lost authenticator cannot lock you out.</p>
<button id="enrol" type="button">Enrol a passkey</button>
</main>
<script>
(function () {
  var params = new URLSearchParams(window.location.search);
  var token = params.get('token') || '';
  var label = params.get('label') || 'CLI enrolment';
  function b64url(buffer) {
    var bytes = new Uint8Array(buffer); var s = '';
    for (var i = 0; i < bytes.length; i++) { s += String.fromCharCode(bytes[i]); }
    return btoa(s).replace(/\\+/g, '-').replace(/\\//g, '_').replace(/=+$/, '');
  }
  function b64urlToBuffer(value) {
    var s = String(value).replace(/-/g, '+').replace(/_/g, '/');
    while (s.length % 4) { s += '='; }
    var raw = atob(s); var bytes = new Uint8Array(raw.length);
    for (var i = 0; i < raw.length; i++) { bytes[i] = raw.charCodeAt(i); }
    return bytes.buffer;
  }
  function credentialToJSON(cred) {
    var data = { id: cred.id, rawId: b64url(cred.rawId), type: cred.type, response: {} };
    var r = cred.response;
    if (r.clientDataJSON) { data.response.clientDataJSON = b64url(r.clientDataJSON); }
    if (r.attestationObject) { data.response.attestationObject = b64url(r.attestationObject); }
    if (r.transports) { data.response.transports = r.transports; }
    return data;
  }
  function show(text) {
    var err = document.getElementById('err');
    err.textContent = text; err.hidden = false;
  }
  document.getElementById('enrol').addEventListener('click', function () {
    if (!window.PublicKeyCredential || !window.isSecureContext) {
      show('This page is not a secure context — passkey enrolment needs HTTPS or \\u200blocalhost. Re-run the CLI with the right origin and open the link there.');
      return;
    }
    fetch('/auth/webauthn/register/options?enrolment=' + encodeURIComponent(token),
          { credentials: 'same-origin' })
      .then(function (resp) {
        if (resp.status === 401) { throw new Error('Enrolment token unknown, used, or expired — run hermes dashboard webauthn enrol again.'); }
        if (!resp.ok) { throw new Error('Could not start enrolment (HTTP ' + resp.status + ').'); }
        return resp.json();
      })
      .then(function (options) {
        // The server serialises {"publicKey": <creation options>}; unwrap and
        // decode the base64url members to ArrayBuffers ourselves — the classic
        // create() path every browser ships; the JSON-parse shortcut here is
        // what threw "challenge … must be an instance of DOMString" in Safari
        // (it never saw a top-level challenge through the wrapper).
        var pk = options.publicKey || options;
        if (!pk || !pk.challenge) { throw new Error('Bad creation options from the server.'); }
        pk.challenge = b64urlToBuffer(pk.challenge);
        if (pk.user && pk.user.id) { pk.user.id = b64urlToBuffer(pk.user.id); }
        (pk.excludeCredentials || []).forEach(function (c) {
          c.id = b64urlToBuffer(c.id);
        });
        return navigator.credentials.create({ publicKey: pk });
      })
      .then(function (attestation) {
        return fetch('/auth/webauthn/register/verify', {
          method: 'POST', credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ credential: credentialToJSON(attestation), label: label, enrolment: token })
        });
      })
      .then(function (resp) {
        if (resp.ok) {
          document.getElementById('ok').hidden = false;
          var btn = document.getElementById('enrol');
          btn.disabled = true; btn.textContent = 'Enrolled';
          return;
        }
        return resp.json().then(function (data) {
          throw new Error((data && data.detail) || ('Enrolment failed (HTTP ' + resp.status + ').'));
        });
      })
      .catch(function (e) { show(String((e && e.message) || e)); });
  });
})();
</script>
</body>
</html>
"""

_ENROL_PAGE_EXPIRED = """\
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Enrolment link expired — Hermes Agent</title></head>
<body style="background:#170d02;color:#fff;font-family:system-ui,sans-serif;display:grid;place-items:center;min-height:100vh">
<main><p>That enrolment link is unknown, already used, or expired.</p>
<p>Run <code>hermes dashboard webauthn enrol</code> again for a fresh one.</p></main>
</body></html>
"""


def _reset_rate_limit_for_tests() -> None:
    """Test-only alias the webauthn suite uses (same store as the login limiter)."""
    _reset_login_rate_limit()