#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from manifest_utils import load_simple_yaml

NGINX_CONF = r'''server {
  listen 80;
  server_name _;
  root /usr/share/nginx/html;
  index index.html;

  add_header X-Content-Type-Options "nosniff" always;
  add_header Referrer-Policy "strict-origin-when-cross-origin" always;
  add_header X-Frame-Options "SAMEORIGIN" always;
  add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;

  location /healthz {
    access_log off;
    default_type text/plain;
    return 200 "delivery-sandbox-ok\n";
  }

  location = /user {
    return 302 /user/;
  }

  location /user/ {
    proxy_pass http://delivery-sandbox-events:8080;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }

  location /api/ {
    proxy_pass http://delivery-sandbox-events:8080;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Authorization $http_authorization;
  }

  location = /_auth_user {
    internal;
    proxy_pass http://delivery-sandbox-events:8080/api/user/session/check;
    proxy_pass_request_body off;
    proxy_set_header Content-Length "";
    proxy_set_header Host $host;
    proxy_set_header Cookie $http_cookie;
  }

  location / {
    try_files $uri $uri/ =404;
  }

  location /download/ {
    autoindex off;
    try_files $uri =404;
  }

  location /w/ {
    try_files $uri $uri/ =404;
  }
}
'''

DOCKERFILE = r"""FROM python:3.12-alpine
WORKDIR /app
COPY server.py /app/server.py
COPY delivery_document_actions.py /app/delivery_document_actions.py
RUN chmod 0644 /app/server.py /app/delivery_document_actions.py
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "/app/server.py"]
"""

SERVER_PY = r'''#!/usr/bin/env python3
"""Minimal delivery sandbox event receiver + authenticated user dashboard.

The public sandbox keeps agent/database secrets out of this container. User
private surfaces are protected with OTP-issued temporary sessions. OTP delivery
is queued for a trusted Hermes-side dispatcher; this service only stores hashes
and append-only audit records.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import os
import re
import secrets
import time
import uuid
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from delivery_document_actions import (
    OTP_REQUIRED_DOCUMENT_EVENT_TYPES,
    build_document_event,
    document_action_allows_direct_post,
    document_action_requires_otp,
    normalize_document_action,
)

EVENT_DIR = Path(os.environ.get("EVENT_DIR", "/data/events"))
PUBLIC_DIR = Path(os.environ.get("PUBLIC_DIR", "/data/public"))
USER_DATA_DIR = Path(os.environ.get("USER_DATA_DIR", "/data/user-data"))
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "65536"))
SESSION_TTL_SECONDS = int(os.environ.get("USER_SESSION_TTL_SECONDS", str(12 * 3600)))
OTP_TTL_SECONDS = int(os.environ.get("USER_OTP_TTL_SECONDS", "600"))
OTP_RATE_LIMIT_SECONDS = int(os.environ.get("USER_OTP_RATE_LIMIT_SECONDS", "45"))
AGENT_ID = os.environ.get("AGENT_ID", "commercial-agent")
AGENT_NAME = os.environ.get("AGENT_NAME", AGENT_ID.replace("-", " ").title())
SESSION_COOKIE_NAME = os.environ.get("USER_SESSION_COOKIE_NAME", f"{AGENT_ID.replace('-', '_')}_user_session")
COOKIE_SECURE = os.environ.get("USER_SESSION_COOKIE_SECURE", "true").lower() not in {"0", "false", "no"}
ALLOWED_HOSTS = {
    host.strip().lower()
    for host in os.environ.get("ALLOWED_HOSTS", "").split(",")
    if host.strip()
}
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
OTP_RE = re.compile(r"^\d{6}$")
VAPI_SERVER_AUTH_TOKEN = os.environ.get("VAPI_SERVER_AUTH_TOKEN", "")
VAPI_ALLOWED_TOOL_NAMES = {
    "capture_voice_lead",
    "schedule_followup",
    "escalate_to_zeus",
    "get_customer_context",
    "send_call_summary",
}

ALLOWED_TYPES = {
    "opened",
    "commented",
    "approved",
    "rejected",
    "signed",
    "paid",
    "payment_failed",
    "change_requested",
    "coach_opened",
    "routine_started",
    "set_completed",
    "workout_finished",
    "meal_photo_requested",
    "meal_logged",
    "barcode_scanned",
    "checkin_submitted",
    "sleep_plan_acknowledged",
}
COACH_EVENT_TYPES = {
    "coach_opened",
    "routine_started",
    "set_completed",
    "workout_finished",
    "meal_photo_requested",
    "meal_logged",
    "barcode_scanned",
    "checkin_submitted",
    "sleep_plan_acknowledged",
}


def _now() -> int:
    return int(time.time())


def _state_path() -> Path:
    return EVENT_DIR / "user_auth_state.json"


def _outbox_path() -> Path:
    return EVENT_DIR / "otp_outbox.jsonl"


def _audit_path() -> Path:
    return EVENT_DIR / "events.jsonl"


def _secret() -> bytes:
    # Not a public API secret. It only signs local dashboard session/OTP hashes.
    seed_path = EVENT_DIR / "user_dashboard_secret"
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    if seed_path.exists():
        raw = seed_path.read_bytes().strip()
        try:
            return base64.urlsafe_b64decode(raw)
        except Exception:
            # Backward-compatible fallback if an older file stored raw bytes.
            return raw
    seed = secrets.token_bytes(32)
    seed_path.write_bytes(base64.urlsafe_b64encode(seed))
    try:
        seed_path.chmod(0o600)
    except OSError:
        pass
    return seed


def _hash(value: str) -> str:
    return hmac.new(_secret(), value.encode("utf-8"), hashlib.sha256).hexdigest()


def _load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {"challenges": {}, "sessions": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"challenges": {}, "sessions": {}}
        data.setdefault("challenges", {})
        data.setdefault("sessions", {})
        return data
    except Exception:
        return {"challenges": {}, "sessions": {}}


def _save_state(state: dict[str, Any]) -> None:
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _state_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(_state_path())


def _cleanup_state(state: dict[str, Any]) -> dict[str, Any]:
    now = _now()
    state["challenges"] = {
        key: val for key, val in state.get("challenges", {}).items()
        if int(val.get("expires_at", 0)) > now and int(val.get("attempts", 0)) < 5
    }
    state["sessions"] = {
        key: val for key, val in state.get("sessions", {}).items()
        if int(val.get("expires_at", 0)) > now
    }
    return state


def _dashboard_config() -> dict[str, Any]:
    path = USER_DATA_DIR / "user_dashboard.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "user_id": os.environ.get("DASHBOARD_USER_ID", "default-user"),
        "display_name": os.environ.get("DASHBOARD_DISPLAY_NAME", "Agent User"),
        "channels": [
            {"id": "whatsapp", "label": "WhatsApp registrado", "target": "whatsapp"},
            {"id": "telegram", "label": "Telegram registrado", "target": "telegram"},
        ],
        "metrics": [],
        "links": [],
        "decisions": [],
    }


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, status: int, body: str, headers: dict[str, str] | None = None) -> None:
    raw = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
    handler.send_header("Cache-Control", "no-store")
    for key, value in (headers or {}).items():
        handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(raw)


def _redirect(handler: BaseHTTPRequestHandler, location: str, headers: dict[str, str] | None = None) -> None:
    handler.send_response(303)
    handler.send_header("Location", location)
    handler.send_header("Cache-Control", "no-store")
    for key, value in (headers or {}).items():
        handler.send_header(key, value)
    handler.end_headers()


def _host_without_port(value: str | None) -> str:
    if not value:
        return ""
    host = value.strip().lower()
    if host.startswith("[") and "]" in host:
        return host.split("]", 1)[0].strip("[]")
    return host.rsplit(":", 1)[0]


def _origin_host(value: str | None) -> str:
    if not value:
        return ""
    return _host_without_port(urlparse(value).netloc)


def _valid_request_origin(handler: BaseHTTPRequestHandler) -> bool:
    """Accept only the configured public workspace domain."""
    if not ALLOWED_HOSTS:
        return True
    host = _host_without_port(handler.headers.get("Host"))
    origin = _origin_host(handler.headers.get("Origin"))
    referer = _origin_host(handler.headers.get("Referer"))
    if host and host not in ALLOWED_HOSTS:
        return False
    if origin and origin not in ALLOWED_HOSTS:
        return False
    if referer and referer not in ALLOWED_HOSTS:
        return False
    return True


def _workspace_manifest(token: str) -> dict[str, Any] | None:
    if not TOKEN_RE.fullmatch(token):
        return None
    workspace = PUBLIC_DIR / "w" / token
    if not workspace.is_dir():
        return None
    manifest = workspace / "workspace.json"
    if not manifest.exists():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _recipient_manifest(token: str) -> dict[str, Any] | None:
    if not TOKEN_RE.fullmatch(token):
        return None
    private_manifest = USER_DATA_DIR / "workspace-recipients" / f"{token}.json"
    if private_manifest.exists():
        try:
            data = json.loads(private_manifest.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None
    return _workspace_manifest(token)


def _workspace_matches_token(token: str, deliverable_id: str, metadata: dict[str, Any]) -> bool:
    """Bind each event to its opaque workspace token and deliverable id."""
    manifest_data = _workspace_manifest(token)
    if manifest_data is None:
        return False
    workspace = PUBLIC_DIR / "w" / token
    if manifest_data:
        expected = str(manifest_data.get("deliverable_id") or manifest_data.get("receipt_id") or manifest_data.get("source_id") or "")
        if expected and expected != deliverable_id:
            return False
    else:
        index = workspace / "index.html"
        if not index.exists():
            return False
        try:
            if deliverable_id not in index.read_text(encoding="utf-8", errors="ignore"):
                return False
        except Exception:
            return False
    for key in ("receipt_id", "quote_id", "invoice_id"):
        value = metadata.get(key)
        if value and str(value) != deliverable_id:
            return False
    return True


def _document_action_recipient(token: str) -> dict[str, str] | None:
    manifest = _recipient_manifest(token)
    if not isinstance(manifest, dict):
        return None
    recipient = manifest.get("recipient") if isinstance(manifest.get("recipient"), dict) else {}
    channel_id = recipient.get("channel_id") or manifest.get("recipient_channel_id") or manifest.get("channel_id")
    target = recipient.get("target") or manifest.get("recipient_target") or manifest.get("customer_phone") or manifest.get("customer_email")
    label = recipient.get("label") or manifest.get("recipient_label") or channel_id
    if not channel_id or not target:
        return None
    return {"channel_id": str(channel_id), "target": str(target), "label": str(label or channel_id)}


def _parse_cookies(header: str | None) -> dict[str, str]:
    if not header:
        return {}
    jar = cookies.SimpleCookie()
    try:
        jar.load(header)
    except cookies.CookieError:
        return {}
    return {key: morsel.value for key, morsel in jar.items()}


def _session_from_request(handler: BaseHTTPRequestHandler) -> dict[str, Any] | None:
    token = _parse_cookies(handler.headers.get("Cookie")).get(SESSION_COOKIE_NAME)
    if not token:
        return None
    state = _cleanup_state(_load_state())
    session = state.get("sessions", {}).get(_hash(token))
    if not session or int(session.get("expires_at", 0)) <= _now():
        _save_state(state)
        return None
    return session


def _session_cookie(token: str, max_age: int = SESSION_TTL_SECONDS) -> str:
    parts = [f"{SESSION_COOKIE_NAME}={token}", "Path=/", "HttpOnly", "SameSite=Lax", f"Max-Age={max_age}"]
    if COOKIE_SECURE:
        parts.append("Secure")
    return "; ".join(parts)


def _expired_cookie() -> str:
    parts = [f"{SESSION_COOKIE_NAME}=", "Path=/", "HttpOnly", "SameSite=Lax", "Max-Age=0"]
    if COOKIE_SECURE:
        parts.append("Secure")
    return "; ".join(parts)


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0 or length > MAX_BODY_BYTES:
        raise ValueError("invalid_body_size")
    return handler.rfile.read(length)


def _read_form(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    raw = _read_body(handler).decode("utf-8")
    if "application/json" in (handler.headers.get("Content-Type") or ""):
        data = json.loads(raw)
        return {str(k): str(v) for k, v in data.items() if v is not None}
    parsed = parse_qs(raw, keep_blank_values=True)
    return {key: vals[-1] if vals else "" for key, vals in parsed.items()}


def _audit(event: dict[str, Any]) -> None:
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_id": str(uuid.uuid4()),
        "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "delivery-sandbox",
        **event,
    }
    with _audit_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _queue_otp(challenge: dict[str, Any], otp: str) -> None:
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_id": str(uuid.uuid4()),
        "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "pending",
        "challenge_id": challenge["challenge_id"],
        "user_id": challenge["user_id"],
        "channel_id": challenge["channel_id"],
        "target": challenge["target"],
        "message": challenge.get("message") or f"Tu código {AGENT_NAME} para entrar al dashboard es: {otp}. Expira en 10 minutos.",
    }
    for key in ("purpose", "event_type", "deliverable_id", "token_ref"):
        if challenge.get(key):
            payload[key] = challenge[key]
    with _outbox_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _layout(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title><style>
:root{{--bg:#070b12;--panel:#101827;--ink:#f7f9fc;--muted:#a8b3c7;--line:rgba(255,255,255,.13);--blue:#85b7ff;--green:#86efac;--orange:#fbbf24;--purple:#c4b5fd;--shadow:rgba(0,0,0,.44)}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 12% 0%,rgba(133,183,255,.20),transparent 30%),radial-gradient(circle at 92% 10%,rgba(134,239,172,.12),transparent 28%),linear-gradient(180deg,#070b12,#0b1020 70%,#070b12);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}a{{color:inherit;text-decoration:none}}.shell{{width:min(1180px,calc(100% - 24px));margin:0 auto;padding:24px 0 44px}}.top{{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:22px}}.brand{{font-size:22px;font-weight:950;letter-spacing:-.05em}}.brand span{{color:var(--blue)}}.pill{{display:inline-flex;gap:8px;align-items:center;padding:8px 12px;border:1px solid var(--line);border-radius:999px;color:var(--muted);font-weight:800;font-size:13px}}.hero,.card,.metric,.module-card{{background:linear-gradient(180deg,rgba(255,255,255,.078),rgba(255,255,255,.035));border:1px solid var(--line);box-shadow:0 22px 70px var(--shadow);border-radius:28px}}.hero{{padding:clamp(24px,4vw,46px);margin-bottom:14px}}h1{{font-size:clamp(42px,7vw,78px);line-height:.92;letter-spacing:-.07em;margin:8px 0 14px;max-width:13ch}}h2{{font-size:clamp(25px,4vw,42px);letter-spacing:-.05em;margin:0 0 14px}}h3{{font-size:22px;letter-spacing:-.035em;margin:0 0 8px}}p{{line-height:1.5}}.muted{{color:var(--muted)}}.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:var(--blue);font-weight:950}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}.metric,.card{{padding:18px}}.metric strong{{display:block;font-size:30px;letter-spacing:-.05em;margin-top:8px}}.metric .icon,.module-card .icon{{font-size:25px;display:inline-flex;align-items:center;justify-content:center;width:42px;height:42px;border-radius:16px;background:rgba(255,255,255,.08);border:1px solid var(--line);margin-bottom:10px}}.modules{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:14px}}.module-card{{display:block;padding:20px;position:relative;overflow:hidden;min-height:170px}}.module-card:hover{{border-color:rgba(133,183,255,.55);transform:translateY(-1px)}}.module-card.primary{{background:linear-gradient(135deg,rgba(133,183,255,.20),rgba(134,239,172,.08)),linear-gradient(180deg,rgba(255,255,255,.085),rgba(255,255,255,.035))}}.module-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}}.status{{font-size:11px;text-transform:uppercase;letter-spacing:.12em;border:1px solid var(--line);border-radius:999px;padding:6px 8px;color:var(--muted);font-weight:900}}.status.active{{color:#bbf7d0;border-color:rgba(134,239,172,.38);background:rgba(134,239,172,.10)}}.status.planned{{color:#fde68a;border-color:rgba(251,191,36,.35);background:rgba(251,191,36,.10)}}button,.button{{border:0;border-radius:999px;background:var(--blue);color:#07111f;padding:12px 17px;font-weight:950;cursor:pointer;display:inline-flex;align-items:center;gap:8px;margin-top:10px}}button.secondary,.button.secondary{{background:rgba(255,255,255,.08);color:var(--ink);border:1px solid var(--line)}}label{{font-weight:900;color:var(--ink)}}select,input{{width:100%;padding:14px 15px;border-radius:18px;border:1px solid rgba(255,255,255,.20);background:#111827;color:var(--ink);margin:8px 0 10px;font-size:16px;accent-color:var(--blue)}}select:focus,input:focus{{outline:2px solid rgba(133,183,255,.55);outline-offset:2px}}select option{{background:#111827;color:#f7f9fc}}.channel-select{{position:relative}}.channel-select:before{{content:'📨';position:absolute;left:15px;top:45px;z-index:1}}.channel-select select{{padding-left:48px;appearance:none;background-color:#111827;background-image:linear-gradient(45deg,transparent 50%,#f7f9fc 50%),linear-gradient(135deg,#f7f9fc 50%,transparent 50%);background-position:calc(100% - 20px) 24px,calc(100% - 14px) 24px;background-size:6px 6px,6px 6px;background-repeat:no-repeat}}.otp-row{{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:10px 0 4px}}.otp-digit{{text-align:center;font-size:24px;font-weight:950;padding:14px 4px;border-radius:16px}}.notice{{padding:12px 14px;border-radius:18px;background:rgba(251,191,36,.12);border:1px solid rgba(251,191,36,.28);color:#fde68a}}.caption{{font-size:13px;color:var(--muted);margin-top:6px}}.decision-list p{{border-top:1px solid var(--line);padding-top:12px}}.hero-actions{{display:flex;flex-wrap:wrap;gap:10px;align-items:center}}@media(max-width:980px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.modules{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:700px){{.grid,.modules{{grid-template-columns:1fr}}.top{{align-items:flex-start;flex-direction:column}}.otp-row{{gap:6px}}}}
</style><script>
function setupOtp(){{
  const form=document.querySelector('[data-otp-form]'); if(!form) return;
  const hidden=form.querySelector('input[name="otp"]'); const boxes=[...form.querySelectorAll('.otp-digit')];
  const sync=()=>{{hidden.value=boxes.map(b=>b.value.replace(/\\D/g,'').slice(0,1)).join('');}};
  boxes.forEach((box,i)=>{{
    box.addEventListener('input',()=>{{box.value=box.value.replace(/\\D/g,'').slice(0,1); sync(); if(box.value && boxes[i+1]) boxes[i+1].focus();}});
    box.addEventListener('keydown',(e)=>{{if(e.key==='Backspace'&&!box.value&&boxes[i-1]) boxes[i-1].focus();}});
    box.addEventListener('paste',(e)=>{{const text=(e.clipboardData||window.clipboardData).getData('text').replace(/\\D/g,'').slice(0,6); if(text){{e.preventDefault(); boxes.forEach((b,j)=>b.value=text[j]||''); sync(); (boxes[Math.min(text.length,6)-1]||box).focus();}}}});
  }});
  form.addEventListener('submit',(e)=>{{sync(); if(!/^\\d{{6}}$/.test(hidden.value)){{e.preventDefault(); boxes.find(b=>!b.value)?.focus();}}}});
}}
document.addEventListener('DOMContentLoaded', setupOtp);
</script></head><body><main class="shell"><div class="top"><a class="brand" href="/user/">{_e(AGENT_NAME)} <span>User</span></a><span class="pill">🔐 Sesión temporal + OTP</span></div>{body}</main></body></html>"""


def _login_page(message: str | None = None, challenge_id: str | None = None) -> str:
    cfg = _dashboard_config()
    options = "".join(
        f"<option value='{_e(ch.get('id'))}'>{_e(ch.get('icon') or '📨')} {_e(ch.get('label') or ch.get('id'))}</option>"
        for ch in cfg.get("channels", [])
    )
    notice = f"<p class='notice'>{_e(message)}</p>" if message else ""
    verify = ""
    if challenge_id:
        digits = "".join(
            f"<input class='otp-digit' inputmode='numeric' pattern='[0-9]' maxlength='1' autocomplete='one-time-code' aria-label='Dígito {i} del OTP'>"
            for i in range(1, 7)
        )
        verify = f"""
        <section class="card" style="margin-top:14px"><span class="eyebrow">Paso 2</span><h2>Ingresa el código</h2>
          <p class="muted">Escribe o pega los 6 dígitos exactos. El formulario avanza automáticamente entre casillas.</p>
          <form method="post" action="/user/verify" data-otp-form><input type="hidden" name="challenge_id" value="{_e(challenge_id)}"><input type="hidden" name="otp" value=""><div class="otp-row">{digits}</div><p class="caption">El código expira en 10 minutos. Si pides otro demasiado rápido, se mantiene esta casilla para que puedas usar el código vigente.</p><button>Entrar</button></form>
        </section>"""
    body = f"""
    <section class="hero"><span class="eyebrow">Dashboard privado</span><h1>Entrar sin fricción</h1><p class="muted">Te enviamos un OTP al canal registrado que elijas. La sesión queda temporalmente activa en este navegador y los espacios privados quedan detrás de esa sesión.</p>{notice}</section>
    <section class="card"><span class="eyebrow">Paso 1</span><h2>Solicitar OTP</h2><form method="post" action="/user/request-otp"><div class="channel-select"><label>Canal registrado</label><select name="channel_id">{options}</select></div><button>Enviar código</button></form></section>{verify}
    """
    return _layout(f"{AGENT_NAME} User - Login", body)


def _dashboard_page(session: dict[str, Any]) -> str:
    cfg = _dashboard_config()
    metrics = cfg.get("metrics") or []
    modules = cfg.get("modules") or cfg.get("links") or []
    decisions = cfg.get("decisions") or []
    metric_html = "".join(
        f"<div class='metric'><span class='icon'>{_e(m.get('icon') or '•')}</span><span class='eyebrow'>{_e(m.get('label'))}</span><strong>{_e(m.get('value'))}</strong><p class='muted'>{_e(m.get('note'))}</p></div>"
        for m in metrics
    )
    module_html = "".join(
        f"<a class='module-card {'primary' if mod.get('primary') else ''}' href='{_e(mod.get('href') or '#')}'><div class='module-top'><span class='icon'>{_e(mod.get('icon') or '◦')}</span><span class='status {_e(mod.get('status') or 'planned')}'>{_e(mod.get('status_label') or mod.get('status') or 'planned')}</span></div><span class='eyebrow'>{_e(mod.get('kind') or 'módulo')}</span><h3>{_e(mod.get('title'))}</h3><p class='muted'>{_e(mod.get('description'))}</p><p class='caption'>{_e(mod.get('metric') or '')}</p></a>"
        for mod in modules
    )
    decision_html = "".join(f"<p>• <strong>{_e(d.get('title'))}</strong>: {_e(d.get('summary'))}</p>" for d in decisions)
    body = f"""
    <section class="hero"><span class="eyebrow">{_e(cfg.get('display_name') or session.get('user_id'))}</span><h1>Mapa del agente</h1><p class="muted">Vista ejecutiva de lo que el agente puede operar por chat: agenda, CRM, cotizaciones, invoices, documentos firmados, ventas, productos y módulos especializados como Fitness Coach. No es una UI de gestión pesada: es un panel claro para mostrar capacidades, métricas y accesos protegidos.</p><div class="hero-actions"><a class="button" href="/w/VtV636xEVsdDGmzSHys6vrko/coach/">Abrir Fitness Coach</a><a class="button secondary" href="/user/logout">Cerrar sesión</a></div></section>
    <section class="grid">{metric_html}</section>
    <section class="card" style="margin-top:14px"><span class="eyebrow">Menú protegido</span><h2>Funcionalidades del agente</h2><p class="muted">Cada tarjeta lleva a un espacio privado o presenta el estado de una capacidad clave del agente.</p><div class="modules">{module_html}</div></section>
    <section class="card decision-list" style="margin-top:14px"><span class="eyebrow">Decisiones y contexto</span><h2>Lo importante</h2>{decision_html or '<p class="muted">Aún no hay decisiones registradas.</p>'}</section>
    """
    return _layout(f"{AGENT_NAME} User", body)


def _handle_user_get(handler: BaseHTTPRequestHandler, path: str, query: dict[str, list[str]]) -> None:
    if path in {"/user/logout", "/user/logout/"}:
        _redirect(handler, "/user/login", {"Set-Cookie": _expired_cookie()})
        return
    if path in {"/user/login", "/user/login/"}:
        _html_response(handler, 200, _login_page())
        return
    if path in {"/user", "/user/"}:
        session = _session_from_request(handler)
        if not session:
            _redirect(handler, "/user/login")
            return
        _html_response(handler, 200, _dashboard_page(session))
        return
    _json_response(handler, 404, {"ok": False, "error": "not_found"})


def _handle_request_otp(handler: BaseHTTPRequestHandler) -> None:
    if not _valid_request_origin(handler):
        _json_response(handler, 403, {"ok": False, "error": "invalid_origin"})
        return
    form = _read_form(handler)
    cfg = _dashboard_config()
    channel_id = form.get("channel_id", "").strip()
    channel = next((ch for ch in cfg.get("channels", []) if ch.get("id") == channel_id), None)
    if not channel:
        _html_response(handler, 422, _login_page("Canal no registrado."))
        return
    state = _cleanup_state(_load_state())
    now = _now()
    recent = [c for c in state.get("challenges", {}).values() if c.get("channel_id") == channel_id and now - int(c.get("created_at", 0)) < OTP_RATE_LIMIT_SECONDS]
    if recent:
        current = sorted(recent, key=lambda c: int(c.get("created_at", 0)), reverse=True)[0]
        _html_response(handler, 429, _login_page("Ya hay un código vigente para ese canal. Puedes introducirlo abajo o esperar unos segundos para pedir otro.", str(current.get("challenge_id") or "")))
        return
    otp = f"{secrets.randbelow(1000000):06d}"
    challenge_id = secrets.token_urlsafe(18)
    challenge = {
        "challenge_id": challenge_id,
        "user_id": cfg.get("user_id", "default-user"),
        "channel_id": channel_id,
        "target": channel.get("target") or channel_id,
        "otp_hash": _hash(f"{challenge_id}:{otp}"),
        "created_at": now,
        "expires_at": now + OTP_TTL_SECONDS,
        "attempts": 0,
    }
    state.setdefault("challenges", {})[challenge_id] = challenge
    _save_state(state)
    _queue_otp(challenge, otp)
    _audit({"event_type": "user_otp_requested", "user_id": challenge["user_id"], "actor_type": "user", "actor_ref": channel_id, "status": "pending_otp_dispatch"})
    _html_response(handler, 200, _login_page(f"Código solicitado por {channel.get('label')}. Revisa ese canal.", challenge_id))


def _handle_verify_otp(handler: BaseHTTPRequestHandler) -> None:
    if not _valid_request_origin(handler):
        _json_response(handler, 403, {"ok": False, "error": "invalid_origin"})
        return
    form = _read_form(handler)
    challenge_id = form.get("challenge_id", "").strip()
    otp = form.get("otp", "").strip()
    state = _cleanup_state(_load_state())
    challenge = state.get("challenges", {}).get(challenge_id)
    if not challenge or not OTP_RE.fullmatch(otp):
        _html_response(handler, 401, _login_page("Código inválido o expirado.", challenge_id if challenge_id else None))
        return
    challenge["attempts"] = int(challenge.get("attempts", 0)) + 1
    if not hmac.compare_digest(challenge.get("otp_hash", ""), _hash(f"{challenge_id}:{otp}")):
        state["challenges"][challenge_id] = challenge
        _save_state(state)
        _html_response(handler, 401, _login_page("Código incorrecto.", challenge_id if challenge["attempts"] < 5 else None))
        return
    token = secrets.token_urlsafe(32)
    session = {
        "user_id": challenge.get("user_id"),
        "channel_id": challenge.get("channel_id"),
        "created_at": _now(),
        "expires_at": _now() + SESSION_TTL_SECONDS,
    }
    state.setdefault("sessions", {})[_hash(token)] = session
    state.get("challenges", {}).pop(challenge_id, None)
    _save_state(state)
    _audit({"event_type": "user_session_started", "user_id": session["user_id"], "actor_type": "user", "actor_ref": session["channel_id"], "status": "active"})
    _redirect(handler, "/user/", {"Set-Cookie": _session_cookie(token)})


def _bearer_token(handler: BaseHTTPRequestHandler) -> str:
    raw = handler.headers.get("Authorization", "")
    if raw.lower().startswith("bearer "):
        return raw.split(" ", 1)[1].strip()
    return ""


def _vapi_authorized(handler: BaseHTTPRequestHandler) -> bool:
    if not VAPI_SERVER_AUTH_TOKEN:
        return False
    return hmac.compare_digest(_bearer_token(handler), VAPI_SERVER_AUTH_TOKEN)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    payload = json.loads(_read_body(handler).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid_json_object")
    return payload


def _document_event_token_ref(token: str) -> str:
    return token[:10] + "..." if len(token) > 10 else token


def _validate_document_action_payload(payload: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]] | None:
    event_type = normalize_document_action(payload.get("event_type") or payload.get("action"))
    deliverable_id = str(payload.get("deliverable_id", "")).strip()
    token = str(payload.get("token", "")).strip()
    metadata = payload.get("metadata") or {}
    if not event_type:
        return None
    if not deliverable_id or not token or not isinstance(metadata, dict):
        return None
    if not _workspace_matches_token(token, deliverable_id, metadata):
        return None
    return event_type, deliverable_id, token, metadata


def _create_document_action_challenge(payload: dict[str, Any], recipient: dict[str, str]) -> str:
    event_type = normalize_document_action(payload.get("event_type") or payload.get("action"))
    deliverable_id = str(payload.get("deliverable_id") or "").strip()
    token = str(payload.get("token") or "").strip()
    otp = f"{secrets.randbelow(1000000):06d}"
    challenge_id = secrets.token_urlsafe(18)
    now = _now()
    challenge = {
        "challenge_id": challenge_id,
        "purpose": "document_action",
        "user_id": recipient["target"],
        "channel_id": recipient["channel_id"],
        "target": recipient["target"],
        "event_type": event_type,
        "deliverable_id": deliverable_id,
        "token_ref": _document_event_token_ref(token),
        "pending_payload": payload,
        "otp_hash": _hash(f"{challenge_id}:{otp}"),
        "created_at": now,
        "expires_at": now + OTP_TTL_SECONDS,
        "attempts": 0,
        "message": f"Tu código {AGENT_NAME} para confirmar {event_type} en {deliverable_id} es: {otp}. Expira en 10 minutos.",
    }
    state = _cleanup_state(_load_state())
    state.setdefault("challenges", {})[challenge_id] = challenge
    _save_state(state)
    _queue_otp(challenge, otp)
    return challenge_id


def _handle_document_action_request_otp(handler: BaseHTTPRequestHandler) -> None:
    if not _valid_request_origin(handler):
        _json_response(handler, 403, {"ok": False, "error": "invalid_origin"})
        return
    payload = _read_json_body(handler)
    validation = _validate_document_action_payload(payload)
    if not validation:
        _json_response(handler, 422, {"ok": False, "error": "invalid_document_action"})
        return
    event_type, deliverable_id, token, _metadata = validation
    if not document_action_requires_otp(event_type):
        _json_response(handler, 422, {"ok": False, "error": "otp_not_required"})
        return
    recipient = _document_action_recipient(token)
    if not recipient:
        _json_response(handler, 422, {"ok": False, "error": "missing_recipient_channel"})
        return
    state = _cleanup_state(_load_state())
    now = _now()
    recent = [
        c for c in state.get("challenges", {}).values()
        if c.get("purpose") == "document_action"
        and c.get("target") == recipient["target"]
        and c.get("event_type") == event_type
        and c.get("deliverable_id") == deliverable_id
        and now - int(c.get("created_at", 0)) < OTP_RATE_LIMIT_SECONDS
    ]
    if recent:
        current = sorted(recent, key=lambda c: int(c.get("created_at", 0)), reverse=True)[0]
        _json_response(handler, 429, {"ok": False, "error": "otp_recently_requested", "challenge_id": current.get("challenge_id")})
        return
    challenge_id = _create_document_action_challenge(payload, recipient)
    _audit({
        "event_type": "document_action_otp_requested",
        "deliverable_id": deliverable_id,
        "actor_type": "customer",
        "actor_ref": recipient["channel_id"],
        "metadata": {"document_event_type": event_type, "token_ref": _document_event_token_ref(token)},
        "status": "pending_otp_dispatch",
    })
    _json_response(handler, 202, {"ok": True, "challenge_id": challenge_id, "status": "otp_sent"})


def _handle_document_action_verify_otp(handler: BaseHTTPRequestHandler) -> None:
    if not _valid_request_origin(handler):
        _json_response(handler, 403, {"ok": False, "error": "invalid_origin"})
        return
    form = _read_form(handler)
    challenge_id = form.get("challenge_id", "").strip()
    otp = form.get("otp", "").strip()
    state = _cleanup_state(_load_state())
    challenge = state.get("challenges", {}).get(challenge_id)
    if not challenge or challenge.get("purpose") != "document_action" or not OTP_RE.fullmatch(otp):
        _json_response(handler, 401, {"ok": False, "error": "invalid_or_expired_otp"})
        return
    challenge["attempts"] = int(challenge.get("attempts", 0)) + 1
    if not hmac.compare_digest(challenge.get("otp_hash", ""), _hash(f"{challenge_id}:{otp}")):
        state["challenges"][challenge_id] = challenge
        _save_state(state)
        _json_response(handler, 401, {"ok": False, "error": "incorrect_otp"})
        return
    pending_payload = challenge.get("pending_payload") if isinstance(challenge.get("pending_payload"), dict) else {}
    validation = _validate_document_action_payload(pending_payload)
    if not validation:
        _json_response(handler, 422, {"ok": False, "error": "invalid_document_action"})
        return
    event_type, _deliverable_id, token, _metadata = validation
    event = build_document_event(
        pending_payload,
        token_ref=_document_event_token_ref(token),
        ip_address=handler.client_address[0] if handler.client_address else None,
        user_agent=handler.headers.get("User-Agent"),
    )
    event["metadata"].update({
        "otp_verified": True,
        "otp_challenge_id": challenge_id,
        "otp_channel_id": challenge.get("channel_id"),
        "otp_target_hash": _hash(str(challenge.get("target") or "")),
    })
    state.get("challenges", {}).pop(challenge_id, None)
    _save_state(state)
    _audit(event)
    _json_response(handler, 202, {"ok": True, "event_id": "queued", "status": event["status"], "event_type": event_type})


def _handle_document_action(handler: BaseHTTPRequestHandler) -> None:
    if not _valid_request_origin(handler):
        _json_response(handler, 403, {"ok": False, "error": "invalid_origin"})
        return
    payload = _read_json_body(handler)
    validation = _validate_document_action_payload(payload)
    if not validation:
        _json_response(handler, 422, {"ok": False, "error": "invalid_document_action"})
        return
    event_type, _deliverable_id, token, _metadata = validation
    if document_action_requires_otp(event_type):
        _json_response(handler, 401, {"ok": False, "error": "otp_required", "request_otp": "/api/document-actions/request-otp", "verify_otp": "/api/document-actions/verify-otp"})
        return
    if not document_action_allows_direct_post(event_type):
        _json_response(handler, 422, {"ok": False, "error": "unsupported_document_action"})
        return
    event = build_document_event(
        payload,
        token_ref=_document_event_token_ref(token),
        ip_address=handler.client_address[0] if handler.client_address else None,
        user_agent=handler.headers.get("User-Agent"),
    )
    _audit(event)
    _json_response(handler, 202, {"ok": True, "event_id": "queued", "status": event["status"], "event_type": event_type})


def _vapi_message(payload: dict[str, Any]) -> dict[str, Any]:
    message = payload.get("message") if isinstance(payload.get("message"), dict) else payload
    return message if isinstance(message, dict) else {}


def _vapi_call(payload: dict[str, Any]) -> dict[str, Any]:
    message = _vapi_message(payload)
    call = message.get("call") if isinstance(message.get("call"), dict) else payload.get("call")
    return call if isinstance(call, dict) else {}


def _vapi_payload_type(payload: dict[str, Any]) -> str | None:
    message = _vapi_message(payload)
    raw = message.get("type") or payload.get("type")
    return str(raw) if raw else None


def _tool_calls_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    message = _vapi_message(payload)
    calls = message.get("toolCalls") or message.get("tool_calls") or payload.get("toolCalls") or []
    return calls if isinstance(calls, list) else []


def _tool_name(call: dict[str, Any]) -> str:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    return str(function.get("name") or call.get("name") or "").strip()


def _tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    raw = function.get("arguments") or call.get("arguments") or {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            return {"raw": raw}
    return raw if isinstance(raw, dict) else {}


def _digits(value: Any) -> str:
    return re.sub(r"\D+", "", str(value or ""))


def _customer_context_path() -> Path:
    return Path(os.environ.get("VAPI_CUSTOMER_CONTEXT_FILE", str(EVENT_DIR / "customer-context.json")))


def _lookup_customer_context(args: dict[str, Any], call_context: dict[str, Any]) -> str:
    """Return a sanitized caller context snapshot for Sophie.

    The public callback service must not hold broad CRM credentials. A trusted
    Zeus-side worker can publish a narrow, sanitized JSON cache keyed by phone,
    email, name, or company. Sophie gets just enough context to continue a
    returning conversation while Zeus keeps the canonical CRM as source of truth.
    """
    path = _customer_context_path()
    if not path.exists():
        return "No encontré contexto previo publicado para este contacto. Continúa como lead nuevo y captura los datos mínimos para Zeus."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "No pude leer el contexto previo. Continúa como lead nuevo y registra el resumen para Zeus."
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        return "El contexto publicado no tiene formato válido. Continúa como lead nuevo y registra el resumen para Zeus."
    customer = call_context.get("customer") if isinstance(call_context.get("customer"), dict) else {}
    needles = {
        "phone": _digits(args.get("phone") or customer.get("number")),
        "email": str(args.get("email") or "").strip().lower(),
        "name": str(args.get("name") or "").strip().lower(),
        "company": str(args.get("company") or "").strip().lower(),
    }
    for rec in records:
        if not isinstance(rec, dict):
            continue
        phones = [_digits(x) for x in ([rec.get("phone")] + list(rec.get("phones") or []))]
        emails = [str(x).strip().lower() for x in ([rec.get("email")] + list(rec.get("emails") or [])) if x]
        names = [str(x).strip().lower() for x in [rec.get("name"), rec.get("contact_name")] if x]
        companies = [str(x).strip().lower() for x in [rec.get("company"), rec.get("organization")] if x]
        matched = (needles["phone"] and needles["phone"] in phones) or (needles["email"] and needles["email"] in emails) or (needles["name"] and needles["name"] in names) or (needles["company"] and needles["company"] in companies)
        if matched:
            summary = str(rec.get("summary") or rec.get("context") or "Hay contexto previo registrado.").strip()
            next_step = str(rec.get("next_step") or rec.get("pending_action") or "Confirmar el próximo paso con el cliente.").strip()
            return f"Contexto encontrado: {summary} Próximo paso pendiente: {next_step}"
    return "No encontré contexto previo para este identificador. Continúa como lead nuevo y captura los datos mínimos para Zeus."


def _handle_vapi_tools(handler: BaseHTTPRequestHandler) -> None:
    if not _vapi_authorized(handler):
        _json_response(handler, 401, {"ok": False, "error": "unauthorized"})
        return
    payload = _read_json_body(handler)
    calls = _tool_calls_from_payload(payload)
    results = []
    call_context = _vapi_call(payload)
    payload_type = _vapi_payload_type(payload)
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = _tool_name(call)
        tool_call_id = str(call.get("id") or call.get("toolCallId") or name or uuid.uuid4())
        args = _tool_arguments(call)
        status = "queued_for_agent_ingest" if name in VAPI_ALLOWED_TOOL_NAMES else "unknown_tool"
        event = {
            "event_type": "vapi_tool_call",
            "actor_type": "voice_call",
            "actor_ref": call_context.get("id"),
            "ip_address": handler.client_address[0] if handler.client_address else None,
            "user_agent": handler.headers.get("User-Agent"),
            "metadata": {"tool_name": name, "tool_call_id": tool_call_id, "arguments": args, "vapi_payload_type": payload_type, "call_id": call_context.get("id")},
            "status": status,
        }
        _audit(event)
        if status == "unknown_tool":
            results.append({"toolCallId": tool_call_id, "name": name, "error": f"Tool no soportada: {name}"})
        else:
            result = "Listo. Registré la información para que Zeus le dé seguimiento."
            if name == "get_customer_context":
                result = _lookup_customer_context(args, call_context)
            elif name == "schedule_followup":
                result = "Listo. Registré la solicitud de seguimiento para que Zeus la procese."
            elif name == "escalate_to_zeus":
                result = "Listo. Escalé el caso a Zeus para decisión y acción posterior."
            results.append({"toolCallId": tool_call_id, "name": name, "result": result})
    _json_response(handler, 200, {"results": results})


def _handle_vapi_webhook(handler: BaseHTTPRequestHandler) -> None:
    if not _vapi_authorized(handler):
        _json_response(handler, 401, {"ok": False, "error": "unauthorized"})
        return
    payload = _read_json_body(handler)
    message = _vapi_message(payload)
    event_type = str(message.get("type") or payload.get("type") or "vapi_webhook")
    call = _vapi_call(payload)
    status = str(message.get("status") or "").strip().lower()
    normalized_event_type = "vapi_" + event_type.replace("-", "_")
    if event_type == "status-update" and status == "ended" and message.get("artifact"):
        # Some Vapi calls provide the final transcript/artifact on status-update
        # without sending a separate end-of-call-report callback. Keep a distinct
        # audit event so the trusted ingestion side can persist final call data.
        normalized_event_type = "vapi_call_ended"
    _audit({
        "event_type": normalized_event_type,
        "actor_type": "voice_call",
        "actor_ref": call.get("id"),
        "ip_address": handler.client_address[0] if handler.client_address else None,
        "user_agent": handler.headers.get("User-Agent"),
        "metadata": payload,
        "status": "pending_agent_ingest",
    })
    _json_response(handler, 200, {"ok": True, "status": "queued"})


class Handler(BaseHTTPRequestHandler):
    server_version = f"{AGENT_NAME}DeliverySandboxEvents/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:  # keep stdout concise
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in {"/healthz", "/api/healthz"}:
            _json_response(self, 200, {"ok": True, "service": f"{AGENT_ID}-delivery-sandbox-events"})
            return
        if path in {"/api/user/session/check", "/user/session/check"}:
            if _session_from_request(self):
                self.send_response(204)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
            else:
                _json_response(self, 401, {"ok": False, "error": "unauthenticated"})
            return
        if path.startswith("/user"):
            _handle_user_get(self, path, query)
            return
        _json_response(self, 404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/api/voice/vapi/tools", "/voice/vapi/tools"}:
            try:
                _handle_vapi_tools(self)
            except ValueError as exc:
                _json_response(self, 413, {"ok": False, "error": str(exc)})
            except Exception:
                _json_response(self, 500, {"ok": False, "error": "vapi_tool_failed"})
            return
        if path in {"/api/voice/vapi/webhook", "/voice/vapi/webhook"}:
            try:
                _handle_vapi_webhook(self)
            except ValueError as exc:
                _json_response(self, 413, {"ok": False, "error": str(exc)})
            except Exception:
                _json_response(self, 500, {"ok": False, "error": "vapi_webhook_failed"})
            return
        if path in {"/api/document-actions/request-otp", "/document-actions/request-otp"}:
            try:
                _handle_document_action_request_otp(self)
            except ValueError as exc:
                _json_response(self, 413, {"ok": False, "error": str(exc)})
            except Exception:
                _json_response(self, 500, {"ok": False, "error": "document_action_otp_request_failed"})
            return
        if path in {"/api/document-actions/verify-otp", "/document-actions/verify-otp"}:
            try:
                _handle_document_action_verify_otp(self)
            except ValueError as exc:
                _json_response(self, 413, {"ok": False, "error": str(exc)})
            except Exception:
                _json_response(self, 500, {"ok": False, "error": "document_action_otp_verify_failed"})
            return
        if path in {"/api/document-actions", "/document-actions"}:
            try:
                _handle_document_action(self)
            except ValueError as exc:
                _json_response(self, 413, {"ok": False, "error": str(exc)})
            except Exception:
                _json_response(self, 500, {"ok": False, "error": "document_action_failed"})
            return
        if path in {"/user/request-otp", "/api/user/request-otp"}:
            try:
                _handle_request_otp(self)
            except ValueError as exc:
                _json_response(self, 413, {"ok": False, "error": str(exc)})
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": "otp_request_failed"})
            return
        if path in {"/user/verify", "/api/user/verify-otp"}:
            try:
                _handle_verify_otp(self)
            except ValueError as exc:
                _json_response(self, 413, {"ok": False, "error": str(exc)})
            except Exception:
                _json_response(self, 500, {"ok": False, "error": "otp_verify_failed"})
            return
        if path not in {"/events", "/api/events"}:
            _json_response(self, 404, {"ok": False, "error": "not_found"})
            return
        if not _valid_request_origin(self):
            _json_response(self, 403, {"ok": False, "error": "invalid_origin"})
            return

        try:
            payload = json.loads(_read_body(self).decode("utf-8"))
        except ValueError:
            _json_response(self, 413, {"ok": False, "error": "invalid_body_size"})
            return
        except Exception:
            _json_response(self, 400, {"ok": False, "error": "invalid_json"})
            return

        event_type = str(payload.get("event_type", "")).strip()
        deliverable_id = str(payload.get("deliverable_id", "")).strip()
        token = str(payload.get("token", "")).strip()
        actor_type = str(payload.get("actor_type", "client")).strip() or "client"
        metadata = payload.get("metadata") or {}

        if event_type not in ALLOWED_TYPES:
            _json_response(self, 422, {"ok": False, "error": "invalid_event_type"})
            return
        if event_type in COACH_EVENT_TYPES and not _session_from_request(self):
            _json_response(self, 401, {"ok": False, "error": "coach_session_required"})
            return
        if not deliverable_id:
            _json_response(self, 422, {"ok": False, "error": "missing_deliverable_id"})
            return
        if not token:
            _json_response(self, 422, {"ok": False, "error": "missing_token"})
            return
        if not isinstance(metadata, dict):
            _json_response(self, 422, {"ok": False, "error": "invalid_metadata"})
            return
        if not _workspace_matches_token(token, deliverable_id, metadata):
            _json_response(self, 403, {"ok": False, "error": "invalid_token_scope"})
            return
        if event_type in OTP_REQUIRED_DOCUMENT_EVENT_TYPES:
            _json_response(self, 401, {"ok": False, "error": "otp_required", "request_otp": "/api/document-actions/request-otp", "verify_otp": "/api/document-actions/verify-otp"})
            return

        event = {
            "event_type": event_type,
            "deliverable_id": deliverable_id,
            "token_ref": token[:10] + "..." if len(token) > 10 else token,
            "actor_type": actor_type,
            "actor_ref": payload.get("actor_ref"),
            "ip_address": self.client_address[0] if self.client_address else None,
            "user_agent": self.headers.get("User-Agent"),
            "comment": payload.get("comment"),
            "metadata": metadata,
            "status": "pending_agent_ingest",
        }
        _audit(event)
        _json_response(self, 202, {"ok": True, "event_id": "queued", "status": event["status"]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
'''


def public_index(agent_name: str) -> str:
    safe = agent_name.replace("<", "").replace(">", "")
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>{safe} Workspace</title>
<style>:root{{color-scheme:dark;--fg:#f6fbff;--muted:#9fb3c8;--line:#264357}}body{{margin:0;min-height:100vh;display:grid;place-items:center;font-family:Inter,system-ui,sans-serif;background:radial-gradient(circle at 20% 0%,#143044 0,#071016 50%,#03070a 100%);color:var(--fg)}}main{{width:min(920px,92vw);border:1px solid var(--line);border-radius:28px;padding:34px;background:rgba(14,27,37,.78);box-shadow:0 24px 80px rgba(0,0,0,.35)}}h1{{margin:0 0 12px;font-size:clamp(34px,6vw,64px);letter-spacing:-.06em}}p{{color:var(--muted);line-height:1.6;font-size:18px}}a{{color:#071016;background:linear-gradient(135deg,#20d17d,#6de2ff);display:inline-block;padding:13px 18px;border-radius:999px;font-weight:800;text-decoration:none;margin-top:16px}}</style>
</head><body><main><h1>{safe}</h1><p>Superficie pública de entregas y espacios de trabajo tokenizados. Los enlaces privados del agente están protegidos por OTP.</p><a href="/user/">Entrar al dashboard privado</a></main></body></html>
"""


def compose(agent_id: str, agent_name: str, bind_ip: str, public_port: int, event_port: int, allowed_hosts: str) -> str:
    return f"""services:
  delivery-sandbox-web:
    image: nginx:1.27-alpine
    container_name: {agent_id}-delivery-sandbox-web
    restart: unless-stopped
    ports:
      - "{bind_ip}:{public_port}:80"
    volumes:
      - ./public:/usr/share/nginx/html:ro
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    networks:
      - delivery_sandbox
    tmpfs:
      - /var/cache/nginx
      - /var/run
      - /tmp
    security_opt:
      - no-new-privileges:true

  delivery-sandbox-events:
    build: ./event-server
    container_name: {agent_id}-delivery-sandbox-events
    restart: unless-stopped
    # Match the standard derived-agent runtime uid/gid on SitioUno VMs so
    # bind-mounted event/OTP state remains writable without privileged writes.
    user: "1001:1002"
    environment:
      AGENT_ID: "{agent_id}"
      AGENT_NAME: "{agent_name}"
      ALLOWED_HOSTS: "{allowed_hosts}"
      VAPI_SERVER_AUTH_TOKEN: "${{VAPI_SERVER_AUTH_TOKEN:-}}"
    env_file:
      - path: ./vapi-public.env
        required: false
    ports:
      - "{bind_ip}:{event_port}:8080"
    volumes:
      - ./events:/data/events
      - ./public:/data/public:ro
      - ./user-data:/data/user-data:ro
    networks:
      - delivery_sandbox
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL

networks:
  delivery_sandbox:
    name: {agent_id}_delivery_sandbox
    driver: bridge
"""


def write(path: Path, content: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish SitioUno commercial delivery sandbox assets")
    parser.add_argument("manifest", help="Path to agent.yaml")
    parser.add_argument("--target", required=True, help="Delivery sandbox directory")
    parser.add_argument("--bind-ip", required=True, help="IP to bind public/event ports")
    parser.add_argument("--public-port", type=int, default=9323)
    parser.add_argument("--event-port", type=int, default=9324)
    parser.add_argument("--allowed-hosts", default="", help="Comma-separated public hostnames allowed for dashboard/event POST origin checks")
    args = parser.parse_args()

    manifest = load_simple_yaml(args.manifest)
    agent_id = str(manifest.get("agent_id", "commercial-agent")).replace("_", "-")
    agent_name = str(manifest.get("agent_name", agent_id.title()))
    allowed_hosts = str(args.allowed_hosts or manifest.get("public_host") or manifest.get("domain") or "")
    target = Path(args.target)

    write(target / "docker-compose.yml", compose(agent_id, agent_name, args.bind_ip, args.public_port, args.event_port, allowed_hosts))
    write(target / "nginx.conf", NGINX_CONF)
    write(target / "public/index.html", public_index(agent_name))
    write(target / "event-server/Dockerfile", DOCKERFILE)
    write(target / "event-server/server.py", SERVER_PY)
    write(target / "event-server/delivery_document_actions.py", Path(__file__).with_name("delivery_document_actions.py").read_text(encoding="utf-8"))
    (target / "events").mkdir(parents=True, exist_ok=True)
    (target / "user-data").mkdir(parents=True, exist_ok=True)
    if not (target / "vapi-public.env").exists():
        write(target / "vapi-public.env", "# Runtime-only secrets for public callback surface. Do not commit generated files.\n# VAPI_SERVER_AUTH_TOKEN=***\n", 0o600)
    print(f"published delivery sandbox for {agent_id} at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
