"""The page a browser gets for a Webapp document request the gateway cannot serve yet.

Two refusals answer 503: ``gateway/run_api.py`` while the gateway drains for a restart or
``hermes update``, and ``_serve_index`` while the Webapp bundle is stale after a failed rebuild. API
callers get the JSON body; a browser's page load gets a page that says what is happening and reloads
once the app answers again.
"""
from __future__ import annotations

import base64
import hashlib
import html

from starlette.responses import HTMLResponse

from hermes_cli.dashboard_auth.prefix import normalise_prefix

# Title and body per page kind. The body is trusted HTML (it may hold <code>); the title is escaped.
_COPY = {
    "restarting": ("Hermes is restarting",
                   "The Hermes backend is restarting or still starting, usually to finish an update. This "
                   "page reloads by itself once Hermes answers again."),
    "not_rebuilt": ("Hermes was updated, but its web app was not rebuilt",
                    "Run <code>hermes webapp</code> on the machine running Hermes. This page reloads by "
                    "itself once the web app is rebuilt."),
}
_KIND_HEADER = "X-Hermes-Restart-Page"
_NO_STORE = "no-store, no-cache, must-revalidate"  # the Cache-Control value web_server_dashboard sends

# Polls the app root, never this URL: a page load with a one-time effect (``/auth/callback``'s code)
# must run once, on the reload. The root is the proxy prefix the SPA's base path comes from
# (``X-Forwarded-Prefix``) plus ``/``. It reloads once the root answers OK, redirects (the app is up but
# signed out; following the redirect would start a sign-in and replace the callback's PKCE state), or
# answers another page kind (the restart is over but the bundle is stale: show that page instead).
# A redirect may also be the proxy's own (``/hermes/`` -> ``/hermes``) while Hermes is still down, so
# the poll count survives the reload in sessionStorage: a page that comes back refused within 30 s
# keeps backing off instead of restarting at one second. An older count (a later restart) is ignored.
# The root and kind come from ``<main>``'s data attributes, so the script and its hash are constant.
_SCRIPT = (
    "(function(){var c=document.querySelector('main').dataset,K='hermes-restart-page-tries',tries=0;"
    "function store(f){try{f(window.sessionStorage);}catch(e){}}"
    "store(function(s){var v=(s.getItem(K)||'').split(':');if(Date.now()-v[1]<30000)tries=+v[0]||0;});"
    "function later(){setTimeout(poll,Math.min(15000,1000+tries*1000));}"
    "function poll(){tries++;store(function(s){s.setItem(K,tries+':'+Date.now());});"
    "fetch(c.root,{cache:'no-store',credentials:'same-origin',redirect:'manual',"
    "headers:{accept:'text/html'}}).then(function(r){var k=r.headers.get('" + _KIND_HEADER.lower() + "');"
    "if(r.ok||r.type==='opaqueredirect'||(k&&k!==c.kind)){location.reload();}else{later();}},later);}"
    "later();})();"
)
_CSP = (
    "default-src 'none'; "
    f"script-src 'sha256-{base64.b64encode(hashlib.sha256(_SCRIPT.encode()).digest()).decode()}'; "
    "style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


def _header(scope, name: bytes) -> bytes:
    return next((value for key, value in scope.get("headers") or () if key == name), b"")


def _wants_document(scope) -> bool:
    """A browser's page load: a GET outside ``/api`` whose ``Accept`` names ``text/html``.
    A browser sends that on navigation; ``fetch``/XHR (every Desktop API call) does not.
    ``/api/`` anywhere counts, so a proxy that keeps its prefix (``/hermes/api/...``) stays JSON."""
    if scope.get("type") != "http" or scope.get("method") != "GET":
        return False
    path = scope.get("path") or ""
    if path == "/api" or "/api/" in path:
        return False
    return b"text/html" in _header(scope, b"accept")


def document_refusal(scope, kind: str) -> HTMLResponse | None:
    """The 503 page of *kind* (a key of ``_COPY``) for a browser's page load, else None (the
    caller answers JSON)."""
    if not _wants_document(scope):
        return None
    title, body = _COPY[kind]
    root = normalise_prefix(_header(scope, b"x-forwarded-prefix").decode("latin-1")) + "/"
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>"
        "body{margin:0;min-height:100vh;display:grid;place-items:center;background:#111113;color:#e7e5df;"
        "font:15px/1.6 system-ui,-apple-system,'Segoe UI',sans-serif}"
        "main{max-width:30rem;padding:2rem;text-align:center}h1{font-size:1.25rem;font-weight:600;margin:0 0 .75rem}"
        "p{margin:0;color:#b8b5ad}"
        "code{font:.9em ui-monospace,Menlo,monospace;background:#26262a;padding:1px 5px;border-radius:4px}"
        "@media(prefers-color-scheme:light){body{background:#faf9f5;color:#1b1b1a}p{color:#4a4945}"
        "code{background:#ecebe5}}</style></head>"
        f'<body><main data-root="{html.escape(root)}" data-kind="{html.escape(kind)}">'
        f"<h1>{html.escape(title)}</h1><p>{body}</p></main>"
        f"<script>{_SCRIPT}</script></body></html>"
    )
    return HTMLResponse(page, status_code=503, headers={
        "Retry-After": "2",
        "Cache-Control": _NO_STORE,
        "Content-Security-Policy": _CSP,
        "X-Content-Type-Options": "nosniff",
        _KIND_HEADER: kind,
    })
