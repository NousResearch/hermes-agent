"""The page a browser gets for a Webapp document request the gateway cannot serve yet.

While the gateway drains for a restart or ``hermes update``, ``gateway/run_api.py`` answers every
request it refuses with ``503 {"error":"gateway_not_ready"}``, and after an update whose Webapp rebuild
failed ``_serve_index`` answers ``/`` with a 503 JSON reason. A tab reloaded then showed only that JSON.
API callers keep the JSON; a browser's page load gets the same 503 as a page that says what is happening
and reloads once the app answers again.
"""
from __future__ import annotations

import html
import json

from starlette.responses import HTMLResponse

from hermes_cli.webapp_page_policy import _ScriptHashes

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
_NO_STORE = "no-store, no-cache, must-revalidate"  # same value as web_server_dashboard._NO_STORE

# Polls the app root, never this URL: a page load with a one-time effect (``/auth/callback``'s code)
# must run once, on the reload. The root is found by stripping PATH (the path the gateway saw) from the
# browser's path, which keeps a proxy prefix. It reloads once the root answers OK, or answers another
# page kind (the restart is over but the bundle is stale: show that page instead).
_SCRIPT = (
    "(function(){var p=location.pathname,s=PATH,base=p.slice(-s.length)===s?p.slice(0,p.length-s.length):'';"
    "var tries=0;function later(){setTimeout(poll,Math.min(15000,1000+tries*1000));}"
    "function poll(){tries++;fetch(base+'/',{cache:'no-store',credentials:'same-origin',"
    "headers:{accept:'text/html'}}).then(function(r){var k=r.headers.get('" + _KIND_HEADER.lower() + "');"
    "if(r.ok||(k&&k!==KIND)){location.reload();}else{later();}},later);}"
    "later();})();"
)


def _wants_document(scope) -> bool:
    """A browser's page load: a GET outside ``/api`` whose ``Accept`` names ``text/html``.
    A browser sends that on navigation; ``fetch``/XHR (every Desktop API call) does not.
    ``/api/`` anywhere counts, so a proxy that keeps its prefix (``/hermes/api/...``) stays JSON."""
    if scope.get("type") != "http" or scope.get("method") != "GET":
        return False
    path = scope.get("path") or ""
    if path == "/api" or "/api/" in path:
        return False
    accept = next((value for key, value in scope.get("headers") or () if key == b"accept"), b"")
    return b"text/html" in accept


def document_refusal(scope, kind: str) -> HTMLResponse | None:
    """The 503 page of *kind* (a key of ``_COPY``) for a browser's page load, else None (the
    caller answers JSON)."""
    if not _wants_document(scope):
        return None
    title, body = _COPY[kind]
    path = json.dumps(scope.get("path") or "/").replace("</", "<\\/")
    script = _SCRIPT.replace("PATH", path, 1).replace("KIND", json.dumps(kind), 1)
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
        f"<body><main><h1>{html.escape(title)}</h1><p>{body}</p></main>"
        f"<script>{script}</script></body></html>"
    )
    return HTMLResponse(page, status_code=503, headers={
        "Retry-After": "2",
        "Cache-Control": _NO_STORE,
        "Content-Security-Policy": (f"default-src 'none'; script-src {_ScriptHashes(page).hashes[0]}; "
                                    "style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; "
                                    "form-action 'none'; frame-ancestors 'none'"),
        "X-Content-Type-Options": "nosniff",
        _KIND_HEADER: kind,
    })
