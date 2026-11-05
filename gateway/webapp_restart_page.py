"""The page a browser gets for a Webapp document request the gateway cannot serve yet.

While the gateway drains for a restart or ``hermes update``, ``gateway/run_api.py`` answers every
request it refuses with ``503 {"error":"gateway_not_ready"}``, and after an update whose Webapp rebuild
failed ``_serve_index`` answers ``/`` with a 503 JSON reason. A browser tab that reloads then shows only
that JSON, with nothing telling the person what is happening or that it will pass. API callers keep the
JSON; a request for a document (``GET``, ``Accept: text/html``, outside ``/api``) gets this page
instead: same status, a plain sentence, and a reload once the page itself answers again.
"""
from __future__ import annotations

import html

# What the tab can be waiting for, in the person's words; the key is what refused the request.
_COPY = {
    "restarting": ("Hermes is restarting",
                   "The Hermes backend is restarting, usually to finish an update. This page reloads by "
                   "itself as soon as it is back. Running chats continue after it returns."),
    "not_rebuilt": ("Hermes was updated, but its web app was not rebuilt",
                    "The update finished, but the web app could not be rebuilt for the new version. Run "
                    "<code>hermes webapp</code> on the machine running Hermes; this page reloads once it is "
                    "rebuilt."),
}

# Asks for this same page again (not /api/health: health answers 200 during a restart's after-turn
# wait, so it would reload straight back onto this page) and reloads once the answer is the app.
_SCRIPT = (
    "(function(){var tries=0;function later(){setTimeout(poll,Math.min(15000,1000+tries*1000));}"
    "function poll(){tries++;fetch(location.href,{cache:'no-store',credentials:'same-origin',"
    "headers:{accept:'text/html'}}).then(function(r){if(r.ok){location.reload();}else{later();}},later);}"
    "later();})();"
)


def wants_document(scope) -> bool:
    """True for a browser's page load: a GET outside ``/api`` whose ``Accept`` names ``text/html``.
    A browser sends that header on navigation; ``fetch``/XHR (every Desktop API call) does not."""
    if scope.get("type") != "http" or scope.get("method") != "GET":
        return False
    path = scope.get("path") or ""
    if path == "/api" or "/api/" in path:
        return False
    accept = next((value for key, value in scope.get("headers") or () if key == b"accept"), b"")
    return b"text/html" in accept


def restart_page(kind: str, detail: str = "") -> tuple[str, dict[str, str]]:
    """``(html, headers)`` for the 503 page of *kind* (a key of ``_COPY``); *detail* is shown as text."""
    title, body = _COPY[kind]
    extra = f'<p class="d">{html.escape(detail)}</p>' if detail else ""
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>"
        "body{margin:0;min-height:100vh;display:grid;place-items:center;background:#111113;color:#e7e5df;"
        "font:15px/1.6 system-ui,-apple-system,'Segoe UI',sans-serif}"
        "main{max-width:30rem;padding:2rem;text-align:center}h1{font-size:1.25rem;font-weight:600;margin:0 0 .75rem}"
        "p{margin:0 0 .75rem;color:#b8b5ad}.d{font-size:.8rem;color:#8a877f}"
        "code{font:.9em ui-monospace,Menlo,monospace;background:#26262a;padding:1px 5px;border-radius:4px}"
        "@media(prefers-color-scheme:light){body{background:#faf9f5;color:#1b1b1a}p{color:#4a4945}"
        "code{background:#ecebe5}}</style></head>"
        f"<body><main><h1>{html.escape(title)}</h1><p>{body}</p>{extra}</main>"
        f"<script>{_SCRIPT}</script></body></html>"
    )
    headers = {
        "Retry-After": "2",
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Content-Security-Policy": ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                                    "connect-src 'self'; base-uri 'none'; form-action 'none'; "
                                    "frame-ancestors 'none'"),
        "X-Content-Type-Options": "nosniff",
    }
    return page, headers
