"""CSP for the browser-hosted Desktop, independently of its preview documents."""

import base64
import hashlib
from html.parser import HTMLParser

from fastapi import Request


class _ScriptHashes(HTMLParser):
    """Hash the trusted build's inline scripts and the server's bootstrap verbatim."""

    def __init__(self, html: str):
        super().__init__(convert_charrefs=False)
        self.script = None
        self.hashes = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "script" and "src" not in dict(attrs):
            self.script = ""

    def handle_data(self, data):
        if self.script is not None:
            self.script += data

    def handle_endtag(self, tag):
        if tag == "script" and self.script is not None:
            digest = base64.b64encode(hashlib.sha256(self.script.encode("utf-8")).digest()).decode("ascii")
            self.hashes.append(f"'sha256-{digest}'")
            self.script = None


def page_headers(request: Request, html: str, headers: dict) -> dict:
    """Only gateway-served Webapp HTML gets this policy; native Desktop is not served here."""
    if getattr(request.app.state, "ui_surface", None) != "webapp":
        return headers
    hashes = " ".join(_ScriptHashes(html).hashes)
    # Keep fetch and media sources open for voice, catalogs, Radio, update notes,
    # and local servers or sockets behind a Host-rewriting proxy.
    policy = "; ".join((
        "default-src 'none'",
        f"script-src 'self' blob: 'wasm-unsafe-eval' {hashes}",
        "script-src-attr 'none'",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' data: https://fonts.gstatic.com",
        "connect-src * ws: wss: data: blob:",
        "img-src 'self' data: blob: https: http:",
        "media-src * data: blob:",
        "worker-src 'self' blob:",
        "frame-src 'self' blob: data: https: http:",
        "manifest-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    ))
    return {**headers, "Content-Security-Policy": policy, "X-Content-Type-Options": "nosniff"}
