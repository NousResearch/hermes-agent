"""The Content-Security-Policy the gateway's Webapp page (the Desktop renderer in a browser) is served under.

Each source is one the built renderer (``apps/desktop/dist-webapp``) uses:

- ``script-src``: its entry and lazy chunks (``'self'``); runtime plugins and the SDK shims they import, which
  load as ``blob:`` modules (``contrib/runtime-loader.ts``); the Oniguruma WebAssembly code highlighting
  compiles (``'wasm-unsafe-eval'``); and inline scripts (``'unsafe-inline'``). Inline scripts cannot be
  narrowed to a nonce: the page's own pre-paint and bootstrap scripts could carry one, but the live HTML
  previews (``::preview`` and artifacts) run their scripts in sandboxed ``srcdoc`` frames, which inherit this
  policy, and a nonce would stop them. No ``'unsafe-eval'``: the bundle evaluates no strings.
- ``style-src``: its stylesheets, the inline styles React, KaTeX, Mermaid and Shiki write, and a theme's
  Google Fonts stylesheet; ``font-src`` the bundled fonts, ``data:`` fonts and Google's font files.
- ``connect-src``: the gateway's REST routes and its sockets on this same origin (``'self'`` covers
  ``ws:``/``wss:`` there), ``blob:``/``data:`` reads, and the public HTTPS APIs plugins call (Radio's station
  directory and track lookups).
- ``img-src``/``media-src``: gateway media, ``data:``/``blob:`` previews and recordings, images a message links
  and Radio's HTTPS streams; ``worker-src`` same-origin and ``blob:`` workers.
- ``frame-src``: previews in ``blob:``/``data:``/``srcdoc`` frames, embeds (YouTube, Spotify, maps, posts) and
  the sandboxed preview of a site (``http:`` for a local dev server).

Everything else stays shut: no plugins (``object-src``), no ``<base>`` or form posts to another origin, and no
page may frame the Webapp (``frame-ancestors``).
"""
from __future__ import annotations

CONTENT_SECURITY_POLICY = "; ".join((
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' blob: 'wasm-unsafe-eval'",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' data: https://fonts.gstatic.com",
    "connect-src 'self' blob: data: https:",
    "img-src 'self' data: blob: https: http:",
    "media-src 'self' data: blob: https:",
    "worker-src 'self' blob:",
    "frame-src 'self' blob: data: https: http:",
    "manifest-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
))


def page_headers(state, headers: dict) -> dict:
    """*headers* for the renderer's page on *state*'s app, with :data:`CONTENT_SECURITY_POLICY` when that app
    serves the Webapp. The dashboard's page keeps main's headers."""
    if getattr(state, "ui_surface", "dashboard") != "webapp":
        return headers
    return {**headers, "Content-Security-Policy": CONTENT_SECURITY_POLICY}
