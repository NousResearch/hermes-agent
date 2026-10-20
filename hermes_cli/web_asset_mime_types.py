"""MIME types for served web assets, normalised in the serving layer.

Starlette's ``StaticFiles`` and ``FileResponse`` derive ``Content-Type`` from
``mimetypes.guess_type``. On Windows, ``mimetypes.init()`` reads the registry,
where installers (or user-scoped ``HKCU\\Software\\Classes`` entries) can map
``.js``/``.mjs`` to ``text/plain``. Browsers then refuse the dashboard's ESM
bundle under strict module MIME checking and the SPA stays blank (#28987).

Fix: pin the types we serve with explicit ``mimetypes.add_type`` calls so the
served values are platform-independent, regardless of the host map.
"""

from __future__ import annotations

import mimetypes

# RFC 9239: ``text/javascript`` is the IANA-blessed JavaScript type and the
# WHATWG HTML spec's canonical essence for module scripts.
WEB_ASSET_MIME_TYPES: dict[str, str] = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".cjs": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
    ".webp": "image/webp",
    ".webmanifest": "application/manifest+json",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ico": "image/x-icon",
}


def normalize_web_asset_mime_types() -> None:
    """Pin web-standard ``Content-Type`` values for dashboard assets.

    Idempotent; overrides any host-provided (e.g. Windows registry) mapping.
    Called at ``web_server_dashboard`` import so every ``StaticFiles`` mount
    and ``FileResponse`` built afterwards sees the corrected map (#28987).
    """
    mimetypes.init()
    for ext, mime in WEB_ASSET_MIME_TYPES.items():
        mimetypes.add_type(mime, ext, strict=True)
