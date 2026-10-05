"""MIME types the served front-ends need, independent of the host's registry.

``mimetypes`` answers out of the OS registry on Windows (``HKCR\\.js`` and friends), and that
mapping can be missing on a real machine. Starlette's ``StaticFiles``/``FileResponse`` then serve
the dashboard bundle with whatever the host says, so the SPA's ES modules go out as
``text/plain`` and a browser refuses to execute them ("Expected a JavaScript module script but
the server responded with a MIME type of 'text/plain'") — the dashboard renders blank at :9119.

Registering the types here makes the served type a property of the app rather than of the host.
"""

from __future__ import annotations

import mimetypes

# The dashboard bundle's extensions. ``.mjs`` is what Vite emits for lazy chunks on some builds
# and ``.css`` is rewritten by the SPA route, so both are served from the same directory.
STATIC_MIME_TYPES = (
    (".js", "application/javascript"),
    (".mjs", "application/javascript"),
    (".css", "text/css"),
)


def register_static_mime_types() -> None:
    """Register the bundle's MIME types globally; idempotent per process."""
    for ext, mime in STATIC_MIME_TYPES:
        if mimetypes.guess_type("probe" + ext)[0] != mime:
            mimetypes.add_type(mime, ext)
