"""Shared markup for the browser-facing OAuth callback pages.

Every OAuth flow in Hermes ends with the provider redirecting the user's
browser at a page we serve — a loopback listener (``hermes mcp login``, the
TUI/desktop gateway flows, Spotify, honcho) or the dashboard's callback route.
Each of those sites grew its own one-line ``<h1>`` string, so the last thing a
user sees after authorizing is unstyled Times New Roman in the top-left corner
of a blank tab. This module is the single place that markup lives.

Branding is user-supplied and deliberately lives OUTSIDE the repo so it
survives ``hermes update``: drop an image at
``$HERMES_HOME/branding/oauth-logo.png`` (``.svg``/``.jpg``/``.webp``/``.gif``
also work) and it is inlined into the page as a ``data:`` URI. Add
``oauth-logo-dark.*`` to swap in a different mark under
``prefers-color-scheme: dark``. Inlining rather than linking is required, not
cosmetic: the callback page must render with no network and no second request
back to a listener that closes itself the moment the redirect lands.

Callers pass plain text. Headings, messages, and provider-supplied error
strings are HTML-escaped here, so a hostile ``?error=`` parameter cannot inject
markup into the page.
"""

from __future__ import annotations

import base64
import html
import logging
from pathlib import Path
from typing import Literal

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

Status = Literal["success", "error", "pending"]

# Extensions we will inline, in preference order: vector first, then the
# lossless raster formats, then lossy. Keyed to the media type so the data URI
# and the sniff share one table.
_LOGO_MEDIA_TYPES: dict[str, str] = {
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
}

# A logo is inlined into every callback page, so an accidentally-dropped
# multi-megabyte asset would be pasted into the response in base64 (~4/3 size).
# Skip anything implausible for a wordmark rather than serve a huge page.
_MAX_LOGO_BYTES = 1024 * 1024

_STATUS_GLYPHS: dict[Status, str] = {
    "success": "&#10003;",  # check
    "error": "&#10005;",  # cross
    "pending": "&#8230;",  # ellipsis
}

_STYLES = """
:root {
  color-scheme: light dark;
  --bg: #eef1f6;
  --card: #ffffff;
  --fg: #16233a;
  --muted: #5a6b85;
  --edge: rgba(22, 35, 58, 0.10);
  --shadow: 0 1px 2px rgba(16, 24, 40, 0.05), 0 16px 40px -16px rgba(16, 24, 40, 0.22);
  --success: #4f9d3f;
  --success-bg: rgba(109, 190, 95, 0.16);
  --error: #c0392b;
  --error-bg: rgba(192, 57, 43, 0.12);
  --pending: #3d6bb3;
  --pending-bg: rgba(61, 107, 179, 0.12);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0b1120;
    --card: #141d2f;
    --fg: #e9eef7;
    --muted: #9aabc4;
    --edge: rgba(255, 255, 255, 0.09);
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.3), 0 16px 40px -16px rgba(0, 0, 0, 0.6);
    --success: #7ecd6c;
    --success-bg: rgba(109, 190, 95, 0.18);
    --error: #f0837a;
    --error-bg: rgba(240, 131, 122, 0.14);
    --pending: #8fb2e8;
    --pending-bg: rgba(143, 178, 232, 0.14);
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
  background: var(--bg);
  color: var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter,
        Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
main {
  width: 100%;
  max-width: 25rem;
  padding: 2.75rem 2rem 2.5rem;
  border: 1px solid var(--edge);
  border-radius: 16px;
  background: var(--card);
  box-shadow: var(--shadow);
  text-align: center;
}
.logo {
  display: block;
  width: auto;
  max-width: 11.5rem;
  max-height: 4.25rem;
  margin: 0 auto 2rem;
}
.logo-dark { display: none; }
@media (prefers-color-scheme: dark) {
  .has-dark-logo .logo-light { display: none; }
  .has-dark-logo .logo-dark { display: block; }
}
.glyph {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.75rem;
  height: 2.75rem;
  margin: 0 auto 1.125rem;
  border-radius: 50%;
  font-size: 1.25rem;
  line-height: 1;
}
.glyph-success { color: var(--success); background: var(--success-bg); }
.glyph-error { color: var(--error); background: var(--error-bg); }
.glyph-pending { color: var(--pending); background: var(--pending-bg); }
h1 {
  margin: 0 0 0.5rem;
  font-size: 1.1875rem;
  font-weight: 600;
  letter-spacing: -0.01em;
}
p { margin: 0; color: var(--muted); }
"""


def _logo_data_uri(path: Path) -> str | None:
    """Read *path* and return it as a ``data:`` URI, or None if unusable.

    Never raises: a missing, oversized, or unreadable branding file degrades to
    a logo-less page rather than failing the OAuth callback the user is
    currently waiting on.
    """
    media_type = _LOGO_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        return None
    try:
        if path.stat().st_size > _MAX_LOGO_BYTES:
            logger.warning(
                "Skipping OAuth callback logo %s: larger than %d bytes",
                path, _MAX_LOGO_BYTES,
            )
            return None
        raw = path.read_bytes()
    except OSError as exc:
        logger.debug("Could not read OAuth callback logo %s: %s", path, exc)
        return None
    if not raw:
        return None
    return f"data:{media_type};base64,{base64.b64encode(raw).decode('ascii')}"


def _find_logo(stem: str) -> str | None:
    """Find ``$HERMES_HOME/branding/<stem>.<ext>`` and inline it."""
    try:
        branding_dir = get_hermes_home() / "branding"
    except Exception as exc:  # pragma: no cover - get_hermes_home is defensive
        logger.debug("Could not resolve branding directory: %s", exc)
        return None
    for extension in _LOGO_MEDIA_TYPES:
        data_uri = _logo_data_uri(branding_dir / f"{stem}{extension}")
        if data_uri:
            return data_uri
    return None


def _logo_markup() -> tuple[str, str]:
    """Return ``(body_class, logo_html)`` for the active branding, if any.

    Resolved per call rather than cached: the pages render at most once per
    OAuth flow, and a process-wide cache would pin one profile's branding for
    every other profile the gateway serves.
    """
    light = _find_logo("oauth-logo")
    if not light:
        return "", ""
    dark = _find_logo("oauth-logo-dark")
    if not dark:
        return "", f'<img class="logo logo-light" src="{light}" alt="">'
    return (
        ' class="has-dark-logo"',
        f'<img class="logo logo-light" src="{light}" alt="">'
        f'<img class="logo logo-dark" src="{dark}" alt="">',
    )


def render_callback_page(
    heading: str,
    message: str,
    *,
    status: Status = "success",
    auto_close: bool = False,
) -> str:
    """Render a centered, branded OAuth callback page.

    Args:
        heading: Short outcome line, e.g. ``"Authorization received"``. Used as
            the page title too. Escaped.
        message: Supporting sentence telling the user what to do next. Escaped.
        status: Picks the glyph and its accent color.
        auto_close: Attempt ``window.close()`` shortly after paint. Only honor
            this for tabs Hermes opened itself (``window.open``-created tabs are
            the only ones browsers let script close).

    Returns:
        A complete, self-contained HTML document.
    """
    safe_heading = html.escape(heading)
    safe_message = html.escape(message)
    glyph = _STATUS_GLYPHS.get(status, _STATUS_GLYPHS["success"])
    body_class, logo = _logo_markup()
    closer = (
        "<script>setTimeout(function(){window.close()},1200)</script>"
        if auto_close
        else ""
    )
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{safe_heading}</title>\n"
        f"<style>{_STYLES}</style>\n"
        "</head>\n"
        f"<body{body_class}>\n"
        "<main>\n"
        f"{logo}\n"
        f'<div class="glyph glyph-{status}" aria-hidden="true">{glyph}</div>\n'
        f"<h1>{safe_heading}</h1>\n"
        f"<p>{safe_message}</p>\n"
        "</main>\n"
        f"{closer}\n"
        "</body>\n"
        "</html>\n"
    )


def render_callback_page_bytes(
    heading: str,
    message: str,
    *,
    status: Status = "success",
    auto_close: bool = False,
) -> bytes:
    """UTF-8 encoded :func:`render_callback_page`, for ``wfile.write``."""
    return render_callback_page(
        heading, message, status=status, auto_close=auto_close
    ).encode("utf-8")
