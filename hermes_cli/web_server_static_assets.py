"""Static asset lookups the SPA catch-all must not answer with ``index.html``.

The sign-in page (``dashboard_auth/login_page.py``) loads its brand fonts from ``/fonts/``. Those
files ship in the dashboard's ``web/public`` tree, which the dashboard build copies into its own
dist, but the Webapp serves ``apps/desktop/dist-webapp`` and has no ``fonts/`` dir. Both surfaces
run from the same tree, so a font missing from the served dist is read from ``web/public`` itself.
"""
from __future__ import annotations

from pathlib import Path

# Requests under these prefixes are files, never client-side routes, so a miss is a 404 rather
# than the SPA shell (a font answered with HTML renders in a fallback face, silently).
STATIC_ASSET_PREFIXES: tuple[str, ...] = ("assets/", "ds-assets/", "fonts/", "fonts-terminal/")
_PUBLIC_FONT_DIRS: tuple[str, ...] = ("fonts", "fonts-terminal")
_WEB_PUBLIC = Path(__file__).resolve().parent.parent / "web" / "public"


def is_static_asset_path(full_path: str) -> bool:
    return full_path.startswith(STATIC_ASSET_PREFIXES)


def public_font_file(dist: Path, full_path: str) -> Path | None:
    """The served dist's copy of ``fonts/<name>`` (or ``fonts-terminal/<name>``), else the
    ``web/public`` copy the dashboard build takes it from; ``None`` when neither has it."""
    folder, _, name = full_path.partition("/")
    if folder not in _PUBLIC_FONT_DIRS or not name:
        return None
    for root in (dist / folder, _WEB_PUBLIC / folder):
        candidate = root / name
        if candidate.is_file() and candidate.resolve().is_relative_to(root.resolve()):
            return candidate
    return None
