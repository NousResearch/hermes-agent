"""Regression coverage for dashboard PWA installability metadata."""

import json
import re
import struct
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"
PUBLIC_DIR = WEB_DIR / "public"


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    # PNG IHDR stores width/height as big-endian uint32 after signature + chunk length/type.
    return struct.unpack(">II", data[16:24])


def test_dashboard_index_exposes_pwa_metadata():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

    assert '<link rel="manifest" href="/manifest.json" />' in html
    assert '<link rel="apple-touch-icon" sizes="180x180" href="/icon-180.png" />' in html
    assert 'viewport-fit=cover' in html
    assert '<meta name="theme-color" content="#041C1C" />' in html
    assert '<meta name="apple-mobile-web-app-capable" content="yes" />' in html
    assert '<meta name="apple-mobile-web-app-title" content="Hermes" />' in html
    assert '<meta name="apple-mobile-web-app-status-bar-style" content="default" />' in html
    assert re.search(r"<title>\s*Hermes Agent - Dashboard\s*</title>", html)


def test_dashboard_manifest_is_installable_and_scoped():
    manifest = json.loads((PUBLIC_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["name"] == "Hermes Agent Dashboard"
    assert manifest["short_name"] == "Hermes"
    assert manifest["start_url"] == "/"
    assert manifest["scope"] == "/"
    assert manifest["display"] == "standalone"
    assert "standalone" in manifest["display_override"]
    assert manifest["background_color"] == "#041C1C"
    assert manifest["theme_color"] == "#041C1C"

    icons = {icon["src"]: icon for icon in manifest["icons"]}
    for src in [
        "/icon-192.png",
        "/icon-512.png",
        "/icon-maskable-192.png",
        "/icon-maskable-512.png",
    ]:
        assert src in icons
        assert icons[src]["type"] == "image/png"

    assert icons["/icon-maskable-192.png"]["purpose"] == "maskable"
    assert icons["/icon-maskable-512.png"]["purpose"] == "maskable"


def test_dashboard_pwa_icons_have_declared_dimensions():
    expected = {
        "icon-180.png": (180, 180),
        "icon-192.png": (192, 192),
        "icon-512.png": (512, 512),
        "icon-maskable-192.png": (192, 192),
        "icon-maskable-512.png": (512, 512),
    }

    for filename, size in expected.items():
        assert _png_size(PUBLIC_DIR / filename) == size
