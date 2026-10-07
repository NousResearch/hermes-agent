#!/usr/bin/env python3
"""Generate every app icon in the repo from the Rabbit masters + platform backgrounds.

Usage (from repo root):
    node scripts/generate-icons.mjs           # write
    node scripts/generate-icons.mjs --check   # verify structure

Sources of truth — two axes, composed per target:
  Logo art (raster):  assets/icon-master.svg        white squircle + black logo
                      assets/icon-master-dark.svg   #0d1117 squircle + white logo
                      The two masters are the COMMITTED brand artifacts: each
                      embeds its logo PNG (base64) at the registered rect
                      (102.4, 102.4, 819.2x819.2 on the 1024 canvas). Light
                      surfaces carry the black silhouette of the logo, dark
                      surfaces the white art.

  Backgrounds (per platform surface, light/dark):
                      assets/backgrounds/squircle-light.svg   white rounded
                      assets/backgrounds/squircle-dark.svg    #0d1117 rounded
                      assets/backgrounds/squircle-mac-light.svg   mac HIG grid
                      assets/backgrounds/squircle-mac-dark.svg    mac HIG grid

  The light master drives every squircle target; the dark master drives the
  dark-appearance targets. macOS is the exception: its icns targets render
  from an in-memory mac master that puts the same squircle on Apple's
  824x824 grid — centered in 1024 with 100px margins — so the icon matches
  the size of Apple-template neighbors. macOS 26 additionally gets
  apps/desktop/assets/icon.icon, an Icon Composer package of canvas-filling
  layers (logo light/dark, plus a grayscale mono layer for Clear and Tinted)
  that the system masks into its own squircle; electron-builder compiles it
  to Assets.car next to the .icns, which macOS <= 15 keeps showing.

Desktop build identity comes from RABBIT_PAYLOAD_TAG / RABBIT_BUILD_COMMIT:
Canary uses yellow/dark-yellow backgrounds. Commit builds use red/dark-red
and a seven-character SHA badge. The logo and tile geometry do not change.
Only apps/desktop outputs use this identity. Website, bootstrap, dashboard,
and the shared master SVGs retain the default brand.

GENERATED OUTPUTS ARE COMMITTED. Regular builds and installs consume them and
never render; flavored release bundles (canary/commit) render to a product dir.
icons-freshness-check.yml regenerates, runs --check, and fails on any diff.

Rendering: resvg (resvg-py) for SVG -> PNG fidelity at every size.
Containers: Pillow for multi-size .ico and .icns.

Dependencies:
    Pillow and resvg-py are core runtime dependencies; run this file with a
    Rabbit runtime interpreter (scripts/generate-icons.mjs uses RABBIT_PYTHON).

Outputs:
  assets/icon-master.svg / icon-master-dark.svg   committed brand masters (input = output)
  apps/desktop/assets/icon.png                    1024x1024 squircle (light)
  apps/desktop/assets/icon.ico                    16,24,32,48,64,128,256
  apps/desktop/assets/icon.icns                   16..1024 (real ICNS)
  apps/desktop/assets/icon-dark.png/.ico/.icns    dark-appearance twins
  apps/desktop/packaging/dmg-volume.icns          16..1024, from assets/dmg-volume.png (DMG volume)
  apps/desktop/assets/icon.icon/icon.json         Icon Composer manifest (macOS 26)
  apps/desktop/assets/icon.icon/Assets/art-*.png  1024 logo (+ commit badge), light/dark
  apps/desktop/assets/icon.icon/Assets/mono.png   1024 Clear/Tinted material (grayscale + opacity)
  apps/bootstrap-installer/src-tauri/icons/**     same set, unbranded (Tauri bundle.icon)
  apps/desktop/assets/appx/<Logo>[.scale-N].png   MSIX logos at 100/125/150/200/400%
  apps/desktop/public/apple-touch-icon.png        1024x1024 squircle
  apps/desktop/public/brand-icon.png              256x256 squircle, black logo (light mark)
  apps/desktop/public/brand-icon-dark.png         256x256 squircle, white logo (dark mark)
  apps/bootstrap-installer/public/brand-icon.png  256x256 squircle mark (light)
  website/static/img/logo.png / logo-dark.png     1772x1799 logo alone, transparent
  website/static/img/favicon-*                    16/32/180px, ico, svg copy of the light master
  web/public/favicon.ico                          16,32,48
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

try:
    import resvg_py
except ImportError:
    sys.exit(
        "resvg-py is missing: run the generator with a Rabbit runtime interpreter\n"
        "  (RABBIT_PYTHON=<rabbit venv python> node scripts/generate-icons.mjs)"
    )

# Copy of rabbit_cli.update_channel._CANARY_TAG_RE: builders run this renderer
# on the runtime dependencies without the application package installed
# (Docker, bundles). tests/scripts/test_icon_flavors.py pins it to the canonical one.
_CANARY_TAG_RE = re.compile(
    r"^v(?:0|[1-9]\d{0,2})\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"\+canary\.20\d{6}T\d{6}Z$"
)

# The dark tile background (#0d1117) — fixed dark tile/background everywhere.
DARK_HEX = "#0d1117"
DARK_RGB = (13, 17, 23)
BORDER_FRACTION = 0.0407747197
# The mac HIG plate: 824 inset by 100 on the 1024 canvas.
MAC_PLATE = 824 / 1024
BORDER_ENABLED = False
# The brand ring is switched off everywhere (tiles and the macOS 26 layers):
# flat plate + logo read better beside macOS 26's own icons. The ring code
# stays so it can be turned back on in one place.

# ─── MSIX (Windows) asset catalog ───────────────────────────────────────────
# The manifest names only the base files; Windows resolves each through
# resource qualifiers. targetsize-* is what the taskbar, Start and search
# draw — an exact bitmap per slot, so nothing is ever upscaled (the bare 44px
# tile was) — and altform-unplated / altform-lightunplated are the dark- and
# light-theme forms Windows requires to exist even when identical (without
# them it plates the icon itself). electron-builder runs makepri + `makeappx
# /l` as soon as one qualified asset is staged (app-builder-lib
# winAppUtil.isScaledAssetsProvided), so the qualifiers alone wire this in.
APPX_DIR = "apps/desktop/assets/appx"
APPX_SCALES = (100, 125, 150, 200, 400)
APPX_TARGET_SIZES = (16, 20, 24, 30, 32, 36, 40, 48, 60, 64, 72, 80, 96, 256)
# Base size at scale-100; the wide tile keeps its 100px squircle centered.
APPX_LOGOS: dict[str, int | tuple[int, int]] = {
    "Square44x44Logo": 44,
    "Square150x150Logo": 150,
    "StoreLogo": 50,
    "Wide310x150Logo": (310, 150),
}
# Theme-qualified taskbar forms: (qualifier suffix, render kind). Dark theme
# gets the dark tile like macOS dark mode does; the unqualified file stays the
# light tile for hosts that ignore qualifiers.
APPX_ALTFORMS = (("", "png"), ("_altform-unplated", "png_dark"), ("_altform-lightunplated", "png"))


def appx_scaled(base: int, scale: int) -> int:
    # Microsoft's tables round up (150 @ 125% = 188, 50 @ 125% = 63).
    return math.ceil(base * scale / 100 - 1e-9)


def appx_targets() -> list[tuple[str, str, object]]:
    targets: list[tuple[str, str, object]] = []
    for name, base in APPX_LOGOS.items():
        for scale in APPX_SCALES:
            qualifier = "" if scale == 100 else f".scale-{scale}"
            if isinstance(base, tuple):
                w, h = base
                arg: object = (appx_scaled(w, scale), appx_scaled(h, scale), appx_scaled(100, scale))
                targets.append((f"{APPX_DIR}/{name}{qualifier}.png", "wide", arg))
            else:
                targets.append((f"{APPX_DIR}/{name}{qualifier}.png", "png", appx_scaled(base, scale)))
    for size in APPX_TARGET_SIZES:
        for suffix, kind in APPX_ALTFORMS:
            targets.append((f"{APPX_DIR}/Square44x44Logo.targetsize-{size}{suffix}.png", kind, size))
    return targets


APPX_TARGETS = appx_targets()


def _appx_check_sizes() -> dict[str, tuple[str, tuple[int, int]]]:
    sizes: dict[str, tuple[str, tuple[int, int]]] = {}
    for rel, kind, arg in APPX_TARGETS:
        if kind == "wide":
            w, h, _tile = arg  # type: ignore[misc]
            sizes[rel] = ("PNG", (w, h))
        else:
            sizes[rel] = ("PNG", (arg, arg))  # type: ignore[arg-type]
    return sizes


# Target sizes for --check's structural verification: relpath -> (format, size)
CHECK_SIZES: dict[str, tuple[str, tuple[int, int]]] = {
    **_appx_check_sizes(),
    "apps/desktop/assets/icon.png": ("PNG", (1024, 1024)),
    "apps/desktop/assets/icon-dark.png": ("PNG", (1024, 1024)),
    "apps/desktop/packaging/dmg-volume.icns": ("ICNS", (1024, 1024)),
    **({"apps/desktop/assets/icon.icon/Assets/border-light.png": ("PNG", (1024, 1024)),
        "apps/desktop/assets/icon.icon/Assets/border-dark.png": ("PNG", (1024, 1024))} if BORDER_ENABLED else {}),
    "apps/desktop/assets/icon.icon/Assets/mono.png": ("PNG", (1024, 1024)),
    "apps/bootstrap-installer/src-tauri/icons/icon.icon/Assets/art-light.png": ("PNG", (1024, 1024)),
    "apps/bootstrap-installer/src-tauri/icons/icon.icon/Assets/art-dark.png": ("PNG", (1024, 1024)),
    "apps/bootstrap-installer/src-tauri/icons/icon.icon/Assets/mono.png": ("PNG", (1024, 1024)),
    "apps/desktop/assets/icon.icon/Assets/art-light.png": ("PNG", (1024, 1024)),
    "apps/desktop/assets/icon.icon/Assets/art-dark.png": ("PNG", (1024, 1024)),
    "apps/desktop/public/apple-touch-icon.png": ("PNG", (1024, 1024)),
    "apps/desktop/public/brand-icon.png": ("PNG", (256, 256)),
    "apps/desktop/public/brand-icon-dark.png": ("PNG", (256, 256)),
    "apps/bootstrap-installer/src-tauri/icons/32x32.png": ("PNG", (32, 32)),
    "apps/bootstrap-installer/src-tauri/icons/128x128.png": ("PNG", (128, 128)),
    "apps/bootstrap-installer/src-tauri/icons/128x128@2x.png": ("PNG", (256, 256)),
    "apps/bootstrap-installer/public/brand-icon.png": ("PNG", (256, 256)),
    "website/static/img/logo.png": ("PNG", (1772, 1799)),
    "website/static/img/logo-dark.png": ("PNG", (1772, 1799)),
    "website/static/img/favicon-16x16.png": ("PNG", (16, 16)),
    "website/static/img/favicon-32x32.png": ("PNG", (32, 32)),
    "website/static/img/apple-touch-icon.png": ("PNG", (180, 180)),
}

# (relpath, kind, arg)
TARGETS: list[tuple[str, str, object]] = [
    ("assets/icon-master.svg", "svg", None),
    ("assets/icon-master-dark.svg", "svg_dark", None),
    ("apps/desktop/assets/icon.png", "png", 1024),
    ("apps/desktop/assets/icon.ico", "ico", [16, 24, 32, 48, 64, 128, 256]),
    ("apps/desktop/assets/icon.icns", "icns", None),
    ("apps/desktop/assets/icon-dark.png", "png_dark", 1024),
    ("apps/desktop/assets/icon-dark.ico", "ico_dark", [16, 24, 32, 48, 64, 128, 256]),
    ("apps/desktop/assets/icon-dark.icns", "icns_dark", None),
    # The DMG volume icon: our own drive artwork with the logo on its face,
    # one artwork for light and dark (volume icons have no appearance variants).
    ("apps/desktop/packaging/dmg-volume.icns", "icns_from_png", "assets/dmg-volume.png"),
    ("apps/desktop/assets/icon.icon/icon.json", "icon_manifest", None),
    *([("apps/desktop/assets/icon.icon/Assets/border-light.png", "icon_border", "#000000"),
       ("apps/desktop/assets/icon.icon/Assets/border-dark.png", "icon_border", "#ffffff")] if BORDER_ENABLED else []),
    ("apps/desktop/assets/icon.icon/Assets/art-light.png", "icon_art", "black"),
    ("apps/desktop/assets/icon.icon/Assets/art-dark.png", "icon_art", "white"),
    ("apps/desktop/assets/icon.icon/Assets/mono.png", "icon_mono", None),
    *APPX_TARGETS,
    ("apps/desktop/public/apple-touch-icon.png", "png", 1024),
    # The dev-run Dock icon (app.dock.setIcon): same mac grid as the icns.
    ("apps/desktop/assets/icon-mac.png", "png_mac", 1024),
    ("apps/desktop/public/brand-icon.png", "brand_light", 256),
    ("apps/desktop/public/brand-icon-dark.png", "brand_dark", 256),
    ("apps/bootstrap-installer/src-tauri/icons/32x32.png", "png", 32),
    ("apps/bootstrap-installer/src-tauri/icons/128x128.png", "png", 128),
    ("apps/bootstrap-installer/src-tauri/icons/128x128@2x.png", "png", 256),
    ("apps/bootstrap-installer/src-tauri/icons/icon.ico", "ico", [16, 32, 64, 128, 256]),
    ("apps/bootstrap-installer/src-tauri/icons/icon.icns", "icns", None),
    # Tauri's bundler compiles an Icon Composer package from `bundle.icon`
    # (actool >= 26) into Assets.car; the unbranded twin of the desktop package.
    ("apps/bootstrap-installer/src-tauri/icons/icon.icon/icon.json", "icon_manifest", None),
    ("apps/bootstrap-installer/src-tauri/icons/icon.icon/Assets/art-light.png", "icon_art", "black"),
    ("apps/bootstrap-installer/src-tauri/icons/icon.icon/Assets/art-dark.png", "icon_art", "white"),
    ("apps/bootstrap-installer/src-tauri/icons/icon.icon/Assets/mono.png", "icon_mono", None),
    ("apps/bootstrap-installer/public/brand-icon.png", "brand_light", 256),
    ("website/static/img/logo.png", "logo", None),
    ("website/static/img/logo-dark.png", "logo_dark", None),
    ("website/static/img/favicon-16x16.png", "png", 16),
    ("website/static/img/favicon-32x32.png", "png", 32),
    ("website/static/img/apple-touch-icon.png", "png", 180),
    ("website/static/img/favicon.ico", "ico", [16, 32, 48]),
    ("website/static/img/favicon.svg", "svg_copy", None),
    ("web/public/favicon.ico", "ico", [16, 32, 48]),
]

# ─── logo art extraction ────────────────────────────────────────────────────

def load_master(path: Path) -> dict:
    """Parse a committed master SVG: plate fill + embedded logo PNG + its rect.

    The masters are the source of truth for the brand artwork: each is a
    squircle plate with the logo PNG embedded (base64) at the registered rect.
    """
    text = path.read_text(encoding="utf-8-sig")
    rect = re.search(r'<rect\b[^>]*fill="(#[0-9a-fA-F]{6})"', text)
    assert rect, f"no plate rect found in {path.name}"
    image = re.search(r"<image\b[^>]*>", text)
    assert image, f"no <image> found in {path.name}"
    attrs = dict(re.findall(r'(\w+(?::\w+)?)="([^"]*)"', image.group(0)))
    href = attrs.get("xlink:href") or attrs.get("href") or ""
    m = re.fullmatch(r"data:image/png;base64,(.+)", href, re.S)
    assert m, f"logo <image> in {path.name} is not an embedded PNG"
    return {
        "bg": rect.group(1),
        "png": base64.b64decode(m.group(1)),
        "rect": tuple(float(attrs[k]) for k in ("x", "y", "width", "height")),
    }


class IconArt:
    """One generation's rendering inputs and caches; never writes to source."""

    def __init__(self, source: Path, *, colors: tuple[str, str] | None = None, commit: str = ""):
        assets = source / "assets"
        self.source = source
        self.colors = colors
        self.commit = commit
        self.backgrounds = assets / "backgrounds"
        light = load_master(assets / "icon-master.svg")
        dark = load_master(assets / "icon-master-dark.svg")
        # The black silhouette rides light plates, the white art dark ones.
        self.logo = {"black": light["png"], "white": dark["png"]}
        assert light["rect"] == dark["rect"], "light/dark masters disagree on the logo rect"
        self.rect = light["rect"]
        self.master = compose_svg(self, "black", "squircle-light.svg")
        self.master_dark = compose_svg(self, "white", "squircle-dark.svg")
        # macOS icons sit on Apple's 824-on-1024 grid, not the full-bleed
        # squircle: same art, mac-grid backgrounds, icns targets only.
        self.master_mac = compose_svg(self, "black", "squircle-mac-light.svg")
        self.master_mac_dark = compose_svg(self, "white", "squircle-mac-dark.svg")
        # Badge-free twins for direct renders below BADGE_MIN_SIZE.
        self.master_small = compose_svg(self, "black", "squircle-light.svg", badge=False)
        self.master_dark_small = compose_svg(self, "white", "squircle-dark.svg", badge=False)


def logo_layer(art: IconArt, ink: str, bg: str) -> str:
    """The embedded logo PNG placed at the masters' registered rect.

    The rect is registered on the 1024 canvas; mac plates re-map the canvas
    onto Apple's 824 grid, and the macOS 26 layered icon's canvas IS the
    plate (the system maps it back), so only the mac squircles transform.
    """
    x, y, w, h = art.rect
    transform = f' transform="translate(100 100) scale({MAC_PLATE})"' if "-mac-" in bg else ""
    b64 = base64.b64encode(art.logo[ink]).decode("ascii")
    return (
        f'<g{transform}><image x="{x}" y="{y}" width="{w}" height="{h}" '
        f'preserveAspectRatio="xMidYMid meet" xlink:href="data:image/png;base64,{b64}"/></g>'
    )


def background_inner(art: IconArt, name: str) -> tuple[str, int, int]:
    """Inner content + (width, height) of a background SVG asset."""
    text = (art.backgrounds / name).read_text(encoding="utf-8-sig")
    if art.colors:
        text = text.replace('fill="#ffffff"', f'fill="{art.colors[0]}"')
        text = text.replace(f'fill="{DARK_HEX}"', f'fill="{art.colors[1]}"')
    root = ET.fromstring(text)
    m = re.fullmatch(r"0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)", root.get("viewBox", ""))
    assert m, f"cannot parse viewBox of {name}"
    w, h = float(m.group(1)), float(m.group(2))
    # Editor exports include XML declarations and root-scoped namespaces.
    # Parse away the prolog and retain child namespaces when embedding.
    inner = "".join(ET.tostring(child, encoding="unicode") for child in root)
    return inner, int(w), int(h)


# Five-by-seven lowercase hexadecimal glyphs, one five-bit row at a time.
# Vector cells keep release builds deterministic without any installed fonts.
HEX_GLYPHS = {
    "0": (14, 17, 19, 21, 25, 17, 14),
    "1": (4, 12, 4, 4, 4, 4, 14),
    "2": (14, 17, 1, 2, 4, 8, 31),
    "3": (30, 1, 1, 14, 1, 1, 30),
    "4": (2, 6, 10, 18, 31, 2, 2),
    "5": (31, 16, 16, 30, 1, 1, 30),
    "6": (14, 16, 16, 30, 17, 17, 14),
    "7": (31, 1, 2, 4, 8, 8, 8),
    "8": (14, 17, 17, 14, 17, 17, 14),
    "9": (14, 17, 17, 15, 1, 1, 14),
    "a": (0, 0, 14, 1, 15, 17, 15),
    "b": (16, 16, 30, 17, 17, 17, 30),
    "c": (0, 0, 14, 16, 16, 17, 14),
    "d": (1, 1, 15, 17, 17, 17, 15),
    "e": (0, 0, 14, 17, 31, 16, 14),
    "f": (6, 9, 8, 28, 8, 8, 8),
}


# Commit badge plate and glyph origin on the 1024 canvas. The logo sits at
# the masters' registered rect, so the badge is scaled into the free band
# above it: BADGE_TRANSFORM maps the design rect onto (336, 14, 352, 76).
BADGE_RECT = (160, 28, 704, 152)
BADGE_GLYPH_ORIGIN = (184, 48)
BADGE_TRANSFORM = (256, 0, 0.5)  # translate-x, translate-y, scale
# Below this edge length a direct render carries no badge: the plate's top
# row is then under one device pixel from the badge, whose anti-aliased edge
# would stack coverage on the tile's own corner pixels and change the alpha
# channel between flavors. Nobody reads a SHA at 16px anyway.
BADGE_MIN_SIZE = 32


def commit_layer(commit: str, bg: str) -> str:
    cells = []
    gx, gy = BADGE_GLYPH_ORIGIN
    for index, char in enumerate(commit[:7]):
        for row, bits in enumerate(HEX_GLYPHS[char]):
            for col in range(5):
                if bits & (1 << (4 - col)):
                    x, y = gx + (index * 6 + col) * 16, gy + row * 16
                    cells.append(f"M{x} {y}h16v16h-16z")
    # The badge rides in the band above the logo rect; on mac tiles it also
    # follows the tile's HIG inset, never the outer canvas.
    tx, ty, scale = BADGE_TRANSFORM
    transform = f' translate({tx} {ty}) scale({scale})'
    if "-mac-" in bg:
        transform = f' translate(100 100) scale({824 / 1024})' + transform
    bx, by, bw, bh = BADGE_RECT
    return (
        f'<g transform="{transform.strip()}"><rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="24" fill="#000000"/>'
        f'<path fill="#ffffff" d="{"".join(cells)}"/></g>'
    )


def compose_svg(art: IconArt, ink: str, bg: str, *, badge: bool = True) -> str:
    """Full svg text: background + logo layer, in the background's native
    coordinate space (resvg scales to whatever output size is requested, so
    the composition is size-agnostic — no manual box scaling)."""
    inner, w, h = background_inner(art, bg)
    background = ET.fromstring(f"<g>{inner}</g>")
    tile = background.find("{http://www.w3.org/2000/svg}rect")
    assert tile is not None, f"no background rectangle in {bg}"
    geometry = {key: float(tile.attrib[key]) for key in ("x", "y", "width", "height", "rx")}
    silhouette = ET.Element("rect", {key: str(value) for key, value in geometry.items()})
    if BORDER_ENABLED:
        thickness = geometry["width"] * BORDER_FRACTION
        # An inward stroke keeps the outer platform geometry unchanged. Subtracting
        # the same inset from rx (not scaling rx) keeps the corner thickness uniform.
        inset = {"x": 1, "y": 1, "width": -2, "height": -2, "rx": -1}
        for key, value in geometry.items():
            tile.set(key, str(value + inset[key] * thickness / 2))
        tile.set("stroke", "#000000" if ink == "black" else "#ffffff")
        tile.set("stroke-width", str(thickness))
    inner = "".join(ET.tostring(child, encoding="unicode") for child in background)
    clip = ET.tostring(silhouette, encoding="unicode")
    # The badge shares the logo's clip so it can never paint past the plate.
    badge_svg = commit_layer(art.commit, bg) if art.commit and badge else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {w} {h}">\n'
        f'  <defs><clipPath id="icon-silhouette">{clip}</clipPath></defs>\n'
        f"  {inner.strip()}\n"
        f'  <g clip-path="url(#icon-silhouette)">{badge_svg}{logo_layer(art, ink, bg)}</g>\n'
        "</svg>\n"
    )


# ─── macOS 26 layered icon (.icon → Assets.car) ─────────────────────────────
#
# macOS 26 clips every app icon to its own mask and shows legacy .icns art
# inside a glass plate, so a ring drawn around our tile no longer follows the
# visible outline. The .icon package gives the system canvas-filling layers
# to mask itself: the border layer is a band whose OUTER edge is the canvas
# (the system's mask becomes the outline) and whose INNER edge is the mask
# offset inward by the border thickness — constant thickness by construction.
#
# The mask is Apple's smoothed rounded square: a circular arc flanked by two
# cubic easing segments (the Figma corner-smoothing construction). Fitted to
# macOS 26.6.2's own render of a system icon on the 824-on-1024 grid:
# radius 214px, smoothing 0.645 (rms 0.32px, worst 0.81px). Layers are
# placed on that grid by the system, so in layer coordinates the mask fills
# the canvas.
MAC_MASK_RADIUS_FRACTION = 214.0 / 824.0
MAC_MASK_SMOOTHING = 0.645
ICON_CANVAS = 1024


def _cubic(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    u = 1.0 - t
    return (u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1])


def _mask_corner(side: float, samples: int) -> list[tuple[float, float]]:
    """Top-left corner with the tile corner at the origin, running clockwise
    from the left edge (0, p) around to the top edge (p, 0)."""
    radius = side * MAC_MASK_RADIUS_FRACTION
    smoothing = MAC_MASK_SMOOTHING
    p = min((1 + smoothing) * radius, side / 2)
    arc_measure = 90 * (1 - smoothing)
    arc_len = math.sin(math.radians(arc_measure / 2)) * radius * math.sqrt(2)
    angle_alpha = (90 - arc_measure) / 2
    p3p4 = radius * math.tan(math.radians(angle_alpha / 2))
    angle_beta = 45 * smoothing
    c = p3p4 * math.cos(math.radians(angle_beta))
    d = c * math.tan(math.radians(angle_beta))
    b = (p - arc_len - c - d) / 3
    a = 2 * b
    # Trace the construction's top-right corner (corner at the origin, x <= 0):
    # easing cubic, circular arc, easing cubic.
    pts: list[tuple[float, float]] = []
    p0 = (-p, 0.0)
    p1, p2, p3 = (p0[0] + a, 0.0), (p0[0] + a + b, 0.0), (p0[0] + a + b + c, d)
    for i in range(samples):
        pts.append(_cubic(p0, p1, p2, p3, i / samples))
    length = math.hypot(c, d)
    nx, ny = -d / length, c / length
    cx, cy = p3[0] + nx * radius, p3[1] + ny * radius
    start = math.atan2(p3[1] - cy, p3[0] - cx)
    for i in range(samples):
        angle = start + math.radians(arc_measure) * i / samples
        pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    p4 = (p3[0] + arc_len, p3[1] + arc_len)
    p5, p6, p7 = (p4[0] + d, p4[1] + c), (p4[0] + d, p4[1] + b + c), (0.0, p)
    for i in range(samples + 1):
        pts.append(_cubic(p4, p5, p6, p7, i / samples))
    # Mirror into the top-left corner and reverse so the run is clockwise.
    return [(-x, y) for x, y in reversed(pts)]


def mac_mask_outline(side: float, samples: int = 96) -> list[tuple[float, float]]:
    """Closed clockwise outline of the macOS 26 icon mask for a tile of `side`."""
    tl = _mask_corner(side, samples)
    tr = [(side - y, x) for x, y in tl]
    br = [(side - x, side - y) for x, y in tl]
    bl = [(y, side - x) for x, y in tl]
    outline: list[tuple[float, float]] = []
    for point in tl + tr + br + bl:
        if not outline or math.hypot(point[0] - outline[-1][0], point[1] - outline[-1][1]) > 1e-9:
            outline.append(point)
    return outline


def offset_inward(points: list[tuple[float, float]], distance: float) -> list[tuple[float, float]]:
    """Parallel curve `distance` inside a convex closed outline."""
    count = len(points)
    cx = sum(x for x, _ in points) / count
    cy = sum(y for _, y in points) / count
    result = []
    for i, (x, y) in enumerate(points):
        px, py = points[i - 1]
        qx, qy = points[(i + 1) % count]
        tx, ty = qx - px, qy - py
        length = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / length, tx / length
        if (cx - x) * nx + (cy - y) * ny < 0:
            nx, ny = -nx, -ny
        result.append((x + nx * distance, y + ny * distance))
    return result


def closed_path(points: list[tuple[float, float]]) -> str:
    """SVG path through the points as a closed Catmull-Rom spline (cubic Beziers)."""
    count = len(points)
    parts = [f"M{points[0][0]:.3f} {points[0][1]:.3f}"]
    for i in range(count):
        p0, p1 = points[(i - 1) % count], points[i]
        p2, p3 = points[(i + 1) % count], points[(i + 2) % count]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        parts.append(f"C{c1[0]:.3f} {c1[1]:.3f} {c2[0]:.3f} {c2[1]:.3f} {p2[0]:.3f} {p2[1]:.3f}")
    parts.append("Z")
    return "".join(parts)


def icon_border_svg(ink: str) -> str:
    """Border layer: the canvas minus the mask's inward offset (even-odd)."""
    thickness = ICON_CANVAS * BORDER_FRACTION
    inner = closed_path(offset_inward(mac_mask_outline(float(ICON_CANVAS)), thickness))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_CANVAS} {ICON_CANVAS}">\n'
        f'  <path d="M0 0H{ICON_CANVAS}V{ICON_CANVAS}H0Z {inner}" fill="{ink}" fill-rule="evenodd"/>\n'
        "</svg>\n"
    )


def icon_art_svg(art: IconArt, ink: str) -> str:
    """Art layer: the logo at the masters' registered rect on the
    canvas-filling plate (the system supplies the grid); the commit badge
    rides along for commit builds."""
    badge = f"  {commit_layer(art.commit, 'icon.icon')}\n" if art.commit else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {ICON_CANVAS} {ICON_CANVAS}">\n'
        f"{badge}"
        f"  {logo_layer(art, ink, 'icon.icon')}\n"
        "</svg>\n"
    )


def icon_color(hex_color: str) -> str:
    """Icon Composer colour literal for an opaque sRGB hex colour."""
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return f"srgb:{r:.5f},{g:.5f},{b:.5f},1.00000"


# Mono (Clear/Tinted) materials, as luminance + opacity: the dark-ink parts of
# the logo become a near-black frosted glass instead of black, its white parts
# stay white, and the ring (when enabled) is a soft light glass.
MONO_INK = (0.13, 0.90)
MONO_RING = (0.82, 0.75)


def icon_mono_image(art: IconArt) -> Image.Image:
    """The single grayscale layer behind Clear light/dark and Tinted light/dark
    (Apple's "tinted" specialization). One image, each pixel painted once, so
    translucent materials never stack where the logo meets the ring."""
    def coverage(svg: str) -> Image.Image:
        return render_svg(svg, ICON_CANVAS).convert("RGBA").getchannel("A")

    def material(tone: float, opacity: float, alpha: Image.Image) -> Image.Image:
        grey = round(tone * 255)
        layer = Image.new("RGBA", alpha.size, (grey, grey, grey, 0))
        layer.putalpha(alpha.point(lambda v: round(v * opacity)))
        return layer

    out = Image.new("RGBA", (ICON_CANVAS, ICON_CANVAS), (0, 0, 0, 0))
    paints = [(coverage(icon_art_svg(art, "black")), MONO_INK), (coverage(icon_art_svg(art, "white")), (1.0, 1.0))]
    if BORDER_ENABLED:
        paints.append((coverage(icon_border_svg("#ffffff")), MONO_RING))
    for alpha, (tone, opacity) in paints:
        # paste, not composite: a later material replaces an earlier one
        out.paste(material(tone, opacity, alpha), (0, 0), alpha.point(lambda v: 255 if v > 127 else 0))
    return out


def icon_manifest(art: IconArt) -> str:
    """icon.json: flat brand fill (flavoured for canary/commit builds), the
    art layer (and ring, when enabled) with light/dark variants, and the mono
    layer that replaces them under Clear and Tinted."""
    light, dark = art.colors or ("#ffffff", DARK_HEX)

    def layer(name: str) -> dict:
        # No fixed "image-name": actool treats it as the image for every
        # appearance and drops the specializations, so the dark variant would
        # never reach Assets.car (black logo on the dark fill).
        return {
            "name": name,
            "image-name-specializations": [
                {"value": f"{name}-light.png"},
                {"appearance": "dark", "value": f"{name}-dark.png"},
            ],
            "glass": False,
            "hidden-specializations": [{"value": False}, {"appearance": "tinted", "value": True}],
        }

    # "tinted" is the single mono annotation behind Clear light/dark and
    # Tinted light/dark: the system reads luminance as visibility and tints it.
    mono = {
        "name": "mono",
        "image-name": "mono.png",
        "glass": True,
        "hidden-specializations": [{"value": True}, {"appearance": "tinted", "value": False}],
    }
    layers = ([layer("border")] if BORDER_ENABLED else []) + [layer("art"), mono]

    manifest = {
        "fill-specializations": [
            {"value": {"solid": icon_color(light)}},
            {"appearance": "dark", "value": {"solid": icon_color(dark)}},
        ],
        "groups": [{
            "lighting": "individual",
            "specular": False,
            "translucency": {"enabled": False, "value": 0},
            "shadow": {"kind": "none", "opacity": 0},
            "blur-material": None,
            "layers": layers,
        }],
        "supported-platforms": {"squares": "shared"},
    }
    return json.dumps(manifest, indent=2) + "\n"


# ─── rendering ──────────────────────────────────────────────────────────────

def render(master: str, size: int, *, background: str | None = None) -> Image.Image:
    """Render a master to an RGBA PNG of `size`x`size`."""
    data = resvg_py.svg_to_bytes(
        svg_string=master, width=size, height=size, background=background
    )
    return Image.open(io.BytesIO(data)).convert("RGBA")


def render_svg(svg: str, size: int | tuple[int, int]) -> Image.Image:
    if isinstance(size, int):
        w = h = size
    else:
        w, h = size
    data = resvg_py.svg_to_bytes(svg_string=svg, width=w, height=h)
    return Image.open(io.BytesIO(data)).convert("RGBA")


def paste_centered(canvas: Image.Image, img: Image.Image) -> None:
    x = (canvas.width - img.width) // 2
    y = (canvas.height - img.height) // 2
    canvas.alpha_composite(img, (x, y))


def save_png(img: Image.Image, buf: io.BytesIO) -> None:
    """Save with alpha preserved. Flat art quantizes losslessly to an 8-bit
    palette (tRNS per-index alpha keeps the AA edges), so use that when the
    palette round-trips pixel-identically; fall back to RGBA otherwise."""
    if img.mode != "RGBA":
        img.convert("RGBA").save(buf, "PNG", optimize=True)
        return
    quantized = img.quantize(colors=256, method=Image.FASTOCTREE, dither=Image.NONE)
    if quantized.convert("RGBA").tobytes() == img.tobytes():
        quantized.save(buf, "PNG", optimize=True)
    else:
        img.save(buf, "PNG", optimize=True)


def brand_mark(art: IconArt, kind: str, size: int) -> Image.Image:
    """The logo in the app-icon squircle (BrandMark asset) — the mark IS the
    icon shape. brand_light: black logo on white squircle.  brand_dark: white
    logo on #0d1117 squircle."""
    if kind == "brand_light":
        ink, bg = "black", "squircle-light.svg"
    else:
        ink, bg = "white", "squircle-dark.svg"
    return render_svg(compose_svg(art, ink, bg), size)


def build_logo_image(art: IconArt, dark: bool = False) -> Image.Image:
    """1772x1799 wordmark: the logo alone on transparency (no frame), centered.
    Light = black logo, dark = white logo — the consuming surface's background
    (navbar light/dark) shows through."""
    W, H = 1772, 1799
    b64 = base64.b64encode(art.logo["white" if dark else "black"]).decode("ascii")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}">\n'
        f'  <image x="0" y="0" width="{W}" height="{H}" preserveAspectRatio="xMidYMid meet" '
        f'xlink:href="data:image/png;base64,{b64}"/>\n</svg>\n'
    )
    return render_svg(svg, (W, H))


def target_bytes(art: IconArt, kind: str, arg: object) -> bytes:
    """Produce the exact bytes for one target. Shared by write + check."""
    if kind == "svg":
        return (art.source / "assets" / "icon-master.svg").read_bytes()
    if kind == "svg_dark":
        return (art.source / "assets" / "icon-master-dark.svg").read_bytes()
    if kind == "svg_copy":
        return (art.source / "assets" / "icon-master.svg").read_bytes()
    if kind == "icon_manifest":
        return icon_manifest(art).encode("utf-8")

    buf = io.BytesIO()
    if kind == "png":
        save_png(render(art.master if arg >= BADGE_MIN_SIZE else art.master_small, arg), buf)
    elif kind == "png_mac":
        save_png(render(art.master_mac, arg), buf)
    elif kind == "png_dark":
        save_png(render(art.master_dark if arg >= BADGE_MIN_SIZE else art.master_dark_small, arg), buf)
    elif kind in ("brand_light", "brand_dark"):
        save_png(brand_mark(art, kind, arg), buf)
    elif kind == "icon_border":
        save_png(render_svg(icon_border_svg(arg), ICON_CANVAS), buf)
    elif kind == "icon_art":
        save_png(render_svg(icon_art_svg(art, arg), ICON_CANVAS), buf)
    elif kind == "icon_mono":
        save_png(icon_mono_image(art), buf)
    elif kind == "ico":
        img = render(art.master, max(arg))
        img.save(buf, format="ICO", sizes=[(s, s) for s in arg])
    elif kind == "ico_dark":
        img = render(art.master_dark, max(arg))
        img.save(buf, format="ICO", sizes=[(s, s) for s in arg])
    elif kind == "icns":
        img = render(art.master_mac, 1024)
        frames = [img.resize((s, s), Image.LANCZOS) for s in (16, 32, 64, 128, 256, 512, 1024)]
        img.save(buf, format="ICNS", append_images=frames[1:])
    elif kind == "icns_dark":
        img = render(art.master_mac_dark, 1024)
        frames = [img.resize((s, s), Image.LANCZOS) for s in (16, 32, 64, 128, 256, 512, 1024)]
        img.save(buf, format="ICNS", append_images=frames[1:])
    elif kind == "icns_from_png":
        # Hand-made artwork (the DMG volume: logo on a drive) shipped as ICNS.
        img = Image.open(art.source / arg).convert("RGBA")
        assert img.size == (1024, 1024), f"{arg} must be 1024x1024"
        frames = [img.resize((s, s), Image.LANCZOS) for s in (16, 32, 64, 128, 256, 512, 1024)]
        img.save(buf, format="ICNS", append_images=frames[1:])
    elif kind == "wide":
        w, h, tile = arg
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        paste_centered(canvas, render(art.master, tile))
        canvas.save(buf, "PNG")
    elif kind == "logo":
        build_logo_image(art, dark=False).save(buf, "PNG")
    elif kind == "logo_dark":
        build_logo_image(art, dark=True).save(buf, "PNG")
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return buf.getvalue()


def build_art(source: Path) -> tuple[IconArt, IconArt]:
    """Only desktop outputs carry build identity. Shared branding stays stable."""
    art = IconArt(source)
    tag = os.environ.get("RABBIT_PAYLOAD_TAG", "")
    commit = os.environ.get("RABBIT_BUILD_COMMIT", "")
    if commit:
        if tag:
            raise ValueError("Commit builds cannot also select RABBIT_PAYLOAD_TAG")
        if not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise ValueError("RABBIT_BUILD_COMMIT requires an exact full 40-character SHA")
        return art, IconArt(source, colors=("#e34850", "#4a1117"), commit=commit)
    if _CANARY_TAG_RE.match(tag.strip()):
        return art, IconArt(source, colors=("#f5cc32", "#443808"))
    return art, art


def cmd_write(source: Path, out: Path) -> int:
    """Return failure when any target cannot be generated or verified."""
    art, desktop_art = build_art(source)
    written = 0
    failures = 0
    for rel, kind, arg in TARGETS:
        path = out / rel
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            selected = desktop_art if rel.startswith("apps/desktop/") else art
            path.write_bytes(target_bytes(selected, kind, arg))
            written += 1
        except Exception as exc:  # noqa: BLE001 - report all, then fail
            failures += 1
            print(f"  !! {rel}: FAILED ({exc})")
    print(f"[write] wrote {written}/{len(TARGETS)} files")

    print("\n[verify]")
    for rel, kind, arg in TARGETS:
        path = out / rel
        try:
            if kind in ("svg", "svg_dark", "svg_copy", "icon_manifest"):
                print(f"  {rel}: {path.stat().st_size} bytes {'JSON' if kind == 'icon_manifest' else 'SVG'}")
                continue
            im = Image.open(path)
            if kind in ("ico", "ico_dark"):
                sizes = []
                try:
                    for i in range(im.n_frames):
                        im.seek(i)
                        sizes.append(im.size)
                except Exception:
                    sizes = [im.size]
                print(f"  {rel}: ICO {sorted(set(sizes))}")
            elif kind in ("icns", "icns_dark", "icns_from_png"):
                print(f"  {rel}: ICNS {im.size} (container)")
            else:
                print(f"  {rel}: {im.format} {im.size}")
        except Exception as exc:
            failures += 1
            print(f"  {rel}: VERIFY FAILED ({exc})")

    return int(failures > 0)


def cmd_check(source: Path, out: Path) -> int:
    """Structural verification: regenerate every target in memory and assert
    the invariants that actually matter (there are no committed bytes to
    byte-compare — outputs are generated on demand)."""
    art, desktop_art = build_art(source)
    problems: list[str] = []

    # every target must generate without error
    for rel, kind, arg in TARGETS:
        try:
            selected = desktop_art if rel.startswith("apps/desktop/") else art
            target_bytes(selected, kind, arg)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{rel}: REGENERATE FAILED ({exc})")

    # PNG targets must have the expected format + size
    for rel, (fmt, size) in CHECK_SIZES.items():
        path = out / rel
        if not path.exists():
            problems.append(f"{rel}: MISSING (expected generated file)")
            continue
        try:
            im = Image.open(path)
            if im.format != fmt or im.size != size:
                problems.append(f"{rel}: got {im.format} {im.size}, expected {fmt} {size}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{rel}: UNREADABLE ({exc})")

    # squircles must keep transparent corners (alpha extrema include 0)
    for rel in (
        "apps/desktop/assets/icon.png",
        "apps/desktop/assets/icon-dark.png",
        "apps/desktop/public/brand-icon.png",
        "apps/desktop/public/brand-icon-dark.png",
        "apps/desktop/public/apple-touch-icon.png",
    ):
        path = out / rel
        if not path.exists():
            continue
        alpha = Image.open(path).convert("RGBA").getchannel("A")
        lo, hi = alpha.getextrema()
        if lo != 0 or hi != 255:
            problems.append(f"{rel}: alpha extrema {alpha.getextrema()}, expected (0, 255) transparent corners")

    # containers must have the right frame sets (parse ICO headers directly —
    # PIL's ICO n_frames is unreliable across versions)
    def ico_sizes(path: Path) -> list[int]:
        data = path.read_bytes()
        count = int.from_bytes(data[4:6], "little")
        sizes = []
        for i in range(count):
            entry = data[6 + i * 16 : 6 + (i + 1) * 16]
            w = entry[0] or 256
            h = entry[1] or 256
            sizes.append(w)
        return sorted(set(sizes))

    for rel, sizes in (
        ("apps/desktop/assets/icon.ico", [16, 24, 32, 48, 64, 128, 256]),
        ("apps/desktop/assets/icon-dark.ico", [16, 24, 32, 48, 64, 128, 256]),
        ("apps/bootstrap-installer/src-tauri/icons/icon.ico", [16, 32, 64, 128, 256]),
    ):
        path = out / rel
        if not path.exists():
            problems.append(f"{rel}: MISSING")
            continue
        try:
            got = ico_sizes(path)
            if got != sizes:
                problems.append(f"{rel}: ICO frames {got}, expected {sizes}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{rel}: UNREADABLE ({exc})")

    if not problems:
        print(f"[ok] all {len(TARGETS)} targets generate and pass structural checks")
        return 0

    print(f"[check] {len(problems)} problem(s):")
    for line in problems:
        print(f"  - {line}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, help="Output root (defaults to source for developer builds)")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    out = (args.out or source).resolve()
    command = cmd_check if args.check else cmd_write
    sys.exit(command(source, out))


if __name__ == "__main__":
    main()
