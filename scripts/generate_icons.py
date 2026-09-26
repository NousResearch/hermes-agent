#!/usr/bin/env python3
"""Generate every app icon in the repo from the nous-girl art + platform backgrounds.

Usage (from repo root):
    node scripts/generate-icons.mjs           # write
    node scripts/generate-icons.mjs --check   # verify structure

Sources of truth — two axes, composed per target:
  Girl art (vector):  assets/nous-girl-black.svg  (black positive space)
                      assets/nous-girl-white.svg  (white positive space)
                      straight from the Nous brand kit (Inkscape exports,
                      5487^2 viewBox, one path each).

  Backgrounds (per platform surface, light/dark):
                      assets/backgrounds/squircle-light.svg   white rounded
                      assets/backgrounds/squircle-dark.svg    #0d1117 rounded
                      assets/backgrounds/squircle-mac-light.svg   mac HIG grid
                      assets/backgrounds/squircle-mac-dark.svg    mac HIG grid

  The master SVGs (assets/icon-master.svg light, assets/icon-master-dark.svg
  dark) are GENERATED artifacts — squircle background + scaled girl artwork.
  The light master drives every squircle target;
  the dark master drives the dark-appearance targets. macOS is the exception:
  its icns targets render from an in-memory mac master that puts the same
  squircle on Apple's 824x824 (r=185.4) grid — centered in 1024 with 100px
  margins — so the icon matches the size of Apple-template neighbors.
  macOS 26 additionally gets apps/desktop/assets/icon.icon, an Icon Composer
  package of canvas-filling layers (border band + girl, light/dark) that the
  system masks into its own squircle; electron-builder compiles it to
  Assets.car next to the .icns, which macOS <= 15 keeps showing.

Desktop build identity comes from HERMES_PAYLOAD_TAG / HERMES_BUILD_COMMIT:
Canary uses yellow/dark-yellow backgrounds. Commit builds use red/dark-red
and a seven-character SHA badge. The girl and tile geometry do not change.
Only apps/desktop outputs use this identity. Website, bootstrap, dashboard,
and the shared master SVGs retain the default brand.

The girl's position and uniform scale are registered to the reference artwork.
She renders in front of the border, clipped only to the outer rounded silhouette.
Only nodes near her bottom edge extend to the border; the fitted face and hair stay fixed.
Standalone wordmarks remain centered and have no border.

GENERATED OUTPUTS ARE COMMITTED. Regular builds and installs consume them and
never render; flavored release bundles (canary/commit) render to a product dir.
icons-freshness-check.yml regenerates, runs --check, and fails on any diff.

Rendering: resvg (resvg-py) for SVG -> PNG fidelity at every size.
Containers: Pillow for multi-size .ico and .icns.

Dependencies:
    Pillow and resvg-py are core runtime dependencies; run this file with a
    Hermes runtime interpreter (scripts/generate-icons.mjs uses HERMES_PYTHON).

Outputs (40 files):
  assets/icon-master.svg                              generated light master
  assets/icon-master-dark.svg                         generated dark master
  apps/desktop/assets/icon.png                        1024x1024 squircle (light)
  apps/desktop/assets/icon.ico                        16,24,32,48,64,128,256
  apps/desktop/assets/icon.icns                       16..1024 (real ICNS)
  apps/desktop/assets/icon-dark.png                   1024x1024 squircle (dark)
  apps/desktop/assets/icon-dark.ico                   16,24,32,48,64,128,256
  apps/desktop/assets/icon-dark.icns                  16..1024 (real ICNS)
  apps/desktop/assets/icon.icon/icon.json             Icon Composer manifest (macOS 26)
  apps/desktop/assets/icon.icon/Assets/border-*.png   1024 border band, light/dark ink
  apps/desktop/assets/icon.icon/Assets/art-*.png      1024 girl (+ commit badge), light/dark
  apps/desktop/assets/appx/Wide310x150Logo.png        310x150, squircle 100 centered
  apps/desktop/assets/appx/StoreLogo.png              50x50 squircle
  apps/desktop/assets/appx/Square44x44Logo.png        44x44 squircle
  apps/desktop/assets/appx/Square150x150Logo.png      150x150 squircle
  apps/desktop/assets/appx/*-dark.png                 dark-appearance logos
  apps/desktop/public/apple-touch-icon.png            1024x1024 squircle
  apps/desktop/public/nous-girl.png                   256x256 squircle, black girl (light mark)
  apps/desktop/public/nous-girl-dark.png              256x256 squircle, white girl (dark mark)
  apps/bootstrap-installer/src-tauri/icons/32x32.png       32x32
  apps/bootstrap-installer/src-tauri/icons/128x128.png     128x128
  apps/bootstrap-installer/src-tauri/icons/128x128@2x.png  256x256
  apps/bootstrap-installer/src-tauri/icons/icon.ico        16,32,64,128,256
  apps/bootstrap-installer/src-tauri/icons/icon.icns       16..1024
  apps/bootstrap-installer/public/nous-girl.png   256x256 squircle mark (light)
  website/static/img/logo.png                     1772x1799 girl alone, transparent (light)
  website/static/img/logo-dark.png                1772x1799 girl alone, transparent (dark)
  website/static/img/nous-logo.png                150x150 on white (opaque)
  website/static/img/nous-logo-dark.png           150x150 on #0d1117 (opaque)
  website/static/img/favicon-16x16.png            16x16
  website/static/img/favicon-32x32.png            32x32
  website/static/img/apple-touch-icon.png         180x180
  website/static/img/favicon.ico                  16,32,48
  website/static/img/favicon.svg                  copy of the light master
  web/public/favicon.ico                          16,32,48
"""

from __future__ import annotations

import argparse
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
        "resvg-py is missing: run the generator with a Hermes runtime interpreter\n"
        "  (HERMES_PYTHON=<hermes venv python> node scripts/generate-icons.mjs)"
    )

# Copy of hermes_cli.update_channel._CANARY_TAG_RE: builders run this renderer
# on the runtime dependencies without the application package installed
# (Docker, bundles). tests/scripts/test_icon_flavors.py pins it to the canonical one.
_CANARY_TAG_RE = re.compile(
    r"^v(?:0|[1-9]\d{0,2})\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"\+canary\.20\d{6}T\d{6}Z$"
)

# The nous dark background (#0d1117) — fixed dark tile/background everywhere.
DARK_HEX = "#0d1117"
DARK_RGB = (13, 17, 23)
BORDER_FRACTION = 0.0407747197

# Portrait boxes fitted to the reference at equal visible tile width, with
# uniform scaling about the tile center followed by an up-left translation.
# Keep their y coordinate: bottom anchoring would undo the registration.
GIRL_BOXES = {
    "squircle-light.svg": (72.149433, 104.703674, 872.767801, 872.767801),
    "squircle-dark.svg": (72.149433, 104.703674, 872.767801, 872.767801),
    "squircle-mac-light.svg": (157.949166, 184.039504, 702.522501, 702.522501),
    "squircle-mac-dark.svg": (157.949166, 184.039504, 702.522501, 702.522501),
}
# The brand-kit SVG canvas (both girl svgs share this viewBox).
GIRL_VIEWBOX = 5487.0615

# Target sizes for --check's structural verification: relpath -> (format, size)
CHECK_SIZES: dict[str, tuple[str, tuple[int, int]]] = {
    "apps/desktop/assets/icon.png": ("PNG", (1024, 1024)),
    "apps/desktop/assets/icon-dark.png": ("PNG", (1024, 1024)),
    "apps/desktop/assets/icon.icon/Assets/border-light.png": ("PNG", (1024, 1024)),
    "apps/desktop/assets/icon.icon/Assets/border-dark.png": ("PNG", (1024, 1024)),
    "apps/desktop/assets/icon.icon/Assets/art-light.png": ("PNG", (1024, 1024)),
    "apps/desktop/assets/icon.icon/Assets/art-dark.png": ("PNG", (1024, 1024)),
    "apps/desktop/assets/appx/Wide310x150Logo.png": ("PNG", (310, 150)),
    "apps/desktop/assets/appx/Wide310x150Logo-dark.png": ("PNG", (310, 150)),
    "apps/desktop/assets/appx/StoreLogo.png": ("PNG", (50, 50)),
    "apps/desktop/assets/appx/StoreLogo-dark.png": ("PNG", (50, 50)),
    "apps/desktop/assets/appx/Square44x44Logo.png": ("PNG", (44, 44)),
    "apps/desktop/assets/appx/Square44x44Logo-dark.png": ("PNG", (44, 44)),
    "apps/desktop/assets/appx/Square150x150Logo.png": ("PNG", (150, 150)),
    "apps/desktop/assets/appx/Square150x150Logo-dark.png": ("PNG", (150, 150)),
    "apps/desktop/public/apple-touch-icon.png": ("PNG", (1024, 1024)),
    "apps/desktop/public/nous-girl.png": ("PNG", (256, 256)),
    "apps/desktop/public/nous-girl-dark.png": ("PNG", (256, 256)),
    "apps/bootstrap-installer/src-tauri/icons/32x32.png": ("PNG", (32, 32)),
    "apps/bootstrap-installer/src-tauri/icons/128x128.png": ("PNG", (128, 128)),
    "apps/bootstrap-installer/src-tauri/icons/128x128@2x.png": ("PNG", (256, 256)),
    "apps/bootstrap-installer/public/nous-girl.png": ("PNG", (256, 256)),
    "website/static/img/logo.png": ("PNG", (1772, 1799)),
    "website/static/img/logo-dark.png": ("PNG", (1772, 1799)),
    "website/static/img/nous-logo.png": ("PNG", (150, 150)),
    "website/static/img/nous-logo-dark.png": ("PNG", (150, 150)),
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
    ("apps/desktop/assets/icon.icon/icon.json", "icon_manifest", None),
    ("apps/desktop/assets/icon.icon/Assets/border-light.png", "icon_border", "#000000"),
    ("apps/desktop/assets/icon.icon/Assets/border-dark.png", "icon_border", "#ffffff"),
    ("apps/desktop/assets/icon.icon/Assets/art-light.png", "icon_art", "black"),
    ("apps/desktop/assets/icon.icon/Assets/art-dark.png", "icon_art", "white"),
    ("apps/desktop/assets/appx/Wide310x150Logo.png", "wide", (310, 150)),
    ("apps/desktop/assets/appx/StoreLogo.png", "png", 50),
    ("apps/desktop/assets/appx/Square44x44Logo.png", "png", 44),
    ("apps/desktop/assets/appx/Square150x150Logo.png", "png", 150),
    ("apps/desktop/assets/appx/Wide310x150Logo-dark.png", "wide_dark", (310, 150)),
    ("apps/desktop/assets/appx/StoreLogo-dark.png", "png_dark", 50),
    ("apps/desktop/assets/appx/Square44x44Logo-dark.png", "png_dark", 44),
    ("apps/desktop/assets/appx/Square150x150Logo-dark.png", "png_dark", 150),
    ("apps/desktop/public/apple-touch-icon.png", "png", 1024),
    ("apps/desktop/public/nous-girl.png", "girl_light", 256),
    ("apps/desktop/public/nous-girl-dark.png", "girl_dark", 256),
    ("apps/bootstrap-installer/src-tauri/icons/32x32.png", "png", 32),
    ("apps/bootstrap-installer/src-tauri/icons/128x128.png", "png", 128),
    ("apps/bootstrap-installer/src-tauri/icons/128x128@2x.png", "png", 256),
    ("apps/bootstrap-installer/src-tauri/icons/icon.ico", "ico", [16, 32, 64, 128, 256]),
    ("apps/bootstrap-installer/src-tauri/icons/icon.icns", "icns", None),
    ("apps/bootstrap-installer/public/nous-girl.png", "girl_light", 256),
    ("website/static/img/logo.png", "logo", None),
    ("website/static/img/logo-dark.png", "logo_dark", None),
    ("website/static/img/nous-logo.png", "png_white", 150),
    ("website/static/img/nous-logo-dark.png", "png_dark_white", 150),
    ("website/static/img/favicon-16x16.png", "png", 16),
    ("website/static/img/favicon-32x32.png", "png", 32),
    ("website/static/img/apple-touch-icon.png", "png", 180),
    ("website/static/img/favicon.ico", "ico", [16, 32, 48]),
    ("website/static/img/favicon.svg", "svg_copy", None),
    ("web/public/favicon.ico", "ico", [16, 32, 48]),
]

# ─── girl art extraction ────────────────────────────────────────────────────

class IconArt:
    """One generation's rendering inputs and caches; never writes to source."""

    def __init__(self, source: Path, *, colors: tuple[str, str] | None = None, commit: str = ""):
        assets = source / "assets"
        self.colors = colors
        self.commit = commit
        self.girls = {color: assets / f"nous-girl-{color}.svg" for color in ("black", "white")}
        self.backgrounds = assets / "backgrounds"
        self.paths: dict[str, str] = {}
        self.bboxes: dict[str, tuple[float, float, float, float]] = {}
        self.master = compose_svg(self, "black", "squircle-light.svg")
        self.master_dark = compose_svg(self, "white", "squircle-dark.svg")
        # macOS icons sit on Apple's 824-on-1024 grid, not the full-bleed
        # squircle: same art, mac-grid backgrounds, icns targets only.
        self.master_mac = compose_svg(self, "black", "squircle-mac-light.svg")
        self.master_mac_dark = compose_svg(self, "white", "squircle-mac-dark.svg")


def girl_path(art: IconArt, girl: str) -> str:
    """The girl `<path>` element with editor metadata stripped (resvg rejects
    undeclared inkscape/sodipodi prefixes)."""
    if girl not in art.paths:
        src = art.girls[girl].read_text(encoding="utf-8-sig")
        m = re.search(r"<path\b.*?/>", src, re.S)
        assert m, f"no <path> found in {art.girls[girl].name}"
        path = re.sub(r'\s+(inkscape|sodipodi):[a-zA-Z-]+="[^"]*"', "", m.group(0))
        art.paths[girl] = path
    return art.paths[girl]


def girl_bbox(art: IconArt, girl: str) -> tuple[float, float, float, float]:
    """Art bounding box in the girl SVG's coordinate space, measured by
    rendering once and taking the alpha bbox (robust to art changes)."""
    if girl not in art.bboxes:
        data = resvg_py.svg_to_bytes(svg_path=str(art.girls[girl]), width=512, height=512)
        im = Image.open(io.BytesIO(data))
        bx, by, bx2, by2 = im.getchannel("A").point(lambda v: 255 if v > 0 else 0).getbbox()
        s = GIRL_VIEWBOX / 512.0
        art.bboxes[girl] = (bx * s, by * s, (bx2 - bx) * s, (by2 - by) * s)
    return art.bboxes[girl]


def girl_layer(
    art: IconArt, girl: str, box: tuple[float, float, float, float],
    *, align: str = "xMidYMid",
) -> str:
    """Nested-svg layer: girl art (bbox as viewBox) placed into `box` — the
    box's aspect is preserved via 'meet', so the girl never distorts."""
    bx, by, bw, bh = girl_bbox(art, girl)
    x, y, w, h = box
    return (
        f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="{bx} {by} {bw} {bh}" '
        f'preserveAspectRatio="{align} meet">\n'
        f"    {girl_path(art, girl)}\n"
        "  </svg>"
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


def commit_layer(commit: str, bg: str) -> str:
    cells = []
    for index, char in enumerate(commit[:7]):
        for row, bits in enumerate(HEX_GLYPHS[char]):
            for col in range(5):
                if bits & (1 << (4 - col)):
                    x, y = 184 + (index * 6 + col) * 16, 48 + row * 16
                    cells.append(f"M{x} {y}h16v16h-16z")
    # The badge follows the tile's mac HIG inset, never the outer canvas.
    transform = f' transform="translate(100 100) scale({824 / 1024})"' if "-mac-" in bg else ""
    return (
        f'<g{transform}><rect x="160" y="28" width="704" height="152" rx="24" fill="#29090c"/>'
        f'<path fill="#ffffff" d="{"".join(cells)}"/></g>'
    )


def drag_bottom_nodes(path: ET.Element, *, cutoff: float, band: float, distance: float) -> None:
    """Drag lower nodes and curve handles, tapering to zero above the bottom band."""
    y_scale, y_offset = 1.0, 0.0
    transform = path.get("transform")
    if transform:
        matrix = re.fullmatch(r"matrix\(([^)]+)\)", transform)
        if matrix is None:
            raise ValueError("bottom node edits require an axis-aligned matrix")
        _, b, c, y_scale, _, y_offset = map(float, re.split(r"[\s,]+", matrix[1].strip()))
        if b != 0 or c != 0 or y_scale <= 0:
            raise ValueError("bottom node edits require an upright axis-aligned matrix")

    # The brand exports use explicit absolute M/L/C commands. Reject other
    # commands rather than silently corrupting relative coordinates or arcs.
    tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", path.attrib["d"])
    counts = {"M": 2, "L": 2, "C": 6, "z": 0, "Z": 0}
    index = 0
    while index < len(tokens):
        command = tokens[index]
        if command not in counts:
            raise ValueError("bottom node edits require explicit absolute M/L/C commands")
        count = counts[command]
        if index + count >= len(tokens):
            raise ValueError("incomplete SVG path command")
        for offset in range(2, count + 1, 2):
            token_index = index + offset
            raw_y = float(tokens[token_index])
            amount = min(1.0, max(0.0, (raw_y * y_scale + y_offset - cutoff) / band))
            if amount:
                weight = amount * amount * (3 - 2 * amount)
                tokens[token_index] = f"{raw_y + distance * weight / y_scale:.12g}"
        index += count + 1
    path.set("d", " ".join(tokens))


def portrait_layer(art: IconArt, girl: str, bg: str, join_bottom: float) -> ET.Element:
    """The registered girl with her lowest nodes dragged down to `join_bottom`
    (the border's inner edge) so hair meets the ring instead of floating."""
    box = GIRL_BOXES[bg]
    portrait = ET.fromstring(girl_layer(art, girl, box, align="xMidYMax"))
    _, y, portrait_width, portrait_height = box
    _, by, bw, bh = girl_bbox(art, girl)
    scale = min(portrait_width / bw, portrait_height / bh)
    drag_bottom_nodes(
        portrait[0], cutoff=by + bh * 0.97, band=bh * 0.02,
        distance=max(0.0, join_bottom - (y + portrait_height)) / scale,
    )
    # Keep the fitted viewBox fixed, but let edited nodes reach into the border.
    portrait.set("overflow", "visible")
    return portrait


def compose_svg(art: IconArt, girl: str, bg: str) -> str:
    """Full svg text: background + girl layer, in the background's native
    coordinate space (resvg scales to whatever output size is requested, so
    the composition is size-agnostic — no manual box scaling)."""
    inner, w, h = background_inner(art, bg)
    background = ET.fromstring(f"<g>{inner}</g>")
    tile = background.find("{http://www.w3.org/2000/svg}rect")
    assert tile is not None, f"no background rectangle in {bg}"
    geometry = {key: float(tile.attrib[key]) for key in ("x", "y", "width", "height", "rx")}
    thickness = geometry["width"] * BORDER_FRACTION
    # An inward stroke keeps the outer platform geometry unchanged. Subtracting
    # the same inset from rx (not scaling rx) keeps the corner thickness uniform.
    inset = {"x": 1, "y": 1, "width": -2, "height": -2, "rx": -1}
    silhouette = ET.Element("rect", {key: str(value) for key, value in geometry.items()})
    for key, value in geometry.items():
        tile.set(key, str(value + inset[key] * thickness / 2))
    tile.set("stroke", "#000000" if girl == "black" else "#ffffff")
    tile.set("stroke-width", str(thickness))
    inner = "".join(ET.tostring(child, encoding="unicode") for child in background)
    clip = ET.tostring(silhouette, encoding="unicode")
    portrait = portrait_layer(art, girl, bg, geometry["y"] + geometry["height"] - thickness + 10)
    badge = f"  {commit_layer(art.commit, bg)}\n" if art.commit else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">\n'
        f'  <defs><clipPath id="icon-silhouette">{clip}</clipPath></defs>\n'
        f"  {inner.strip()}\n"
        f"{badge}"
        f'  <g clip-path="url(#icon-silhouette)">{ET.tostring(portrait, encoding="unicode")}</g>\n'
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


def icon_art_svg(art: IconArt, girl: str) -> str:
    """Art layer: the girl registered as on the canvas-filling squircle, joined
    to the border band; the commit badge rides along for commit builds."""
    thickness = ICON_CANVAS * BORDER_FRACTION
    bg = "squircle-light.svg"  # canvas-filling registration; the system supplies the grid
    portrait = portrait_layer(art, girl, bg, ICON_CANVAS - thickness + 10)
    badge = f"  {commit_layer(art.commit, bg)}\n" if art.commit else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_CANVAS} {ICON_CANVAS}">\n'
        f"{badge}"
        f"  {ET.tostring(portrait, encoding='unicode')}\n"
        "</svg>\n"
    )


def icon_color(hex_color: str) -> str:
    """Icon Composer colour literal for an opaque sRGB hex colour."""
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return f"srgb:{r:.5f},{g:.5f},{b:.5f},1.00000"


def icon_manifest(art: IconArt) -> str:
    """icon.json: flat brand fill (flavoured for canary/commit builds), a
    border layer and an art layer, each with a dark-appearance variant."""
    light, dark = art.colors or ("#ffffff", DARK_HEX)

    def layer(name: str) -> dict:
        # No fixed "image-name": actool treats it as the image for every
        # appearance and drops the specializations, so the dark variant would
        # never reach Assets.car (black girl on the dark fill).
        return {
            "name": name,
            "image-name-specializations": [
                {"value": f"{name}-light.png"},
                {"appearance": "dark", "value": f"{name}-dark.png"},
            ],
            "glass": False,
            "hidden": False,
        }

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
            "layers": [layer("border"), layer("art")],
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


def girl_mark(art: IconArt, kind: str, size: int) -> Image.Image:
    """The girl in the app-icon squircle (BrandMark asset) — the mark IS the
    icon shape. girl_light: black girl on white squircle.  girl_dark: white
    girl on #0d1117 squircle."""
    if kind == "girl_light":
        girl, bg = "black", "squircle-light.svg"
    else:
        girl, bg = "white", "squircle-dark.svg"
    return render_svg(compose_svg(art, girl, bg), size)


def build_logo_image(art: IconArt, dark: bool = False) -> Image.Image:
    """1772x1799 wordmark: the girl alone on transparency (no frame), centered.
    Light = black girl, dark = white girl — the consuming surface's background
    (navbar light/dark) shows through."""
    W, H = 1772, 1799
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">\n  {girl_layer(art, "white" if dark else "black", (0, 0, W, H))}\n</svg>\n'
    return render_svg(svg, (W, H))


def target_bytes(art: IconArt, kind: str, arg: object) -> bytes:
    """Produce the exact bytes for one target. Shared by write + check."""
    if kind == "svg":
        return art.master.encode("utf-8")
    if kind == "svg_dark":
        return art.master_dark.encode("utf-8")
    if kind == "svg_copy":
        return art.master.encode("utf-8")
    if kind == "icon_manifest":
        return icon_manifest(art).encode("utf-8")

    buf = io.BytesIO()
    if kind == "png":
        save_png(render(art.master, arg), buf)
    elif kind == "png_dark":
        save_png(render(art.master_dark, arg), buf)
    elif kind == "png_white":
        render(art.master, arg, background="#ffffff").convert("RGB").save(buf, "PNG", optimize=True)
    elif kind == "png_dark_white":
        render(art.master_dark, arg, background=DARK_HEX).convert("RGB").save(buf, "PNG", optimize=True)
    elif kind in ("girl_light", "girl_dark"):
        save_png(girl_mark(art, kind, arg), buf)
    elif kind == "icon_border":
        save_png(render_svg(icon_border_svg(arg), ICON_CANVAS), buf)
    elif kind == "icon_art":
        save_png(render_svg(icon_art_svg(art, arg), ICON_CANVAS), buf)
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
    elif kind == "wide":
        w, h = arg
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        paste_centered(canvas, render(art.master, 100))
        canvas.save(buf, "PNG")
    elif kind == "wide_dark":
        w, h = arg
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        paste_centered(canvas, render(art.master_dark, 100))
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
    tag = os.environ.get("HERMES_PAYLOAD_TAG", "")
    commit = os.environ.get("HERMES_BUILD_COMMIT", "")
    if commit:
        if tag:
            raise ValueError("Commit builds cannot also select HERMES_PAYLOAD_TAG")
        if not re.fullmatch(r"[a-f0-9]{40}", commit):
            raise ValueError("HERMES_BUILD_COMMIT requires an exact full 40-character SHA")
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
            elif kind in ("icns", "icns_dark"):
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
        "apps/desktop/public/nous-girl.png",
        "apps/desktop/public/nous-girl-dark.png",
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
