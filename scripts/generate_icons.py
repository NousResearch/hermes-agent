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
  dark) are GENERATED artifacts — squircle background + girl nested into the
  824px HIG content safe zone. The light master drives every squircle target;
  the dark master drives the dark-appearance targets. macOS is the exception:
  its icns targets render from an in-memory mac master that puts the same
  squircle on Apple's 824x824 (r=185.4) grid — centered in 1024 with 100px
  margins — so the icon matches the size of Apple-template neighbors.

The girl is nested via its art bbox as viewBox, so it always lands centered in
the box (824 safe zone for squircles / height-fitted for the marks) without
distortion. The girl art corners sit ~185px from the squircle arc centers vs
the 245px radius, so no art touches the rounded corners on any OS mask.

GENERATED OUTPUTS ARE NOT COMMITTED. Everything this script writes is
gitignored and regenerated on demand by the consuming pipelines (website
prebuild, desktop prebuild/predev, installer prebuild, web prebuild — via
scripts/generate-icons.mjs). The freshness lane (icons-freshness-check.yml)
runs --check, which regenerates in memory and asserts structural invariants
(sizes, transparency, container frame sets) — there are no committed bytes to
byte-compare against.

Rendering: resvg (resvg-py) for SVG -> PNG fidelity at every size.
Containers: Pillow for multi-size .ico and .icns.

Build dependencies:
    The runner uses the locked icon-build group in an isolated environment.
    resvg-py is not a runtime extra and does not enter production payloads.

Outputs (30 files):
  assets/icon-master.svg                              generated light master
  assets/icon-master-dark.svg                         generated dark master
  apps/desktop/assets/icon.png                        1024x1024 squircle (light)
  apps/desktop/assets/icon.ico                        16,24,32,48,64,128,256
  apps/desktop/assets/icon.icns                       16..1024 (real ICNS)
  apps/desktop/assets/icon-dark.png                   1024x1024 squircle (dark)
  apps/desktop/assets/icon-dark.ico                   16,24,32,48,64,128,256
  apps/desktop/assets/icon-dark.icns                  16..1024 (real ICNS)
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
import re
import sys
from pathlib import Path

from PIL import Image

try:
    import resvg_py
except ImportError:
    sys.exit(
        "resvg-py is build-only. Run the isolated generator with:\n"
        "  node scripts/generate-icons.mjs"
    )

# The nous dark background (#0d1117) — fixed dark tile/background everywhere.
DARK_HEX = "#0d1117"
DARK_RGB = (13, 17, 23)

# Girl placement per background: (x, y, w, h) in that background's coordinate
# space. Full-bleed squircles put the girl in the 824px HIG content safe zone
# (centered, 100px pad on a 1024 canvas); the mac-grid squircle is itself 824
# on 1024, so the girl box scales by 824/1024 to keep the same relative size
# inside the shape. Marks reuse the full-bleed squircles.
GIRL_BOXES = {
    "squircle-light.svg": (100, 100, 824, 824),
    "squircle-dark.svg": (100, 100, 824, 824),
    "squircle-mac-light.svg": (180.5, 180.5, 663, 663),
    "squircle-mac-dark.svg": (180.5, 180.5, 663, 663),
}
# The brand-kit SVG canvas (both girl svgs share this viewBox).
GIRL_VIEWBOX = 5487.0615

# Target sizes for --check's structural verification: relpath -> (format, size)
CHECK_SIZES: dict[str, tuple[str, tuple[int, int]]] = {
    "apps/desktop/assets/icon.png": ("PNG", (1024, 1024)),
    "apps/desktop/assets/icon-dark.png": ("PNG", (1024, 1024)),
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

    def __init__(self, source: Path):
        assets = source / "assets"
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
        src = art.girls[girl].read_text(encoding="utf-8")
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


def girl_layer(art: IconArt, girl: str, box: tuple[float, float, float, float]) -> str:
    """Nested-svg layer: girl art (bbox as viewBox) placed into `box` — the
    box's aspect is preserved via 'meet', so the girl never distorts."""
    bx, by, bw, bh = girl_bbox(art, girl)
    x, y, w, h = box
    return (
        f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="{bx} {by} {bw} {bh}">\n'
        f"    {girl_path(art, girl)}\n"
        "  </svg>"
    )


def background_inner(art: IconArt, name: str) -> tuple[str, int, int]:
    """Inner content + (width, height) of a background SVG asset."""
    text = (art.backgrounds / name).read_text(encoding="utf-8")
    m = re.search(r'<svg\b[^>]*viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"[^>]*>', text)
    assert m, f"cannot parse viewBox of {name}"
    w, h = float(m.group(1)), float(m.group(2))
    inner = re.sub(r"^.*?>\s*", "", text, count=1, flags=re.S)
    inner = re.sub(r"\s*</svg>\s*$", "", inner, flags=re.S)
    return inner, int(w), int(h)


def compose_svg(art: IconArt, girl: str, bg: str) -> str:
    """Full svg text: background + girl layer, in the background's native
    coordinate space (resvg scales to whatever output size is requested, so
    the composition is size-agnostic — no manual box scaling)."""
    inner, w, h = background_inner(art, bg)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">\n'
        f"  {inner.strip()}\n"
        f"  {girl_layer(art, girl, GIRL_BOXES[bg])}\n"
        "</svg>\n"
    )


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


def cmd_write(source: Path, out: Path) -> int:
    """Return failure when any target cannot be generated or verified."""
    art = IconArt(source)
    written = 0
    failures = 0
    for rel, kind, arg in TARGETS:
        path = out / rel
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(target_bytes(art, kind, arg))
            written += 1
        except Exception as exc:  # noqa: BLE001 - report all, then fail
            failures += 1
            print(f"  !! {rel}: FAILED ({exc})")
    print(f"[write] wrote {written}/{len(TARGETS)} files")

    print("\n[verify]")
    for rel, kind, arg in TARGETS:
        path = out / rel
        try:
            if kind in ("svg", "svg_dark", "svg_copy"):
                print(f"  {rel}: {path.stat().st_size} bytes SVG")
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
    art = IconArt(source)
    problems: list[str] = []

    # every target must generate without error
    for rel, kind, arg in TARGETS:
        try:
            target_bytes(art, kind, arg)
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
