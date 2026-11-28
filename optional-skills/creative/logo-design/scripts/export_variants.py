#!/usr/bin/env python3
"""Export the standard delivery variants of a logo from one master SVG (no required dependencies).

SVG variants (next to the master, or in --out-dir):
  <name>-black.svg          one-colour black (all paint -> #000000; gradients flattened)
  <name>-white.svg          one-colour white, for dark backgrounds
  <name>-mono-<hex>.svg     one-colour in a brand colour (repeatable --mono)
  <name>-square.svg         mark centred on a 256×256 canvas with padding (avatar / social master)
  <name>-favicon.svg        tight square version for browser tabs (works as favicon.svg)
  <name>-app-icon.svg       mark in white (or --icon-fg) on a rounded-square tile (--icon-bg)

Raster (needs any backend of render_png.py: cairosvg, rsvg-convert, inkscape, Chrome/Chromium, or macOS Quick Look):
  --png 32 512 1024         PNGs of every SVG variant written (transparent)
  --web-icons               favicon.ico (16/32/48), favicon-16/32/48.png, apple-touch-icon.png (180),
                            icon-192.png, icon-512.png, maskable-512.png, plus site.webmanifest and
                            head-snippet.html — the full web/PWA icon set

Usage:
  python3 scripts/export_variants.py brand-symbol.svg --mono "#0F7C80" --icon-bg "#0F7C80" --title "Brand"
  python3 scripts/export_variants.py brand-horizontal.svg --only black white mono --mono "#0F7C80"
  python3 scripts/export_variants.py brand-symbol.svg --web-icons --icon-bg "#0F7C80" --out-dir dist/web
  python3 scripts/export_variants.py brand-symbol.svg --favicon-source brand-symbol-small.svg --web-icons

Notes:
- One-colour conversion replaces every fill/stroke/stop colour. White shapes used as fake cut-outs become
  blobs: fix the master with real holes (fill-rule="evenodd"), or use --keep-white.
- Reversed (white) marks look heavier (irradiation); for critical uses thin the white geometry by hand.
- Small sizes deserve a simplified drawing: pass it with --favicon-source (used for favicon/app icon/web icons).
- Optical centring: --optical-offset -0.02 nudges square/app-icon/favicon marks up by 2 % of the canvas.
"""
import argparse
import json
import os
import re
import sys
import tempfile

sys.dont_write_bytecode = True  # keep the skill folder clean (no __pycache__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import svglib  # noqa: E402
import render_png  # noqa: E402

PAINT_ATTR = re.compile(r'\b(fill|stroke|stop-color)\s*=\s*"([^"]*)"')
PAINT_CSS = re.compile(r'\b(fill|stroke|stop-color)\s*:\s*([^;"}]+)')


def recolor(raw, color, keep_white=False):
    def swap(value):
        v = value.strip()
        low = v.lower()
        if low in ("none", "transparent") or low.startswith("url("):
            return v
        norm = svglib.normalize_color(v) or (low if low == "currentcolor" else None)
        if norm is None:
            return v
        if keep_white and norm == "#ffffff":
            return v
        return color

    raw = PAINT_ATTR.sub(lambda m: f'{m.group(1)}="{swap(m.group(2))}"', raw)
    raw = PAINT_CSS.sub(lambda m: f"{m.group(1)}:{swap(m.group(2))}", raw)
    # Unpainted shapes default to black; give the root a fill so they follow the new colour.
    raw = re.sub(r"<svg\b(?![^>]*\bfill=)", f'<svg fill="{color}"', raw, count=1)
    return raw


def set_title(raw, title):
    if not title:
        return raw
    raw = re.sub(r"<title>.*?</title>", "", raw, flags=re.S)
    return re.sub(r"(<svg\b[^>]*>)", lambda m: m.group(1) + f"<title>{title}</title>", raw, count=1)


def inner_markup(raw):
    m = re.search(r"<svg\b[^>]*>(.*)</svg>", raw, re.S)
    body = m.group(1) if m else raw
    return re.sub(r"<title>.*?</title>", "", body, flags=re.S)


def square_wrap(raw, root, padding=0.1, bg=None, radius=0.0, fg=None, symbol_scale=None, offset=0.0, title=None):
    """Centre the artwork's real bounding box on a 256×256 canvas (optionally on a rounded tile)."""
    vb = svglib.view_box(root)
    geo = svglib.document_geometry(root)
    bb = geo["bbox"] or (vb[0], vb[1], vb[0] + vb[2], vb[1] + vb[3])
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    side = max(w, h)
    canvas = 256.0
    target = symbol_scale if symbol_scale else (1 - 2 * padding)
    s = round(canvas * target / side, 3)
    tx = round((canvas - w * s) / 2 - bb[0] * s, 2)
    ty = round((canvas - h * s) / 2 - bb[1] * s + offset * canvas, 2)
    src = recolor(raw, fg) if fg else raw
    body = inner_markup(src)
    root_tag = re.search(r"<svg\b[^>]*>", src)
    carried = ""
    if root_tag:
        for attr in ("fill", "stroke", "fill-rule", "clip-rule", "stroke-width", "stroke-linecap", "stroke-linejoin", "style"):
            m = re.search(r'\s%s\s*=\s*"([^"]*)"' % re.escape(attr), root_tag.group(0))
            if m:
                carried += f' {attr}="{m.group(1)}"'
    tile = f'<rect width="256" height="256" rx="{canvas * radius:g}" fill="{bg}"/>' if bg else ""
    t = f"<title>{title}</title>" if title else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">{t}'
            f'{tile}<g transform="translate({tx:g} {ty:g}) scale({s:g})"{carried}>{body}</g></svg>')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("master")
    ap.add_argument("--out-dir")
    ap.add_argument("--name", help="base name for outputs (default: master file name)")
    ap.add_argument("--title", help="accessible <title> for every output, e.g. 'Harbor logo'")
    ap.add_argument("--mono", action="append", default=[], help="extra one-colour version(s), e.g. '#0F7C80'")
    ap.add_argument("--icon-bg", default="#111111", help="app-icon / apple-touch / maskable tile colour")
    ap.add_argument("--icon-fg", default="#ffffff", help="mark colour on tiles ('keep' = original colours)")
    ap.add_argument("--icon-scale", type=float, default=0.62, help="mark size on the app-icon tile (0.55–0.7 typical)")
    ap.add_argument("--optical-offset", type=float, default=0.0, help="vertical nudge for square outputs (fraction; negative = up)")
    ap.add_argument("--favicon-source", help="simplified small-size drawing used for favicon/app-icon/web icons")
    ap.add_argument("--keep-white", action="store_true", help="leave pure-white paint untouched in one-colour versions")
    ap.add_argument("--only", nargs="*", help="subset: black white mono square favicon app-icon")
    ap.add_argument("--png", nargs="*", type=int, default=[], help="PNG sizes for every SVG variant written")
    ap.add_argument("--web-icons", action="store_true", help="favicon.ico + PNG icon set + webmanifest + head snippet")
    a = ap.parse_args()

    raw, root = svglib.load_svg(a.master)
    small_raw, small_root = svglib.load_svg(a.favicon_source) if a.favicon_source else (raw, root)
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.master))
    os.makedirs(out_dir, exist_ok=True)
    base = a.name or os.path.splitext(os.path.basename(a.master))[0]
    want = set(a.only) if a.only else {"black", "white", "mono", "square", "favicon", "app-icon"}
    fg = None if a.icon_fg == "keep" else a.icon_fg
    written = []

    def write(suffix, content):
        p = os.path.join(out_dir, f"{base}-{suffix}.svg")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(content)
        written.append(p)
        return p

    if "black" in want:
        write("black", set_title(recolor(raw, "#000000", a.keep_white), a.title))
    if "white" in want:
        write("white", set_title(recolor(raw, "#ffffff"), a.title))
    if "mono" in want:
        for c in a.mono:
            hx = svglib.normalize_color(c)
            if not hx:
                print("skip invalid colour", c)
                continue
            write(f"mono-{hx[1:]}", set_title(recolor(raw, hx, a.keep_white), a.title))
    if "square" in want:
        write("square", square_wrap(raw, root, padding=0.08, offset=a.optical_offset, title=a.title))
    if "favicon" in want:
        write("favicon", square_wrap(small_raw, small_root, padding=0.02, offset=a.optical_offset, title=a.title))
    if "app-icon" in want:
        write("app-icon", square_wrap(small_raw, small_root, bg=a.icon_bg, radius=0.225, fg=fg,
                                      symbol_scale=a.icon_scale, offset=a.optical_offset, title=a.title))

    colors = svglib.collect_colors(root)
    if any(svglib.lightness(c) > 0.97 for c in colors) and len(colors) > 1:
        print("note: master contains white paint — check the one-colour files for blobs (fake cut-outs).")

    raster = []
    if a.png:
        for p in list(written):
            for size in a.png:
                txt = open(p, encoding="utf-8").read()
                vb = re.search(r'viewBox\s*=\s*"[\d.\-]+[ ,]+[\d.\-]+[ ,]+([\d.]+)[ ,]+([\d.]+)"', txt)
                ratio = float(vb.group(1)) / float(vb.group(2)) if vb else 1.0
                w, h = (size, max(1, round(size / ratio))) if ratio >= 1 else (max(1, round(size * ratio)), size)
                png = p[:-4] + f"-{size}.png"
                if render_png.render(p, png, w, h):
                    raster.append(png)
                else:
                    print("PNG export skipped: no rendering backend (see: python3 scripts/render_png.py --which)")
                    break

    if a.web_icons:
        with tempfile.TemporaryDirectory() as tmp:
            fav_svg = os.path.join(tmp, "fav.svg")
            with open(fav_svg, "w", encoding="utf-8") as fh:
                fh.write(square_wrap(small_raw, small_root, padding=0.02, offset=a.optical_offset, title=a.title))
            tile_svg = os.path.join(tmp, "tile.svg")
            with open(tile_svg, "w", encoding="utf-8") as fh:
                fh.write(square_wrap(small_raw, small_root, bg=a.icon_bg, radius=0.0, fg=fg,
                                     symbol_scale=a.icon_scale, offset=a.optical_offset))
            mask_svg = os.path.join(tmp, "mask.svg")  # maskable: keep the mark inside the 80 % safe zone
            with open(mask_svg, "w", encoding="utf-8") as fh:
                fh.write(square_wrap(small_raw, small_root, bg=a.icon_bg, radius=0.0, fg=fg,
                                     symbol_scale=min(a.icon_scale, 0.5), offset=a.optical_offset))
            jobs = [(fav_svg, "favicon-16.png", 16), (fav_svg, "favicon-32.png", 32), (fav_svg, "favicon-48.png", 48),
                    (tile_svg, "apple-touch-icon.png", 180), (tile_svg, "icon-192.png", 192),
                    (tile_svg, "icon-512.png", 512), (mask_svg, "maskable-512.png", 512)]
            ok = True
            for src, fname, size in jobs:
                dst = os.path.join(out_dir, fname)
                if render_png.render(src, dst, size, size):
                    raster.append(dst)
                else:
                    ok = False
                    print(f"web icons skipped: could not render {fname} (see: python3 scripts/render_png.py --which)")
                    break
            if ok:
                ico = os.path.join(out_dir, "favicon.ico")
                render_png.write_ico([os.path.join(out_dir, f"favicon-{s}.png") for s in (16, 32, 48)], ico)
                raster.append(ico)
                fav_copy = os.path.join(out_dir, "favicon.svg")
                with open(fav_svg, encoding="utf-8") as src, open(fav_copy, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
                raster.append(fav_copy)
                manifest = {"name": a.title or base, "icons": [
                    {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
                    {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
                    {"src": "/maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}],
                    "theme_color": a.icon_bg, "background_color": a.icon_bg, "display": "standalone"}
                with open(os.path.join(out_dir, "site.webmanifest"), "w", encoding="utf-8") as fh:
                    json.dump(manifest, fh, indent=2)
                with open(os.path.join(out_dir, "head-snippet.html"), "w", encoding="utf-8") as fh:
                    fh.write('<link rel="icon" href="/favicon.ico" sizes="48x48">\n'
                             '<link rel="icon" href="/favicon.svg" type="image/svg+xml">\n'
                             '<link rel="apple-touch-icon" href="/apple-touch-icon.png">\n'
                             '<link rel="manifest" href="/site.webmanifest">\n'
                             f'<meta name="theme-color" content="{a.icon_bg}">\n')
                raster += [os.path.join(out_dir, "site.webmanifest"), os.path.join(out_dir, "head-snippet.html")]

    for p in written + raster:
        print("wrote", p)


if __name__ == "__main__":
    main()
