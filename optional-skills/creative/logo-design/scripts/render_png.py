#!/usr/bin/env python3
"""Render SVG logos to PNG (transparent by default) and build favicon.ico files — no required dependencies.

Use it to *look* at your work (render, then view the PNG) and to produce raster deliverables.
The SVG is fitted ("contain") into an exact WIDTH×HEIGHT box with optional padding and background.

Backends are tried in order until one works:
  1. cairosvg (python package)   2. rsvg-convert   3. inkscape
  4. headless Chrome / Chromium / Edge / Brave     5. macOS Quick Look (qlmanage) — last resort
Quick Look renders on opaque white; transparency is recovered by rendering on white and black
("difference matting") when Pillow is installed, otherwise the PNG stays opaque white.

Usage:
  python3 scripts/render_png.py logo.svg -o logo.png --size 512
  python3 scripts/render_png.py lockup.svg -o lockup.png --width 1200 --height 400 --padding 0.05
  python3 scripts/render_png.py symbol.svg -o preview.png --size 256 --bg "#ffffff"
  python3 scripts/render_png.py a.svg b.svg c.svg --out-dir renders --size 400      # batch, for viewing
  python3 scripts/render_png.py favicon.svg --ico favicon.ico --ico-sizes 16 32 48
  python3 scripts/render_png.py preview.html -o preview.png --width 1400 --height 3000  # screenshot an HTML page (Chrome)
  python3 scripts/render_png.py --which                                             # show available backends
"""
import argparse
import base64
import contextlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True  # keep the skill folder clean (no __pycache__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import svglib  # noqa: E402,F401  (sets UTF-8 console output)

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "brave-browser",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


# ----------------------------------------------------------------------------- helpers

def png_size(path):
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def wrap_svg(svg_path, width, height, padding=0.0, bg=None):
    """Return SVG markup that fits the source into an exact width×height canvas (contain + padding)."""
    with open(svg_path, encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()
    raw = re.sub(r"<\?xml[^>]*\?>", "", raw)
    raw = re.sub(r"<!DOCTYPE[^>]*>", "", raw, flags=re.I)
    m = re.search(r"<svg\b[^>]*>", raw)
    if not m:
        raise ValueError(f"not an SVG: {svg_path}")
    tag = m.group(0)
    vb = re.search(r'viewBox\s*=\s*"([^"]+)"', tag)
    if not vb:
        w = re.search(r'\swidth\s*=\s*"([\d.]+)', tag)
        h = re.search(r'\sheight\s*=\s*"([\d.]+)', tag)
        if w and h:
            tag2 = tag[:-1] + f' viewBox="0 0 {w.group(1)} {h.group(1)}">'
            raw = raw.replace(tag, tag2, 1)
            tag = tag2
    pw, ph = width * padding, height * padding
    new_tag = re.sub(r'\s(width|height|x|y|preserveAspectRatio)\s*=\s*"[^"]*"', "", tag)
    new_tag = new_tag[:-1].rstrip("/") + (f' x="{pw:g}" y="{ph:g}" width="{width - 2 * pw:g}" height="{height - 2 * ph:g}"'
                                         ' preserveAspectRatio="xMidYMid meet">')
    raw = raw.replace(tag, new_tag, 1)
    bg_rect = f'<rect width="100%" height="100%" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">{bg_rect}{raw}</svg>')


def find_chrome():
    for c in CHROME_CANDIDATES:
        if os.path.isabs(c) and os.path.exists(c):
            return c
        if not os.path.isabs(c) and shutil.which(c):
            return shutil.which(c)
    return None


def available_backends():
    out = []
    try:
        import cairosvg  # noqa: F401
        out.append("cairosvg")
    except Exception:
        pass
    if shutil.which("rsvg-convert"):
        out.append("rsvg-convert")
    if shutil.which("inkscape"):
        out.append("inkscape")
    if find_chrome():
        out.append("chrome")
    if shutil.which("qlmanage"):
        out.append("qlmanage")
    return out


# ----------------------------------------------------------------------------- backends

def _cairosvg(svg_file, png, w, h):
    import cairosvg  # type: ignore
    cairosvg.svg2png(url=svg_file, write_to=png, output_width=w, output_height=h)


def _rsvg(svg_file, png, w, h):
    subprocess.run(["rsvg-convert", "-w", str(w), "-h", str(h), "-o", png, svg_file], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _inkscape(svg_file, png, w, h):
    subprocess.run(["inkscape", svg_file, "--export-type=png", f"--export-filename={png}", "-w", str(w), "-h", str(h)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@contextlib.contextmanager
def _tempdir():
    """Like tempfile.TemporaryDirectory, but cleanup never raises (Windows can keep Chrome's files locked)."""
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        for _ in range(10):
            shutil.rmtree(path, ignore_errors=True)
            if not os.path.exists(path):
                break
            time.sleep(0.3)


def _stop(proc):
    """Stop a browser process and its children (Chrome keeps helper processes alive on Windows)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _chrome(svg_file, png, w, h):
    with open(svg_file, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    page = (f"<!doctype html><html><head><style>html,body{{margin:0;padding:0;background:transparent;width:{w}px;"
            f"height:{h}px;overflow:hidden}}img{{display:block;width:{w}px;height:{h}px}}</style></head>"
            f"<body><img src='data:image/svg+xml;base64,{data}'></body></html>")
    with _tempdir() as tmp:
        html_path = os.path.join(tmp, "render.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(page)
        screenshot_html(html_path, png, w, h)


def screenshot_html(html_path, png, w, h):
    """Screenshot an HTML page (viewport w×h, transparent where the page is) with headless Chrome/Chromium."""
    chrome = find_chrome()
    if not chrome:
        raise RuntimeError("no Chromium-based browser found for HTML screenshots")
    html_path = os.path.abspath(html_path)
    with _tempdir() as tmp:
        shot = os.path.join(tmp, "shot.png")  # never poll the destination: an older file may already exist there
        cmd = [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
               "--no-default-browser-check", "--use-mock-keychain", "--password-store=basic", "--disable-extensions",
               "--default-background-color=00000000", f"--window-size={w},{h}", f"--screenshot={shot}",
               f"--user-data-dir={os.path.join(tmp, 'profile')}", "file://" + html_path]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 45
        last = -1
        while time.time() < deadline:
            if os.path.exists(shot):
                size = os.path.getsize(shot)
                if size > 0 and size == last:
                    break
                last = size
            if proc.poll() is not None and os.path.exists(shot):
                break
            time.sleep(0.25)
        try:
            proc.wait(timeout=0.5)  # it may exit by itself after --screenshot; otherwise stop the whole tree
        except subprocess.TimeoutExpired:
            pass
        _stop(proc)
        if not os.path.exists(shot):
            raise RuntimeError("chrome produced no file")
        shutil.move(shot, png)


def _qlmanage_once(svg_markup, w, h, tmp, name):
    side = max(w, h)
    p = os.path.join(tmp, name + ".svg")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(svg_markup)
    subprocess.run(["qlmanage", "-t", "-s", str(side), "-o", tmp, p], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    out = p + ".png"
    if not os.path.exists(out):
        raise RuntimeError("qlmanage produced no file")
    return out


def _qlmanage(svg_file, png, w, h, want_alpha=True):
    # Quick Look crops non-square canvases and ignores transparency: render a square canvas, crop after.
    side = max(w, h)
    with open(svg_file, encoding="utf-8") as fh:
        markup = fh.read()
    ox, oy = (side - w) / 2, (side - h) / 2
    square = re.sub(r"^<svg\b[^>]*>", f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" height="{side}" '
                                      f'viewBox="{-ox:g} {-oy:g} {side} {side}">', markup, count=1)
    with tempfile.TemporaryDirectory() as tmp:
        white = _qlmanage_once(square, w, h, tmp, "white")
        try:
            from PIL import Image  # type: ignore
        except Exception:
            shutil.copy(white, png)
            print("  note: Quick Look output is opaque white (install Pillow for transparency).", file=sys.stderr)
            return
        black_markup = re.sub(r"(<svg\b[^>]*>)", r'\1<rect x="-100000" y="-100000" width="200000" height="200000" fill="#000"/>',
                              square, count=1)
        black = _qlmanage_once(black_markup, w, h, tmp, "black")
        iw = Image.open(white).convert("RGB")
        ib = Image.open(black).convert("RGB").resize(iw.size)
        if want_alpha:
            flat = lambda im: list(im.get_flattened_data() if hasattr(im, "get_flattened_data") else im.getdata())
            wd, bd = flat(iw), flat(ib)
            px = []
            for (wr, wg, wb), (br, bg_, bb) in zip(wd, bd):
                a = 255 - max(wr - br, wg - bg_, wb - bb)
                if a >= 255:
                    px.append((br, bg_, bb, 255))
                elif a <= 0:
                    px.append((0, 0, 0, 0))
                else:
                    px.append((min(255, br * 255 // a), min(255, bg_ * 255 // a), min(255, bb * 255 // a), a))
            out = Image.new("RGBA", iw.size)
            out.putdata(px)
        else:
            out = iw
        s = out.size[0] / side
        box = (round(ox * s), round(oy * s), round((ox + w) * s), round((oy + h) * s))
        out.crop(box).resize((w, h)).save(png)


BACKENDS = {"cairosvg": _cairosvg, "rsvg-convert": _rsvg, "inkscape": _inkscape, "chrome": _chrome, "qlmanage": _qlmanage}


def render(svg_path, png_path, width, height, padding=0.0, bg=None, backend=None):
    """Render svg_path into png_path at exactly width×height. Returns the backend used, or None."""
    markup = wrap_svg(svg_path, width, height, padding, bg)
    order = [backend] if backend else available_backends()
    with _tempdir() as tmp:
        wrapped = os.path.join(tmp, "wrapped.svg")
        with open(wrapped, "w", encoding="utf-8") as fh:
            fh.write(markup)
        for name in order:
            for _attempt in range(2 if name == "chrome" else 1):  # a browser can fail once on a busy machine
                try:
                    BACKENDS[name](wrapped, png_path, width, height)
                    if os.path.exists(png_path):
                        return name
                except Exception:  # retry, then try the next backend
                    continue
    return None


def write_ico(png_paths, ico_path):
    """Write a .ico that embeds PNG images (supported by all modern browsers and Windows Vista+)."""
    entries, blobs = [], []
    for p in png_paths:
        with open(p, "rb") as fh:
            blobs.append(fh.read())
        w, h = png_size(p)
        entries.append((w, h))
    offset = 6 + 16 * len(blobs)
    header = struct.pack("<HHH", 0, 1, len(blobs))
    dirs = b""
    for (w, h), blob in zip(entries, blobs):
        dirs += struct.pack("<BBBBHHII", w if w < 256 else 0, h if h < 256 else 0, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
    with open(ico_path, "wb") as fh:
        fh.write(header + dirs + b"".join(blobs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("-o", "--out", help="output PNG (single input)")
    ap.add_argument("--out-dir", help="output folder for batch rendering")
    ap.add_argument("--size", type=int, help="square size in px")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--padding", type=float, default=0.0, help="padding as a fraction of the canvas (0–0.4)")
    ap.add_argument("--bg", help="background colour (default transparent)")
    ap.add_argument("--backend", choices=sorted(BACKENDS), help="force a backend")
    ap.add_argument("--ico", help="write a favicon .ico from the (first) input")
    ap.add_argument("--ico-sizes", nargs="*", type=int, default=[16, 32, 48])
    ap.add_argument("--which", action="store_true", help="list available backends and exit")
    a = ap.parse_args()

    if a.which:
        found = available_backends()
        print("available backends:", ", ".join(found) if found else "none — install cairosvg (pip install cairosvg) "
              "or librsvg, or use a Chromium-based browser")
        return 0
    if not a.files:
        ap.error("give at least one SVG")

    if a.ico:
        pngs = []
        with _tempdir() as tmp:
            for s in a.ico_sizes:
                p = os.path.join(tmp, f"ico-{s}.png")
                if not render(a.files[0], p, s, s, a.padding, a.bg, a.backend):
                    print("could not render; available backends:", available_backends() or "none")
                    return 1
                pngs.append(p)
            write_ico(pngs, a.ico)
        print("wrote", a.ico, f"({', '.join(map(str, a.ico_sizes))} px)")
        return 0

    rc = 0
    for f in a.files:
        if f.lower().endswith((".html", ".htm")):
            out = a.out if (a.out and len(a.files) == 1) else os.path.join(a.out_dir or os.path.dirname(os.path.abspath(f)),
                                                                          os.path.splitext(os.path.basename(f))[0] + ".png")
            w, h = a.width or 1400, a.height or 900
            try:
                screenshot_html(f, out, w, h)
                print(f"wrote {out}  {w}×{h}  via chrome (HTML screenshot)")
            except Exception as exc:
                rc = 1
                print(f"could not screenshot {f}: {exc}")
            continue
        if a.out and len(a.files) == 1:
            out = a.out
        else:
            folder = a.out_dir or os.path.dirname(os.path.abspath(f))
            os.makedirs(folder, exist_ok=True)
            out = os.path.join(folder, os.path.splitext(os.path.basename(f))[0] + ".png")
        if a.width and a.height:
            w, h = a.width, a.height
        else:
            side = a.size or 512
            w, h = side, side
            if a.width or a.height:  # keep the SVG's aspect ratio
                txt = open(f, encoding="utf-8", errors="ignore").read()
                vb = re.search(r'viewBox\s*=\s*"[\d.\-]+[ ,]+[\d.\-]+[ ,]+([\d.]+)[ ,]+([\d.]+)"', txt)
                ratio = float(vb.group(1)) / float(vb.group(2)) if vb else 1.0
                if a.width:
                    w, h = a.width, max(1, round(a.width / ratio))
                else:
                    w, h = max(1, round(a.height * ratio)), a.height
        used = render(f, out, w, h, a.padding, a.bg, a.backend)
        if used:
            print(f"wrote {out}  {w}×{h}  via {used}")
        else:
            rc = 1
            print(f"could not render {f}; available backends: {available_backends() or 'none'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
