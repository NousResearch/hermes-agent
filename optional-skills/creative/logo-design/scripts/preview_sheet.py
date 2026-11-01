#!/usr/bin/env python3
"""Generate an HTML test sheet for one or more logo SVGs (no dependencies).

Shows every concept at a ladder of sizes (16 px upward), a pixel-level favicon test, on light / dark /
brand-colour / photographic / patterned backgrounds, in treatments that expose weaknesses (greyscale,
one-colour black and white, squint blur, mirror, 180° rotation), in real contexts (browser tab, app icon,
avatar crop, website header, business card), side by side with other concepts, and on a "shelf" next to
reference logos (competitors or library examples) to judge distinctiveness.

Usage:
  python3 scripts/preview_sheet.py concept-a.svg concept-b.svg -o preview.html
  python3 scripts/preview_sheet.py logo.svg --brand-color "#0F7C80" --name "Harbor" -o preview.html
  python3 scripts/preview_sheet.py logo.svg --refs-industry payments-fintech -o shelf.html
  python3 scripts/preview_sheet.py logo.svg --refs a.svg b.svg c.svg -o shelf.html
  python3 scripts/preview_sheet.py v1.svg v2.svg v3.svg --compare-only -o compare.html

Open the HTML in a browser (agents: use the browser/screenshot tool) and actually look at it.
"""
import argparse
import html
import json
import os
import random
import sys

sys.dont_write_bytecode = True  # keep the skill folder clean (no __pycache__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import svglib  # noqa: E402

SIZES = [16, 20, 24, 32, 48, 64, 96, 128, 256]

CSS = """
:root{--ink:#1b1b1b;--muted:#6d6d6d;--line:#e4e4e0;--paper:#fafaf8}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--paper)}
header{padding:20px 24px;border-bottom:1px solid var(--line);background:#fff}h1{margin:0 0 4px;font-size:20px}
h2{font-size:17px;margin:28px 0 10px}h3{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:18px 0 8px}
.wrap{padding:0 24px 40px;max-width:1400px}.row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end}.row>.cell{flex:0 0 auto;max-width:100%}
.cell{display:flex;flex-direction:column;align-items:center;gap:6px}.cap{font-size:11px;color:var(--muted);text-align:center}
.tile{display:flex;align-items:center;justify-content:center;border-radius:10px;border:1px solid var(--line);overflow:hidden}
.checklist{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-top:14px;columns:2;font-size:13px}
.checklist li{margin:2px 0}
.concept{border-top:2px solid var(--ink);margin-top:34px}
.pix canvas{image-rendering:pixelated;image-rendering:crisp-edges;border:1px solid var(--line);background:#fff}
.tab{width:260px;height:38px;background:#dfe1e5;border-radius:10px 10px 0 0;display:flex;align-items:center;gap:8px;padding:0 12px;font-size:12px;color:#333}
.tab img{width:16px;height:16px;object-fit:contain}
.phone{width:250px;padding:18px;border-radius:28px;background:linear-gradient(160deg,#3a4a5e,#1d2733);display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.app{display:flex;flex-direction:column;align-items:center;gap:4px;font-size:9px;color:#fff}
.app .ico{width:46px;height:46px;border-radius:11px;background:rgba(255,255,255,.18)}
.nav{width:520px;max-width:100%;height:64px;background:#fff;border:1px solid var(--line);border-radius:8px;display:flex;align-items:center;padding:0 18px;gap:26px;font-size:13px;color:#444}
.nav .sp{flex:1}.nav .btn{background:#1b1b1b;color:#fff;padding:7px 12px;border-radius:6px}
.card{width:336px;height:192px;border-radius:6px;box-shadow:0 6px 24px rgba(0,0,0,.15);background:#fff;padding:22px;display:flex;flex-direction:column;justify-content:space-between}
.card .t{font-size:11px;color:#444;line-height:1.5}
.avatar{width:96px;height:96px;border-radius:50%;overflow:hidden;display:flex;align-items:center;justify-content:center;border:1px solid var(--line)}
.shelf{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.shelf .cell{align-items:stretch}.shelf .tile{height:110px;width:100%;background:#fff}.shelf .me{outline:3px solid #e8505b;outline-offset:2px}
.grey img{filter:grayscale(1)}
img.mono-black{filter:brightness(0)}img.mono-white{filter:brightness(0) invert(1)}
"""

CHECKLIST = """<ul class="checklist">
<li>Is the idea still clear at 16–24 px? Which details disappear first?</li>
<li>Does the silhouette survive the squint (blur) test?</li>
<li>Does the one-colour black version still carry the concept?</li>
<li>Does the reversed (white on dark) version look heavier? (irradiation → thin it)</li>
<li>Any unintended readings when mirrored or rotated 180°?</li>
<li>Does it hold up on the brand colour, a photo and a pattern?</li>
<li>In the favicon/app-icon tests, is the symbol optically sized and centred?</li>
<li>On the shelf: does it stand out from the references, or blend in?</li>
<li>Greyscale: do colour segments still separate by value?</li>
<li>Would you recognise it again tomorrow from memory?</li></ul>"""

PIX_JS = """
document.querySelectorAll('canvas[data-src]').forEach(c=>{
  const img=new Image(); const n=+c.dataset.n, k=+c.dataset.k; c.width=n*k; c.height=n*k;
  img.onload=()=>{const t=document.createElement('canvas'); t.width=n; t.height=n; const x=t.getContext('2d');
    x.fillStyle=c.dataset.bg; x.fillRect(0,0,n,n);
    const r=Math.min(n/img.width,n/img.height)*0.9, w=img.width*r, h=img.height*r; x.drawImage(img,(n-w)/2,(n-h)/2,w,h);
    const g=c.getContext('2d'); g.imageSmoothingEnabled=false; g.drawImage(t,0,0,n*k,n*k);};
  img.src=c.dataset.src;});
"""


def img(uri, h=None, w=None, cls="", style=""):
    dims = ""
    if h:
        dims += f"height:{h}px;"
    if w:
        dims += f"max-width:{w}px;"
    return f'<img src="{uri}" class="{cls}" style="{dims}{style}" alt="">'


def tile(inner, w, h, bg, extra=""):
    return f'<div class="tile" style="width:{w}px;height:{h}px;background:{bg};{extra}">{inner}</div>'


def concept_block(name, uri, aspect, brand, idx):
    square = aspect is None or aspect <= 1.35
    out = [f'<section class="concept"><h2>{html.escape(name)}</h2>']
    # scale ladder
    out.append("<h3>Scale ladder (height in px)</h3><div class='row'>")
    for s in SIZES:
        out.append(f"<div class='cell'>{tile(img(uri, h=s), max(40, int(s * (aspect or 1)) + 16), s + 16, '#fff')}"
                   f"<div class='cap'>{s}px</div></div>")
    out.append("</div>")
    # pixel test
    out.append("<h3>Pixel test — rendered at 16 / 32 / 48 px, enlarged</h3><div class='row pix'>")
    for n, k in ((16, 8), (32, 4), (48, 3)):
        for bg in ("#ffffff", "#111111"):
            out.append(f"<div class='cell'><canvas data-src='{uri}' data-n='{n}' data-k='{k}' data-bg='{bg}'></canvas>"
                       f"<div class='cap'>{n}px on {bg}</div></div>")
    out.append("</div>")
    # backgrounds
    photo = "radial-gradient(circle at 30% 30%,#e7b889,#8a5a3c 45%,#2d3b2f 80%)"
    pattern = "repeating-linear-gradient(45deg,#e9e4da 0 10px,#d4cbbd 10px 20px)"
    bgs = [("#ffffff", "white", ""), ("#f1f1ee", "light grey", ""), ("#111111", "black", "mono-white"),
           ("#111111", "black (original colours)", ""), (brand, f"brand {brand}", ""), (brand, "brand (white)", "mono-white"),
           (photo, "photo-like", ""), (pattern, "pattern", "")]
    out.append("<h3>Backgrounds</h3><div class='row'>")
    for bg, cap, cls in bgs:
        out.append(f"<div class='cell'>{tile(img(uri, h=72, w=200, cls=cls), 220, 120, bg)}<div class='cap'>{cap}</div></div>")
    out.append("</div>")
    # treatments
    treats = [("original", "", ""), ("greyscale", "", "filter:grayscale(1)"), ("one-colour black", "mono-black", ""),
              ("one-colour white", "mono-white", ""), ("squint (blur 2px)", "", "filter:blur(2px)"),
              ("squint small (blur 1px @48)", "", "filter:blur(1px)"), ("mirrored", "", "transform:scaleX(-1)"),
              ("rotated 180°", "", "transform:rotate(180deg)")]
    out.append("<h3>Treatments</h3><div class='row'>")
    for cap, cls, st in treats:
        bg = "#111" if cls == "mono-white" else "#fff"
        h = 48 if "@48" in cap else 96
        out.append(f"<div class='cell'>{tile(img(uri, h=h, w=200, cls=cls, style=st), 220, 140, bg)}<div class='cap'>{cap}</div></div>")
    out.append("</div>")
    # contexts
    out.append("<h3>Contexts</h3><div class='row'>")
    out.append(f"<div class='cell'><div class='tab'>{img(uri)}<span>{html.escape(name)} — Home</span>"
               f"<span style='margin-left:auto'>✕</span></div><div class='cap'>browser tab (16 px favicon)</div></div>")
    icon = (f"<div class='ico' style='background:{brand};display:flex;align-items:center;justify-content:center'>"
            f"{img(uri, h=28, w=30, cls='mono-white')}</div>")
    apps = "".join("<div class='app'><div class='ico'></div>app</div>" for _ in range(6))
    out.append(f"<div class='cell'><div class='phone'>{apps}<div class='app'>{icon}{html.escape(name[:10])}</div>{apps[:0]}"
               + "".join("<div class='app'><div class='ico'></div>app</div>" for _ in range(5))
               + "</div><div class='cap'>app icon (white symbol on brand colour)</div></div>")
    out.append(f"<div class='cell'><div class='avatar' style='background:#fff'>{img(uri, h=58, w=70)}</div>"
               f"<div class='cap'>avatar circle crop</div></div>")
    out.append(f"<div class='cell'><div class='nav'>{img(uri, h=30 if not square else 34, w=180)}<span class='sp'></span>"
               f"<span>Product</span><span>Pricing</span><span>About</span><span class='btn'>Sign up</span></div>"
               f"<div class='cap'>website header</div></div>")
    out.append(f"<div class='cell'><div class='card'>{img(uri, h=40, w=160)}<div class='t'><b>Alex Morgan</b><br>Founder<br>"
               f"alex@example.com · +00 000 000 000</div></div><div class='cap'>business card (3.5 × 2 in)</div></div>")
    out.append("</div></section>")
    return "\n".join(out)


def compare_block(items):
    out = ["<section><h2>Side by side</h2>"]
    for h, cls, bg, cap in ((96, "", "#fff", "96 px"), (32, "", "#fff", "32 px"), (64, "mono-black", "#fff", "one-colour"),
                            (64, "mono-white", "#111", "reversed")):
        out.append(f"<h3>{cap}</h3><div class='row'>")
        for name, uri, aspect in items:
            w = max(60, int(h * (aspect or 1)) + 30)
            out.append(f"<div class='cell'>{tile(img(uri, h=h, cls=cls), w, h + 30, bg)}<div class='cap'>{html.escape(name)}</div></div>")
        out.append("</div>")
    out.append("</section>")
    return "\n".join(out)


def shelf_block(items, refs):
    cells = []
    for name, uri, _ in items:
        cells.append((name, uri, True))
    for p in refs:
        cells.append((os.path.basename(p), svglib.svg_data_uri(p), False))
    random.Random(7).shuffle(cells)
    out = ["<section><h2>Shelf test</h2><p class='cap' style='text-align:left'>Your concept(s) outlined in red among "
           "reference logos at equal height. Squint: does it stand out, or blend in? Second row repeats in greyscale.</p>"]
    for grey in (False, True):
        out.append(f"<div class='shelf{' grey' if grey else ''}' style='margin-bottom:12px'>")
        for name, uri, me in cells:
            out.append(f"<div class='cell'><div class='tile{' me' if me else ''}'>{img(uri, h=56, w=130)}</div>"
                       f"<div class='cap'>{html.escape(name)}</div></div>")
        out.append("</div>")
    out.append("</section>")
    return "\n".join(out)


def refs_for_industry(industry, n):
    with open(svglib.CATALOG_PATH, encoding="utf-8") as fh:
        cat = json.load(fh)
    rows = [r for r in cat if r.get("industry") == industry and r.get("variant") == "main"]
    rows.sort(key=lambda r: (not r.get("exemplary"), r["file"]))
    return [os.path.join(svglib.LIBRARY_SVG_DIR, r["file"]) for r in rows[:n]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="logo SVG(s) to test")
    ap.add_argument("-o", "--out", default="logo-preview.html")
    ap.add_argument("--name", help="brand name (used in mock contexts)")
    ap.add_argument("--names", nargs="*", help="labels per file (default: file names)")
    ap.add_argument("--brand-color", help="brand colour for background/app-icon tests (default: most saturated colour in the first logo, else dark grey)")
    ap.add_argument("--refs", nargs="*", default=[], help="reference SVGs for the shelf test")
    ap.add_argument("--refs-industry", help="pull shelf references from the library by industry")
    ap.add_argument("--refs-count", type=int, default=11)
    ap.add_argument("--compare-only", action="store_true", help="only the side-by-side section")
    a = ap.parse_args()

    items = []
    for i, p in enumerate(a.files):
        _, root = svglib.load_svg(p)
        vb = svglib.view_box(root)
        aspect = (vb[2] / vb[3]) if vb and vb[3] else None
        label = (a.names[i] if a.names and i < len(a.names) else os.path.splitext(os.path.basename(p))[0])
        items.append((label, svglib.svg_data_uri(p), aspect))
    brand_name = a.name or items[0][0]
    color_note = "given"
    if not a.brand_color:
        _, r0 = svglib.load_svg(a.files[0])
        cols = [c for c in svglib.collect_colors(r0) if 0.12 < svglib.lightness(c) < 0.9]
        if cols:
            import colorsys
            a.brand_color = max(cols, key=lambda c: colorsys.rgb_to_hls(*[v / 255 for v in svglib.hex_to_rgb(c)])[2])
            color_note = "taken from the logo"
        else:
            a.brand_color, color_note = "#3a3a3a", "none given — neutral grey; pass --brand-color"
    refs = list(a.refs)
    if a.refs_industry:
        refs += refs_for_industry(a.refs_industry, a.refs_count)

    body = [f"<header><h1>Logo test sheet — {html.escape(brand_name)}</h1>"
            f"<div class='cap' style='text-align:left'>{len(items)} file(s) · brand colour {a.brand_color} ({color_note}). "
            f"Look critically; fix what fails, then re-run.</div>{CHECKLIST}</header><div class='wrap'>"]
    if len(items) > 1 or a.compare_only:
        body.append(compare_block(items))
    if not a.compare_only:
        for i, (label, uri, aspect) in enumerate(items):
            body.append(concept_block(label, uri, aspect, a.brand_color, i))
    if refs:
        body.append(shelf_block(items, refs))
    body.append("</div>")
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Logo test — {html.escape(brand_name)}</title>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{CSS}</style></head>"
           f"<body>{''.join(body)}<script>{PIX_JS}</script></body></html>")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {a.out}  ({len(items)} logo(s), {len(refs)} reference(s)) — open it in a browser")


if __name__ == "__main__":
    main()
