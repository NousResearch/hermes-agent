#!/usr/bin/env python3
"""Palette-variety demo — same transplant technique, 5 distinct on-brand palettes.

Addresses the "one-trick-pony palette" problem: the cohesive thread is the craft
(analog texture, type hierarchy, designed density), NOT a single colour scheme.
Each job: baoyu LAYOUT anchor (structure) + baoyu STYLE anchor (palette/look) +
explicit palette hexes + topic. Proves the feed can stay varied.

Usage (env must carry FAL_KEY):
    cd content_engine && PYTHONPATH=. ../.venv/bin/python tools/palette_demo.py
"""
from __future__ import annotations
import base64, os, sys, time, uuid
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import budget
import postprocess as pp
from PIL import Image

CR = Path.home() / "content-references"
LAY = CR / "baoyu-screenshots" / "infographic-layouts"
STY = CR / "baoyu-screenshots" / "infographic-styles"
ASTY = CR / "baoyu-screenshots" / "article-illustrator-styles"
OUT = Path(__file__).resolve().parent.parent / "output" / "palette"
OUT.mkdir(parents=True, exist_ok=True)
FAL_KEY = os.getenv("FAL_KEY", "")
EDIT_ID = "fal-ai/nano-banana-pro/edit"
COST = 0.12

CRAFT = ("Keep our brand craft constant: analog texture (film grain, halftone, "
         "subtle print/scanline), strong type hierarchy (distressed condensed "
         "display + clean labels + mono), designed information density, restrained "
         "composition. No human/anime character. All text crisp, no gibberish.")

# id, layout, style-anchor, palette-hex, light?, title, labels
JOBS = [
    ("p1_cyberneon", LAY/"iceberg.webp", STY/"cyberpunk-neon.webp", False,
     "PALETTE: near-black #00000E background, electric blue #3847FF, magenta #BD2EFF, cyan #2EE6FF accents.",
     "WHAT YOU SEE vs WHAT THE AGENT DOES",
     "Iceberg: above water 'One daily post'. Below: 'Signal scan','Dedup','Draft','Quality gates','Illustrate','Publish'."),
    ("p2_acidduo", LAY/"funnel.webp", STY/"bold-graphic.webp", False,
     "PALETTE: black background, cobalt blue #1D4ED8 and acid yellow #E3FF00 duotone, white. Bold high-energy.",
     "FROM 200 SIGNALS TO 1 POST",
     "Funnel stages top to bottom: '200 raw signals','Score & rank','Dedup 30d','Top signal','1 daily article'."),
    ("p3_synthwave", LAY/"timeline-horizontal.webp", ASTY/"retro.webp", False,
     "PALETTE: deep purple #1A0B2E background, hot pink #FF6AC1, sunset orange #FF8A3D, cyan #00E0FF. Retro synthwave.",
     "HOW HERMES GOT BUILT",
     "Horizontal timeline: 'Single script','Multi-agent','Kanban loop','Skill library','Self-hosted', each a milestone."),
    ("p4_blueprint", LAY/"pyramid.webp", STY/"technical-schematic.webp", False,
     "PALETTE: dark navy #0A1A2F blueprint background, cyan #5FE0FF technical linework, off-white #E6F2FF, faint grid.",
     "5 LEVELS OF AGENT AUTONOMY",
     "Five tiers bottom-up: L1 Scripted, L2 AI-assisted, L3 Supervised, L4 Autonomous+gate, L5 Self-improving."),
    ("p5_warmedit", LAY/"comparison-table.webp", STY/"aged-academia.webp", True,
     "PALETTE: warm parchment/cream #F2E8D5 background, charcoal #1A1A1A ink, one deep red #C0392B accent. Vintage editorial letterpress, LIGHT not dark.",
     "CODEX vs CLAUDE FOR AGENT LOOPS",
     "Two-column comparison: rows 'Cost','Tool-calling','Long context','Reasoning','Refusals'; tidy ticks/crosses per side."),
]


def _uri(p: Path) -> str:
    m = {"png":"image/png","webp":"image/webp"}.get(p.suffix.lower().lstrip("."), "image/jpeg")
    return f"data:{m};base64," + base64.b64encode(p.read_bytes()).decode()

def _upload(p: Path) -> str:
    try:
        i = requests.post("https://rest.alpha.fal.ai/storage/upload/initiate",
            headers={"Authorization":f"Key {FAL_KEY}","Content-Type":"application/json"},
            json={"content_type":"image/jpeg","file_name":f"{uuid.uuid4().hex}.jpg"}, timeout=30)
        if i.status_code==200:
            d=i.json()
            if requests.put(d["upload_url"],data=p.read_bytes(),headers={"Content-Type":"image/jpeg"},timeout=60).status_code in (200,201):
                return d["file_url"]
    except Exception as e:  # noqa: BLE001
        print(f"  upload {e}")
    return _uri(p)

def _gen(urls, prompt) -> bytes|None:
    pay={"prompt":prompt,"image_urls":urls,"aspect_ratio":"4:5","num_images":1,"resolution":"2K"}
    h={"Authorization":f"Key {FAL_KEY}","Content-Type":"application/json"}
    try:
        r=requests.post(f"https://fal.run/{EDIT_ID}",json=pay,headers=h,timeout=220)
        if r.status_code==200:
            im=r.json().get("images",[])
            if im: return requests.get(im[0]["url"],timeout=90).content
            print(f"  no imgs {list(r.json())[:6]}")
        else: print(f"  {r.status_code} {r.text[:140]}")
    except Exception as e:  # noqa: BLE001
        print(f"  gen {e}")
    return None

def _finish(raw: Path, dst: Path, light: bool) -> Path:
    img=Image.open(raw).convert("RGB")
    img=pp._contrast(img,1.1)
    img=pp._halftone(img,0.22)
    if light:
        img=pp._paper_texture(img,0.5); img=pp._film_grain(img,0.5)
    else:
        img=pp._scanlines(img,0.25); img=pp._film_grain(img,0.7); img=pp._vignette(img,0.4)
    img.save(dst,quality=92); return dst

def run():
    if not FAL_KEY: print("no FAL_KEY"); return
    cells,total=[],0.0
    for jid,lay,sty,light,pal,title,labels in JOBS:
        print(f"[palette] {jid}")
        lu=_upload(lay); su=_upload(sty)
        prompt=(f"Create an infographic. Use the FIRST image ONLY as the LAYOUT/structure "
                f"template (same designed density). Use the SECOND image ONLY as the visual "
                f"STYLE and colour reference. Title: \"{title}\". Content: {labels}\n\n{pal}\n\n{CRAFT}")
        t0=time.time(); data=_gen([lu,su],prompt); dt=time.time()-t0
        if not data: print(f"  FAIL ({dt:.0f}s)"); cells.append((jid,lay,None,None,"fail")); continue
        raw=OUT/f"{jid}_raw.png"; raw.write_bytes(data)
        fin=_finish(raw,OUT/f"{jid}_final.png",light)
        budget.record(COST,label=f"palette:{jid}"); total+=COST
        cells.append((jid,lay,raw,fin,f"{dt:.0f}s")); print(f"  OK ({dt:.0f}s)")
    def b(p): return base64.b64encode(Path(p).read_bytes()).decode() if p and Path(p).exists() else ""
    def im(p,m="png"): s=b(p); return f'<img src="data:image/{m};base64,{s}"/>' if s else '<div class=miss>x</div>'
    html=["""<!doctype html><meta charset=utf-8><title>Palette variety</title>
<style>body{background:#0c0c0c;color:#eee;font:14px system-ui;margin:24px}
.grid{display:flex;flex-wrap:wrap;gap:16px}.c{width:300px}
.c img,.miss{width:300px;border-radius:8px;background:#000}.lab{color:#9cf;font-weight:600;margin-top:6px}</style>
<h1>Palette variety — same craft DNA, five palettes</h1><div class=grid>"""]
    for jid,lay,raw,fin,note in cells:
        html.append(f"<div class=c>{im(fin)}<div class=lab>{jid}</div><div style=color:#888;font-size:12px>{note}</div></div>")
    html.append("</div>"); (OUT/"index.html").write_text("\n".join(html))
    print("="*50); print(f"Gallery: {OUT/'index.html'}  £{total:.2f}")

if __name__=="__main__":
    run()
