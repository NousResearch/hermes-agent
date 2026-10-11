#!/usr/bin/env python3
"""Build a client-ready concept presentation (single HTML file, no dependencies).

Follows references/presentation-delivery.md: title, "what we heard", then each concept with its
one-sentence idea, rationale, the mark large / small / reversed, and six mockups chosen for the client's
industry, followed by a side-by-side comparison and a recommendation. Greyscale by default for a first
round; set "greyscale": false for colour, and "final": true for a single final-design board.

Usage:
  python3 scripts/presentation_board.py spec.json -o presentation.html
  python3 scripts/presentation_board.py spec.json -o board.html --png-dir slides   # + one PNG per slide
  python3 scripts/presentation_board.py --list-mockups

spec.json (paths relative to the spec file; see templates/presentation-spec.example.json):
{
  "brand": "Kiln", "tagline": "Small-batch roasters", "brief": "…", "adjectives": ["warm", "crafted"],
  "industry": "coffee",                    # picks relevant mockups (see --list-mockups)
  "mockups": ["cup", "bag", "social"],     # optional explicit list (overrides industry preset)
  "nav": ["Shop", "Cafés", "Story"],       # optional website menu items
  "brand_color": "#B5532F", "greyscale": true, "final": false, "round": 1,
  "concepts": [
    {"name": "Kiln K", "symbol": "b-symbol.svg", "lockup": "b-horizontal.svg", "stacked": "b-stacked.svg", "avatar": "b-avatar.svg",
     "idea": "One sentence.", "rationale": ["…", "…"]}
  ],
  "recommendation": "…"
}
Each concept needs at least one of symbol / lockup / avatar; "stacked" (optional) is used on bags, signs and badges.
Optional colour artwork (otherwise the mark is forced to white on those surfaces):
  "symbol_on_tile" / "stacked_on_tile" (or "lockup_on_tile") — for brand-colour tiles (cards, app icon, cup, shirt, bag);
  "symbol_on_dark" / "lockup_on_dark" / "stacked_on_dark" — for the near-black signage, README, terminal and reversed strip.
Top-level "tile_color" overrides brand_color for the tiles (e.g. a dark ink so a bright second colour can show).
Missing ones fall back sensibly
(a wordmark-only concept can provide just "lockup" plus an "avatar" for small uses).
"""
import argparse
import datetime
import html
import json
import os
import re
import sys
import unicodedata

sys.dont_write_bytecode = True  # keep the skill folder clean (no __pycache__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import svglib  # noqa: E402

PRESETS = {
    "software": ["readme", "app-icon", "website", "terminal", "sticker", "social"],
    "consumer-app": ["app-icon", "website", "social", "favicon-tab", "sticker", "business-card"],
    "coffee": ["cup", "bag", "signage", "business-card", "social", "tote"],
    "food": ["bag", "cup", "signage", "box", "social", "business-card"],
    "retail": ["shopping-bag", "signage", "website", "box", "social", "tote"],
    "fashion": ["shopping-bag", "tshirt", "signage", "social", "business-card", "tote"],
    "services": ["business-card", "website", "signage", "social", "badge", "app-icon"],
    "event": ["badge", "tshirt", "website", "social", "sticker", "signage"],
    "education": ["website", "badge", "tshirt", "social", "business-card", "app-icon"],
    "health": ["website", "app-icon", "signage", "business-card", "social", "box"],
    "finance": ["payment-card", "app-icon", "website", "social", "business-card", "favicon-tab"],
    "saas": ["website", "favicon-tab", "app-icon", "social", "sticker", "business-card"],
}
ALIASES = {
    "fintech": "finance", "bank": "finance", "banking": "finance", "payments": "finance", "insurance": "finance",
    "accounting": "finance", "b2b-saas": "saas", "startup": "saas",
    "developer": "software", "developer-tools": "software", "devtool": "software", "tech": "software",
    "open-source": "software", "ai": "software", "app": "consumer-app", "mobile": "consumer-app", "cafe": "coffee",
    "café": "coffee", "roaster": "coffee", "restaurant": "food", "bakery": "food", "beverage": "food", "drink": "food",
    "ecommerce": "retail", "shop": "retail", "store": "retail", "agency": "services", "consulting": "services",
    "legal": "services", "conference": "event", "festival": "event", "school": "education",
    "university": "education", "clinic": "health", "wellness": "health",
}
DEFAULT_NAV = {"software": ["Docs", "Pricing", "GitHub"], "coffee": ["Shop", "Cafés", "Story"],
               "food": ["Menu", "Order", "About"], "retail": ["Shop", "New in", "Stores"]}

CSS = """
:root{--ink:#161616;--muted:#6b6b6b;--line:#e6e6e2;--paper:#f7f7f4}
*{box-sizing:border-box}body{margin:0;background:#dcdcd6;font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink)}
.slide{width:1280px;min-height:720px;margin:28px auto;background:#fff;box-shadow:0 8px 30px rgba(0,0,0,.12);padding:56px 64px 70px;position:relative;overflow:hidden}
.slide h1{font-size:44px;margin:0 0 8px;letter-spacing:-.02em}.slide h2{font-size:30px;margin:0 0 6px;letter-spacing:-.01em}
.kicker{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:14px}
.idea{font-size:22px;max-width:900px;margin:6px 0 18px}.muted{color:var(--muted)}
.foot{position:absolute;left:64px;right:64px;bottom:24px;display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}
.hero{display:grid;grid-template-columns:1.3fr 1fr;gap:40px;align-items:center}
.stage{height:420px;background:var(--paper);border-radius:14px;display:flex;align-items:center;justify-content:center}
.stage img{max-width:70%;max-height:62%}
ul.rat{padding-left:18px;margin:0}ul.rat li{margin:6px 0}
.sizes{display:flex;gap:18px;align-items:flex-end;margin-top:18px}.sizes div{display:flex;flex-direction:column;align-items:center;gap:4px;font-size:11px;color:var(--muted)}
.rev{background:#161616;border-radius:10px;padding:18px 22px;display:inline-flex;margin-top:14px}
.mock{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:10px}
.m{height:270px;border-radius:12px;overflow:hidden;position:relative;display:flex;align-items:center;justify-content:center}
.m .lbl{position:absolute;left:12px;bottom:10px;font-size:11px;color:rgba(0,0,0,.6);background:rgba(255,255,255,.75);padding:2px 8px;border-radius:20px}
.bc{width:250px;height:143px;background:#fff;border-radius:5px;box-shadow:0 10px 26px rgba(0,0,0,.22);padding:16px;display:flex;flex-direction:column;justify-content:space-between;transform:translate(-50px,-38px) rotate(-4deg)}
.bc2{position:absolute;width:250px;height:143px;border-radius:5px;transform:translate(70px,52px) rotate(6deg);box-shadow:0 10px 26px rgba(0,0,0,.25);display:flex;align-items:center;justify-content:center}
.bc .t{font-size:9px;line-height:1.5;color:#333}
.phone{width:150px;height:224px;margin-bottom:14px;border-radius:24px;background:#10151c;padding:9px;box-shadow:0 10px 30px rgba(0,0,0,.3)}
.screen{width:100%;height:100%;border-radius:16px;background:linear-gradient(170deg,#e9edf1,#cfd7df);display:grid;grid-template-columns:repeat(3,1fr);gap:9px;padding:16px 10px;align-content:start}
.ai{width:32px;height:32px;border-radius:8px;background:rgba(0,0,0,.08);justify-self:center}
.bigapp{display:flex;flex-direction:column;align-items:center;gap:10px;font-size:12px;font-weight:600;color:#333;margin-left:30px}
.bigapp .bi{width:108px;height:108px;border-radius:25px;display:flex;align-items:center;justify-content:center;box-shadow:0 12px 26px rgba(0,0,0,.2)}
.web{width:340px;height:210px;background:#fff;border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.15);overflow:hidden}
.web .bar{height:22px;background:#eceff2;display:flex;gap:5px;align-items:center;padding:0 8px}.web .bar i{width:7px;height:7px;border-radius:50%;background:#c9ced4}
.web .nav{height:44px;display:flex;align-items:center;padding:0 16px;gap:14px;font-size:9px;color:#666;border-bottom:1px solid #eee}
.web .hero2{padding:18px 16px}.web .hero2 b{display:block;font-size:17px;line-height:1.2;margin-bottom:8px}.web .hero2 span{display:block;height:6px;background:#eee;border-radius:3px;margin:5px 0;width:80%}
.sign{width:300px;height:190px;background:linear-gradient(#8c8f93,#6f7276);display:flex;flex-direction:column;align-items:center;padding-top:18px}
.sign .board{width:230px;height:74px;border-radius:6px;display:flex;align-items:center;justify-content:center;box-shadow:0 6px 16px rgba(0,0,0,.3)}
.sign .door{width:80px;height:80px;margin-top:18px;background:#3d4146;border-radius:3px 3px 0 0}
.tote{width:180px;height:210px;background:#efe7d8;border-radius:4px 4px 10px 10px;position:relative;display:flex;align-items:center;justify-content:center;box-shadow:0 8px 20px rgba(0,0,0,.15);margin-top:40px}
.tote:before{content:"";position:absolute;top:-46px;left:50%;transform:translateX(-50%);width:84px;height:70px;border:9px solid #e3d8c3;border-bottom:none;border-radius:50px 50px 0 0}
.social{width:300px;background:#fff;border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.14);overflow:hidden}
.social .cover{height:70px}.social .av{width:74px;height:74px;border-radius:50%;background:#fff;border:4px solid #fff;margin:-38px 0 0 16px;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(0,0,0,.12);overflow:hidden}
.social .meta{padding:8px 16px 16px;font-size:11px;color:#555}.social .meta b{font-size:14px;color:#111;display:block}
.cup{position:relative;width:150px;height:200px;margin-top:30px}
.cup .body{position:absolute;inset:0;background:#f4f1ea;clip-path:polygon(0 0,100% 0,86% 100%,14% 100%);display:flex;align-items:center;justify-content:center;box-shadow:inset -18px 0 24px rgba(0,0,0,.06)}
.cup .lid{position:absolute;left:-8px;right:-8px;top:-22px;height:24px;border-radius:8px 8px 3px 3px;background:#2b2b2b}
.cup .sleeve{position:absolute;left:6%;right:6%;top:36%;height:34%;clip-path:polygon(0 0,100% 0,96% 100%,4% 100%);display:flex;align-items:center;justify-content:center}
.bag{position:relative;width:170px;height:230px;background:#e9e2d6;border-radius:6px 6px 10px 10px;box-shadow:0 12px 26px rgba(0,0,0,.2);display:flex;flex-direction:column;align-items:center;padding-top:38px;gap:14px}
.bag:before{content:"";position:absolute;top:0;left:0;right:0;height:22px;background:repeating-linear-gradient(90deg,rgba(0,0,0,.08) 0 4px,transparent 4px 8px);border-radius:6px 6px 0 0}
.bag .label{width:118px;height:118px;border-radius:50%;background:#fff;display:flex;align-items:center;justify-content:center}
.bag .t{font-size:8px;letter-spacing:.14em;color:#555;text-transform:uppercase;padding:0 12px;text-align:center;line-height:1.4}
.sbag{position:relative;width:180px;height:210px;box-shadow:0 12px 26px rgba(0,0,0,.2);display:flex;align-items:center;justify-content:center;margin-top:30px}
.sbag:before{content:"";position:absolute;top:-40px;left:50%;transform:translateX(-50%);width:70px;height:56px;border:5px solid #333;border-bottom:none;border-radius:40px 40px 0 0}
.box{width:220px;height:150px;background:#f3efe6;box-shadow:14px 14px 0 rgba(0,0,0,.08),0 12px 26px rgba(0,0,0,.18);display:flex;align-items:center;justify-content:center;border-top:14px solid rgba(0,0,0,.06)}
.readme{width:360px;background:#0d1117;border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.25);padding:18px 20px;color:#c9d1d9;font-size:10px}
.readme .rh{display:flex;align-items:center;gap:10px;border-bottom:1px solid #30363d;padding-bottom:12px;margin-bottom:10px}
.readme .badges{display:flex;gap:6px;margin:8px 0}.readme .badges i{display:block;width:54px;height:12px;border-radius:3px;background:#238636}.readme .badges i:nth-child(2){background:#1f6feb}.readme .badges i:nth-child(3){background:#6e7681}
.readme .ln{height:6px;background:#21262d;border-radius:3px;margin:6px 0}
.term{width:360px;background:#1e1e1e;border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.3);overflow:hidden;font:10px/1.6 ui-monospace,Menlo,monospace;color:#d4d4d4}
.term .tb{height:22px;background:#2d2d2d;display:flex;gap:5px;align-items:center;padding:0 8px}.term .tb i{width:8px;height:8px;border-radius:50%;background:#555}
.term .tbody{padding:14px 16px;display:flex;gap:14px;align-items:center}
.sticker{width:250px;height:170px;background:linear-gradient(135deg,#9aa0a6,#6b7177);border-radius:12px;display:flex;align-items:center;justify-content:center;box-shadow:0 10px 24px rgba(0,0,0,.25)}
.sticker .s{width:100px;height:100px;border-radius:50%;background:#fff;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 6px rgba(0,0,0,.25)}
.shirt{width:210px;height:220px;display:flex;align-items:center;justify-content:center;position:relative}
.shirt svg{position:absolute;inset:0}
.badge{width:170px;height:235px;background:#fff;border-radius:10px;box-shadow:0 10px 24px rgba(0,0,0,.2);display:flex;flex-direction:column;align-items:center;padding:20px 14px;gap:14px;position:relative;margin-top:30px}
.badge:before{content:"";position:absolute;top:-60px;left:50%;width:22px;height:62px;transform:translateX(-50%)}
.badge .hole{width:40px;height:8px;border-radius:4px;background:#ddd}.badge .nm{font-size:15px;font-weight:600}.badge .rl{font-size:10px;color:#777}
.pcard{position:relative;width:300px;height:189px;border-radius:14px;box-shadow:0 14px 30px rgba(0,0,0,.28);font:600 12px ui-monospace,Menlo,monospace}
.pcard .chip{position:absolute;left:22px;top:76px;width:40px;height:30px;border-radius:6px;background:linear-gradient(135deg,#e9d8a6,#c9a95c)}
.pcard .num{position:absolute;left:22px;top:122px;letter-spacing:.12em;font-size:14px}
.pcard .holder{position:absolute;left:22px;bottom:18px;font-size:10px;letter-spacing:.14em;opacity:.85}
.pcard .brandline{position:absolute;right:20px;bottom:16px;font:700 13px system-ui,sans-serif;opacity:.9}
.tabbar{width:340px;background:#dfe1e5;border-radius:10px;padding:8px 8px 0}
.tab{width:230px;height:36px;background:#fff;border-radius:10px 10px 0 0;display:flex;align-items:center;gap:8px;padding:0 12px;font-size:11px;color:#333}
.addr{height:40px;background:#fff;display:flex;align-items:center;padding:0 12px;font-size:10px;color:#777;border-top:1px solid #eee}
.cmp{display:grid;gap:22px;margin-top:20px}.cmp .c{background:var(--paper);border-radius:12px;padding:22px;display:flex;flex-direction:column;align-items:center;gap:12px}
.cmp img{max-height:120px;max-width:80%}
img.g{filter:grayscale(1)}img.w{filter:brightness(0) invert(1)}img.k{filter:brightness(0)}
.adj span{display:inline-block;border:1px solid var(--line);border-radius:20px;padding:4px 14px;margin:0 8px 8px 0;font-size:14px}
"""


def uri(path, base):
    p = path if os.path.isabs(path) else os.path.join(base, path)
    return svglib.svg_data_uri(p)


def text_on(bg):
    """White text while it keeps 3:1 on the background, near-black below that (white if the colour can't be parsed)."""
    h = (bg or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    try:
        rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    except ValueError:
        return "#fff"
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#fff" if 1.05 / (lum + 0.05) >= 3 else "#161616"


def mock_html(kind, c):
    """c: dict with sym, lock, avatar, color, cls, name, tag, nav."""
    sym, lock, av, cls, name, tag = c["sym"], c["lock"], c["avatar"], c["cls"], c["name"], c["tag"]
    stack = c.get("stacked") or lock   # stacked lockup for tall/boxy surfaces (bags, signs, badges) when provided
    # Artwork on brand-colour tiles ("on_tile") and on the fixed near-black surfaces ("on_dark"): use the dedicated
    # colour files when given (keeps multi-colour marks), otherwise force the mark to white.
    avt, wat = (c["avatar_tile"], "") if c.get("avatar_tile") else (av, "w")
    avd, wav = (c["avatar_dark"], "") if c.get("avatar_dark") else (av, "w")
    skt, wst = (c["stack_tile"], "") if c.get("stack_tile") else (stack, "w")
    lkd, wlk = (c["lock_dark"], "") if c.get("lock_dark") else (lock, "w")
    skd, wsk = ((c.get("stack_dark") or c["lock_dark"]), "") if (c.get("stack_dark") or c.get("lock_dark")) else (stack, "w")
    tile = c["tile"]
    # e-mail / social handle / domain: plain ASCII letters and digits ("Marlow & Finch" -> "marlowfinch")
    handle = re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower())
    n = html.escape(name)
    if kind == "business-card":
        return (f'<div class="m" style="background:#d9d4cb"><div class="bc"><img class="{cls}" src="{lock}" style="max-height:34px;max-width:170px">'
                f'<div class="t"><b>Alex Morgan</b><br>Founder<br>alex@{handle}.com</div></div>'
                f'<div class="bc2" style="background:{tile}"><img class="{wat}" src="{avt}" style="max-height:60px;max-width:120px"></div><span class="lbl">Business cards</span></div>')
    if kind == "app-icon":
        cells = "".join('<div class="ai"></div>' for _ in range(4))
        icon = (f'<div class="ai" style="background:{tile};display:flex;align-items:center;justify-content:center">'
                f'<img class="{wat}" src="{avt}" style="max-height:19px;max-width:21px"></div>')
        # the home screen shows the icon among others; the large tile beside it shows it at store size
        big = (f'<div class="bigapp"><div class="bi" style="background:{tile}"><img class="{wat}" src="{avt}" '
               f'style="max-height:62px;max-width:70px"></div>{n}</div>')
        return (f'<div class="m" style="background:#eef0f2"><div class="phone"><div class="screen">{cells}{icon}'
                + "".join('<div class="ai"></div>' for _ in range(7)) + f'</div></div>{big}<span class="lbl">App icon</span></div>')
    if kind == "website":
        nav = "".join(f"<span>{html.escape(x)}</span>" for x in c["nav"])
        return (f'<div class="m" style="background:#e4e8ec"><div class="web"><div class="bar"><i></i><i></i><i></i></div>'
                f'<div class="nav"><img class="{cls}" src="{lock}" style="max-height:20px;max-width:110px"><span style="flex:1"></span>{nav}</div>'
                f'<div class="hero2"><b>{html.escape(tag) or n}</b><span></span><span style="width:60%"></span></div></div><span class="lbl">Website</span></div>')
    if kind == "signage":
        return (f'<div class="m" style="background:#b9bcc0"><div class="sign"><div class="board" style="background:#161616">'
                f'<img class="{wsk}" src="{skd}" style="max-height:{58 if stack != lock else 34}px;max-width:190px"></div><div class="door"></div></div><span class="lbl">Storefront / signage</span></div>')
    if kind == "tote":
        return (f'<div class="m" style="background:#cfc6b6"><div class="tote"><img class="k" src="{av}" style="max-height:80px;max-width:110px;opacity:.88"></div>'
                '<span class="lbl">Tote bag</span></div>')
    if kind == "social":
        return (f'<div class="m" style="background:#e9e9ef"><div class="social"><div class="cover" style="background:{tile}"></div>'
                f'<div class="av"><img class="{cls}" src="{av}" style="max-height:42px;max-width:50px"></div>'
                f'<div class="meta"><b>{n}</b>@{handle} · {html.escape(tag)}</div></div><span class="lbl">Social profile</span></div>')
    if kind == "cup":
        return (f'<div class="m" style="background:#d8cfc2"><div class="cup"><div class="lid"></div><div class="body"></div>'
                f'<div class="sleeve" style="background:{tile}"><img class="{wat}" src="{avt}" style="max-height:44px;max-width:80px"></div></div>'
                '<span class="lbl">Paper cup</span></div>')
    if kind == "bag":
        return (f'<div class="m" style="background:#cdbfa9"><div class="bag"><div class="label"><img class="{cls}" src="{av}" style="max-height:60px;max-width:84px"></div>'
                f'<img class="{cls}" src="{lock}" style="max-height:18px;max-width:120px"><div class="t">{html.escape(tag)[:40]}</div></div>'
                '<span class="lbl">Packaging bag</span></div>')
    if kind == "shopping-bag":
        return (f'<div class="m" style="background:#e3ded6"><div class="sbag" style="background:{tile}"><img class="{wst}" src="{skt}" style="max-height:{96 if stack != lock else 40}px;max-width:130px"></div>'
                '<span class="lbl">Shopping bag</span></div>')
    if kind == "box":
        return (f'<div class="m" style="background:#d6d9dc"><div class="box"><img class="{cls}" src="{lock}" style="max-height:44px;max-width:160px"></div>'
                '<span class="lbl">Packaging</span></div>')
    if kind == "readme":
        return (f'<div class="m" style="background:#e6e8eb"><div class="readme"><div class="rh"><img class="{wlk}" src="{lkd}" style="max-height:26px;max-width:180px"></div>'
                f'<div>{html.escape(tag) or n}</div><div class="badges"><i></i><i></i><i></i></div><div class="ln"></div><div class="ln" style="width:70%"></div>'
                '<div class="ln" style="width:85%"></div></div><span class="lbl">GitHub README</span></div>')
    if kind == "terminal":
        return (f'<div class="m" style="background:#cfd3d8"><div class="term"><div class="tb"><i></i><i></i><i></i></div>'
                f'<div class="tbody"><img class="{wav}" src="{avd}" style="max-height:42px;max-width:42px"><div>$ {handle} --version<br>'
                f'<span style="color:#6a9955">{n} 1.0.0</span><br>$ {handle} search "error"</div></div></div><span class="lbl">Terminal</span></div>')
    if kind == "sticker":
        return (f'<div class="m" style="background:#e1e3e6"><div class="sticker"><div class="s"><img class="{cls}" src="{av}" style="max-height:58px;max-width:66px"></div></div>'
                '<span class="lbl">Laptop sticker</span></div>')
    if kind == "tshirt":
        shirt = ('<svg viewBox="0 0 210 220"><path d="M60 10 L20 35 L0 80 L35 95 L45 75 L45 215 L165 215 L165 75 L175 95 L210 80 '
                 f'L190 35 L150 10 Q105 40 60 10 Z" fill="{tile}"/></svg>')
        return (f'<div class="m" style="background:#e4e1dc"><div class="shirt">{shirt}<img class="{wat}" src="{avt}" style="position:relative;max-height:56px;max-width:70px;margin-top:10px"></div>'
                '<span class="lbl">T-shirt</span></div>')
    if kind == "badge":
        return (f'<div class="m" style="background:#d9dde2"><div class="badge"><div class="hole"></div><img class="{cls}" src="{stack}" style="max-height:{70 if stack != lock else 34}px;max-width:130px">'
                '<div class="nm">Alex Morgan</div><div class="rl">Speaker</div></div><span class="lbl">Event badge</span></div>')
    if kind == "payment-card":
        return (f'<div class="m" style="background:#dfe3e8"><div class="pcard" style="background:{tile};color:{text_on(tile)}">'
                f'<img class="{wat}" src="{avt}" style="position:absolute;left:22px;top:20px;max-height:34px;max-width:70px">'
                f'<div class="chip"></div><div class="num">•••• •••• •••• 4821</div>'
                f'<div class="holder">ALEX MORGAN</div><div class="brandline">{n}</div></div>'
                '<span class="lbl">Payment card</span></div>')
    if kind == "favicon-tab":
        return (f'<div class="m" style="background:#eceff1"><div class="tabbar"><div class="tab"><img class="{cls}" src="{av}" style="width:16px;height:16px;object-fit:contain">'
                f'{n} — Home<span style="margin-left:auto">✕</span></div><div class="addr">https://{handle}.com</div></div><span class="lbl">Browser tab</span></div>')
    raise KeyError(kind)


ALL_MOCKUPS = ["payment-card", "business-card", "app-icon", "website", "signage", "tote", "social", "cup", "bag", "shopping-bag", "box",
               "readme", "terminal", "sticker", "tshirt", "badge", "favicon-tab"]


def pick_mockups(spec):
    if spec.get("mockups"):
        chosen = [m for m in spec["mockups"] if m in ALL_MOCKUPS]
    else:
        ind = str(spec.get("industry", "")).lower()
        key = ind if ind in PRESETS else ALIASES.get(ind, ind)   # a real preset name always wins over an alias
        chosen = PRESETS.get(key, PRESETS["services"])
    return chosen[:6] if len(chosen) >= 6 else (chosen + [m for m in PRESETS["services"] if m not in chosen])[:6]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs="?")
    ap.add_argument("-o", "--out", default="presentation.html")
    ap.add_argument("--list-mockups", action="store_true")
    ap.add_argument("--png-dir", help="also export every slide as slide-NN.png (needs Chrome/Chromium)")
    a = ap.parse_args()
    if a.list_mockups:
        print("mockups:", ", ".join(ALL_MOCKUPS))
        for k, v in PRESETS.items():
            print(f"  industry '{k}': {', '.join(v)}")
        print("  aliases:", ", ".join(f"{k}→{v}" for k, v in ALIASES.items()))
        return
    if not a.spec:
        ap.error("spec.json required")
    with open(a.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    base = os.path.dirname(os.path.abspath(a.spec))
    brand = spec.get("brand", "Brand")
    color = spec.get("brand_color", "#161616")
    grey = spec.get("greyscale", True)
    final = spec.get("final", False)
    cls = "g" if grey else ""
    tile = "#161616" if grey else (spec.get("tile_color") or color)
    date = spec.get("date") or datetime.date.today().isoformat()
    designer = spec.get("designer", "")
    concepts = spec.get("concepts", [])
    ind = str(spec.get("industry", "")).lower()
    nav = spec.get("nav") or DEFAULT_NAV.get(ALIASES.get(ind, ind), ["Products", "About", "Contact"])
    mockups = pick_mockups(spec)
    deck_title = spec.get("deck_title", "final identity" if final else "identity concepts")

    def foot(i):
        return f'<div class="foot"><span>{html.escape(brand)} — {html.escape(deck_title)}</span><span>{i}</span></div>'

    slides = []
    custom_title = spec.get("deck_title")
    cover_mark = ""
    if final and concepts:
        c0 = concepts[0]
        first = c0.get("lockup") or c0.get("symbol") or c0.get("avatar")
        if first:
            cover_mark = (f'<div style="position:absolute;right:64px;top:50%;transform:translateY(-50%);width:520px;'
                          f'height:300px;display:flex;align-items:center;justify-content:center">'
                          f'<img class="{cls}" src="{uri(first, base)}" style="max-width:100%;max-height:100%"></div>')
    kicker = (custom_title.capitalize() if (final and custom_title) else "Final identity") if final \
        else f"Logo design · round {spec.get('round', 1)}"
    note = ("Concepts are shown in greyscale so the conversation stays on form and idea; colour follows once a "
            "direction is chosen." if grey and not final else "")
    slides.append(f"""<section class="slide"><div class="kicker">{kicker}</div>
<h1>{html.escape(brand)}</h1><p class="idea muted">{html.escape(deck_title.capitalize())}{(' · ' + html.escape(designer)) if designer else ''} · {date}</p>
{cover_mark}<p class="muted" style="max-width:720px;margin-top:120px">{note}</p>{foot(1)}</section>""")
    adj = "".join(f"<span>{html.escape(x)}</span>" for x in spec.get("adjectives", []))
    crit = spec.get("criteria", "Success: distinctive in its category, recognisable at 16 px, works in one colour, "
                                "and feels like the words above.")
    slides.append(f"""<section class="slide"><div class="kicker">What we heard</div><h2>The brief</h2>
<p class="idea">{html.escape(spec.get('brief', ''))}</p><div class="adj">{adj}</div>
<p class="muted" style="margin-top:28px;max-width:860px">{html.escape(crit)}</p>{foot(2)}</section>""")
    items = []
    for i, c in enumerate(concepts):
        paths = [c.get("symbol"), c.get("lockup"), c.get("avatar")]
        if not any(paths):
            print(f"skip concept {i}: needs symbol, lockup or avatar")
            continue
        sym = uri(c.get("symbol") or c.get("avatar") or c.get("lockup"), base)
        lock = uri(c.get("lockup") or c.get("symbol") or c.get("avatar"), base)
        av = uri(c.get("avatar") or c.get("symbol") or c.get("lockup"), base)
        stacked = uri(c["stacked"], base) if c.get("stacked") else None
        dark = {}
        if not grey:
            pick = lambda *keys: next((c[k] for k in keys if c.get(k)), None)
            if c.get("symbol_on_dark") and not c.get("symbol_on_tile"):
                print(f"note: concept {chr(65 + i)} has no symbol_on_tile; symbol_on_dark is used on the {tile} tiles "
                      "(app icon, cards) — check its contrast there, or pass a dedicated tile file")
            for k, src in (("avatar_tile", pick("symbol_on_tile", "symbol_on_dark")),
                           ("stack_tile", pick("stacked_on_tile", "lockup_on_tile")),
                           ("avatar_dark", pick("symbol_on_dark")), ("lock_dark", pick("lockup_on_dark")),
                           ("stack_dark", pick("stacked_on_dark"))):
                if src:
                    dark[k] = uri(src, base)
        items.append((c, sym, lock))
        label = spec.get("label") or ((custom_title.capitalize() if custom_title else "Final design") if final
                                      else f"Concept {chr(65 + i)}")
        rat = "".join(f"<li>{html.escape(r)}</li>" for r in c.get("rationale", []))
        sizes = "".join(f'<div><img class="{cls}" src="{av}" style="height:{s}px">{s}px</div>' for s in (16, 24, 32, 64))
        slides.append(f"""<section class="slide"><div class="kicker">{label}</div><h2>{html.escape(c.get('name', ''))}</h2>
<p class="idea">{html.escape(c.get('idea', ''))}</p>
<div class="hero"><div class="stage"><img class="{cls}" src="{lock}"></div>
<div><ul class="rat">{rat}</ul><div class="sizes">{sizes}</div><div class="rev"><img class="{'' if (c.get('lockup_on_dark') and not grey) else 'w'}" src="{uri(c['lockup_on_dark'], base) if (c.get('lockup_on_dark') and not grey) else lock}" style="max-height:44px;max-width:300px"></div></div></div>
{foot(len(slides) + 1)}</section>""")
        ctx = {"sym": sym, "lock": lock, "avatar": av, "cls": cls, "name": brand, "tag": spec.get("tagline", ""),
               "tile": tile, "nav": nav, "stacked": stacked, **dark}
        cells = "".join(mock_html(k, ctx) for k in mockups)
        slides.append(f"""<section class="slide"><div class="kicker">{label} · in use</div><h2>{html.escape(c.get('name', ''))}</h2>
<div class="mock">{cells}</div>{foot(len(slides) + 1)}</section>""")
    if len(items) > 1:
        cells = "".join(f'<div class="c"><img class="{cls}" src="{lock}"><img class="{cls}" src="{sym}" style="height:32px">'
                        f'<b>{chr(65 + i)} · {html.escape(c.get("name", ""))}</b><span class="muted" style="text-align:center">{html.escape(c.get("idea", ""))}</span></div>'
                        for i, (c, sym, lock) in enumerate(items))
        slides.append(f"""<section class="slide"><div class="kicker">Side by side</div><h2>{len(items)} directions</h2>
<div class="cmp" style="grid-template-columns:repeat({len(items)},1fr)">{cells}</div>{foot(len(slides) + 1)}</section>""")
    if spec.get("recommendation"):
        steps = spec.get("next_steps") or ([] if final else ["Choose a direction (or name what you like in each).",
                                                             "We refine it: geometry, optical corrections, lockups.",
                                                             "Colour and typography options for the chosen direction.",
                                                             "Final artwork, variants and mini guidelines."])
        title = "Notes" if final else "Our recommendation"
        slides.append(f"""<section class="slide"><div class="kicker">{'Next steps' if final else 'Recommendation & next steps'}</div><h2>{title}</h2>
<p class="idea">{html.escape(spec['recommendation'])}</p>
<ul class="rat" style="margin-top:30px">{''.join(f'<li>{html.escape(s)}</li>' for s in steps)}</ul>{foot(len(slides) + 1)}</section>""")
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(brand)} — {html.escape(deck_title)}</title>"
           f"<style>{CSS}</style></head><body>{''.join(slides)}</body></html>")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {a.out} ({len(slides)} slides; mockups: {', '.join(mockups)})")
    if a.png_dir:
        import tempfile
        import render_png
        os.makedirs(a.png_dir, exist_ok=True)
        solo_css = CSS + "body{background:#fff}.slide{margin:0;box-shadow:none;min-height:800px}"
        for i, sl in enumerate(slides, 1):
            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as fh:
                fh.write(f"<!doctype html><html><head><meta charset='utf-8'><style>{solo_css}</style></head><body>{sl}</body></html>")
                tmp_html = fh.name
            out = os.path.join(a.png_dir, f"slide-{i:02d}.png")
            try:
                render_png.screenshot_html(tmp_html, out, 1280, 800)
                print("wrote", out)
            except Exception as exc:
                print(f"slide PNG export failed ({exc}); open the HTML instead")
                break
            finally:
                os.unlink(tmp_html)


if __name__ == "__main__":
    main()
