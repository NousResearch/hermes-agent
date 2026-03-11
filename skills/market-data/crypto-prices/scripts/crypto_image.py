#!/usr/bin/env python3
"""
Crypto Market Card Generator
Fetches live top-10 data from CoinGecko and renders a styled PNG.
Usage: python3 crypto_image.py
Output: ~/crypto_market.png
"""

import io, json, math, os, sys, urllib.request, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
import requests
requests.packages.urllib3.disable_warnings()

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

OUT_PATH   = os.path.expanduser("~/crypto_market.png")
FONT_REG   = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD  = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_BLACK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"

BG          = (6,  9, 18)
BG_ROW_ALT  = (11, 15, 28)
BG_CARD     = (15, 20, 38)
BG_HEADER   = (10, 14, 26)
CYAN        = (0,  220, 255)
CYAN_DIM    = (0,  120, 160)
GREEN       = (0,  255, 130)
GREEN_DIM   = (0,  180,  80)
RED         = (255,  55,  90)
RED_DIM     = (180,  20,  50)
GOLD        = (255, 200,  50)
WHITE       = (255, 255, 255)
GRAY        = (160, 170, 195)
DIMGRAY     = ( 80,  90, 115)
BORDER      = ( 25,  32,  55)

W, H        = 1200, 900
ROW_H       = 68
HEADER_H    = 130
COL_H_H     = 44
FOOTER_H    = 44
LOGO_SIZE   = 42

X_RANK      = 28
X_LOGO      = 68
X_NAME      = 128
X_PRICE     = 470
X_CHANGE    = 630
X_MCAP      = 810
X_VOL       = 1000

def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def fetch_market_data():
    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        "?vs_currency=usd&order=market_cap_desc&per_page=10&page=1"
        "&sparkline=false&price_change_percentage=24h"
    )
    r = requests.get(url, timeout=12, verify=False,
                     headers={"User-Agent": "crypto-card/1.0"})
    r.raise_for_status()
    return r.json()

def fetch_logo(url):
    try:
        r = requests.get(url, timeout=8, verify=False)
        img = Image.open(io.BytesIO(r.content)).convert("RGBA")
        img = img.resize((LOGO_SIZE, LOGO_SIZE), Image.LANCZOS)
        return img
    except Exception:
        return None

def circle_mask(size):
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size-1, size-1), fill=255)
    return mask

def make_circle_logo(img):
    base = Image.new("RGBA", (LOGO_SIZE, LOGO_SIZE), (20, 25, 45, 255))
    if img:
        base.paste(img, (0, 0), img)
    base.putalpha(circle_mask(LOGO_SIZE))
    return base

def glow_text(draw_ref_img, pos, text, font, color, glow_radius=6, glow_alpha=120):
    tmp_probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bb = tmp_probe.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = glow_radius * 3
    tmp = Image.new("RGBA", (tw + pad*2 + 10, th + pad*2 + 10), (0, 0, 0, 0))
    td  = ImageDraw.Draw(tmp)
    td.text((pad, pad), text, font=font, fill=(*color, glow_alpha))
    blurred = tmp.filter(ImageFilter.GaussianBlur(radius=glow_radius))
    td2 = ImageDraw.Draw(blurred)
    td2.text((pad, pad), text, font=font, fill=(*color, 255))
    return blurred, (pos[0] - pad, pos[1] - pad)

def fmt_price(v):
    if v >= 10000: return f"${v:,.0f}"
    if v >= 100:   return f"${v:,.1f}"
    if v >= 1:     return f"${v:,.3f}"
    return f"${v:.5f}"

def fmt_mcap(v):
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    if v >= 1e6:  return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"

def fmt_vol(v):
    if v >= 1e9: return f"${v/1e9:.1f}B"
    if v >= 1e6: return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"

def draw_rounded_rect(draw, xy, radius, fill, outline=None, outline_width=1):
    x0,y0,x1,y1 = xy
    draw.rounded_rectangle([x0,y0,x1,y1], radius=radius, fill=fill,
                            outline=outline, width=outline_width)

def draw_badge(img_draw, x, y, text, bg, fg, font, padding=(14, 6), radius=8):
    tw = int(img_draw.textlength(text, font=font))
    bw = tw + padding[0]*2
    bh = 28 + padding[1]
    draw_rounded_rect(img_draw, (x, y, x+bw, y+bh), radius, fill=bg)
    img_draw.text((x+padding[0], y+padding[1]//2+2), text, font=font, fill=fg)
    return bw

def draw_gradient_line(draw, x0, y0, x1, steps=200):
    for i in range(steps):
        t  = i / steps
        r  = int(0   + 80*math.sin(math.pi*t))
        g  = int(220 * (1 - 0.6*t))
        b  = int(255)
        x  = int(x0 + (x1-x0)*t)
        draw.line([(x, y0), (x+6, y0)], fill=(r,g,b))

def build_background(w, h):
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    for i in range(120):
        alpha = int(30 * (1 - i/120))
        col   = tuple(max(0, c - alpha) for c in BG)
        draw.rectangle([i, i, w-i, h-i], outline=col)
    for gx in range(0, w, 60):
        for gy in range(HEADER_H, h - FOOTER_H, 60):
            draw.point((gx, gy), fill=(25, 32, 52))
    return img, draw

def add_background_glow(img, x, y, color, radius=200, alpha=18):
    glow = Image.new("RGB", img.size, (0,0,0))
    gd   = ImageDraw.Draw(glow)
    gd.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=radius//2))
    return Image.blend(img, glow, alpha/255)

def main():
    print("Fetching live crypto data...")
    coins = fetch_market_data()
    print(f"Got {len(coins)} coins. Downloading logos...")

    logos = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_logo, c["image"]): c["id"] for c in coins}
        for f in as_completed(futures):
            logos[futures[f]] = f.result()
    print("Logos done. Rendering image...")

    f_title   = load_font(FONT_BLACK, 38)
    f_sub     = load_font(FONT_BOLD,  16)
    f_col_hdr = load_font(FONT_BOLD,  15)
    f_name    = load_font(FONT_BOLD,  17)
    f_sym     = load_font(FONT_REG,   14)
    f_price   = load_font(FONT_BOLD,  19)
    f_change  = load_font(FONT_BOLD,  15)
    f_mcap    = load_font(FONT_REG,   15)
    f_rank    = load_font(FONT_BOLD,  14)
    f_footer  = load_font(FONT_REG,   13)

    actual_h = HEADER_H + COL_H_H + len(coins)*ROW_H + FOOTER_H + 20
    img, draw = build_background(W, actual_h)

    img = add_background_glow(img, 200,  80, (0, 60, 120), 280, 30)
    img = add_background_glow(img, 1000, 700, (60, 0, 100), 260, 25)
    img = add_background_glow(img, 600,  actual_h//2, (0, 40, 80), 350, 15)
    draw = ImageDraw.Draw(img)

    draw_rounded_rect(draw, (16, 12, W-16, HEADER_H-8), radius=14,
                      fill=BG_HEADER, outline=BORDER, outline_width=1)

    # --- NOUS RESEARCH branding: top-left neon cyan ---
    nous_txt = "NOUS RESEARCH"
    nous_layer, nous_pos = glow_text(img, (44, 14), nous_txt, f_sub, CYAN,
                                     glow_radius=8, glow_alpha=160)
    img.paste(nous_layer, nous_pos, nous_layer)
    draw = ImageDraw.Draw(img)
    draw.text((44, 14), nous_txt, font=f_sub, fill=CYAN)

    title_txt = "CRYPTO MARKET"
    glow_layer, gpos = glow_text(img, (44, 34), title_txt, f_title, CYAN,
                                 glow_radius=10, glow_alpha=100)
    img.paste(glow_layer, gpos, glow_layer)
    draw = ImageDraw.Draw(img)
    draw.text((44, 34), title_txt, font=f_title, fill=WHITE)
    draw.text((46, 82), "LIVE PRICES", font=f_sub, fill=CYAN)
    now_str = datetime.now(timezone.utc).strftime("%b %d, %Y  %H:%M UTC")
    draw.text((46, 100), now_str, font=f_sym, fill=DIMGRAY)

    pill_coins = [c for c in coins if c["symbol"] in ("btc","eth","sol")][:3]
    px = W - 30
    for pc in reversed(pill_coins):
        chg  = pc.get("price_change_percentage_24h_in_currency") or 0
        bg   = (0, 55, 30) if chg >= 0 else (55, 8, 18)
        col  = GREEN if chg >= 0 else RED
        sign = "+" if chg >= 0 else ""
        txt  = f"{pc['symbol'].upper()}  {sign}{chg:.2f}%"
        tw   = int(draw.textlength(txt, font=f_sub)) + 28
        px  -= tw + 10
        draw_rounded_rect(draw, (px, 35, px+tw, 65), radius=10, fill=bg,
                          outline=col, outline_width=1)
        draw.text((px+14, 40), txt, font=f_sub, fill=col)
    draw = ImageDraw.Draw(img)

    draw_gradient_line(draw, 16, HEADER_H, W-16, steps=300)

    yh = HEADER_H + 10
    for label, x, anchor in [
        ("#",       X_RANK,    "lm"),
        ("COIN",    X_NAME,    "lm"),
        ("PRICE",   X_PRICE,   "rm"),
        ("24H",     X_CHANGE+40, "mm"),
        ("MKT CAP", X_MCAP,   "rm"),
        ("VOLUME",  X_VOL+80, "rm"),
    ]:
        draw.text((x, yh+COL_H_H//2), label, font=f_col_hdr,
                  fill=DIMGRAY, anchor=anchor)

    y_start = HEADER_H + COL_H_H
    for i, coin in enumerate(coins):
        y  = y_start + i * ROW_H
        yc = y + ROW_H // 2

        row_fill = BG_ROW_ALT if i % 2 == 0 else BG
        draw.rectangle([16, y+2, W-16, y+ROW_H-2], fill=row_fill)

        rank   = coin.get("market_cap_rank", i+1)
        name   = coin.get("name", "")
        symbol = coin.get("symbol", "").upper()
        price  = coin.get("current_price", 0) or 0
        chg    = coin.get("price_change_percentage_24h_in_currency", 0) or 0
        mcap   = coin.get("market_cap", 0) or 0
        vol    = coin.get("total_volume", 0) or 0

        chg_col = GREEN if chg >= 0 else RED
        chg_dim = GREEN_DIM if chg >= 0 else RED_DIM
        chg_bg  = (0, 40, 18) if chg >= 0 else (40, 6, 14)
        arrow   = "▲" if chg >= 0 else "▼"
        sign    = "+" if chg >= 0 else ""

        rk_col = GOLD if rank <= 3 else DIMGRAY
        draw.text((X_RANK, yc), str(rank), font=f_rank, fill=rk_col, anchor="lm")

        logo_raw = logos.get(coin["id"])
        circ = make_circle_logo(logo_raw)
        ring = Image.new("RGBA", (LOGO_SIZE+6, LOGO_SIZE+6), (0,0,0,0))
        rd   = ImageDraw.Draw(ring)
        rd.ellipse([0,0,LOGO_SIZE+5,LOGO_SIZE+5], outline=(*chg_dim,180), width=2)
        ring_b = ring.filter(ImageFilter.GaussianBlur(2))
        img.paste(ring_b, (X_LOGO-3, yc-LOGO_SIZE//2-3), ring_b)
        img.paste(circ, (X_LOGO, yc-LOGO_SIZE//2), circ)
        draw = ImageDraw.Draw(img)

        max_name = 16
        disp_name = name[:max_name] + ("…" if len(name)>max_name else "")
        draw.text((X_NAME, yc-10), disp_name, font=f_name, fill=WHITE, anchor="lm")
        draw.text((X_NAME, yc+10), symbol, font=f_sym, fill=DIMGRAY, anchor="lm")

        price_str = fmt_price(price)
        if abs(chg) > 5:
            pg, gpos2 = glow_text(img, (X_PRICE, yc-10), price_str, f_price,
                                  chg_col, glow_radius=5, glow_alpha=60)
            img.paste(pg, gpos2, pg)
            draw = ImageDraw.Draw(img)
        draw.text((X_PRICE, yc-1), price_str, font=f_price, fill=WHITE, anchor="rm")

        chg_str = f"  {arrow} {sign}{chg:.2f}%  "
        draw_badge(draw, X_CHANGE, yc-14, chg_str, chg_bg, chg_col,
                   f_change, padding=(10,4), radius=8)

        draw.text((X_MCAP, yc-1), fmt_mcap(mcap), font=f_mcap, fill=GRAY, anchor="rm")
        draw.text((X_VOL+80, yc-1), fmt_vol(vol), font=f_mcap, fill=DIMGRAY, anchor="rm")
        draw.line([(28, y+ROW_H-1), (W-28, y+ROW_H-1)], fill=BORDER)

    fy = y_start + len(coins)*ROW_H + 10
    draw_gradient_line(draw, 16, fy, W-16, steps=300)
    draw.text((28, fy+14), "Data: CoinGecko API  •  Prices in USD",
              font=f_footer, fill=DIMGRAY)
    draw.text((W-28, fy+14), "Built with Hermes Agent",
              font=f_footer, fill=DIMGRAY, anchor="rm")
    draw.text((W-28, fy+28), "by buraq_c",
              font=f_footer, fill=CYAN, anchor="rm")

    img = ImageEnhance.Contrast(img).enhance(1.08)
    img.save(OUT_PATH, "PNG", optimize=True)
    print(f"Saved: {OUT_PATH}  ({img.size[0]}x{img.size[1]})")
    return OUT_PATH

if __name__ == "__main__":
    main()
