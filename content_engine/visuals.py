"""Pillow visual card generators per brand."""
import random
import textwrap
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from config import BRANDS, OUTPUT_DIR, font_path


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(font_path("DejaVuSans-Bold")), size)
    except Exception:
        return ImageFont.load_default()


def _font_light(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(font_path("DejaVuSans")), size)
    except Exception:
        return ImageFont.load_default()


def make_card(
    brand: str,
    body_text: str,
    title: Optional[str] = None,
    platform: str = "twitter",
    out_filename: Optional[str] = None,
    content_type: str = "text",
    pillar: str = "",
) -> str:
    """Render a visual card with content-type-aware layout variation.

    Content type affects background pattern, layout, and framing so the
    multimodal embedder doesn't see the same template card each time.
    Returns output path.
    """
    cfg = BRANDS.get(brand, BRANDS["sahil_twitter"])

    # Dimensions: 1080x1080 for IG/Twitter, 1080x1920 for Stories/TikTok
    if platform in ("tiktok", "stories"):
        w, h = 1080, 1920
    else:
        w, h = 1080, 1080

    img = Image.new("RGB", (w, h), color=cfg["bg"])
    draw = ImageDraw.Draw(img)

    # ── Content-type-aware visual variation ──
    # Different layouts so the multimodal embedder sees semantic variety

    if pillar in ("data", "data_driven"):
        # Data layout: stat number large at top, smaller explanation below
        draw.rectangle([0, 0, w, 8], fill=cfg["accent"])
        badge_font = _font(28)
        draw.text((60, 50), cfg["display"].upper(), fill=cfg["accent"], font=badge_font)

        body_font = _font_light(36)
        wrapped = textwrap.wrap(body_text, width=28)
        y = 200
        for line in wrapped:
            draw.text((60, y), line, fill=cfg["colour"], font=body_font)
            y += 50
            if y > h - 120:
                break

    elif pillar in ("tutorial", "howto"):
        # Tutorial layout: numbered-list friendly, wider margins, code-style bg accent
        draw.rectangle([0, 0, w, 8], fill=cfg["accent"])
        draw.rectangle([40, 120, w - 40, h - 40], outline="#333333", width=2)
        badge_font = _font(28)
        draw.text((60, 140), cfg["display"].upper(), fill=cfg["accent"], font=badge_font)
        draw.text((60, 180), cfg["handle"], fill="#666666", font=_font_light(22))

        body_font = _font_light(30)
        wrapped = textwrap.wrap(body_text, width=30)
        y = 260
        for line in wrapped:
            draw.text((60, y), line, fill=cfg["colour"], font=body_font)
            y += 44
            if y > h - 100:
                break

    elif pillar in ("promotion", "sly_product"):
        # Promotion layout: centered, bold, problem-to-solution framing
        draw.rectangle([0, 0, w, 8], fill=cfg["accent"])
        badge_font = _font(28)
        draw.text((60, 50), cfg["display"].upper(), fill=cfg["accent"], font=badge_font)

        body_font = _font_light(34)
        wrapped = textwrap.wrap(body_text, width=26)
        y = 200
        for line in wrapped:
            draw.text((60, y), line, fill=cfg["colour"], font=body_font)
            y += 48
            if y > h - 120:
                break

    else:
        # Default layout (build updates, wry observations, football reactions)
        draw.rectangle([0, 0, w, 8], fill=cfg["accent"])
        badge_font = _font(28)
        draw.text((60, 50), cfg["display"].upper(), fill=cfg["accent"], font=badge_font)
        handle_font = _font_light(22)
        draw.text((60, 90), cfg["handle"], fill="#666666", font=handle_font)

        y_cursor = 180
        if title:
            title_font = _font(42)
            wrapped = textwrap.wrap(title, width=22)
            for line in wrapped:
                draw.text((60, y_cursor), line, fill="#FFFFFF", font=title_font)
                y_cursor += 56
            y_cursor += 30

        body_font = _font_light(32)
        wrapped_body = textwrap.wrap(body_text, width=30)
        for line in wrapped_body:
            draw.text((60, y_cursor), line, fill=cfg["colour"], font=body_font)
            y_cursor += 46
            if y_cursor > h - 120:
                break

    # Footer decoration
    draw.rectangle([60, h - 60, 200, h - 56], fill=cfg["accent"])

    out_dir = OUTPUT_DIR / brand / platform
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_filename or f"{brand}_{random.randint(1000,9999)}.png"
    out_path = out_dir / fname
    img.save(str(out_path))
    return str(out_path)


def make_leaderboard_card(
    brand: str,
    players: list,
    week: str = "GW38",
    platform: str = "twitter",
) -> str:
    """MMaestro-specific: mini-leaderboard snippet."""
    cfg = BRANDS.get("matchdaymaestro")
    w, h = 1080, 1080
    img = Image.new("RGB", (w, h), color=cfg["bg"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, 8], fill=cfg["accent"])

    draw.text((60, 50), "MATCHDAYMAESTRO", fill=cfg["accent"], font=_font(28))
    draw.text((60, 90), f"MINI-LEAGUE — {week}", fill="#FFFFFF", font=_font(36))

    y = 200
    for i, p in enumerate(players[:5]):
        rank = i + 1
        colour = "#FFD700" if rank == 1 else "#C0C0C0" if rank == 2 else "#CD7F32" if rank == 3 else cfg["colour"]
        draw.text((60, y), f"#{rank}", fill=colour, font=_font(40))
        draw.text((140, y), p["name"], fill="#FFFFFF", font=_font_light(36))
        draw.text((800, y), f"{p['score']} pts", fill=cfg["colour"], font=_font_light(36))
        y += 90

    out_dir = OUTPUT_DIR / brand / platform
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mm_leaderboard_{week}.png"
    img.save(str(out_path))
    return str(out_path)


def make_streak_card(
    brand: str,
    streak: int,
    player_name: str = "You",
    platform: str = "twitter",
) -> str:
    """MMaestro-specific: streak milestone card."""
    cfg = BRANDS.get("matchdaymaestro")
    w, h = 1080, 1080
    img = Image.new("RGB", (w, h), color=cfg["bg"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, 8], fill=cfg["accent"])

    draw.text((60, 50), "MATCHDAYMAESTRO", fill=cfg["accent"], font=_font(28))
    draw.text((60, 120), f"{player_name}'s streak", fill="#FFFFFF", font=_font(36))

    big_font = _font(180)
    draw.text((60, 280), str(streak), fill=cfg["colour"], font=big_font)
    draw.text((60, 500), "days", fill="#888888", font=_font(60))

    draw.text((60, 700), "Keep the Streak Shield alive.", fill="#FFFFFF", font=_font_light(36))

    out_dir = OUTPUT_DIR / brand / platform
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mm_streak_{streak}.png"
    img.save(str(out_path))
    return str(out_path)


def make_stat_reveal(
    brand: str,
    stat_line: str,
    sub_line: str,
    platform: str = "twitter",
) -> str:
    """Generic stat mic-drop card. Works for any brand."""
    cfg = BRANDS.get(brand, BRANDS["sahil_twitter"])
    w, h = 1080, 1080
    img = Image.new("RGB", (w, h), color=cfg["bg"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, 8], fill=cfg["accent"])

    draw.text((60, 50), cfg["display"].upper(), fill=cfg["accent"], font=_font(28))

    stat_font = _font(64)
    wrapped_stat = textwrap.wrap(stat_line, width=20)
    y = 220
    for line in wrapped_stat:
        draw.text((60, y), line, fill=cfg["colour"], font=stat_font)
        y += 82

    y += 30
    sub_font = _font_light(36)
    wrapped_sub = textwrap.wrap(sub_line, width=30)
    for line in wrapped_sub:
        draw.text((60, y), line, fill="#FFFFFF", font=sub_font)
        y += 50

    out_dir = OUTPUT_DIR / brand / platform
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{brand}_stat_{random.randint(1000,9999)}.png"
    img.save(str(out_path))
    return str(out_path)
