#!/usr/bin/env python3
"""Hybrid Visual Pipeline — reference implementation for social content generation.
Demonstrates the three-tier stack: Pillow (free static), FFMPEG (free motion), FAL.ai (premium).
Copy and adapt for your brand."""
import os
import subprocess
import textwrap
import uuid
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

# ── Tier 1: Pillow Static Cards ──────────────────────────────────────────

def make_text_card(
    brand_name: str,
    body: str,
    accent: str = "#FBBF24",
    bg: str = "#0A0A0A",
    size: tuple = (1080, 1080),
    out_dir: str = "./output",
) -> str:
    """Render a brand-aligned text card. Returns output path."""
    if Image is None:
        raise RuntimeError("Pillow not installed: pip install Pillow")

    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)

    # Accent bar at top
    draw.rectangle([0, 0, size[0], 8], fill=accent)

    # Brand badge
    try:
        badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except Exception:
        badge_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.text((60, 50), brand_name.upper(), fill=accent, font=badge_font)

    y = 180
    for line in textwrap.wrap(body, width=30):
        draw.text((60, y), line, fill="#FFFFFF", font=body_font)
        y += 46

    out = Path(out_dir) / f"{brand_name}_{uuid.uuid4().hex[:8]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))
    return str(out)


# ── Tier 2: FFMPEG Text Animations ────────────────────────────────────────

def make_stat_reveal_video(
    stat_line: str,
    sub_line: str,
    accent: str = "#FBBF24",
    bg: str = "#0A0A0A",
    duration: int = 4,
    size: str = "1080x1080",
    out_dir: str = "./output",
) -> str:
    """FFMPEG animated stat reveal. Returns output path."""
    out = Path(out_dir) / f"stat_reveal_{uuid.uuid4().hex[:8]}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Escape drawtext specials
    stat = stat_line.replace("'", r"'\\''")
    sub = sub_line.replace("'", r"'\\''")

    vf = (
        f"drawtext=text='{stat}':fontcolor={accent}:fontsize=72:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-40:"
        f"alpha='if(lt(t,0.5),t*2,if(lt(t,3.5),1,(4-t)*2))':"
        f"font=DejaVuSans-Bold,"
        f"drawtext=text='{sub}':fontcolor=white:fontsize=36:"
        f"x=(w-text_w)/2:y=(h-text_h)/2+60:"
        f"alpha='if(lt(t,1.0),0,if(lt(t,1.5),(t-1)*2,if(lt(t,3.5),1,(4-t)*2)))':"
        f"font=DejaVuSans"
    )

    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={bg}:s={size}:r=30:d={duration}",
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-t", str(duration),
        str(out),
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=120)
    return str(out)


# ── Tier 3: FAL.ai Premium Images/Video ──────────────────────────────────

FAL_KEY = os.getenv("FAL_KEY", "")


def fal_image(prompt: str, model: str = "fal-ai/flux-2/klein/9b", aspect: str = "square_hd") -> str:
    """Generate image via FAL.ai. Returns file path."""
    if not FAL_KEY or requests is None:
        raise RuntimeError("FAL_KEY not set or requests unavailable")

    url = f"https://fal.run/{model}"
    r = requests.post(
        url,
        json={"prompt": prompt, "image_size": aspect, "num_images": 1, "enable_safety_checker": False},
        headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    image_url = r.json()["images"][0]["url"]

    img_r = requests.get(image_url, timeout=60)
    img_r.raise_for_status()

    out = Path("./output/fal_images") / f"fal_{uuid.uuid4().hex[:8]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(img_r.content)
    return str(out)


def fal_video(prompt: str, model: str = "fal-ai/wan/v2.2/480p", duration: int = 5) -> str:
    """Generate video via FAL.ai (async queue + poll). Returns file path."""
    if not FAL_KEY or requests is None:
        raise RuntimeError("FAL_KEY not set or requests unavailable")

    # 1. Submit to queue
    qurl = f"https://queue.fal.run/{model}"
    qr = requests.post(
        qurl,
        json={"prompt": prompt, "duration": duration, "aspect_ratio": "16:9"},
        headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
        timeout=30,
    )
    qr.raise_for_status()
    req_id = qr.json()["request_id"]

    # 2. Poll for result
    import time
    poll_url = f"https://queue.fal.run/{model}/requests/{req_id}"
    for _ in range(40):
        pr = requests.get(poll_url, headers={"Authorization": f"Key {FAL_KEY}"}, timeout=15)
        data = pr.json()
        if data.get("status") == "COMPLETED":
            video_url = data.get("video", {}).get("url") or data.get("output", {}).get("video", {}).get("url")
            if video_url:
                vr = requests.get(video_url, timeout=120)
                vr.raise_for_status()
                out = Path("./output/fal_videos") / f"fal_video_{uuid.uuid4().hex[:8]}.mp4"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(vr.content)
                return str(out)
            break
        time.sleep(5)
    raise RuntimeError("FAL video generation timed out")


# ── Example usage ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Free tier
    card = make_text_card("MatchdayMaestro", "Day 14 of your prediction streak. One more for the Streak Shield.")
    print("Card:", card)

    video = make_stat_reveal_video("89% predicted Liverpool", "The 12% got it right")
    print("Video:", video)

    # Premium tier (requires FAL_KEY)
    if FAL_KEY:
        img = fal_image("Minimalist football card, dark navy, golden accents, no text")
        print("AI Image:", img)

        vid = fal_video("Cinematic football match highlight, slow motion goal celebration")
        print("AI Video:", vid)
