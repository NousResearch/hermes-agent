from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
import yaml
from PIL import Image, ImageDraw, ImageFont

BASE = Path("/home/orbibot/.zeroclaw/workspace/hermes-agent/docs/plans/orbi-romance-webtoon-20260421/webtoon/ep001")
OUT = BASE / "generated_fal_live_ep001"
MANIFEST_PATH = BASE / "generated_fal_live_manifest.json"
PANEL_W = 720
PANEL_H = 1080

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for ch in text:
        trial = current + ch
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def fal_generate(prompt: str) -> str:
    import fal_client

    result = fal_client.subscribe(
        "fal-ai/flux-2-pro",
        arguments={
            "prompt": prompt,
            "image_size": {"width": PANEL_W, "height": PANEL_H},
            "num_images": 1,
            "output_format": "png",
        },
    )
    return result["images"][0]["url"]


def download(url: str, path: Path) -> None:
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    path.write_bytes(resp.content)


def style_prompt(panel_data: dict[str, Any]) -> str:
    positive = panel_data["style_anchor"]["positive"]
    negative = ", ".join(panel_data["style_anchor"]["negative"])
    return f"{positive}, clean Korean romance webtoon cartoon illustration, 2D cel shading, polished digital manhwa style, no readable text, no speech bubbles, no letters, avoid {negative}"


def build_prompt(panel_data: dict[str, Any], panel_spec: dict[str, Any]) -> str:
    chars = []
    for key in panel_spec.get("visible_characters", []):
        info = panel_data["characters"].get(key, {})
        chars.append(f"{info.get('role', key)} with {info.get('visual', '')}")
    char_block = ", ".join(chars)
    return ", ".join(
        p for p in [
            style_prompt(panel_data),
            panel_spec["prompt"],
            char_block,
            "mobile vertical webtoon panel with strong facial acting and clean composition",
        ] if p
    )


def render_overlays(img_path: Path, panel_id: str, lettering: dict[str, Any]) -> None:
    image = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    caption_font = load_font(28)
    balloon_font = load_font(30)

    captions = {c["panel_id"]: c["text"] for c in lettering.get("captions", [])}
    balloons = [b for b in lettering.get("balloons", []) if b["panel_id"] == panel_id]

    if panel_id in captions:
        text = captions[panel_id]
        lines = wrap_text(draw, text, caption_font, PANEL_W - 140)
        box_h = 26 + len(lines) * 36 + 24
        box = (50, 56, PANEL_W - 50, 56 + box_h)
        draw.rounded_rectangle(box, radius=24, fill=(10, 10, 10, 200))
        y = box[1] + 20
        for line in lines:
            draw.text((box[0] + 24, y), line, font=caption_font, fill=(255, 255, 255, 255))
            y += 36

    top = PANEL_H - 250
    for balloon in balloons:
        lines = wrap_text(draw, balloon["text"], balloon_font, PANEL_W - 260)
        box_h = 30 + len(lines) * 38 + 28
        box = (100, top, PANEL_W - 100, top + box_h)
        draw.rounded_rectangle(box, radius=46, fill=(248, 248, 248, 235), outline=(20, 20, 20, 255), width=3)
        y = top + 22
        for line in lines:
            draw.text((box[0] + 40, y), line, font=balloon_font, fill=(25, 25, 25, 255))
            y += 38
        tail = [(box[0] + 90, box[3] - 8), (box[0] + 135, box[3] - 8), (box[0] + 112, box[3] + 40)]
        draw.polygon(tail, fill=(248, 248, 248, 235), outline=(20, 20, 20, 255))
        top -= box_h + 36

    image.convert("RGB").save(img_path)


def compose_longscroll(panel_paths: list[Path], scroll_plan: dict[str, Any]) -> Path:
    spacing_map = {block["block_id"]: block.get("spacing", "medium") for block in scroll_plan["blocks"]}
    gap_px = {"tight": 30, "medium": 70, "tall_drop": 180, "end_cliff": 260}
    images = []
    total_h = 0
    panel_ids = [f"p{i:02d}" for i in range(1, len(panel_paths) + 1)]
    blocks = {f"p{i:02d}": block["block_id"] for i, block in enumerate(scroll_plan["blocks"], start=1)}

    for idx, panel_path in enumerate(panel_paths):
        img = Image.open(panel_path).convert("RGB")
        panel_id = panel_ids[idx]
        gap = gap_px.get(spacing_map.get(blocks[panel_id], "medium"), 70)
        images.append((img, gap))
        total_h += img.height
        if idx < len(panel_paths) - 1:
            total_h += gap

    canvas = Image.new("RGB", (PANEL_W, total_h), (244, 244, 244))
    y = 0
    for idx, (img, gap) in enumerate(images):
        canvas.paste(img, (0, y))
        y += img.height
        if idx < len(images) - 1:
            y += gap

    out = OUT / "ep001_fal_live_longscroll.png"
    canvas.save(out)
    return out


def main() -> None:
    panel_data = load_yaml(BASE / "panel_prompts.yaml")
    lettering = load_yaml(BASE / "lettering_script.yaml")
    scroll_plan = load_yaml(BASE / "scroll_plan.yaml")
    OUT.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"episode": "ep001", "mode": "fal_live_flux2_pro", "panels": []}
    panel_paths: list[Path] = []

    for panel_spec in panel_data["panels"]:
        panel_id = panel_spec["panel_id"]
        prompt = build_prompt(panel_data, panel_spec)
        url = fal_generate(prompt)
        out_path = OUT / f"{panel_id}.png"
        download(url, out_path)
        render_overlays(out_path, panel_id, lettering)
        panel_paths.append(out_path)
        manifest["panels"].append({
            "panel_id": panel_id,
            "prompt": prompt,
            "url": url,
            "path": str(out_path.resolve()),
        })

    longscroll = compose_longscroll(panel_paths, scroll_plan)
    manifest["longscroll"] = str(longscroll.resolve())
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"panel_count": len(panel_paths), "longscroll": str(longscroll.resolve()), "manifest": str(MANIFEST_PATH.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
