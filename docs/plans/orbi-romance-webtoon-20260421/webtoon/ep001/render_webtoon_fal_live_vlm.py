from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
import yaml
from PIL import Image, ImageDraw, ImageFont

BASE = Path("/home/orbibot/.zeroclaw/workspace/hermes-agent/docs/plans/orbi-romance-webtoon-20260421/webtoon/ep001")
SOURCE_MANIFEST_PATH = BASE / "generated_fal_live_manifest.json"
SOURCE_RENDERED_DIR = BASE / "generated_fal_live_ep001"
SOURCE_RAW_DIR = BASE / "generated_fal_live_ep001_raw"
OBSERVATION_PATH = BASE / "vlm_observations_ep001.yaml"
LETTERING_PATH = BASE / "lettering_script.yaml"
SCROLL_PLAN_PATH = BASE / "scroll_plan.yaml"
OUT = BASE / "generated_fal_live_ep001_vlm"
MANIFEST_PATH = BASE / "generated_fal_live_manifest_vlm.json"

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
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


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    *,
    start_size: int,
    min_size: int,
    line_gap: int,
) -> tuple[ImageFont.ImageFont, list[str], int]:
    for size in range(start_size, min_size - 1, -2):
        font = load_font(size)
        lines = wrap_text(draw, text, font, max_width)
        line_height = size + line_gap
        total_height = len(lines) * line_height - line_gap
        longest = max((draw.textlength(line, font=font) for line in lines), default=0)
        if total_height <= max_height and longest <= max_width:
            return font, lines, line_height
    font = load_font(min_size)
    lines = wrap_text(draw, text, font, max_width)
    return font, lines, min_size + line_gap


def normalized_box(zone: dict[str, Any], image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    x1 = int(zone["x"] * width)
    y1 = int(zone["y"] * height)
    x2 = int((zone["x"] + zone["w"]) * width)
    y2 = int((zone["y"] + zone["h"]) * height)
    return x1, y1, x2, y2


def normalized_point(point: dict[str, Any], image: Image.Image) -> tuple[int, int]:
    width, height = image.size
    return int(point["x"] * width), int(point["y"] * height)


def build_lettering_index(lettering: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    panel_items: dict[str, list[dict[str, Any]]] = {}
    for caption in lettering.get("captions", []):
        panel_items.setdefault(caption["panel_id"], []).append(
            {
                "item_id": f"{caption['panel_id']}_caption",
                "item_kind": "caption",
                "panel_id": caption["panel_id"],
                "speaker": "narration",
                "text": caption["text"],
            }
        )
    for balloon in lettering.get("balloons", []):
        panel_items.setdefault(balloon["panel_id"], []).append(
            {
                "item_id": balloon["id"],
                "item_kind": "speech",
                "panel_id": balloon["panel_id"],
                "speaker": balloon.get("speaker", "dialogue"),
                "text": balloon["text"],
            }
        )
    for items in panel_items.values():
        items.sort(key=lambda item: (0 if item["item_kind"] == "caption" else 1, item["item_id"]))
    return panel_items


def observation_index(observations: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {panel["panel_id"]: panel for panel in observations.get("panels", [])}


def observation_target(panel_obs: dict[str, Any], item_id: str) -> dict[str, Any] | None:
    for target in panel_obs.get("item_targets", []):
        if target["item_id"] == item_id:
            return target
    return None


def observation_zone(panel_obs: dict[str, Any], zone_id: str) -> dict[str, Any] | None:
    for zone in panel_obs.get("candidate_zones", []):
        if zone["zone_id"] == zone_id:
            return zone
    return None


def observation_anchor(panel_obs: dict[str, Any], anchor_id: str | None) -> dict[str, Any] | None:
    if not anchor_id:
        return None
    for anchor in panel_obs.get("anchor_points", []):
        if anchor["anchor_id"] == anchor_id:
            return anchor
    return None


def default_caption_box(image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    return 50, 56, width - 50, min(height - 60, 176)


def default_speech_box(image: Image.Image, index: int) -> tuple[int, int, int, int]:
    width, height = image.size
    base_h = 150
    gap = 36
    bottom = height - 120 - index * (base_h + gap)
    top = max(70, bottom - base_h)
    return 100, top, width - 100, bottom


def build_tail(box: tuple[int, int, int, int], anchor: tuple[int, int] | None) -> list[tuple[int, int]] | None:
    if anchor is None:
        return None
    x1, y1, x2, y2 = box
    anchor_x, anchor_y = anchor
    mid_x = (x1 + x2) // 2
    if anchor_y < y1:
        base_y = y1 + 4
        center_x = min(max(x1 + 40, anchor_x), x2 - 40)
        return [(center_x - 22, base_y), (center_x + 22, base_y), (anchor_x, anchor_y)]
    if anchor_y > y2:
        base_y = y2 - 4
        center_x = min(max(x1 + 40, anchor_x), x2 - 40)
        return [(center_x - 22, base_y), (center_x + 22, base_y), (anchor_x, anchor_y)]
    if anchor_x < mid_x:
        base_x = x1 + 8
        center_y = min(max(y1 + 40, anchor_y), y2 - 40)
        return [(base_x, center_y - 22), (base_x, center_y + 22), (anchor_x, anchor_y)]
    base_x = x2 - 8
    center_y = min(max(y1 + 40, anchor_y), y2 - 40)
    return [(base_x, center_y - 22), (base_x, center_y + 22), (anchor_x, anchor_y)]


def draw_caption(
    image: Image.Image,
    box: tuple[int, int, int, int],
    text: str,
) -> dict[str, Any]:
    draw = ImageDraw.Draw(image)
    padding_x = 24
    padding_y = 18
    max_width = max(80, box[2] - box[0] - padding_x * 2)
    max_height = max(40, box[3] - box[1] - padding_y * 2)
    font, lines, line_height = fit_font(draw, text, max_width, max_height, start_size=28, min_size=20, line_gap=8)
    draw.rounded_rectangle(box, radius=24, fill=(10, 10, 10, 224))
    text_height = len(lines) * line_height - 8
    y = box[1] + max(padding_y, (box[3] - box[1] - text_height) // 2)
    for line in lines:
        draw.text((box[0] + padding_x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height
    return {"font_size": getattr(font, "size", None), "line_count": len(lines)}


def draw_speech(
    image: Image.Image,
    box: tuple[int, int, int, int],
    text: str,
    anchor: tuple[int, int] | None,
) -> dict[str, Any]:
    draw = ImageDraw.Draw(image)
    padding_x = 40
    padding_y = 22
    max_width = max(80, box[2] - box[0] - padding_x * 2)
    max_height = max(50, box[3] - box[1] - padding_y * 2)
    font, lines, line_height = fit_font(draw, text, max_width, max_height, start_size=30, min_size=20, line_gap=8)
    draw.rounded_rectangle(box, radius=46, fill=(248, 248, 248, 240), outline=(20, 20, 20, 255), width=3)
    tail = build_tail(box, anchor)
    if tail:
        draw.polygon(tail, fill=(248, 248, 248, 240), outline=(20, 20, 20, 255))
    text_height = len(lines) * line_height - 8
    y = box[1] + max(padding_y, (box[3] - box[1] - text_height) // 2)
    for line in lines:
        draw.text((box[0] + padding_x, y), line, font=font, fill=(25, 25, 25, 255))
        y += line_height
    return {"font_size": getattr(font, "size", None), "line_count": len(lines), "tail_points": tail}


def ensure_source_panels(manifest: dict[str, Any]) -> dict[str, Path]:
    SOURCE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    panel_paths: dict[str, Path] = {}
    for panel in manifest.get("panels", []):
        panel_id = panel["panel_id"]
        out_path = SOURCE_RAW_DIR / f"{panel_id}.png"
        if not out_path.exists():
            response = requests.get(panel["url"], timeout=180)
            response.raise_for_status()
            out_path.write_bytes(response.content)
        panel_paths[panel_id] = out_path
    return panel_paths


def compose_longscroll(panel_paths: list[Path], scroll_plan: dict[str, Any]) -> Path:
    spacing_map = {block["block_id"]: block.get("spacing", "medium") for block in scroll_plan["blocks"]}
    gap_px = {"tight": 30, "medium": 70, "tall_drop": 180, "end_cliff": 260}
    panel_to_block = {f"p{i:02d}": block["block_id"] for i, block in enumerate(scroll_plan["blocks"], start=1)}

    images: list[tuple[Image.Image, int]] = []
    total_h = 0
    for panel_path in panel_paths:
        panel_id = panel_path.stem
        img = Image.open(panel_path).convert("RGB")
        gap = gap_px.get(spacing_map.get(panel_to_block.get(panel_id, ""), "medium"), 70)
        images.append((img, gap))
        total_h += img.height
    total_h += sum(gap for _, gap in images[:-1])

    panel_w = images[0][0].width if images else 720
    canvas = Image.new("RGB", (panel_w, total_h), (244, 244, 244))
    y = 0
    for idx, (img, gap) in enumerate(images):
        canvas.paste(img, (0, y))
        y += img.height
        if idx < len(images) - 1:
            y += gap

    out = OUT / "ep001_fal_live_longscroll.png"
    canvas.save(out)
    return out


def render_panel(
    source_path: Path,
    panel_id: str,
    items: list[dict[str, Any]],
    panel_obs: dict[str, Any] | None,
    output_path: Path,
) -> dict[str, Any]:
    image = Image.open(source_path).convert("RGBA")
    placements: list[dict[str, Any]] = []
    speech_index = 0

    for item in items:
        target = observation_target(panel_obs, item["item_id"]) if panel_obs else None
        zone = observation_zone(panel_obs, target["zone_id"]) if panel_obs and target else None
        anchor_meta = observation_anchor(panel_obs, target.get("anchor_id")) if panel_obs and target else None

        if zone:
            box = normalized_box(zone, image)
            placement_source = "vlm_observation"
        elif item["item_kind"] == "caption":
            box = default_caption_box(image)
            placement_source = "fallback_default"
        else:
            box = default_speech_box(image, speech_index)
            placement_source = "fallback_default"

        anchor = normalized_point(anchor_meta, image) if anchor_meta else None

        if item["item_kind"] == "caption":
            draw_meta = draw_caption(image, box, item["text"])
        else:
            draw_meta = draw_speech(image, box, item["text"], anchor)
            speech_index += 1

        placements.append(
            {
                "item_id": item["item_id"],
                "item_kind": item["item_kind"],
                "text": item["text"],
                "placement_source": placement_source,
                "zone_id": zone.get("zone_id") if zone else None,
                "zone_kind": zone.get("kind") if zone else None,
                "zone_confidence": zone.get("confidence") if zone else None,
                "target_confidence": target.get("confidence") if target else None,
                "target_rationale": target.get("rationale") if target else None,
                "box_px": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
                "anchor_id": anchor_meta.get("anchor_id") if anchor_meta else None,
                "anchor_px": {"x": anchor[0], "y": anchor[1]} if anchor else None,
                **draw_meta,
            }
        )

    image.convert("RGB").save(output_path)
    return {
        "panel_id": panel_id,
        "source_panel_path": str(source_path.resolve()),
        "output_panel_path": str(output_path.resolve()),
        "observation_used": bool(panel_obs),
        "placements": placements,
    }


def main() -> None:
    lettering = load_yaml(LETTERING_PATH)
    observations = load_yaml(OBSERVATION_PATH)
    scroll_plan = load_yaml(SCROLL_PLAN_PATH)
    source_manifest = load_json(SOURCE_MANIFEST_PATH)
    panel_items = build_lettering_index(lettering)
    panel_obs_index = observation_index(observations)
    source_panels = ensure_source_panels(source_manifest)

    OUT.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "episode": "ep001",
        "mode": "fal_live_vlm_overlay_mvp",
        "source_manifest": str(SOURCE_MANIFEST_PATH.resolve()),
        "source_rendered_dir": str(SOURCE_RENDERED_DIR.resolve()),
        "source_raw_dir": str(SOURCE_RAW_DIR.resolve()),
        "observation_path": str(OBSERVATION_PATH.resolve()),
        "panels": [],
    }

    output_paths: list[Path] = []
    for panel in source_manifest.get("panels", []):
        panel_id = panel["panel_id"]
        out_path = OUT / f"{panel_id}.png"
        panel_manifest = render_panel(
            source_panels[panel_id],
            panel_id,
            panel_items.get(panel_id, []),
            panel_obs_index.get(panel_id),
            out_path,
        )
        panel_manifest["source_url"] = panel["url"]
        manifest["panels"].append(panel_manifest)
        output_paths.append(out_path)

    longscroll = compose_longscroll(output_paths, scroll_plan)
    manifest["longscroll"] = str(longscroll.resolve())
    manifest["placement_sources"] = sorted(
        {
            placement["placement_source"]
            for panel in manifest["panels"]
            for placement in panel["placements"]
        }
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "panel_count": len(output_paths),
                "longscroll": str(longscroll.resolve()),
                "manifest": str(MANIFEST_PATH.resolve()),
                "placement_sources": manifest["placement_sources"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
