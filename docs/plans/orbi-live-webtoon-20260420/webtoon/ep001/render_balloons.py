from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFilter

from balloon_layout_utils import (
    DEFAULT_BOX_GAP,
    DEFAULT_FORBIDDEN_OVERLAP_THRESHOLD,
    SpeakerStyle,
    STYLE_MAP,
    boxes_intersect,
    clamp_box,
    compose_longscroll,
    expand_box,
    fit_text_block,
    intersection_area,
    load_font,
    load_lettering,
    load_scroll_plan,
    normalize_speaker,
    normalized_point_to_pixels,
    normalized_rect_to_pixels,
    overlap_ratio,
    resolve_font_path,
    speech_inner_size_for_text_bbox,
    speech_text_safe_box,
    template_for_item,
)


@dataclass
class PlacementCandidate:
    item: dict[str, Any]
    template: str
    bubble_kind: str
    speaker: str
    box: tuple[int, int, int, int]
    inner_box: tuple[int, int, int, int]
    score: float
    zone_id: str
    zone_kind: str
    confidence: float
    font_size: int
    lines: list[str]
    overlap_metrics: list[dict[str, Any]]
    selected_with_warning: bool
    warning_reason: str | None
    candidate_class: str
    resolved_anchor_id: str | None
    resolved_anchor_role: str | None
    selected_zone_score: float
    score_breakdown: dict[str, float]
    fallback_reason: str | None
    review_reasons: list[str]


TEMPLATE_KIND_MAP = {
    "speech": "speech",
    "caption": "caption",
    "chat_ui": "chat",
    "screen_note": "note",
}

TEMPLATE_ZONE_COMPAT = {
    "speech": {"speech"},
    "caption": {"caption"},
    "chat_ui": {"chat_ui"},
    "screen_note": {"screen_note"},
}

RENDER_ORDER = {
    "chat_ui": 0,
    "screen_note": 1,
    "speech": 2,
    "caption": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render balloon/caption overlays from analysis guidance.")
    parser.add_argument("--input-dir", required=True, help="Directory containing fal v3 panel pngs.")
    parser.add_argument("--analysis", required=True, help="Path to balloon_analysis_ep001.yaml.")
    parser.add_argument("--lettering", required=True, help="Path to lettering_script.yaml.")
    parser.add_argument("--scroll-plan", required=True, help="Path to scroll_plan.yaml.")
    parser.add_argument("--output-dir", required=True, help="Directory where rendered panel pngs are written.")
    parser.add_argument("--font-path", help="Optional font override.")
    parser.add_argument("--compose-longscroll", action="store_true", help="Compose rendered panel images into a longscroll.")
    parser.add_argument("--manifest-name", default="placement_manifest.json", help="Debug manifest filename.")
    return parser.parse_args()


def draw_soft_mask(
    base: Image.Image,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    radius: int,
    blur_radius: int,
) -> None:
    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_layer)
    draw.rounded_rectangle(box, radius=radius, fill=fill)
    softened = mask_layer.filter(ImageFilter.GaussianBlur(radius=max(0, blur_radius)))
    base.alpha_composite(softened)


def load_analysis(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def bubble_style_for_template(speaker: str, template: str) -> SpeakerStyle:
    if template == "caption":
        return STYLE_MAP["caption"]
    if template == "screen_note":
        return STYLE_MAP["internal_note"]
    if template == "chat_ui" and speaker in STYLE_MAP:
        return STYLE_MAP[speaker]
    return STYLE_MAP.get(speaker, STYLE_MAP["mother"])


def resolve_anchor(panel_analysis: dict[str, Any], speaker: str, item_id: str) -> dict[str, Any] | None:
    item_overrides = panel_analysis["render_hints"].get("item_overrides", {}).get(item_id, {})
    forced_anchor_id = item_overrides.get("force_anchor_id")
    anchors = panel_analysis.get("speaker_anchors", [])
    if forced_anchor_id:
        for anchor in anchors:
            if anchor.get("anchor_id") == forced_anchor_id:
                return anchor
    ranked = [anchor for anchor in anchors if anchor.get("speaker") == speaker]
    if ranked:
        ranked.sort(key=lambda item: (-item.get("priority", 0), -item.get("confidence", 0.0)))
        return ranked[0]
    for point in panel_analysis.get("speaker_points", []):
        if point.get("speaker") == speaker:
            return {
                "speaker": point.get("speaker"),
                "anchor_id": point.get("anchor_id"),
                "role": "legacy_point",
                "x": point.get("x"),
                "y": point.get("y"),
                "confidence": point.get("confidence", 0.0),
                "source": point.get("source", "legacy"),
            }
    return None


def measure_box_for_zone(
    item: dict[str, Any],
    style: SpeakerStyle,
    zone_box: tuple[int, int, int, int],
    font_path: str | None,
    item_overrides: dict[str, Any] | None = None,
    zone: dict[str, Any] | None = None,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], list[str], int] | None:
    item_overrides = item_overrides or {}
    zone = zone or {}
    zone_w = zone_box[2] - zone_box[0]
    zone_h = zone_box[3] - zone_box[1]
    max_text_width = zone_w - style.text_padding[0] - style.text_padding[2]
    max_text_height = zone_h - style.text_padding[1] - style.text_padding[3]
    max_text_width = min(max_text_width, int(item_overrides.get("max_text_width", max_text_width)))
    max_text_height = min(max_text_height, int(item_overrides.get("max_text_height", max_text_height)))
    if max_text_width < 50 or max_text_height < 30:
        return None
    fit = fit_text_block(item["text"], style, font_path, max_text_width, max_text_height)
    if fit is None:
        return None
    lines, font, (text_w, text_h) = fit
    inner_w = text_w
    inner_h = text_h
    if style.bubble_kind == "speech":
        inner_w, inner_h = speech_inner_size_for_text_bbox(text_w, text_h)
    box_w = inner_w + style.text_padding[0] + style.text_padding[2]
    box_h = inner_h + style.text_padding[1] + style.text_padding[3]
    horizontal_align = item_overrides.get("box_horizontal_align") or zone.get("box_horizontal_align") or "center"
    vertical_align = item_overrides.get("box_vertical_align") or zone.get("box_vertical_align") or "center"
    offset_x = int(item_overrides.get("box_offset_x", zone.get("box_offset_x", 0)))
    offset_y = int(item_overrides.get("box_offset_y", zone.get("box_offset_y", 0)))
    if horizontal_align == "left":
        pos_x = zone_box[0]
    elif horizontal_align == "right":
        pos_x = zone_box[2] - box_w
    else:
        pos_x = zone_box[0] + max(0, (zone_w - box_w) // 2)
    if vertical_align == "top":
        pos_y = zone_box[1]
    elif vertical_align == "bottom":
        pos_y = zone_box[3] - box_h
    else:
        pos_y = zone_box[1] + max(0, (zone_h - box_h) // 2)
    pos_x = min(max(zone_box[0], pos_x + offset_x), zone_box[2] - box_w)
    pos_y = min(max(zone_box[1], pos_y + offset_y), zone_box[3] - box_h)
    outer_box = (pos_x, pos_y, pos_x + box_w, pos_y + box_h)
    inner_box = (
        outer_box[0] + style.text_padding[0],
        outer_box[1] + style.text_padding[1],
        outer_box[2] - style.text_padding[2],
        outer_box[3] - style.text_padding[3],
    )
    return outer_box, inner_box, lines, getattr(font, "size", style.font_size)


def render_shape(base: Image.Image, placement: PlacementCandidate, font_path: str | None) -> None:
    style = bubble_style_for_template(placement.speaker, placement.template)
    draw = ImageDraw.Draw(base)
    halo_box = expand_box(placement.box, 8)
    if placement.template == "caption":
        halo_fill = (18, 20, 26, 58)
    elif placement.template == "speech":
        halo_fill = (245, 245, 245, 52)
    else:
        halo_fill = (245, 245, 245, 44)
    draw_soft_mask(base, halo_box, halo_fill, radius=style.radius + 8, blur_radius=4)

    if placement.template == "speech":
        draw.ellipse(placement.box, fill=style.fill, outline=style.outline, width=3)
    else:
        draw.rounded_rectangle(placement.box, radius=style.radius, fill=style.fill, outline=style.outline, width=3)

    font = load_font(font_path, placement.font_size)
    text = "\n".join(placement.lines)
    spacing = max(7, int(placement.font_size * 0.34))
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    text_box = speech_text_safe_box(placement.inner_box) if placement.template == "speech" else placement.inner_box
    tx = text_box[0] + (text_box[2] - text_box[0] - text_w) / 2
    ty = text_box[1] + (text_box[3] - text_box[1] - text_h) / 2 - 1
    draw.multiline_text((tx, ty), text, font=font, fill=style.text_fill, spacing=spacing, align="center")


def style_with_item_overrides(style: SpeakerStyle, item_overrides: dict[str, Any]) -> SpeakerStyle:
    if not item_overrides:
        return style
    font_size = int(item_overrides.get("max_font_size", style.font_size))
    min_font_size = int(item_overrides.get("min_font_size", style.min_font_size))
    font_size = max(style.min_font_size, min(style.font_size, font_size))
    min_font_size = max(12, min(font_size, min_font_size))
    return replace(style, font_size=font_size, min_font_size=min_font_size)


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def normalized_anchor_distance(point: tuple[int, int] | None, box: tuple[int, int, int, int], panel_size: tuple[int, int]) -> float:
    if point is None:
        return 1.0
    cx, cy = box_center(box)
    diagonal = max(1.0, math.hypot(panel_size[0], panel_size[1]))
    return math.hypot(cx - point[0], cy - point[1]) / diagonal


def bubble_matches_placement_side(
    box: tuple[int, int, int, int],
    anchor_point: tuple[int, int] | None,
    placement_side: str | None,
    panel_size: tuple[int, int],
) -> bool:
    if anchor_point is None or not placement_side:
        return False
    cx, cy = box_center(box)
    dx = cx - anchor_point[0]
    dy = cy - anchor_point[1]
    x_threshold = panel_size[0] * 0.08
    y_threshold = panel_size[1] * 0.08
    if placement_side == "upper_right":
        return dx >= x_threshold and dy <= -y_threshold
    if placement_side == "upper_left":
        return dx <= -x_threshold and dy <= -y_threshold
    if placement_side == "upper_mid":
        return abs(dx) <= x_threshold and dy <= -y_threshold
    if placement_side == "mid_right":
        return dx >= x_threshold and abs(dy) <= y_threshold
    if placement_side == "mid_left":
        return dx <= -x_threshold and abs(dy) <= y_threshold
    return False


def candidate_zones_for_item(panel_analysis: dict[str, Any], item: dict[str, Any], template: str, speaker: str) -> list[dict[str, Any]]:
    compatible_kinds = TEMPLATE_ZONE_COMPAT[template]
    zones: list[dict[str, Any]] = []
    item_overrides = panel_analysis["render_hints"].get("item_overrides", {}).get(item["id"], {})
    panel_overrides = panel_analysis.get("panel_overrides", {})
    preferred_zone_ids = item_overrides.get("preferred_zone_ids", [])
    disallowed_zone_ids = set(item_overrides.get("disallowed_zone_ids", []))
    allow_generic_speech_fallback = bool(panel_overrides.get("generic_speech_fallback_allowed", True))
    placement_policy = panel_overrides.get("placement_order_policy", "hybrid_ranked")

    for zone in panel_analysis.get("speaker_local_zones", []):
        if zone.get("item_id") and zone["item_id"] != item["id"]:
            continue
        if zone.get("speaker") and zone["speaker"] != speaker:
            continue
        if zone["kind"] not in compatible_kinds:
            continue
        if zone["id"] in disallowed_zone_ids:
            continue
        zones.append({**zone, "candidate_class": "speaker_local"})

    for zone in panel_analysis.get("safe_zones", []):
        if zone["kind"] not in compatible_kinds:
            continue
        if zone["id"] in disallowed_zone_ids:
            continue
        zones.append({**zone, "candidate_class": "generic"})

    if template == "speech" and not allow_generic_speech_fallback:
        zones = [zone for zone in zones if zone.get("candidate_class") == "speaker_local"]

    def hybrid_sort_key(zone: dict[str, Any]) -> tuple[int, int, float, float]:
        preferred_rank = 0 if zone["id"] in preferred_zone_ids else 1
        class_rank = 0 if zone.get("candidate_class") == "speaker_local" else 1
        return preferred_rank, class_rank, -float(zone.get("confidence", 0.0)), float(zone.get("y", 0.0))

    def legacy_sort_key(zone: dict[str, Any]) -> tuple[int, float, float]:
        class_rank = 0 if zone.get("candidate_class") == "generic" else 1
        return class_rank, float(zone.get("y", 0.0)), float(zone.get("x", 0.0))

    sort_key = hybrid_sort_key if placement_policy == "hybrid_ranked" else legacy_sort_key
    return sorted(zones, key=sort_key)


def placement_overlap_metrics(
    box: tuple[int, int, int, int],
    forbidden_abs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for zone in forbidden_abs:
        ratio = overlap_ratio(box, tuple(zone["box"]))
        metrics.append(
            {
                "id": zone["id"],
                "kind": zone["kind"],
                "ratio": round(ratio, 4),
                "intersection_area": intersection_area(box, tuple(zone["box"])),
            }
        )
    return metrics


def attachment_score_breakdown(
    zone: dict[str, Any],
    item_overrides: dict[str, Any],
    panel_overrides: dict[str, Any],
    anchor: dict[str, Any] | None,
    anchor_distance: float,
    preferred_side_bonus: float,
    local_priority_bonus: float,
    max_overlap: float,
    line_count: int,
    line_balance_penalty: float,
    generic_fallback_penalty: float,
) -> dict[str, float]:
    preferred_bonus = 18.0 if zone["id"] in item_overrides.get("preferred_zone_ids", []) else 0.0
    local_bonus = 22.0 if zone.get("candidate_class") == "speaker_local" else 0.0
    anchor_bonus = 12.0 if anchor else -14.0
    panel_threshold = float(panel_overrides.get("attachment_review_threshold", 0.65))
    confidence_score = float(zone.get("confidence", 0.0)) * 100.0
    review_penalty = 12.0 if zone.get("candidate_class") != "speaker_local" and confidence_score / 100.0 < panel_threshold else 0.0
    return {
        "zone_confidence": round(confidence_score, 3),
        "preferred_zone_bonus": round(preferred_bonus, 3),
        "speaker_local_bonus": round(local_bonus, 3),
        "local_priority_bonus": round(local_priority_bonus, 3),
        "preferred_side_bonus": round(preferred_side_bonus, 3),
        "anchor_bonus": round(anchor_bonus, 3),
        "anchor_distance_penalty": round(anchor_distance * 35.0, 3),
        "overlap_penalty": round(max_overlap * 180.0, 3),
        "line_count_penalty": round(line_count * 5.0, 3),
        "line_balance_penalty": round(line_balance_penalty, 3),
        "generic_fallback_penalty": round(generic_fallback_penalty, 3),
        "attachment_review_penalty": round(review_penalty, 3),
    }


def sum_score_breakdown(breakdown: dict[str, float]) -> float:
    positive_keys = {
        "zone_confidence",
        "preferred_zone_bonus",
        "speaker_local_bonus",
        "local_priority_bonus",
        "preferred_side_bonus",
        "anchor_bonus",
    }
    score = 0.0
    for key, value in breakdown.items():
        if key in positive_keys:
            score += value
        else:
            score -= value
    return round(score, 3)


def choose_placement(
    image: Image.Image,
    item: dict[str, Any],
    panel_analysis: dict[str, Any],
    placed_boxes: list[tuple[int, int, int, int]],
    forbidden_abs: list[dict[str, Any]],
    font_path: str | None,
) -> tuple[PlacementCandidate, list[dict[str, Any]]]:
    template = panel_analysis["render_hints"]["item_templates"].get(item["id"], template_for_item(item))
    speaker = normalize_speaker(item.get("speaker"), item)
    item_overrides = panel_analysis["render_hints"].get("item_overrides", {}).get(item["id"], {})
    style = style_with_item_overrides(bubble_style_for_template(speaker, template), item_overrides)
    threshold = panel_analysis["render_hints"].get("forbidden_overlap_threshold", DEFAULT_FORBIDDEN_OVERLAP_THRESHOLD)
    panel_overrides = panel_analysis.get("panel_overrides", {})
    rejected: list[dict[str, Any]] = []
    candidates_scored: list[PlacementCandidate] = []
    warned_fallback: PlacementCandidate | None = None
    anchor = None
    speaker_point = None
    if template == "speech":
        anchor = resolve_anchor(panel_analysis, speaker, item["id"])
        if anchor:
            speaker_point = normalized_point_to_pixels(anchor, image.size)

    candidates = candidate_zones_for_item(panel_analysis, item, template, speaker)
    local_candidates_available = any(zone.get("candidate_class") == "speaker_local" for zone in candidates)
    for zone in candidates:
        zone_box = normalized_rect_to_pixels(zone, image.size)
        measured = measure_box_for_zone(item, style, zone_box, font_path, item_overrides=item_overrides, zone=zone)
        if measured is None:
            rejected.append({"zone_id": zone["id"], "candidate_class": zone.get("candidate_class", "generic"), "reason": "text_does_not_fit"})
            continue
        outer_box, inner_box, lines, font_size = measured
        outer_box = clamp_box(outer_box, image.size)
        if any(boxes_intersect(expand_box(outer_box, DEFAULT_BOX_GAP), expand_box(other, DEFAULT_BOX_GAP)) for other in placed_boxes):
            rejected.append({"zone_id": zone["id"], "candidate_class": zone.get("candidate_class", "generic"), "reason": "collides_with_previous_placement"})
            continue
        overlap_metrics = placement_overlap_metrics(outer_box, forbidden_abs)
        max_overlap = max((metric["ratio"] for metric in overlap_metrics), default=0.0)
        line_balance_penalty = 0.0
        if lines:
            widths = [ImageDraw.Draw(Image.new("L", (8, 8), 0)).textlength(line, font=load_font(font_path, font_size)) for line in lines]
            if len(widths) >= 2:
                line_balance_penalty += (max(widths) - min(widths)) * 0.06
                if widths[-1] < max(widths) * 0.42:
                    line_balance_penalty += (max(widths) * 0.42 - widths[-1]) * 0.12
        review_reasons = ["missing_speaker_anchor"] if template == "speech" and anchor is None else []
        anchor_distance = normalized_anchor_distance(speaker_point, outer_box, image.size)
        local_priority_bonus = min(float(zone.get("priority", 0.0)), 100.0) * 0.3 if zone.get("candidate_class") == "speaker_local" else 0.0
        preferred_side_bonus = 12.0 if bubble_matches_placement_side(outer_box, speaker_point, zone.get("placement_side"), image.size) else 0.0
        generic_fallback_penalty = 18.0 if template == "speech" and zone.get("candidate_class") == "generic" and local_candidates_available else 0.0
        score_breakdown = attachment_score_breakdown(
            zone=zone,
            item_overrides=item_overrides,
            panel_overrides=panel_overrides,
            anchor=anchor,
            anchor_distance=anchor_distance,
            preferred_side_bonus=preferred_side_bonus,
            local_priority_bonus=local_priority_bonus,
            max_overlap=max_overlap,
            line_count=len(lines),
            line_balance_penalty=line_balance_penalty,
            generic_fallback_penalty=generic_fallback_penalty,
        )
        score = sum_score_breakdown(score_breakdown)
        fallback_reason = None
        if zone.get("candidate_class") != "speaker_local":
            fallback_reason = "speech_fell_back_to_generic_zone" if template == "speech" else None
            if template == "speech" and "chat_ui" in panel_analysis.get("overlay_tracks", []):
                fallback_reason = "mixed_mode_attachment_conflict"
                review_reasons = sorted(set(review_reasons + ["mixed_mode_attachment_conflict"]))
        candidate = PlacementCandidate(
            item=item,
            template=template,
            bubble_kind=TEMPLATE_KIND_MAP[template],
            speaker=speaker,
            box=outer_box,
            inner_box=inner_box,
            score=round(score, 3),
            zone_id=zone["id"],
            zone_kind=zone["kind"],
            confidence=zone["confidence"],
            font_size=font_size,
            lines=lines,
            overlap_metrics=overlap_metrics,
            selected_with_warning=max_overlap > threshold,
            warning_reason="forbidden_overlap_threshold_exceeded" if max_overlap > threshold else None,
            candidate_class=zone.get("candidate_class", "generic"),
            resolved_anchor_id=anchor.get("anchor_id") if anchor else None,
            resolved_anchor_role=anchor.get("role") if anchor else None,
            selected_zone_score=score,
            score_breakdown=score_breakdown,
            fallback_reason=fallback_reason,
            review_reasons=review_reasons,
        )
        if warned_fallback is None or candidate.score > warned_fallback.score:
            warned_fallback = candidate
        if max_overlap <= threshold:
            candidates_scored.append(candidate)
        if max_overlap > threshold:
            rejected.append(
                {
                    "zone_id": zone["id"],
                    "candidate_class": zone.get("candidate_class", "generic"),
                    "reason": "exceeds_forbidden_overlap_threshold",
                    "max_overlap": round(max_overlap, 4),
                    "score": score,
                }
            )
        else:
            rejected.append(
                {
                    "zone_id": zone["id"],
                    "candidate_class": zone.get("candidate_class", "generic"),
                    "reason": "ranked_below_selected_candidate",
                    "score": score,
                }
            )
    if candidates_scored:
        fallback_margin = float(panel_overrides.get("generic_fallback_margin", 8.0))

        def candidate_class_rank(candidate: PlacementCandidate) -> int:
            return 0 if candidate.candidate_class == "speaker_local" else 1

        def tie_break_key(candidate: PlacementCandidate) -> tuple[float, float, float, int, str]:
            return (
                candidate.score_breakdown.get("anchor_distance_penalty", 0.0),
                candidate.score_breakdown.get("overlap_penalty", 0.0),
                -candidate.confidence,
                candidate_class_rank(candidate),
                candidate.zone_id,
            )

        candidates_scored.sort(key=lambda candidate: (-candidate.score, tie_break_key(candidate)))
        best = candidates_scored[0]
        best_local = next((candidate for candidate in candidates_scored if candidate.candidate_class == "speaker_local"), None)
        best_generic = next((candidate for candidate in candidates_scored if candidate.candidate_class == "generic"), None)
        if best_local and best_generic and best_generic.score > best_local.score + fallback_margin:
            best = best_generic
        else:
            best = best_local or best
        return best, [entry for entry in rejected if entry["zone_id"] != best.zone_id]
    if warned_fallback is not None:
        warned_fallback.selected_with_warning = True
        warned_fallback.warning_reason = warned_fallback.warning_reason or "forbidden_overlap_threshold_exceeded"
        warned_fallback.review_reasons = sorted(set(warned_fallback.review_reasons + ["forbidden_overlap_threshold_exceeded"]))
        return warned_fallback, [entry for entry in rejected if entry["zone_id"] != warned_fallback.zone_id]
    raise RuntimeError(f"No placement candidate available for item {item['id']}")


def render_panel(
    panel_path: Path,
    output_path: Path,
    panel_analysis: dict[str, Any],
    items: list[dict[str, Any]],
    font_path: str | None,
) -> dict[str, Any]:
    base = Image.open(panel_path).convert("RGBA")
    forbidden_abs = []
    for zone in panel_analysis.get("forbidden_zones", []):
        forbidden_abs.append(
            {
                "id": zone["id"],
                "kind": zone["kind"],
                "box": list(normalized_rect_to_pixels(zone, base.size)),
                "confidence": zone["confidence"],
            }
        )
    placements: list[PlacementCandidate] = []
    rejected_by_item: dict[str, list[dict[str, Any]]] = {}
    manual_review_required = panel_analysis["render_hints"].get("manual_review", False)
    manual_review_reasons = list(panel_analysis["render_hints"].get("manual_review_reasons", []))
    working = base.copy()

    ordered_items = sorted(
        items,
        key=lambda item: (
            RENDER_ORDER[panel_analysis["render_hints"]["item_templates"].get(item["id"], template_for_item(item))],
            -len(item.get("text", "")),
        ),
    )

    for item in ordered_items:
        placement, rejected = choose_placement(working, item, panel_analysis, [entry.box for entry in placements], forbidden_abs, font_path)
        rejected_by_item[item["id"]] = rejected
        if placement.selected_with_warning:
            manual_review_required = True
            manual_review_reasons.append(f"{item['id']}:forbidden_overlap_threshold_exceeded")
        for reason in placement.review_reasons:
            manual_review_required = True
            manual_review_reasons.append(f"{item['id']}:{reason}")
        if placement.fallback_reason:
            manual_review_required = True
            manual_review_reasons.append(f"{item['id']}:{placement.fallback_reason}")
        placements.append(placement)
        render_shape(working, placement, font_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    working.convert("RGB").save(output_path)
    return {
        "panel_id": panel_path.stem,
        "source": str(panel_path.resolve()),
        "output": str(output_path.resolve()),
        "size": list(base.size),
        "panel_mode": panel_analysis["panel_mode"],
        "overlay_tracks": panel_analysis["overlay_tracks"],
        "analysis_safe_zones": panel_analysis["safe_zones"],
        "analysis_forbidden_zones": forbidden_abs,
        "manual_review_required": manual_review_required,
        "manual_review_reasons": sorted(set(manual_review_reasons)),
        "placements": [
            {
                "id": placement.item["id"],
                "kind": placement.item["kind"],
                "speaker": placement.speaker,
                "text": placement.item["text"],
                "template": placement.template,
                "bubble_kind": placement.bubble_kind,
                "zone_id": placement.zone_id,
                "zone_kind": placement.zone_kind,
                "zone_confidence": placement.confidence,
                "score": placement.score,
                "candidate_class": placement.candidate_class,
                "resolved_anchor_id": placement.resolved_anchor_id,
                "resolved_anchor_role": placement.resolved_anchor_role,
                "selected_zone_score": placement.selected_zone_score,
                "score_breakdown": placement.score_breakdown,
                "font_size": placement.font_size,
                "lines": placement.lines,
                "box": list(placement.box),
                "inner_box": list(placement.inner_box),
                "overlap_metrics": placement.overlap_metrics,
                "max_forbidden_overlap": max((metric["ratio"] for metric in placement.overlap_metrics), default=0.0),
                "selected_with_warning": placement.selected_with_warning,
                "warning_reason": placement.warning_reason,
                "fallback_reason": placement.fallback_reason,
                "review_reasons": placement.review_reasons,
                "rejected_candidates": rejected_by_item[placement.item["id"]],
            }
            for placement in placements
        ],
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    analysis_path = Path(args.analysis)
    lettering_path = Path(args.lettering)
    scroll_plan_path = Path(args.scroll_plan)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis file not found: {analysis_path}")

    font_path = resolve_font_path(args.font_path)
    analysis = load_analysis(analysis_path)
    panel_items, lettering_data = load_lettering(lettering_path)
    spacing_map, panel_to_block, _ = load_scroll_plan(scroll_plan_path)
    panel_analysis_by_id = {panel["panel_id"]: panel for panel in analysis.get("panels", [])}

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_outputs: list[dict[str, Any]] = []
    for panel_path in sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix == ".png" and path.stem.startswith("p")):
        panel_analysis = panel_analysis_by_id[panel_path.stem]
        panel_outputs.append(
            render_panel(
                panel_path=panel_path,
                output_path=output_dir / panel_path.name,
                panel_analysis=panel_analysis,
                items=panel_items.get(panel_path.stem, []),
                font_path=font_path,
            )
        )

    longscroll_path = None
    if args.compose_longscroll:
        longscroll_path = compose_longscroll(
            panel_outputs=panel_outputs,
            output_dir=output_dir,
            episode_id=str(lettering_data.get("episode", "episode")),
            spacing_map=spacing_map,
            panel_to_block=panel_to_block,
            longscroll_name="ep001_ballooned_longscroll.png",
        )

    manifest = {
        "episode": lettering_data.get("episode", "episode"),
        "analysis_path": str(analysis_path.resolve()),
        "analysis_source_manifest": analysis.get("source", {}).get("manifest"),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "lettering": str(lettering_path.resolve()),
        "scroll_plan": str(scroll_plan_path.resolve()),
        "font_path": font_path,
        "forbidden_overlap_threshold": DEFAULT_FORBIDDEN_OVERLAP_THRESHOLD,
        "longscroll": str(longscroll_path.resolve()) if longscroll_path else None,
        "panels": panel_outputs,
    }
    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote render manifest: {manifest_path}")
    if longscroll_path:
        print(f"Wrote longscroll: {longscroll_path}")


if __name__ == "__main__":
    main()
