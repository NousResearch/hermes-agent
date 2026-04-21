from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from balloon_layout_utils import (
    DEFAULT_FORBIDDEN_OVERLAP_THRESHOLD,
    candidate_background_score,
    load_lettering,
    load_manifest,
    load_scroll_plan,
    relative_or_absolute,
    template_for_item,
)


SCREEN_HEAVY_SHOTS = {"insert_screen", "split_focus", "hand_closeup", "text_focus_prop", "final_cliffhanger"}
SCREEN_VISUAL_HINTS = ("screen", "laptop", "message", "browser", "memo", "file", "notification", "monitor")

PANEL_FORBIDDEN_OVERRIDES: dict[str, list[dict[str, Any]]] = {
    "p01": [
        {"id": "score_sheet", "kind": "key_prop", "x": 0.18, "y": 0.56, "w": 0.58, "h": 0.24, "confidence": 0.9},
        {"id": "writing_hand", "kind": "hand", "x": 0.52, "y": 0.62, "w": 0.20, "h": 0.18, "confidence": 0.84},
    ],
    "p02": [
        {"id": "mother_face", "kind": "face", "x": 0.62, "y": 0.32, "w": 0.16, "h": 0.14, "confidence": 0.88},
        {"id": "son_head", "kind": "face", "x": 0.24, "y": 0.45, "w": 0.22, "h": 0.16, "confidence": 0.74},
        {"id": "desk_focus", "kind": "key_prop", "x": 0.32, "y": 0.58, "w": 0.44, "h": 0.22, "confidence": 0.66},
    ],
    "p03": [
        {"id": "screen_triptych", "kind": "screen_region", "x": 0.06, "y": 0.10, "w": 0.88, "h": 0.76, "confidence": 0.95},
    ],
    "p04": [
        {"id": "screen_triptych", "kind": "screen_region", "x": 0.08, "y": 0.10, "w": 0.84, "h": 0.76, "confidence": 0.95},
    ],
    "p05": [
        {"id": "screen_region", "kind": "screen_region", "x": 0.22, "y": 0.28, "w": 0.58, "h": 0.38, "confidence": 0.92},
        {"id": "mouse_hand", "kind": "hand", "x": 0.48, "y": 0.48, "w": 0.22, "h": 0.18, "confidence": 0.82},
    ],
    "p06": [
        {"id": "mother_face", "kind": "face", "x": 0.64, "y": 0.25, "w": 0.16, "h": 0.18, "confidence": 0.9},
        {"id": "son_face", "kind": "face", "x": 0.20, "y": 0.34, "w": 0.18, "h": 0.16, "confidence": 0.86},
        {"id": "laptop_glow", "kind": "screen_region", "x": 0.66, "y": 0.46, "w": 0.22, "h": 0.26, "confidence": 0.8},
        {"id": "typing_hands", "kind": "hand", "x": 0.46, "y": 0.58, "w": 0.34, "h": 0.16, "confidence": 0.8},
    ],
    "p07": [
        {"id": "protagonist_face", "kind": "face", "x": 0.25, "y": 0.20, "w": 0.20, "h": 0.16, "confidence": 0.86},
        {"id": "mother_face", "kind": "face", "x": 0.62, "y": 0.16, "w": 0.16, "h": 0.16, "confidence": 0.6},
        {"id": "screen_glow_band", "kind": "screen_region", "x": 0.04, "y": 0.43, "w": 0.92, "h": 0.15, "confidence": 0.94},
    ],
    "p08": [
        {"id": "protagonist_face", "kind": "face", "x": 0.16, "y": 0.30, "w": 0.22, "h": 0.16, "confidence": 0.88},
        {"id": "mother_face", "kind": "face", "x": 0.60, "y": 0.26, "w": 0.16, "h": 0.16, "confidence": 0.9},
        {"id": "phone_glow", "kind": "screen_region", "x": 0.54, "y": 0.56, "w": 0.18, "h": 0.22, "confidence": 0.94},
        {"id": "laptop_glow", "kind": "screen_region", "x": 0.38, "y": 0.52, "w": 0.22, "h": 0.18, "confidence": 0.82},
        {"id": "typing_hands", "kind": "hand", "x": 0.28, "y": 0.58, "w": 0.24, "h": 0.14, "confidence": 0.72},
    ],
}

PANEL_ATTACHMENT_PROFILES: dict[str, dict[str, Any]] = {
    "p02": {
        "speaker_anchors": [
            {
                "speaker": "mother",
                "anchor_id": "mother_mouth_primary",
                "role": "mouth",
                "x": 0.70,
                "y": 0.41,
                "confidence": 0.9,
                "source": "panel_override",
                "priority": 100,
            }
        ],
        "speaker_local_zones": [
            {
                "id": "p02_mother_upper_right",
                "kind": "speech",
                "speaker": "mother",
                "item_id": "l01",
                "zone_ref": "speech_top_right",
                "x": 0.56,
                "y": 0.04,
                "w": 0.30,
                "h": 0.19,
                "placement_side": "upper_right",
                "priority": 100,
                "confidence": 0.94,
                "rationale": "keep the mother's line in the clear space directly above her shoulder",
                "source": "panel_override",
            },
            {
                "id": "p02_mother_upper_mid",
                "kind": "speech",
                "speaker": "mother",
                "item_id": "l01",
                "zone_ref": "speech_upper_mid",
                "x": 0.43,
                "y": 0.05,
                "w": 0.29,
                "h": 0.18,
                "placement_side": "upper_mid",
                "priority": 84,
                "confidence": 0.8,
                "rationale": "secondary shelf that still stays attached to mother",
                "source": "panel_override",
            },
        ],
        "panel_overrides": {
            "placement_order_policy": "hybrid_ranked",
            "generic_speech_fallback_allowed": True,
            "generic_fallback_margin": 8.0,
            "attachment_review_threshold": 0.65,
        },
        "item_overrides": {
            "l01": {
                "preferred_zone_ids": ["p02_mother_upper_right", "speech_top_right", "speech_upper_mid"],
                "disallowed_zone_ids": ["speech_top_left"],
                "force_anchor_id": "mother_mouth_primary",
                "max_font_size": 29,
                "min_font_size": 23,
                "max_text_width": 170,
                "box_horizontal_align": "right",
                "box_vertical_align": "bottom",
                "box_offset_x": -8,
                "box_offset_y": 12,
                "force_manual_review": False,
            }
        },
    },
    "p06": {
        "speaker_anchors": [
            {
                "speaker": "mother",
                "anchor_id": "mother_mouth_primary",
                "role": "mouth",
                "x": 0.71,
                "y": 0.36,
                "confidence": 0.92,
                "source": "panel_override",
                "priority": 100,
            }
        ],
        "speaker_local_zones": [
            {
                "id": "p06_mother_upper_right",
                "kind": "speech",
                "speaker": "mother",
                "item_id": "l02",
                "zone_ref": "speech_top_right",
                "x": 0.57,
                "y": 0.04,
                "w": 0.28,
                "h": 0.17,
                "placement_side": "upper_right",
                "priority": 100,
                "confidence": 0.93,
                "rationale": "top-right pocket above the laptop keeps the accusation attached to mother",
                "source": "panel_override",
            },
            {
                "id": "p06_mother_upper_mid",
                "kind": "speech",
                "speaker": "mother",
                "item_id": "l02",
                "zone_ref": "speech_upper_mid",
                "x": 0.43,
                "y": 0.05,
                "w": 0.26,
                "h": 0.16,
                "placement_side": "upper_mid",
                "priority": 80,
                "confidence": 0.79,
                "rationale": "secondary shelf that keeps the bubble clear of the laptop glow",
                "source": "panel_override",
            },
        ],
        "panel_overrides": {
            "placement_order_policy": "hybrid_ranked",
            "generic_speech_fallback_allowed": True,
            "generic_fallback_margin": 8.0,
            "attachment_review_threshold": 0.68,
        },
        "item_overrides": {
            "l02": {
                "preferred_zone_ids": ["p06_mother_upper_right", "speech_top_right"],
                "disallowed_zone_ids": ["speech_top_left", "caption_top_left"],
                "force_anchor_id": "mother_mouth_primary",
                "max_font_size": 30,
                "min_font_size": 24,
                "max_text_width": 150,
                "box_horizontal_align": "right",
                "box_vertical_align": "bottom",
                "box_offset_x": -10,
                "box_offset_y": 10,
                "force_manual_review": False,
            }
        },
    },
    "p08": {
        "speaker_anchors": [
            {
                "speaker": "mother",
                "anchor_id": "mother_mouth_primary",
                "role": "mouth",
                "x": 0.67,
                "y": 0.34,
                "confidence": 0.94,
                "source": "panel_override",
                "priority": 100,
            }
        ],
        "speaker_local_zones": [
            {
                "id": "p08_mother_upper_right",
                "kind": "speech",
                "speaker": "mother",
                "item_id": "l05",
                "zone_ref": "speech_top_right",
                "x": 0.58,
                "y": 0.02,
                "w": 0.29,
                "h": 0.19,
                "placement_side": "upper_right",
                "priority": 100,
                "confidence": 0.95,
                "rationale": "keep the mother's dialogue over her side without colliding with the phone chat",
                "source": "panel_override",
            },
            {
                "id": "p08_mother_upper_mid",
                "kind": "speech",
                "speaker": "mother",
                "item_id": "l05",
                "zone_ref": "speech_upper_mid",
                "x": 0.31,
                "y": 0.03,
                "w": 0.38,
                "h": 0.22,
                "placement_side": "upper_mid",
                "priority": 82,
                "confidence": 0.77,
                "rationale": "fallback shelf when the chat UI forces a tighter right edge",
                "source": "panel_override",
            },
        ],
        "panel_overrides": {
            "placement_order_policy": "hybrid_ranked",
            "generic_speech_fallback_allowed": True,
            "generic_fallback_margin": 8.0,
            "attachment_review_threshold": 0.7,
        },
        "item_overrides": {
            "l05": {
                "preferred_zone_ids": ["p08_mother_upper_right", "speech_top_right", "speech_upper_mid"],
                "disallowed_zone_ids": ["speech_top_left"],
                "force_anchor_id": "mother_mouth_primary",
                "max_font_size": 22,
                "min_font_size": 20,
                "max_text_width": 132,
                "max_text_height": 150,
                "box_horizontal_align": "right",
                "box_vertical_align": "top",
                "box_offset_x": -8,
                "box_offset_y": -6,
                "force_manual_review": False,
            }
        },
    },
}

BASE_ZONE_LIBRARY: list[dict[str, Any]] = [
    {"id": "speech_top_left", "kind": "speech", "x": 0.05, "y": 0.04, "w": 0.34, "h": 0.18, "bias": 1.0, "rationale": "top-left negative space"},
    {"id": "speech_top_right", "kind": "speech", "x": 0.61, "y": 0.04, "w": 0.34, "h": 0.18, "bias": 1.0, "rationale": "top-right negative space"},
    {"id": "speech_upper_mid", "kind": "speech", "x": 0.32, "y": 0.05, "w": 0.36, "h": 0.17, "bias": 0.92, "rationale": "upper-mid negative space"},
    {"id": "speech_mid_left", "kind": "speech", "x": 0.04, "y": 0.24, "w": 0.30, "h": 0.17, "bias": 0.76, "rationale": "mid-left fallback"},
    {"id": "speech_mid_right", "kind": "speech", "x": 0.66, "y": 0.24, "w": 0.28, "h": 0.17, "bias": 0.76, "rationale": "mid-right fallback"},
    {"id": "caption_top_left", "kind": "caption", "x": 0.05, "y": 0.05, "w": 0.38, "h": 0.14, "bias": 0.98, "rationale": "compact caption banner"},
    {"id": "caption_bottom_left", "kind": "caption", "x": 0.05, "y": 0.80, "w": 0.42, "h": 0.14, "bias": 0.78, "rationale": "bottom caption fallback"},
    {"id": "chat_top_left", "kind": "chat_ui", "x": 0.05, "y": 0.08, "w": 0.28, "h": 0.14, "bias": 1.04, "rationale": "compact edge-aligned chat slot"},
    {"id": "chat_top_right", "kind": "chat_ui", "x": 0.67, "y": 0.08, "w": 0.28, "h": 0.14, "bias": 1.04, "rationale": "compact edge-aligned chat slot"},
    {"id": "chat_mid_right", "kind": "chat_ui", "x": 0.67, "y": 0.26, "w": 0.28, "h": 0.14, "bias": 0.86, "rationale": "secondary chat slot"},
    {"id": "note_top_right", "kind": "screen_note", "x": 0.60, "y": 0.06, "w": 0.30, "h": 0.14, "bias": 0.95, "rationale": "screen note callout"},
    {"id": "note_top_left", "kind": "screen_note", "x": 0.07, "y": 0.06, "w": 0.30, "h": 0.14, "bias": 0.95, "rationale": "screen note callout"},
]

PANEL_SAFE_ZONE_EXTRAS: dict[str, list[dict[str, Any]]] = {
    "p01": [{"id": "caption_upper_mid", "kind": "caption", "x": 0.28, "y": 0.05, "w": 0.44, "h": 0.14, "bias": 1.02, "rationale": "score-sheet opener caption"}],
    "p03": [{"id": "caption_top_band", "kind": "caption", "x": 0.08, "y": 0.01, "w": 0.42, "h": 0.11, "bias": 1.08, "rationale": "top border shelf above screen triptych"}],
    "p04": [
        {"id": "chat_top_band", "kind": "chat_ui", "x": 0.56, "y": 0.01, "w": 0.32, "h": 0.11, "bias": 1.08, "rationale": "top border shelf above split-focus triptych"},
        {"id": "chat_left_band", "kind": "chat_ui", "x": 0.12, "y": 0.01, "w": 0.32, "h": 0.11, "bias": 1.0, "rationale": "alternate top border shelf above split-focus triptych"},
    ],
    "p07": [{"id": "note_top_center", "kind": "screen_note", "x": 0.34, "y": 0.06, "w": 0.30, "h": 0.14, "bias": 1.05, "rationale": "screen note above glow band"}],
    "p08": [{"id": "caption_top_left_wide", "kind": "caption", "x": 0.04, "y": 0.05, "w": 0.42, "h": 0.14, "bias": 1.0, "rationale": "cliffhanger caption shelf"}],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze fal-generated panels and emit balloon-safe zone guidance.")
    parser.add_argument("--manifest", required=True, help="Path to generated_fal_manifest_v3.json")
    parser.add_argument("--lettering", required=True, help="Path to lettering_script.yaml")
    parser.add_argument("--scroll-plan", required=True, help="Path to scroll_plan.yaml")
    parser.add_argument("--output", required=True, help="Output YAML path")
    return parser.parse_args()


def _clamp_norm(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def _normalized_rect(rect: dict[str, Any]) -> dict[str, Any]:
    return {
        "x": _clamp_norm(rect["x"]),
        "y": _clamp_norm(rect["y"]),
        "w": _clamp_norm(rect["w"]),
        "h": _clamp_norm(rect["h"]),
    }


def _normalized_point(point: dict[str, Any]) -> dict[str, float]:
    return {
        "x": _clamp_norm(point["x"]),
        "y": _clamp_norm(point["y"]),
    }


def _rect_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area = max(0.0001, a["w"] * a["h"])
    return intersection / area


def classify_tracks(items: list[dict[str, Any]], shot: str, visual: str) -> tuple[str, list[str]]:
    tracks: list[str] = []
    for item in items:
        template = template_for_item(item)
        if template == "speech":
            tracks.append("dialogue")
        elif template == "caption":
            tracks.append("caption")
        elif template == "chat_ui":
            tracks.append("chat_ui")
        elif template == "screen_note":
            tracks.append("screen_ui")
    visual_lower = visual.lower()
    if shot in SCREEN_HEAVY_SHOTS or any(hint in visual_lower for hint in SCREEN_VISUAL_HINTS):
        tracks.append("screen_ui")
    deduped = list(dict.fromkeys(track for track in tracks if track))
    if not deduped:
        deduped = ["silent"]
    if "dialogue" in deduped:
        panel_mode = "dialogue"
    elif "chat_ui" in deduped:
        panel_mode = "chat_ui"
    elif "screen_ui" in deduped:
        panel_mode = "screen_ui"
    elif "caption" in deduped:
        panel_mode = "caption"
    else:
        panel_mode = "silent"
    return panel_mode, deduped


def build_forbidden_zones(panel_id: str, shot: str, tracks: list[str]) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    if panel_id != "p01":
        zones.append(
            {
                "id": "focal_center",
                "kind": "focal_center",
                "x": 0.28,
                "y": 0.22,
                "w": 0.44,
                "h": 0.34,
                "confidence": 0.78,
                "source": "composition_rule",
            }
        )
    if "screen_ui" in tracks and panel_id not in {"p03", "p04", "p05", "p07", "p08"}:
        zones.append(
            {
                "id": "screen_hint",
                "kind": "screen_region",
                "x": 0.34,
                "y": 0.36,
                "w": 0.34,
                "h": 0.26,
                "confidence": 0.72,
                "source": "shot_heuristic",
            }
        )
    for rect in PANEL_FORBIDDEN_OVERRIDES.get(panel_id, []):
        zone = dict(rect)
        zone.setdefault("source", "panel_override")
        zones.append(zone)
    return zones


def candidate_zone_specs(panel_id: str, panel_mode: str, tracks: list[str]) -> list[dict[str, Any]]:
    specs = [dict(spec) for spec in BASE_ZONE_LIBRARY]
    specs.extend(dict(spec) for spec in PANEL_SAFE_ZONE_EXTRAS.get(panel_id, []))
    if panel_mode == "dialogue":
        specs.append({"id": "speech_top_left_wide", "kind": "speech", "x": 0.04, "y": 0.03, "w": 0.38, "h": 0.20, "bias": 1.02, "rationale": "dialogue-first widened corner"})
    if "chat_ui" in tracks:
        specs.append({"id": "chat_upper_left", "kind": "chat_ui", "x": 0.08, "y": 0.24, "w": 0.28, "h": 0.14, "bias": 0.88, "rationale": "secondary chat stack"})
    if "caption" in tracks:
        specs.append({"id": "caption_top_right", "kind": "caption", "x": 0.55, "y": 0.05, "w": 0.38, "h": 0.14, "bias": 0.9, "rationale": "right caption shelf"})
    return specs


def score_safe_zones(image: Image.Image, specs: list[dict[str, Any]], forbidden_zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    width, height = image.size
    results: list[dict[str, Any]] = []
    for spec in specs:
        rect = _normalized_rect(spec)
        box = (
            int(rect["x"] * width),
            int(rect["y"] * height),
            int((rect["x"] + rect["w"]) * width),
            int((rect["y"] + rect["h"]) * height),
        )
        penalty, background = candidate_background_score(image, box)
        max_overlap = max((_rect_overlap_ratio(rect, zone) for zone in forbidden_zones), default=0.0)
        if max_overlap > 0.22:
            continue
        score = spec.get("bias", 0.8) * 100 - penalty * 0.8 - max_overlap * 180
        confidence = max(0.15, min(0.98, score / 100))
        result = {
            "id": spec["id"],
            "kind": spec["kind"],
            **rect,
            "confidence": round(confidence, 3),
            "rationale": f"{spec['rationale']}; busy={penalty:.1f}; overlap={max_overlap:.2f}",
            "background_metrics": background,
        }
        results.append(result)
    results.sort(key=lambda item: (item["kind"], -item["confidence"], item["y"], item["x"]))
    ranked: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()
    for item in sorted(results, key=lambda item: item["confidence"], reverse=True):
        key = (item["kind"], item["x"], item["y"])
        if key in seen:
            continue
        seen.add(key)
        ranked.append(item)
    return ranked


def build_attachment_contract(panel_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    profile = PANEL_ATTACHMENT_PROFILES.get(panel_id, {})
    item_ids = {item["id"] for item in items if item.get("id")}
    speaker_anchors = []
    for anchor in profile.get("speaker_anchors", []):
        speaker_anchors.append(
            {
                **anchor,
                **_normalized_point(anchor),
                "confidence": round(float(anchor.get("confidence", 0.0)), 3),
            }
        )
    speaker_points = []
    seen_speakers: set[str] = set()
    for anchor in sorted(speaker_anchors, key=lambda item: (-item.get("priority", 0), -item.get("confidence", 0.0))):
        if anchor["speaker"] in seen_speakers:
            continue
        seen_speakers.add(anchor["speaker"])
        speaker_points.append(
            {
                "speaker": anchor["speaker"],
                "anchor_id": anchor["anchor_id"],
                "x": anchor["x"],
                "y": anchor["y"],
                "confidence": anchor["confidence"],
                "source": anchor.get("source", "panel_override"),
            }
        )
    speaker_local_zones = []
    for zone in profile.get("speaker_local_zones", []):
        if zone.get("item_id") and zone["item_id"] not in item_ids:
            continue
        speaker_local_zones.append(
            {
                **zone,
                **_normalized_rect(zone),
                "confidence": round(float(zone.get("confidence", 0.0)), 3),
            }
        )
    item_overrides = {item_id: dict(value) for item_id, value in profile.get("item_overrides", {}).items() if item_id in item_ids}
    return {
        "speaker_points": speaker_points,
        "speaker_anchors": speaker_anchors,
        "speaker_local_zones": speaker_local_zones,
        "panel_overrides": dict(profile.get("panel_overrides", {})),
        "item_overrides": item_overrides,
    }


def build_render_hints(
    items: list[dict[str, Any]],
    panel_mode: str,
    safe_zones: list[dict[str, Any]],
    attachment_contract: dict[str, Any],
) -> dict[str, Any]:
    item_templates = {item["id"]: template_for_item(item) for item in items if item.get("id")}
    required_candidates = sum(1 for template in item_templates.values() if template in {"speech", "chat_ui", "screen_note", "caption"})
    high_confidence = [zone for zone in safe_zones if zone["confidence"] >= 0.55]
    manual_review_reasons: list[str] = []
    if required_candidates and len(high_confidence) < required_candidates:
        manual_review_reasons.append("insufficient_high_confidence_safe_zones")
    if len({template for template in item_templates.values() if template != "caption"}) >= 2 and len(high_confidence) < 3:
        manual_review_reasons.append("mixed_mode_panel_needs_manual_review")
    if any(template == "speech" for template in item_templates.values()) and not attachment_contract["speaker_anchors"]:
        manual_review_reasons.append("missing_speaker_anchor")
    local_item_ids = {zone.get("item_id") for zone in attachment_contract["speaker_local_zones"] if zone.get("item_id")}
    for item in items:
        item_id = item.get("id")
        if item_id and item_templates.get(item_id) == "speech" and item_id not in local_item_ids:
            manual_review_reasons.append("missing_speaker_local_zone")
            break
    return {
        "default_template": "speech" if panel_mode == "dialogue" else ("chat_ui" if panel_mode == "chat_ui" else "caption"),
        "item_templates": item_templates,
        "manual_review": bool(manual_review_reasons),
        "manual_review_reasons": manual_review_reasons,
        "compactness": "balanced" if panel_mode != "chat_ui" else "compact",
        "max_balloons": max(1, len(item_templates)),
        "forbidden_overlap_threshold": DEFAULT_FORBIDDEN_OVERLAP_THRESHOLD,
        "item_overrides": attachment_contract["item_overrides"],
    }


def analyze_panel(
    panel: dict[str, Any],
    items: list[dict[str, Any]],
    block: dict[str, Any],
) -> dict[str, Any]:
    image_path = Path(panel["path"])
    image = Image.open(image_path).convert("RGB")
    visual = str(block.get("visual", ""))
    panel_mode, overlay_tracks = classify_tracks(items, panel.get("shot", ""), visual)
    forbidden_zones = build_forbidden_zones(panel["panel_id"], panel.get("shot", ""), overlay_tracks)
    safe_zones = score_safe_zones(image, candidate_zone_specs(panel["panel_id"], panel_mode, overlay_tracks), forbidden_zones)
    attachment_contract = build_attachment_contract(panel["panel_id"], items)
    render_hints = build_render_hints(items, panel_mode, safe_zones, attachment_contract)
    reading_order = [item["id"] for item in items if item.get("id")]
    return {
        "panel_id": panel["panel_id"],
        "image_path": relative_or_absolute(image_path),
        "panel_mode": panel_mode,
        "overlay_tracks": overlay_tracks,
        "narrative_role": block.get("purpose", ""),
        "shot": panel.get("shot", ""),
        "reading_order": reading_order,
        "safe_zones": safe_zones,
        "forbidden_zones": forbidden_zones,
        "speaker_points": attachment_contract["speaker_points"],
        "speaker_anchors": attachment_contract["speaker_anchors"],
        "speaker_local_zones": attachment_contract["speaker_local_zones"],
        "panel_overrides": attachment_contract["panel_overrides"],
        "render_hints": render_hints,
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    lettering_path = Path(args.lettering)
    scroll_plan_path = Path(args.scroll_plan)
    output_path = Path(args.output)

    manifest = load_manifest(manifest_path)
    per_panel_items, lettering_data = load_lettering(lettering_path)
    _, _, block_map = load_scroll_plan(scroll_plan_path)

    panels: list[dict[str, Any]] = []
    for panel in manifest.get("panels", []):
        block = block_map.get(panel.get("block_id", ""), {})
        panels.append(
            analyze_panel(
                panel=panel,
                items=per_panel_items.get(panel["panel_id"], []),
                block=block,
            )
        )

    payload = {
        "episode": lettering_data.get("episode", "episode"),
        "analysis_version": "v2",
        "source": {
            "manifest": relative_or_absolute(manifest_path),
            "lettering": relative_or_absolute(lettering_path),
            "scroll_plan": relative_or_absolute(scroll_plan_path),
            "default_input_dir": relative_or_absolute(Path(manifest["panels"][0]["path"]).parent) if manifest.get("panels") else "",
        },
        "panels": panels,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Wrote analysis: {output_path}")


if __name__ == "__main__":
    main()
