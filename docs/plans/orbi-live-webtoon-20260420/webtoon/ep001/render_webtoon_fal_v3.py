from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml
from PIL import Image

BASE = Path("/home/orbibot/.zeroclaw/workspace/hermes-agent/docs/plans/orbi-live-webtoon-20260420/webtoon/ep001")
OUT = BASE / "generated_fal_live"
MANIFEST_PATH = BASE / "generated_fal_manifest_live.json"

PANEL_W = 720
PANEL_H = 1072

STYLE_TAIL = "vertical mobile webtoon panel, clean composition, keep open areas free of any text-like marks"
IDENTITY_PRIORITY = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class RefDescriptor:
    url: str
    label: str
    source: str
    source_panel: str | None = None
    reason: str | None = None


@dataclass
class StrongRefState:
    ref: RefDescriptor
    panel_id: str
    view_class: str
    face_visibility: str


@dataclass
class ContinuityState:
    location_master: RefDescriptor | None = None
    character_anchor_urls: dict[str, dict[str, RefDescriptor]] = field(default_factory=dict)
    last_strong_ref_by_character_and_view_class: dict[str, dict[str, StrongRefState]] = field(default_factory=dict)
    last_scene_ref_by_visible_set: dict[frozenset[str], RefDescriptor] = field(default_factory=dict)
    panel_refs: dict[str, RefDescriptor] = field(default_factory=dict)
    panel_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    previous_panel_id: str | None = None
    previous_panel_ref: RefDescriptor | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_episode_data() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_yaml(BASE / "panel_prompts.yaml"), load_yaml(BASE / "scroll_plan.yaml")


def panel_map_from_data(panel_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {panel["panel_id"]: panel for panel in panel_data["panels"]}


def scroll_block_map(scroll: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {block["block_id"]: block for block in scroll["blocks"]}


def spacing_map_from_scroll(scroll: dict[str, Any]) -> dict[str, str]:
    return {block["block_id"]: block.get("spacing", "medium") for block in scroll["blocks"]}


def join_parts(parts: list[str]) -> str:
    return ", ".join(part.strip() for part in parts if part and part.strip())


def list_to_sentence(values: list[str]) -> str:
    return ", ".join(values)


def build_style_prompt(panel_data: dict[str, Any]) -> str:
    style_anchor = panel_data["style_anchor"]
    return join_parts([style_anchor["positive"], STYLE_TAIL])


def build_scene_prompt(panel_spec: dict[str, Any], scroll_block: dict[str, Any]) -> str:
    scene_bits = [
        panel_spec["prompt"],
        f"story beat {scroll_block.get('purpose', '')}".strip(),
        f"emotion {scroll_block.get('emotion', '')}".strip(),
    ]
    return join_parts(scene_bits)


def build_character_lock_block(panel_spec: dict[str, Any], character_sheets: dict[str, Any]) -> str:
    segments: list[str] = []
    for character in panel_spec["visible_characters"]:
        sheet = character_sheets[character]
        identity_core = sheet["identity_core"]
        identity_text = join_parts(
            [
                identity_core["role"],
                identity_core["age_impression"],
                identity_core["body_build"],
            ]
        )
        visual_invariants = sheet["visual_invariants"]
        invariant_text = join_parts(
            [
                list_to_sentence(visual_invariants["hair"]),
                list_to_sentence(visual_invariants["face"]),
                list_to_sentence(visual_invariants["wardrobe"]),
                list_to_sentence(panel_spec["must_keep"][character]),
            ]
        )
        expression_map = sheet["expression_baseline"]
        expression_text = list_to_sentence(list(expression_map.values()))
        segments.append(
            join_parts(
                [
                    f"{character} identity lock",
                    identity_text,
                    invariant_text,
                    expression_text,
                ]
            )
        )
    return join_parts(segments)


def build_negative_lock_block(panel_spec: dict[str, Any], panel_data: dict[str, Any]) -> str:
    negative_terms = list(panel_data["style_anchor"]["negative"])
    for character in panel_spec["visible_characters"]:
        negative_terms.extend(panel_data["character_sheets"][character]["negative_drift_cues"])
    return join_parts(["avoid " + list_to_sentence(negative_terms)])


def build_prompt_parts(
    panel_data: dict[str, Any],
    panel_spec: dict[str, Any],
    scroll_block: dict[str, Any],
) -> dict[str, str]:
    return {
        "style": build_style_prompt(panel_data),
        "scene": build_scene_prompt(panel_spec, scroll_block),
        "identity_lock": build_character_lock_block(panel_spec, panel_data["character_sheets"]),
        "negative_lock": build_negative_lock_block(panel_spec, panel_data),
    }


def ordered_visible_characters(panel_spec: dict[str, Any]) -> list[str]:
    return sorted(
        panel_spec["visible_characters"],
        key=lambda character: (-IDENTITY_PRIORITY.get(panel_spec["identity_focus"].get(character, "low"), 0), character),
    )


def flatten_prompt(prompt_parts: dict[str, str]) -> str:
    return join_parts(
        [
            prompt_parts["style"],
            prompt_parts["scene"],
            prompt_parts["identity_lock"],
            prompt_parts["negative_lock"],
        ]
    )


def anchor_view_map(panel_data: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for character, sheet in panel_data["character_sheets"].items():
        result[character] = {view["view_id"]: view for view in sheet["anchor_views"]}
    return result


def eligible_face_visibility(face_visibility: str, policy: dict[str, Any]) -> bool:
    return face_visibility in set(policy["eligible_strong_ref_face_visibility"])


def is_reset_shot(shot: str, policy: dict[str, Any]) -> bool:
    return shot in set(policy["continuity_reset_shots"])


def make_anchor_descriptor(character: str, view_id: str, url: str) -> RefDescriptor:
    return RefDescriptor(
        url=url,
        label=f"{character}.{view_id}",
        source="character_sheet",
        source_panel=None,
    )


def make_location_descriptor(url: str) -> RefDescriptor:
    return RefDescriptor(url=url, label="location_master", source="anchor")


def select_reference_stack(
    panel_spec: dict[str, Any],
    panel_data: dict[str, Any],
    state: ContinuityState,
) -> dict[str, Any]:
    policy = panel_data["reference_policy_defaults"]
    anchors = anchor_view_map(panel_data)
    used_refs: list[RefDescriptor] = []
    skipped_refs: list[dict[str, str]] = []

    if state.location_master is not None:
        used_refs.append(state.location_master)

    requested_view_classes: dict[str, set[str]] = {}
    ref_injection = panel_spec["reference_injection"]
    for character in ordered_visible_characters(panel_spec):
        requested_views = ref_injection["use_character_sheet_views"].get(character, [])
        requested_view_classes[character] = set()
        for view_id in requested_views:
            descriptor = state.character_anchor_urls[character][view_id]
            requested_view_classes[character].add(anchors[character][view_id]["view_class"])
            used_refs.append(descriptor)

    for character in ordered_visible_characters(panel_spec):
        if not ref_injection["use_recent_strong_refs"].get(character, False):
            continue
        available = state.last_strong_ref_by_character_and_view_class.get(character, {})
        for view_class in requested_view_classes.get(character, set()):
            strong_state = available.get(view_class)
            if strong_state is None:
                skipped_refs.append(
                    {
                        "label": f"{character}.last_strong_ref.{view_class}",
                        "source": "strong_ref",
                        "reason": "missing_strong_ref",
                    }
                )
                continue
            used_refs.append(
                RefDescriptor(
                    url=strong_state.ref.url,
                    label=f"{character}.last_strong_ref.{view_class}",
                    source=f"panel:{strong_state.panel_id}",
                    source_panel=strong_state.panel_id,
                )
            )

    visible_set = frozenset(panel_spec["visible_characters"])
    scene_ref = state.last_scene_ref_by_visible_set.get(visible_set)
    if scene_ref is not None:
        scene_meta = state.panel_meta.get(scene_ref.source_panel or "", {})
        if scene_meta.get("is_reset_shot", False):
            skipped_refs.append(
                {
                    "label": f"scene_ref:{'+'.join(sorted(visible_set))}",
                    "source": f"panel:{scene_ref.source_panel}",
                    "reason": "continuity_reset_shot",
                }
            )
        else:
            used_refs.append(
                RefDescriptor(
                    url=scene_ref.url,
                    label=f"scene_ref:{'+'.join(sorted(visible_set))}",
                    source=f"panel:{scene_ref.source_panel}" if scene_ref.source_panel else scene_ref.source,
                    source_panel=scene_ref.source_panel,
                )
            )
    else:
        skipped_refs.append(
            {
                "label": f"scene_ref:{'+'.join(sorted(visible_set))}",
                "source": "scene_ref",
                "reason": "missing_scene_ref",
            }
        )

    prev_policy = ref_injection["use_previous_panel"]
    if state.previous_panel_ref is not None:
        prev_id = state.previous_panel_id
        prev_meta = state.panel_meta.get(prev_id or "", {})
        overlapping = bool(visible_set & set(prev_meta.get("visible_characters", [])))
        same_characters_required = prev_policy["only_if_previous_panel_has_same_characters"]
        same_characters_ok = (not same_characters_required) or visible_set == set(prev_meta.get("visible_characters", []))
        prev_is_reset = prev_meta.get("is_reset_shot", False)
        allow_prev = prev_policy["mode"] != "never"
        reason = None
        if not allow_prev:
            reason = "previous_panel_disabled"
        elif not overlapping:
            reason = "no_visible_character_overlap"
        elif not same_characters_ok:
            reason = "different_visible_character_set"
        elif prev_is_reset and not prev_policy["allow_if_previous_shot_in_reset_list"]:
            reason = "continuity_reset_shot"

        if reason is None:
            used_refs.append(
                RefDescriptor(
                    url=state.previous_panel_ref.url,
                    label="prev_panel",
                    source=f"panel:{prev_id}",
                    source_panel=prev_id,
                )
            )
        else:
            skipped_refs.append(
                {
                    "label": "prev_panel",
                    "source": f"panel:{prev_id}",
                    "reason": reason,
                }
            )
    else:
        skipped_refs.append(
            {
                "label": "prev_panel",
                "source": "panel:none",
                "reason": "no_previous_panel",
            }
        )

    deduped: list[RefDescriptor] = []
    seen_urls: set[str] = set()
    for descriptor in used_refs:
        if descriptor.url in seen_urls:
            skipped_refs.append(
                {
                    "label": descriptor.label,
                    "source": descriptor.source,
                    "reason": "duplicate_url_preserved_by_precedence",
                }
            )
            continue
        seen_urls.add(descriptor.url)
        deduped.append(descriptor)

    return {
        "continuity_mode": panel_spec["continuity_mode"],
        "used_refs": [
            {
                "label": descriptor.label,
                "source": descriptor.source,
                "source_panel": descriptor.source_panel,
                "url": descriptor.url,
            }
            for descriptor in deduped
        ],
        "skipped_refs": skipped_refs,
    }


def compute_strong_ref_updates(
    panel_spec: dict[str, Any],
    panel_data: dict[str, Any],
    rendered_ref: RefDescriptor,
) -> list[dict[str, Any]]:
    policy = panel_data["reference_policy_defaults"]
    anchors = anchor_view_map(panel_data)
    updates: list[dict[str, Any]] = []
    shot_reset = is_reset_shot(panel_spec["shot"], policy)
    for character in panel_spec["visible_characters"]:
        face_visibility = panel_spec["face_visibility"][character]
        requested_views = panel_spec["reference_injection"]["use_character_sheet_views"].get(character, [])
        if not requested_views:
            updates.append(
                {
                    "character": character,
                    "accepted": False,
                    "reason": "no_requested_views",
                }
            )
            continue
        for view_id in requested_views:
            view_class = anchors[character][view_id]["view_class"]
            if shot_reset:
                updates.append(
                    {
                        "character": character,
                        "view_class": view_class,
                        "source_panel": panel_spec["panel_id"],
                        "accepted": False,
                        "reason": "continuity_reset_shot",
                    }
                )
                continue
            if not eligible_face_visibility(face_visibility, policy):
                updates.append(
                    {
                        "character": character,
                        "view_class": view_class,
                        "source_panel": panel_spec["panel_id"],
                        "accepted": False,
                        "reason": f"face_visibility:{face_visibility}",
                    }
                )
                continue
            suitable_visibility = set(anchors[character][view_id]["suitable_face_visibility"])
            if face_visibility not in suitable_visibility:
                updates.append(
                    {
                        "character": character,
                        "view_class": view_class,
                        "source_panel": panel_spec["panel_id"],
                        "accepted": False,
                        "reason": f"view_visibility_mismatch:{face_visibility}",
                    }
                )
                continue
            updates.append(
                {
                    "character": character,
                    "view_id": view_id,
                    "view_class": view_class,
                    "source_panel": panel_spec["panel_id"],
                    "accepted": True,
                    "face_visibility": face_visibility,
                    "ref": rendered_ref,
                }
            )
    return updates


def apply_strong_ref_updates(state: ContinuityState, updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_updates: list[dict[str, Any]] = []
    for update in updates:
        ref = update.pop("ref", None)
        manifest_update = dict(update)
        if update["accepted"] and ref is not None:
            character = update["character"]
            view_class = update["view_class"]
            state.last_strong_ref_by_character_and_view_class.setdefault(character, {})[view_class] = StrongRefState(
                ref=ref,
                panel_id=update["source_panel"],
                view_class=view_class,
                face_visibility=update["face_visibility"],
            )
        manifest_updates.append(manifest_update)
    return manifest_updates


def build_anchor_prompt(panel_data: dict[str, Any], character: str | None = None, view: dict[str, Any] | None = None) -> str:
    style = build_style_prompt(panel_data)
    if character is None:
        return join_parts([style, panel_data["location_anchor"]["description"], build_negative_lock_block({"visible_characters": []}, panel_data)])

    sheet = panel_data["character_sheets"][character]
    panel_like = {
        "visible_characters": [character],
        "must_keep": {
            character: sheet["visual_invariants"]["hair"][:1]
            + sheet["visual_invariants"]["face"][:1]
            + sheet["visual_invariants"]["wardrobe"][:1]
        },
    }
    identity_lock = build_character_lock_block(panel_like, panel_data["character_sheets"])
    framing = view["framing"] if view else ""
    return join_parts([style, framing, identity_lock, build_negative_lock_block(panel_like, panel_data)])


def generate_anchor_plan(panel_data: dict[str, Any]) -> list[dict[str, Any]]:
    plan = [
        {"anchor_id": "location_master", "kind": "location", "prompt": build_anchor_prompt(panel_data)},
    ]
    for character, sheet in panel_data["character_sheets"].items():
        for view in sheet["anchor_views"]:
            plan.append(
                {
                    "anchor_id": f"{character}.{view['view_id']}",
                    "kind": "character_view",
                    "character": character,
                    "view_id": view["view_id"],
                    "view_class": view["view_class"],
                    "prompt": build_anchor_prompt(panel_data, character, view),
                }
            )
    return plan


def dry_run_url(label: str) -> str:
    return f"dry-run://{label}"


def fal_generate(prompt: str) -> str:
    import fal_client

    return fal_client.subscribe(
        "fal-ai/flux-2-pro",
        arguments={
            "prompt": prompt,
            "image_size": {"width": PANEL_W, "height": PANEL_H},
            "num_images": 1,
            "output_format": "png",
        },
    )["images"][0]["url"]


def fal_edit(prompt: str, image_urls: list[str]) -> str:
    import fal_client

    return fal_client.subscribe(
        "fal-ai/flux-2-pro/edit",
        arguments={
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": {"width": PANEL_W, "height": PANEL_H},
            "num_images": 1,
            "output_format": "png",
        },
    )["images"][0]["url"]


def download(url: str, path: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)


def build_longscroll(rendered_paths: list[tuple[str, Path, str]], spacing_map: dict[str, str]) -> tuple[Path, list[int]]:
    spacing_px = {"tight": 30, "medium": 70, "tall_drop": 180, "end_cliff": 260}
    images = []
    total_h = 0
    for idx, (_panel_id, path, block_id) in enumerate(rendered_paths):
        image = Image.open(path).convert("RGB")
        gap = spacing_px.get(spacing_map.get(block_id, "medium"), 70)
        images.append((image, gap))
        total_h += image.height
        if idx < len(rendered_paths) - 1:
            total_h += gap

    canvas = Image.new("RGB", (PANEL_W, total_h), (244, 244, 244))
    y = 0
    for idx, (image, gap) in enumerate(images):
        canvas.paste(image, (0, y))
        y += image.height
        if idx < len(images) - 1:
            y += gap

    longscroll = OUT / "ep001_fal_live_longscroll.png"
    canvas.save(longscroll)
    return longscroll, list(canvas.size)


def initialize_state(panel_data: dict[str, Any]) -> ContinuityState:
    state = ContinuityState()
    state.character_anchor_urls = {character: {} for character in panel_data["character_sheets"]}
    state.last_strong_ref_by_character_and_view_class = {character: {} for character in panel_data["character_sheets"]}
    return state


def render_episode(dry_run: bool = False, manifest_only: bool = False) -> dict[str, Any]:
    panel_data, scroll = load_episode_data()
    panel_map = panel_map_from_data(panel_data)
    block_map = scroll_block_map(scroll)
    spacing_map = spacing_map_from_scroll(scroll)
    OUT.mkdir(parents=True, exist_ok=True)

    state = initialize_state(panel_data)
    anchor_plan = generate_anchor_plan(panel_data)
    manifest: dict[str, Any] = {
        "mode": "v3-character-consistency-dry-run" if dry_run else "v3-character-consistency",
        "dry_run": dry_run,
        "manifest_only": manifest_only,
        "anchors": {"location_master": None, "character_views": {}},
        "panels": [],
    }

    for anchor in anchor_plan:
        label = anchor["anchor_id"]
        url = dry_run_url(label) if dry_run else fal_generate(anchor["prompt"])
        if anchor["kind"] == "location":
            state.location_master = make_location_descriptor(url)
            manifest["anchors"]["location_master"] = {
                "url": url,
                "prompt": anchor["prompt"],
            }
        else:
            descriptor = make_anchor_descriptor(anchor["character"], anchor["view_id"], url)
            state.character_anchor_urls[anchor["character"]][anchor["view_id"]] = descriptor
            manifest["anchors"]["character_views"].setdefault(anchor["character"], {})[anchor["view_id"]] = {
                "url": url,
                "view_class": anchor["view_class"],
                "prompt": anchor["prompt"],
            }

    rendered_paths: list[tuple[str, Path, str]] = []
    for panel_id in sorted(panel_map):
        panel_spec = panel_map[panel_id]
        prompt_parts = build_prompt_parts(panel_data, panel_spec, block_map[panel_spec["block_id"]])
        final_prompt = flatten_prompt(prompt_parts)
        reference_strategy = select_reference_stack(panel_spec, panel_data, state)
        used_ref_urls = [ref["url"] for ref in reference_strategy["used_refs"]]

        output_url = dry_run_url(f"panel:{panel_id}") if dry_run else (
            fal_generate(final_prompt) if panel_id == "p01" else fal_edit(final_prompt, used_ref_urls)
        )
        output_path = OUT / f"{panel_id}.png"
        if not dry_run and not manifest_only:
            download(output_url, output_path)
            rendered_paths.append((panel_id, output_path, panel_spec["block_id"]))

        rendered_ref = RefDescriptor(
            url=output_url,
            label=f"panel:{panel_id}",
            source=f"panel:{panel_id}",
            source_panel=panel_id,
        )
        raw_updates = compute_strong_ref_updates(panel_spec, panel_data, rendered_ref)
        strong_ref_updates = apply_strong_ref_updates(state, raw_updates)

        visible_set = frozenset(panel_spec["visible_characters"])
        state.last_scene_ref_by_visible_set[visible_set] = rendered_ref
        state.panel_refs[panel_id] = rendered_ref
        state.panel_meta[panel_id] = {
            "visible_characters": list(panel_spec["visible_characters"]),
            "is_reset_shot": is_reset_shot(panel_spec["shot"], panel_data["reference_policy_defaults"]),
            "face_visibility": dict(panel_spec["face_visibility"]),
        }
        state.previous_panel_id = panel_id
        state.previous_panel_ref = rendered_ref

        manifest["panels"].append(
            {
                "panel_id": panel_id,
                "url": output_url,
                "path": str(output_path.resolve()),
                "block_id": panel_spec["block_id"],
                "shot": panel_spec["shot"],
                "visible_characters": list(panel_spec["visible_characters"]),
                "identity_focus": dict(panel_spec["identity_focus"]),
                "face_visibility": dict(panel_spec["face_visibility"]),
                "prompt_parts": prompt_parts,
                "prompt": final_prompt,
                "reference_strategy": {
                    "continuity_mode": reference_strategy["continuity_mode"],
                    "used_refs": reference_strategy["used_refs"],
                    "skipped_refs": reference_strategy["skipped_refs"],
                    "strong_ref_updates": strong_ref_updates,
                },
            }
        )

    manifest["dimensions"] = {"panel": [PANEL_W, PANEL_H]}
    if dry_run or manifest_only:
        manifest["longscroll"] = None
        manifest["dimensions"]["longscroll"] = None
    else:
        longscroll, size = build_longscroll(rendered_paths, spacing_map)
        manifest["longscroll"] = str(longscroll.resolve())
        manifest["dimensions"]["longscroll"] = size

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Build prompt/reference manifest without live FAL calls.")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write the manifest without downloading panel images or assembling longscroll output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = render_episode(dry_run=args.dry_run, manifest_only=args.manifest_only)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
