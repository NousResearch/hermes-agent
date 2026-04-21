from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import jsonschema
import yaml

ALLOWED_REFERENCE_PRIORITY = {
    "previous_panel",
    "scene_anchor",
    "character_anchor",
    "location_anchor",
}
BANNED_PROMPT_TERMS = (
    "storyboard",
    "shot list",
    "placeholder composition",
)
SAFE_POLICY_SUBSTITUTIONS = (
    "abstract admissions chart",
    "unreadable study-app ui",
    "department-building shapes",
    "scholarship-board blocks",
    "exam-prep props",
)
STYLE_TAIL = (
    "clean Korean romance webtoon illustration, polished digital manhwa style, "
    "2D cel shading, expressive acting, vertical mobile webtoon panel, no readable text, "
    "no speech bubbles, keep lettering safe zones clean"
)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(collect_strings(item))
        return strings
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(collect_strings(item))
        return strings
    return []


def format_schema_error(error: jsonschema.ValidationError) -> str:
    path = ".".join(str(part) for part in error.path)
    if not path:
        path = "<root>"
    return f"{path}: {error.message}"


def validate_schema(data: Any, schema: dict[str, Any], label: str, errors: list[str]) -> None:
    validator = jsonschema.Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(data), key=lambda item: list(item.path)):
        errors.append(f"{label}: {format_schema_error(error)}")


def ensure_no_banned_terms(strings: Iterable[str], label: str, errors: list[str]) -> None:
    for value in strings:
        normalized = normalize_text(value)
        for banned in BANNED_PROMPT_TERMS:
            if banned in normalized:
                errors.append(f"{label}: banned wording '{banned}' found in '{value}'")


def sanitize_prompt_for_policy(prompt: str) -> str:
    replacements = {
        "약대 입결표": "abstract admissions chart",
        "입결표": "abstract admissions chart",
        "계약학과": "department-building shapes",
        "장학 조건": "scholarship-board blocks",
        "장학": "scholarship-board blocks",
        "반수 루틴": "exam-prep props",
        "반수": "exam-prep props",
        "단톡방": "unreadable study-app UI",
        "메시지": "unreadable study-app UI",
        "음성메시지": "unreadable study-app UI",
        "노트북 메모": "exam-prep props",
        "강시윤": "college student",
        "윤서하": "college student",
        "한소라": "college student",
        "조민우": "college senior",
        "시윤의 어머니": "middle-aged Korean woman",
        "시윤": "college student",
        "서하": "college student",
        "소라": "college student",
        "민우": "college senior",
        "지방 국립대": "regional university",
        "대학": "campus",
        "admissions-track transfer-prep student": "college student under exam pressure",
        "contract-department scholarship student": "college student under scholarship pressure",
        "male senior student": "college senior",
        "male student": "college student",
        "female student": "college student",
        "grabbing his arm": "reaching toward his sleeve",
        "hand grips his arm": "hand catches his sleeve",
        "heroine grips protagonist arm": "one student reaches toward the other student's sleeve",
        "heroine claims him in public": "one student steps in beside the other in public",
        "physically claiming him in public": "initiating visible public proximity",
        "claims him in public": "steps beside him in public",
        "forces him to play along": "forces a public misunderstanding",
        "gets dragged into the public performance": "gets pulled into the public scene",
        "public performance": "public scene",
        "witness pressure": "crowd attention",
        "public witness": "crowd attention",
        "witnesses nearby": "students nearby",
        "arm-grab": "sleeve-contact",
        "pulling his arm into frame": "drawing him into frame",
        "caught under public attention": "caught in public attention",
        "frozen under witness pressure": "caught in crowd attention",
        "frozen under crowd attention": "caught in crowd attention",
        "physical closeness": "charged proximity",
        "body-memory fragment": "memory fragment",
        "waistline": "side silhouette",
        "waist contact": "side contact",
        "hand placement": "hand presence",
        "almost teasing": "disarming",
        "wounded disbelief": "hurt surprise",
        "vulnerable resolve": "steady resolve",
    }
    sanitized = prompt
    for original, replacement in replacements.items():
        sanitized = sanitized.replace(original, replacement)
    sanitized = re.sub(r"\b\d+세\b", "college-age", sanitized)
    sanitized = re.sub(r"public\s+campus\s+corridor", "campus corridor", sanitized)
    sanitized = re.sub(r"scene characters [^,|]+", "scene characters college student pair", sanitized)
    sanitized = re.sub(r"forbidden absences [^,|]+", "forbidden absences corridor depth and crowd cues", sanitized)
    sanitized = sanitized.replace("must show heroine physically claiming him in public", "must show one student initiating public contact")
    sanitized = sanitized.replace("protagonist frozen under witness pressure", "protagonist frozen under crowd attention")
    sanitized = re.sub(r"\s+", " ", sanitized).strip(" ,")
    return sanitized


def project_paths(project_root: Path, episode: str) -> dict[str, Path]:
    webtoon_root = project_root / "webtoon"
    episode_dir = webtoon_root / episode
    queue_path = episode_dir / "render_queue.yaml"
    queue = load_yaml(queue_path)
    output = queue.get("output", {})
    manifest_path = Path(output["live_manifest"]).resolve() if output.get("live_manifest") else episode_dir / f"generated_fal_live_manifest_{episode}.json"
    return {
        "project_root": project_root,
        "webtoon_root": webtoon_root,
        "episode_dir": episode_dir,
        "panel_prompts": episode_dir / "panel_prompts.yaml",
        "lettering_script": episode_dir / "lettering_script.yaml",
        "scroll_plan": episode_dir / "scroll_plan.yaml",
        "render_queue": queue_path,
        "continuity_bible": webtoon_root / "continuity_bible.yaml",
        "contracts_dir": webtoon_root / "contracts",
        "manifest": manifest_path,
    }


def load_episode_bundle(project_root: Path, episode: str) -> dict[str, Any]:
    paths = project_paths(project_root, episode)
    contracts_dir = paths["contracts_dir"]
    bundle = {
        "paths": paths,
        "continuity": load_yaml(paths["continuity_bible"]),
        "panel_data": load_yaml(paths["panel_prompts"]),
        "lettering": load_yaml(paths["lettering_script"]),
        "scroll_plan": load_yaml(paths["scroll_plan"]),
        "render_queue": load_yaml(paths["render_queue"]),
        "schemas": {
            "continuity": load_yaml(contracts_dir / "continuity_bible.schema.yaml"),
            "shot_spec": load_yaml(contracts_dir / "shot_spec_v2.schema.yaml"),
            "render_queue": load_yaml(contracts_dir / "render_queue_qc.schema.yaml"),
            "manifest": load_yaml(contracts_dir / "qc_manifest.schema.yaml"),
        },
    }
    manifest_path = paths["manifest"]
    if manifest_path.exists():
        bundle["manifest"] = load_json(manifest_path)
    else:
        bundle["manifest"] = None
    return bundle


def panel_map(panel_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {panel["panel_id"]: panel for panel in panel_data.get("panels", [])}


def scroll_block_map(scroll_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {block["block_id"]: block for block in scroll_plan.get("blocks", [])}


def lettering_panel_map(lettering: dict[str, Any]) -> dict[str, dict[str, bool]]:
    result: dict[str, dict[str, bool]] = {}
    for caption in lettering.get("captions", []):
        panel_id = caption["panel_id"]
        result.setdefault(panel_id, {"captions": False, "balloons": False})["captions"] = True
    for balloon in lettering.get("balloons", []):
        panel_id = balloon["panel_id"]
        result.setdefault(panel_id, {"captions": False, "balloons": False})["balloons"] = True
    return result


def zone_summary(zones: list[dict[str, Any]]) -> str:
    if not zones:
        return "none"
    summaries = []
    for zone in zones:
        summaries.append(
            f"{zone['zone_id']} x={zone['x']:.2f} y={zone['y']:.2f} w={zone['w']:.2f} h={zone['h']:.2f}"
        )
    return "; ".join(summaries)


def build_style_prompt(panel_data: dict[str, Any]) -> str:
    style_anchor = panel_data["style_anchor"]
    negative = ", ".join(style_anchor["negative"])
    return f"{style_anchor['positive']}, {STYLE_TAIL}, avoid {negative}"


def build_prompt_parts(
    panel_data: dict[str, Any],
    panel_spec: dict[str, Any],
    continuity: dict[str, Any],
) -> dict[str, str]:
    scene = continuity["scene_links"][panel_spec["scene_id"]]
    location = continuity["location_states"][panel_spec["background_continuity"]["location_state"]]

    shot_design = ", ".join(
        [
            f"beat {panel_spec['beat_purpose']}",
            f"camera {panel_spec['camera']['angle']}",
            f"camera height {panel_spec['camera']['height']}",
            f"lens feel {panel_spec['camera']['lens_feel']}",
            f"motion {panel_spec['camera']['motion']}",
            f"shot size {panel_spec['framing']['shot_size']}",
            f"subject priority {'; '.join(panel_spec['framing']['subject_priority'])}",
            f"depth layers {'; '.join(panel_spec['framing']['depth_layers'])}",
        ]
    )

    acting_bits: list[str] = []
    continuity_bits: list[str] = []
    for character_id in panel_spec["visible_characters"]:
        character = continuity["characters"][character_id]
        outfit_id = panel_spec["continuity_refs"]["outfit_states"][character_id]
        outfit = continuity["outfit_states"][outfit_id]
        acting_bits.append(
            ", ".join(
                [
                    f"{character['canonical_name']} as {character['role']}",
                    panel_spec["gesture"][character_id],
                    panel_spec["micro_expression"][character_id],
                    " / ".join(outfit["garments"]),
                ]
            )
        )
        continuity_bits.append(
            ", ".join(
                [
                    f"{character_id} anchor props {'; '.join(character['anchor_props'])}",
                    f"scene reference priority {'; '.join(panel_spec['continuity_refs']['reference_priority'])}",
                ]
            )
        )

    blocking = ", ".join(
        [
            "blocking",
            "; ".join(
                f"{character}: {position}"
                for character, position in panel_spec["blocking"]["character_positions"].items()
            ),
            "; ".join(
                f"{character}: {eyeline}"
                for character, eyeline in panel_spec["blocking"]["eyelines"].items()
            ),
            f"contact {panel_spec['blocking']['contact_state']}",
        ]
    )

    background = ", ".join(
        [
            f"setting {location['setting']}",
            f"persistent elements {'; '.join(panel_spec['background_continuity']['persistent_elements'])}",
            f"must show {'; '.join(panel_spec['must_show'])}",
            f"must not show {'; '.join(panel_spec['must_not_show'])}",
            f"scene characters {'; '.join(scene['characters'])}",
        ]
    )
    lettering = ", ".join(
        [
            "reserve negative space for later lettering",
            f"caption zones {zone_summary(panel_spec['negative_space_for_lettering']['caption_zones'])}",
            f"balloon zones {zone_summary(panel_spec['negative_space_for_lettering']['balloon_zones'])}",
            f"avoid regions {zone_summary(panel_spec['negative_space_for_lettering']['avoid_regions'])}",
        ]
    )
    negative = ", ".join(
        [
            "avoid " + "; ".join(panel_data["style_anchor"]["negative"]),
            "forbidden drift " + "; ".join(panel_spec["continuity_refs"]["forbidden_drift"]),
            "forbidden absences " + "; ".join(panel_spec["background_continuity"]["forbidden_absences"]),
        ]
    )
    return {
        "style": build_style_prompt(panel_data),
        "shot_design": shot_design,
        "acting": " | ".join(acting_bits),
        "blocking": blocking,
        "continuity": " | ".join(continuity_bits + [background]),
        "lettering": lettering,
        "negative": negative,
    }


def flatten_prompt_parts(prompt_parts: dict[str, str]) -> str:
    return ", ".join(part.strip() for part in prompt_parts.values() if part.strip())


def validate_zone(zone: dict[str, Any], path_label: str, errors: list[str]) -> None:
    for axis in ("x", "y", "w", "h"):
        value = zone[axis]
        if value < 0.0 or value > 1.0:
            errors.append(f"{path_label}: {axis} coordinate {value} outside [0.0, 1.0]")
    if zone["w"] <= 0.0 or zone["h"] <= 0.0:
        errors.append(f"{path_label}: width/height must be > 0")
    if zone["x"] + zone["w"] > 1.0:
        errors.append(f"{path_label}: x + w exceeds 1.0")
    if zone["y"] + zone["h"] > 1.0:
        errors.append(f"{path_label}: y + h exceeds 1.0")


def validate_continuity_bible(continuity: dict[str, Any], errors: list[str]) -> None:
    characters = continuity.get("characters", {})
    outfit_states = continuity.get("outfit_states", {})
    location_states = continuity.get("location_states", {})
    scene_links = continuity.get("scene_links", {})
    drift_policy = continuity.get("drift_policy", {})

    allowed_drift = set(drift_policy.get("allowed_drift", []))
    forbidden_drift = set(drift_policy.get("forbidden_drift", []))
    overlap = sorted(allowed_drift & forbidden_drift)
    if overlap:
        errors.append(f"continuity_bible: drift policy overlaps for {', '.join(overlap)}")

    for outfit_id, outfit in outfit_states.items():
        character_id = outfit["character_id"]
        if character_id not in characters:
            errors.append(f"continuity_bible: outfit state '{outfit_id}' references missing character '{character_id}'")

    incoming_links = {scene_id: 0 for scene_id in scene_links}
    outgoing_links = {scene_id: 0 for scene_id in scene_links}
    for scene_id, scene in scene_links.items():
        location_state = scene["location_state"]
        if location_state not in location_states:
            errors.append(f"continuity_bible: scene '{scene_id}' references missing location state '{location_state}'")
        for character_id in scene["characters"]:
            if character_id not in characters:
                errors.append(f"continuity_bible: scene '{scene_id}' references missing character '{character_id}'")
            outfit_state = scene["outfit_by_character"].get(character_id)
            if outfit_state is None:
                errors.append(f"continuity_bible: scene '{scene_id}' missing outfit for '{character_id}'")
                continue
            if outfit_state not in outfit_states:
                errors.append(f"continuity_bible: scene '{scene_id}' outfit '{outfit_state}' missing")
                continue
            if outfit_states[outfit_state]["character_id"] != character_id:
                errors.append(
                    f"continuity_bible: scene '{scene_id}' outfit '{outfit_state}' does not belong to '{character_id}'"
                )
        for priority in scene["default_reference_priority"]:
            if priority not in ALLOWED_REFERENCE_PRIORITY:
                errors.append(f"continuity_bible: scene '{scene_id}' has invalid reference priority '{priority}'")
        previous_scene_id = scene["previous_scene_id"]
        next_scene_id = scene["next_scene_id"]
        if previous_scene_id is not None:
            if previous_scene_id not in scene_links:
                errors.append(f"continuity_bible: scene '{scene_id}' previous scene '{previous_scene_id}' missing")
            else:
                incoming_links[scene_id] += 1
                outgoing_links[previous_scene_id] += 1
        if next_scene_id is not None:
            if next_scene_id not in scene_links:
                errors.append(f"continuity_bible: scene '{scene_id}' next scene '{next_scene_id}' missing")
            else:
                outgoing_links[scene_id] += 1
                incoming_links[next_scene_id] += 1
        if previous_scene_id is None and next_scene_id is None and len(scene_links) > 1:
            errors.append(f"continuity_bible: scene '{scene_id}' cannot omit both previous and next scene ids")

    for scene_id, scene in scene_links.items():
        previous_scene_id = scene["previous_scene_id"]
        next_scene_id = scene["next_scene_id"]
        if len(scene_links) > 1 and incoming_links[scene_id] > 0 and outgoing_links[scene_id] > 0:
            if previous_scene_id is None or next_scene_id is None:
                errors.append(f"continuity_bible: nonterminal scene '{scene_id}' must include previous and next scene ids")


def validate_shot_spec(
    panel_data: dict[str, Any],
    continuity: dict[str, Any],
    lettering: dict[str, Any],
    strict: bool,
    errors: list[str],
) -> None:
    version = panel_data.get("prompt_schema_version")
    if strict and version != 2:
        errors.append(f"shot_spec: strict mode requires prompt_schema_version 2, found {version}")
        return
    if version != 2:
        return

    characters = continuity["characters"]
    scene_links = continuity["scene_links"]
    location_states = continuity["location_states"]
    lettering_map = lettering_panel_map(lettering)
    panels = panel_data["panels"]

    for index, panel in enumerate(panels):
        panel_id = panel["panel_id"]
        label = f"shot_spec:{panel_id}"
        if "prompt" in panel:
            errors.append(f"{label}: v2 panels must not carry legacy 'prompt' field")
        if panel["scene_id"] not in scene_links:
            errors.append(f"{label}: scene_id '{panel['scene_id']}' missing from continuity bible")
            continue
        if panel["continuity_refs"]["scene_link"] != panel["scene_id"]:
            errors.append(f"{label}: continuity_refs.scene_link must equal scene_id")
        scene = scene_links[panel["scene_id"]]
        location_state = panel["background_continuity"]["location_state"]
        if location_state not in location_states:
            errors.append(f"{label}: background_continuity.location_state '{location_state}' missing")
        elif location_state != scene["location_state"]:
            errors.append(f"{label}: location_state '{location_state}' does not match scene link '{scene['location_state']}'")
        for character_id in panel["visible_characters"]:
            if character_id not in characters:
                errors.append(f"{label}: visible character '{character_id}' missing from continuity bible")
                continue
            if not panel["gesture"].get(character_id):
                errors.append(f"{label}: missing gesture for visible character '{character_id}'")
            if not panel["micro_expression"].get(character_id):
                errors.append(f"{label}: missing micro_expression for visible character '{character_id}'")
            outfit_state = panel["continuity_refs"]["outfit_states"].get(character_id)
            if not outfit_state:
                errors.append(f"{label}: missing outfit state for visible character '{character_id}'")
        overlap = sorted(set(panel["continuity_refs"]["allowed_drift"]) & set(panel["continuity_refs"]["forbidden_drift"]))
        if overlap:
            errors.append(f"{label}: allowed/forbidden drift overlaps for {', '.join(overlap)}")

        expected_previous = None if index == 0 else panels[index - 1]["panel_id"]
        expected_next = None if index == len(panels) - 1 else panels[index + 1]["panel_id"]
        if panel["continuity_refs"]["previous_panel"] != expected_previous:
            errors.append(f"{label}: previous_panel must be '{expected_previous}'")
        if panel["continuity_refs"]["next_panel"] != expected_next:
            errors.append(f"{label}: next_panel must be '{expected_next}'")

        lettering_flags = lettering_map.get(panel_id, {"captions": False, "balloons": False})
        caption_zones = panel["negative_space_for_lettering"]["caption_zones"]
        balloon_zones = panel["negative_space_for_lettering"]["balloon_zones"]
        if lettering_flags["captions"] and not caption_zones:
            errors.append(f"{label}: caption lettering exists but caption_zones are empty")
        if lettering_flags["balloons"] and not balloon_zones:
            errors.append(f"{label}: balloon lettering exists but balloon_zones are empty")
        for zone_group_name, zones in panel["negative_space_for_lettering"].items():
            for zone in zones:
                validate_zone(zone, f"{label}:{zone_group_name}:{zone['zone_id']}", errors)

        authored_strings = collect_strings(panel_data["style_anchor"]) + collect_strings(panel)
        ensure_no_banned_terms(authored_strings, label, errors)
        derived_prompt = flatten_prompt_parts(build_prompt_parts(panel_data, panel, continuity))
        ensure_no_banned_terms([derived_prompt], f"{label}:derived_prompt", errors)


def validate_render_queue(render_queue: dict[str, Any], expected_panel_ids: list[str], errors: list[str]) -> None:
    label = "render_queue"
    if "candidate_policy" not in render_queue:
        errors.append(f"{label}: missing candidate_policy")
        return
    review_required = render_queue["candidate_policy"]["review_required_panels"]
    job_panel_ids = [job["panel_id"] for job in render_queue["jobs"]]
    if sorted(job_panel_ids) != sorted(expected_panel_ids):
        errors.append(f"{label}: jobs do not match shot-spec panel ids")
    unknown_review_panels = sorted(set(review_required) - set(job_panel_ids))
    if unknown_review_panels:
        errors.append(f"{label}: review_required_panels contains unknown ids {', '.join(unknown_review_panels)}")


def _path_exists(path_value: str) -> bool:
    if path_value.startswith("dry-run://"):
        return True
    return Path(path_value).exists()


def validate_manifest(
    manifest: dict[str, Any] | None,
    render_queue: dict[str, Any],
    panel_data: dict[str, Any],
    continuity_path: Path,
    strict: bool,
    errors: list[str],
) -> None:
    if manifest is None:
        if strict:
            errors.append("manifest: strict mode requires a live manifest on disk")
        return

    label = "manifest"
    if "contract_versions" not in render_queue or "candidate_policy" not in render_queue:
        errors.append(f"{label}: render_queue is missing hardened contract metadata")
        return
    if "manifest_version" not in manifest:
        errors.append(f"{label}: missing manifest_version")
    if manifest["prompt_schema_version"] != panel_data["prompt_schema_version"]:
        errors.append(
            f"{label}: prompt_schema_version {manifest['prompt_schema_version']} does not match shot-spec {panel_data['prompt_schema_version']}"
        )
    expected_continuity_ref = str(continuity_path.resolve())
    if manifest["continuity_bible_ref"] != expected_continuity_ref:
        errors.append(f"{label}: continuity_bible_ref must equal {expected_continuity_ref}")
    if manifest["contract_versions"] != render_queue["contract_versions"]:
        errors.append(f"{label}: contract_versions must match render_queue contract_versions")

    review_required = set(render_queue["candidate_policy"]["review_required_panels"])
    panels_by_id = panel_map(panel_data)
    manifest_panel_ids = {panel["panel_id"] for panel in manifest["panels"]}
    if manifest_panel_ids != set(panels_by_id):
        errors.append(f"{label}: panel ids do not match shot spec")
    if manifest["longscroll"] is not None and not _path_exists(manifest["longscroll"]):
        errors.append(f"{label}: longscroll file missing on disk")
    if not _path_exists(manifest["generated_dir"]):
        errors.append(f"{label}: generated_dir missing on disk")
    if not manifest["dry_run"]:
        if str(manifest["generated_dir"]).startswith("dry-run://"):
            errors.append(f"{label}: live manifest cannot use dry-run generated_dir")
        if manifest["longscroll"] is not None and str(manifest["longscroll"]).startswith("dry-run://"):
            errors.append(f"{label}: live manifest cannot use dry-run longscroll path")

    for panel in manifest["panels"]:
        panel_id = panel["panel_id"]
        panel_label = f"{label}:{panel_id}"
        candidate_ids = [candidate["candidate_id"] for candidate in panel["candidates"]]
        if panel["candidate_count"] != len(panel["candidates"]):
            errors.append(f"{panel_label}: candidate_count must equal len(candidates)")
        if panel["selected_candidate"] not in candidate_ids:
            errors.append(f"{panel_label}: selected_candidate not found in candidates")
        missing_rejections = sorted(set(panel["rejected_candidates"]) - set(candidate_ids))
        if missing_rejections:
            errors.append(f"{panel_label}: rejected candidates missing from candidates: {', '.join(missing_rejections)}")
        if not panel["selection_reason"].strip():
            errors.append(f"{panel_label}: selection_reason must not be blank")
        if panel["candidate_count"] == 1 and not panel["rerender_reason"].strip():
            errors.append(f"{panel_label}: rerender_reason must explain single-candidate acceptance")
        for path_key in ("generated_panel_path", "final_panel_path"):
            if not _path_exists(panel[path_key]):
                errors.append(f"{panel_label}: {path_key} missing on disk")
            if not manifest["dry_run"] and str(panel[path_key]).startswith("dry-run://"):
                errors.append(f"{panel_label}: live manifest cannot use dry-run {path_key}")
        for candidate in panel["candidates"]:
            if not _path_exists(candidate["generated_panel_path"]):
                errors.append(f"{panel_label}: candidate path missing for {candidate['candidate_id']}")
            if not manifest["dry_run"]:
                if str(candidate["generated_panel_path"]).startswith("dry-run://"):
                    errors.append(f"{panel_label}: live manifest cannot use dry-run candidate path")
                if str(candidate.get("url", "")).startswith("reused://"):
                    errors.append(f"{panel_label}: live manifest cannot use reused:// candidate url")
                if str(candidate.get("image_url", "")).startswith("reused://"):
                    errors.append(f"{panel_label}: live manifest cannot use reused:// candidate image_url")

        if panel["review_status"] == "reviewed":
            if panel["candidate_count"] < 2:
                errors.append(f"{panel_label}: reviewed panels require at least 2 candidates")
            if panel["selected_scores"] is None:
                errors.append(f"{panel_label}: reviewed panels require selected_scores")
        if panel["review_status"] == "not_reviewed_pilot_exception":
            if panel_id in review_required:
                errors.append(f"{panel_label}: review-required panel cannot be marked not_reviewed_pilot_exception")
            if not panel["rerender_reason"].strip():
                errors.append(f"{panel_label}: pilot exception requires explicit rerender_reason")

        if panel["selected_scores"] is not None:
            for score_name, score_value in panel["selected_scores"].items():
                if score_value < 1.0 or score_value > 5.0:
                    errors.append(f"{panel_label}: {score_name} {score_value} outside 1.0..5.0")

        any_candidate_changed = any(
            candidate["policy_sanitized"] or candidate["final_prompt"] != candidate["prompt"]
            for candidate in panel["candidates"]
        )
        for candidate in panel["candidates"]:
            if candidate.get("final_prompt_changed") != (candidate["final_prompt"] != candidate["prompt"]):
                errors.append(f"{panel_label}: candidate {candidate['candidate_id']} final_prompt_changed mismatch")
        if panel["final_prompt_changed"] != (panel["final_prompt"] != panel["prompt"]):
            errors.append(f"{panel_label}: final_prompt_changed does not match prompt diff")
        if panel["policy_sanitized"] != any_candidate_changed:
            errors.append(f"{panel_label}: policy_sanitized does not match candidate prompt changes")
        if panel["final_prompt_changed"] != any_candidate_changed:
            errors.append(f"{panel_label}: final_prompt_changed must track candidate prompt changes")
        if panel["policy_sanitized"] or panel["final_prompt_changed"]:
            normalized_prompt = normalize_text(panel["final_prompt"])
            if not any(token in normalized_prompt for token in SAFE_POLICY_SUBSTITUTIONS):
                errors.append(f"{panel_label}: sanitized prompt must retain a safe visual substitute")


def validate_episode_contracts(project_root: Path, episode: str, strict: bool = False) -> list[str]:
    bundle = load_episode_bundle(project_root, episode)
    paths = bundle["paths"]
    errors: list[str] = []

    validate_schema(bundle["continuity"], bundle["schemas"]["continuity"], "continuity_bible.schema", errors)
    validate_continuity_bible(bundle["continuity"], errors)

    panel_data = bundle["panel_data"]
    if panel_data.get("prompt_schema_version") == 2:
        validate_schema(panel_data, bundle["schemas"]["shot_spec"], "shot_spec_v2.schema", errors)
    validate_shot_spec(panel_data, bundle["continuity"], bundle["lettering"], strict, errors)

    validate_schema(bundle["render_queue"], bundle["schemas"]["render_queue"], "render_queue_qc.schema", errors)
    validate_render_queue(bundle["render_queue"], [panel["panel_id"] for panel in panel_data.get("panels", [])], errors)

    manifest = bundle["manifest"]
    if manifest is not None:
        validate_schema(manifest, bundle["schemas"]["manifest"], "qc_manifest.schema", errors)
    validate_manifest(manifest, bundle["render_queue"], panel_data, paths["continuity_bible"], strict, errors)
    return errors
