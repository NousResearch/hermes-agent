#!/usr/bin/env python3
"""
Bootstrap a video production kanban from a structured plan JSON.

Reads a plan.json describing the brief + team, expands templates from
../assets/, and writes a setup.sh that creates Hermes profiles and fires the
initial kanban task.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def load_template(name: str) -> str:
    return (ASSETS_DIR / name).read_text(encoding="utf-8")


def require_list_of_strings(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        errors.append(f"{path} must be a list of strings")


def validate_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top = [
        "title", "slug", "tenant", "duration_s", "aspect", "resolution",
        "fps", "team", "scenes", "audio", "deliverables", "brief_extra", "taste"
    ]
    for k in required_top:
        if k not in plan:
            errors.append(f"missing required key: {k}")

    if "slug" in plan and (not isinstance(plan["slug"], str) or not SLUG_RE.match(plan["slug"])):
        errors.append("slug must be lowercase kebab-case matching [a-z0-9][a-z0-9-]{0,63}")

    if "tenant" in plan and (not isinstance(plan["tenant"], str) or not plan["tenant"].strip()):
        errors.append("tenant must be a non-empty string")

    if "duration_s" in plan and not isinstance(plan["duration_s"], int):
        errors.append("duration_s must be an integer")
    if "fps" in plan and not isinstance(plan["fps"], int):
        errors.append("fps must be an integer")

    extra = plan.get("brief_extra", {})
    if not isinstance(extra, dict):
        errors.append("brief_extra must be an object")
    else:
        for k in ["concept_one_liner", "emotional_north_star", "platforms", "tone", "aesthetic_rules"]:
            if not isinstance(extra.get(k), str) or not extra.get(k, "").strip():
                errors.append(f"brief_extra.{k} is required and must be a non-empty string")

    taste = plan.get("taste", {})
    if not isinstance(taste, dict):
        errors.append("taste must be an object")
    else:
        for k in ["brand_guide", "emotional_dna"]:
            if not isinstance(taste.get(k), str) or not taste.get(k, "").strip():
                errors.append(f"taste.{k} is required and must be a non-empty string")

    audio = plan.get("audio", {})
    if not isinstance(audio, dict):
        errors.append("audio must be an object")
    else:
        for k in ["approach", "vo", "music", "sfx"]:
            if not isinstance(audio.get(k), str):
                errors.append(f"audio.{k} must be a string")

    if "team" in plan:
        if not isinstance(plan["team"], list) or not plan["team"]:
            errors.append("team must be a non-empty list")
        else:
            seen_profiles: set[str] = set()
            roles = [t.get("role") for t in plan["team"] if isinstance(t, dict)]
            if "director" not in roles:
                errors.append("team must include a director role")
            for i, t in enumerate(plan["team"]):
                if not isinstance(t, dict):
                    errors.append(f"team[{i}] must be an object")
                    continue
                for k in ["profile", "role", "toolsets", "skills", "responsibilities", "inputs", "outputs"]:
                    if k not in t:
                        errors.append(f"team[{i}] missing {k}")
                profile = t.get("profile")
                if isinstance(profile, str):
                    if not PROFILE_NAME_RE.match(profile):
                        errors.append(f"team[{i}].profile {profile!r} must be lowercase kebab-case")
                    if profile in seen_profiles:
                        errors.append(f"team[{i}].profile {profile!r} is duplicated")
                    seen_profiles.add(profile)
                require_list_of_strings(t.get("toolsets"), f"team[{i}].toolsets", errors)
                require_list_of_strings(t.get("skills"), f"team[{i}].skills", errors)
                for k in ["role", "responsibilities", "inputs", "outputs"]:
                    if k in t and not isinstance(t[k], str):
                        errors.append(f"team[{i}].{k} must be a string")

    if "scenes" in plan:
        if not isinstance(plan["scenes"], list) or not plan["scenes"]:
            errors.append("scenes must be a non-empty list")
        else:
            for i, s in enumerate(plan["scenes"]):
                if not isinstance(s, dict):
                    errors.append(f"scenes[{i}] must be an object")
                    continue
                for k in ["n", "time", "content", "tool", "audio", "notes"]:
                    if k not in s:
                        errors.append(f"scenes[{i}] missing {k}")
                if "n" in s and not isinstance(s["n"], int):
                    errors.append(f"scenes[{i}].n must be an integer")
                for k in ["time", "content", "tool", "audio", "notes"]:
                    if k in s and not isinstance(s[k], str):
                        errors.append(f"scenes[{i}].{k} must be a string")

    if "deliverables" in plan:
        if not isinstance(plan["deliverables"], list) or not plan["deliverables"]:
            errors.append("deliverables must be a non-empty list")
        else:
            for i, d in enumerate(plan["deliverables"]):
                if not isinstance(d, dict):
                    errors.append(f"deliverables[{i}] must be an object")
                    continue
                for k in ["format", "resolution", "notes"]:
                    if not isinstance(d.get(k), str) or not d.get(k, "").strip():
                        errors.append(f"deliverables[{i}].{k} must be a non-empty string")

    if "api_keys_required" in plan:
        require_list_of_strings(plan["api_keys_required"], "api_keys_required", errors)

    assets = plan.get("assets", {})
    if assets and not isinstance(assets, dict):
        errors.append("assets must be an object if present")
    return errors


def shell_single_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def shell_double_quote_expand_vars(s: str) -> str:
    """Quote a trusted shell argument while allowing $VARS to expand.

    Use only for generator-controlled destination paths such as
    "$WORKSPACE/assets/...". User-provided paths must still use
    shell_single_quote().
    """
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`') + '"'


def replace_vars(tmpl: str, values: dict[str, str]) -> str:
    out = tmpl
    for k, v in values.items():
        out = out.replace("{{" + k + "}}", v)
    leftovers = re.findall(r"\{\{[A-Z0-9_]+\}\}", out)
    if leftovers:
        raise ValueError(f"unresolved template placeholders: {sorted(set(leftovers))}")
    return out


def render_brief(plan: dict[str, Any]) -> str:
    tmpl = load_template("brief.md.tmpl")
    extra = plan["brief_extra"]
    scene_rows = "\n".join(
        f"| {s['n']} | {s['time']} | {s['content']} | {s['tool']} | {s['audio']} | {s['notes']} |"
        for s in plan["scenes"]
    )
    deliverable_rows = "\n".join(
        f"| {d['format']} | {d['resolution']} | {d['notes']} |"
        for d in plan["deliverables"]
    )
    return replace_vars(tmpl, {
        "TITLE": plan["title"],
        "SLUG": plan["slug"],
        "TENANT": plan["tenant"],
        "WORKSPACE": f"~/projects/video-pipeline/{plan['slug']}",
        "ONE_LINE_PITCH": extra["concept_one_liner"],
        "EMOTIONAL_NORTH_STAR": extra["emotional_north_star"],
        "DURATION_S": str(plan["duration_s"]),
        "ASPECT": plan["aspect"],
        "RESOLUTION": plan["resolution"],
        "FPS": str(plan["fps"]),
        "PLATFORMS": extra["platforms"],
        "DEADLINE": extra.get("deadline", "_(none)_"),
        "QUALITY_BAR": extra.get("quality_bar", "polished"),
        "VISUAL_REFS": extra.get("visual_refs", "_(none)_"),
        "TONE": extra["tone"],
        "BRAND_CONSTRAINTS": extra.get("brand_constraints", "_(none)_"),
        "AESTHETIC_RULES": extra["aesthetic_rules"],
        "AUDIO_APPROACH": plan["audio"]["approach"],
        "VO_DETAILS": plan["audio"]["vo"],
        "MUSIC_DETAILS": plan["audio"]["music"],
        "SFX_DETAILS": plan["audio"]["sfx"],
        "DELIVERABLE_ROWS": deliverable_rows,
        "SCENE_ROWS": scene_rows,
        "API_KEYS_REQUIRED": ", ".join(plan.get("api_keys_required", [])) or "none",
        "EXT_DEPS": extra.get("ext_deps", "ffmpeg, Python 3.11+"),
        "SOURCE_ASSETS": extra.get("source_assets", "_(none)_"),
    })


def group_team_by_role(team: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for member in team:
        out.setdefault(member["role"], []).append(member)
    return out


def first_profile(team_by_role: dict[str, list[dict[str, Any]]], role: str, default: str | None = None) -> str | None:
    members = team_by_role.get(role, [])
    return members[0]["profile"] if members else default


def render_team_md(plan: dict[str, Any]) -> str:
    team_by_role = group_team_by_role(plan["team"])
    lines = [f"# Team & Task Graph — {plan['title']}", "", "## Team", ""]
    for member in plan["team"]:
        skills = f"loads `{', '.join(member['skills'])}`" if member["skills"] else "no extra skill required"
        lines.append(f"- `{member['profile']}` ({member['role']}) — {member['responsibilities']} ({skills})")

    lines.extend(["", "## Task Graph", "", "```"])
    director = first_profile(team_by_role, "director", "director")
    lines.append(f"T0    {director} — decompose")
    next_id = 1

    def tid() -> str:
        nonlocal next_id
        v = f"T{next_id}"
        next_id += 1
        return v

    writer_id = None
    if first_profile(team_by_role, "writer"):
        writer_id = tid()
        lines.append(f"{writer_id:5} {first_profile(team_by_role, 'writer')} — script / narration draft (parent: T0)")
    elif first_profile(team_by_role, "copywriter"):
        writer_id = tid()
        lines.append(f"{writer_id:5} {first_profile(team_by_role, 'copywriter')} — copy / CTA / VO draft (parent: T0)")

    concept_id = None
    if first_profile(team_by_role, "concept-artist"):
        concept_id = tid()
        parent = writer_id or "T0"
        lines.append(f"{concept_id:5} {first_profile(team_by_role, 'concept-artist')} — style frames / visual direction (parent: {parent})")

    storyboard_id = None
    if first_profile(team_by_role, "storyboarder"):
        storyboard_id = tid()
        parent = concept_id or writer_id or "T0"
        lines.append(f"{storyboard_id:5} {first_profile(team_by_role, 'storyboarder')} — beat-by-beat shot list (parent: {parent})")

    cinematographer_id = None
    if first_profile(team_by_role, "cinematographer"):
        cinematographer_id = tid()
        parent = storyboard_id or concept_id or writer_id or "T0"
        lines.append(f"{cinematographer_id:5} {first_profile(team_by_role, 'cinematographer')} — visual spec for all scenes (parent: {parent})")

    music_id = None
    if first_profile(team_by_role, "music-supervisor"):
        music_id = tid()
        lines.append(f"{music_id:5} {first_profile(team_by_role, 'music-supervisor')} — track analysis + beats.json (parent: T0)")

    image_gen_id = None
    if first_profile(team_by_role, "image-generator"):
        image_gen_id = tid()
        parent = storyboard_id or concept_id or writer_id or "T0"
        lines.append(f"{image_gen_id:5} {first_profile(team_by_role, 'image-generator')} — still generation (parent: {parent})")

    image_to_video_id = None
    if first_profile(team_by_role, "image-to-video-generator"):
        image_to_video_id = tid()
        parent = image_gen_id or storyboard_id or concept_id or "T0"
        lines.append(f"{image_to_video_id:5} {first_profile(team_by_role, 'image-to-video-generator')} — animate stills / image-to-video (parent: {parent})")

    render_scene_ids: list[str] = []
    render_parents = [p for p in [cinematographer_id, storyboard_id, concept_id] if p]
    if not render_parents:
        render_parents = ["T0"]
    if music_id:
        render_parents.append(music_id)
    if image_to_video_id:
        render_parents = [image_to_video_id]

    known_profiles = {m["profile"] for m in plan["team"]}
    for s in plan["scenes"]:
        scene_task = tid()
        assignee = s["tool"] if s["tool"] in known_profiles else s["tool"]
        parent_str = ", ".join(render_parents)
        lines.append(f"{scene_task:5} {assignee} — scene {s['n']}: {s['content'][:60]} (parents: {parent_str})")
        render_scene_ids.append(scene_task)

    voice_id = None
    if first_profile(team_by_role, "voice-talent"):
        voice_id = tid()
        parent = writer_id or "T0"
        lines.append(f"{voice_id:5} {first_profile(team_by_role, 'voice-talent')} — narration / VO generation (parent: {parent})")

    audio_id = None
    if first_profile(team_by_role, "audio-mixer"):
        audio_id = tid()
        parents = [p for p in [music_id, voice_id] if p]
        parent_str = ", ".join(parents) if parents else "T0"
        lines.append(f"{audio_id:5} {first_profile(team_by_role, 'audio-mixer')} — mix audio (parents: {parent_str})")

    editor_id = None
    if first_profile(team_by_role, "editor"):
        editor_id = tid()
        parents = render_scene_ids + [p for p in [audio_id, voice_id, music_id, image_to_video_id] if p and p not in render_scene_ids]
        parent_str = ", ".join(parents) if parents else "T0"
        lines.append(f"{editor_id:5} {first_profile(team_by_role, 'editor')} — assemble + mux (parents: {parent_str})")

    captioner_id = None
    if first_profile(team_by_role, "captioner") and editor_id:
        captioner_id = tid()
        lines.append(f"{captioner_id:5} {first_profile(team_by_role, 'captioner')} — captions / burn-in (parent: {editor_id})")

    masterer_id = None
    if first_profile(team_by_role, "masterer") and (captioner_id or editor_id):
        masterer_id = tid()
        parent = captioner_id or editor_id
        lines.append(f"{masterer_id:5} {first_profile(team_by_role, 'masterer')} — platform variants / mastering (parent: {parent})")

    brand_cop_id = None
    if first_profile(team_by_role, "brand-cop") and (masterer_id or captioner_id or editor_id):
        brand_cop_id = tid()
        parent = masterer_id or captioner_id or editor_id
        lines.append(f"{brand_cop_id:5} {first_profile(team_by_role, 'brand-cop')} — brand compliance review (parent: {parent})")

    reviewer_id = None
    if first_profile(team_by_role, "reviewer") and (brand_cop_id or masterer_id or captioner_id or editor_id):
        reviewer_id = tid()
        parent = brand_cop_id or masterer_id or captioner_id or editor_id
        lines.append(f"{reviewer_id:5} {first_profile(team_by_role, 'reviewer')} — final QA (parent: {parent})")

    lines.extend([
        "```",
        "",
        "## Per-task workspace requirement",
        "",
        "All `kanban_create` calls MUST pass:",
        "```",
        'workspace_kind="dir"',
        f'workspace_path="$HOME/projects/video-pipeline/{plan["slug"]}"',
        f'tenant="{plan["tenant"]}"',
        "```",
    ])
    return "\n".join(lines)


def render_asset_copies(plan: dict[str, Any]) -> str:
    assets = plan.get("assets", {}) or {}
    lines = [
        'copy_one() {',
        '    local src="$1"',
        '    local dst="$2"',
        '    if [ ! -f "$src" ]; then',
        '        echo "✗ Missing asset: $src" >&2',
        '        exit 1',
        '    fi',
        '    cp "$src" "$dst"',
        '    echo " ✓ copied $(basename "$src") -> $dst"',
        '}',
        '',
    ]
    mapping = {
        "audio_track": "$WORKSPACE/audio/track.mp3",
    }
    for key, dst in mapping.items():
        if key in assets and isinstance(assets[key], str) and assets[key].strip():
            lines.append(f'copy_one {shell_single_quote(assets[key])} {shell_double_quote_expand_vars(dst)}')
    multi_map = {
        "logos": "$WORKSPACE/assets/logos/",
        "fonts": "$WORKSPACE/assets/fonts/",
        "existing_footage": "$WORKSPACE/assets/existing-footage/",
        "style_frames": "$WORKSPACE/taste/style-frames/",
        "voiceover_files": "$WORKSPACE/audio/voiceover/",
        "sfx_files": "$WORKSPACE/audio/sfx/",
    }
    for key, dst_dir in multi_map.items():
        values = assets.get(key, [])
        if isinstance(values, list):
            for src in values:
                if isinstance(src, str) and src.strip():
                    lines.append(f'copy_one {shell_single_quote(src)} {shell_double_quote_expand_vars(dst_dir)}')
    if len(lines) == 11:
        return 'echo "  (no assets to copy)"'
    return "\n".join(lines)


def role_rules(member: dict[str, Any], plan: dict[str, Any]) -> str:
    role = member["role"]
    workspace = f"$HOME/projects/video-pipeline/{plan['slug']}"
    if role == "director":
        return "\n".join([
            "- Do not execute the work yourself.",
            "- For every concrete task, create a kanban task and assign it.",
            "- Read `brief.md`, `TEAM.md`, and `taste/` before decomposing.",
            "- Follow the task graph in `TEAM.md`; do not invent extra roles unless truly required.",
            f"- On every `kanban_create`, pass `workspace_kind=\"dir\"`, `workspace_path=\"{workspace}\"`, and `tenant=\"{plan['tenant']}\"`.",
            "- Carry runtime defaults forward: renderer 1800s, editor 600s, voice-talent 300s, image-to-video 900s.",
        ])
    if role == "cinematographer":
        return "\n".join([
            "- Produce coherent visual language and per-scene specs.",
            "- Review renderer outputs for consistency and quality before approval.",
            "- Use `video` / `vision` tools when enabled for review, not for production.",
        ])
    if role.startswith("renderer") or role in {"image-generator", "image-to-video-generator"}:
        return "\n".join([
            "- Read the assigned scene spec before rendering.",
            "- Write predictable outputs under `scenes/scene-NN/`.",
            "- Emit heartbeats for jobs expected to run longer than 5 minutes.",
            "- Save preview frames into `checkpoints/` before final completion when practical.",
        ])
    if role == "editor":
        return "\n".join([
            "- Assemble only from completed scene outputs and approved audio inputs.",
            "- Produce `output/final.mp4` and any required alternates.",
            "- Emit assembly-progress heartbeats on longer jobs.",
        ])
    if role == "audio-mixer":
        return "\n".join([
            "- Produce deterministic filenames under `audio/`.",
            "- Normalize levels and preserve intelligibility of voiceover.",
        ])
    if role in {"reviewer", "brand-cop"}:
        return "\n".join([
            "- Do not create final assets unless explicitly asked; your default output is review feedback.",
            "- Be specific about sync, brand, pacing, and technical defects.",
        ])
    return member.get("role_rules", "- Follow `brief.md`, `TEAM.md`, and the workspace conventions.")


def profile_description(profile: str, plan: dict[str, Any]) -> str:
    for member in plan["team"]:
        if member["profile"] == profile:
            return (
                f"Video pipeline {member['role']} for tenant {plan['tenant']}: "
                f"{member['responsibilities']} Outputs: {member['outputs']}"
            )[:500]
    return f"Video pipeline worker for tenant {plan['tenant']}"


def render_soul_md(member: dict[str, Any], plan: dict[str, Any]) -> str:
    tmpl = load_template("soul.md.tmpl")
    workspace = f"$HOME/projects/video-pipeline/{plan['slug']}"
    child_task_command = "\n".join([
        f"    --workspace dir:\"{workspace}\" \\",
        f"    --tenant {shell_single_quote(plan['tenant'])} \\",
        "    --max-runtime <duration> \\",
        "    --body <clear acceptance criteria>",
    ])
    common_rules = "\n".join([
        "- Read `brief.md`, `TEAM.md`, and the relevant workspace subdirectories before acting.",
        f"- This project uses a shared workspace at `{workspace}`.",
        f"- Use tenant `{plan['tenant']}` on every kanban call.",
        "- Keep filenames predictable so other profiles can consume your outputs.",
        "- Emit heartbeats during long-running work with `hermes kanban heartbeat <task-id> --note \"<progress>\"`.",
        "- When creating child tasks, use this Hermes CLI shape:",
        "  ```bash",
        "  hermes kanban create \"<title>\" \\",
        f"{child_task_command}",
        "  ```",
    ])
    common_commands = "\n".join([
        "```bash",
        "# Inspect a media file",
        "ffprobe -v quiet -show_entries format=duration -show_entries stream=codec_name,width,height,r_frame_rate input.mp4",
        "",
        "# Extract a representative frame",
        "ffmpeg -y -i input.mp4 -vf \"select='eq(n,30)'\" -vsync vfr out.png",
        "```",
    ])
    return replace_vars(tmpl, {
        "ROLE_NAME": member["role"],
        "WORKSPACE_PATH": workspace,
        "ROLE_RESPONSIBILITIES": member["responsibilities"],
        "INPUTS_READ": member.get("inputs", "brief.md, TEAM.md, taste/"),
        "OUTPUTS_PRODUCED": member.get("outputs", "See TEAM.md"),
        "TOOLSETS": ", ".join(member["toolsets"]),
        "SKILLS": ", ".join(member["skills"]) if member["skills"] else "(none)",
        "EXTERNAL_TOOLS": member.get("external_tools", "ffmpeg, ffprobe, provider CLIs/APIs as applicable"),
        "ROLE_RULES": role_rules(member, plan),
        "COMMON_RULES": common_rules,
        "COMMON_COMMANDS": common_commands,
    })


def render_setup_sh(plan: dict[str, Any], brief_md: str, team_md: str) -> str:
    tmpl = load_template("setup.sh.tmpl")
    director_profile = next((m["profile"] for m in plan["team"] if m["role"] == "director"), "director")
    key_checks = "\n".join(
        f'check_key {key} hermes {key} || exit 1' for key in plan.get("api_keys_required", [])
    ) or 'echo "  (no API keys required)"'
    scene_dirs = "\n".join(
        f'mkdir -p "$WORKSPACE/scenes/scene-{s["n"]:02d}/checkpoints"' for s in plan["scenes"]
    )
    def _profile_create_cmd(profile: str) -> str:
        return (
            f'create_profile {shell_single_quote(profile)} '
            f'{shell_single_quote(profile_description(profile, plan))}'
        )

    profile_create_commands = "\n".join(
        _profile_create_cmd(m["profile"]) for m in plan["team"]
    )
    profile_config_commands = "\n".join(
        f"configure_profile {shell_single_quote(m['profile'])} {shell_single_quote(json.dumps(m['toolsets']))} {shell_single_quote(json.dumps(m['skills']))}"
        for m in plan["team"]
    )
    soul_writes = "\n\n".join(
        f'cat > "$HOME/.hermes/profiles/{m["profile"]}/SOUL.md" <<\'SOUL_EOF\'\n{render_soul_md(m, plan)}\nSOUL_EOF\necho " ✓ SOUL.md for {m["profile"]}"'
        for m in plan["team"]
    )
    taste_writes = "\n".join([
        'cat > "$WORKSPACE/taste/brand-guide.md" <<\'BRAND_EOF\'',
        plan["taste"]["brand_guide"],
        'BRAND_EOF',
        '',
        'cat > "$WORKSPACE/taste/emotional-dna.md" <<\'DNA_EOF\'',
        plan["taste"]["emotional_dna"],
        'DNA_EOF',
    ])
    return replace_vars(tmpl, {
        "TITLE": plan["title"],
        "SLUG": plan["slug"],
        "TENANT": plan["tenant"],
        "KEY_CHECKS": key_checks,
        "SCENE_DIRS": scene_dirs,
        "PROFILE_CREATE_COMMANDS": profile_create_commands,
        "PROFILE_CONFIG_COMMANDS": profile_config_commands,
        "SOUL_WRITES": soul_writes,
        "BRIEF_CONTENTS": brief_md,
        "TEAM_CONTENTS": team_md,
        "TASTE_WRITES": taste_writes,
        "ASSET_COPIES": render_asset_copies(plan),
        "DIRECTOR_PROFILE": director_profile,
        "PROFILE_NAMES": " ".join(m["profile"] for m in plan["team"]),
        "GENERATED_AT": datetime.now(timezone.utc).isoformat(),
    })




def build_plan_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.local/kanban-video-orchestrator/plan.schema.json",
        "title": "Kanban Video Orchestrator Plan",
        "type": "object",
        "required": [
            "title", "slug", "tenant", "duration_s", "aspect", "resolution",
            "fps", "team", "scenes", "audio", "deliverables", "brief_extra", "taste"
        ],
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "slug": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]{0,63}$"},
            "tenant": {"type": "string", "minLength": 1},
            "duration_s": {"type": "integer", "minimum": 1},
            "aspect": {"type": "string", "minLength": 1},
            "resolution": {"type": "string", "minLength": 1},
            "fps": {"type": "integer", "minimum": 1},
            "api_keys_required": {
                "type": "array",
                "items": {"type": "string"},
                "default": []
            },
            "team": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["profile", "role", "toolsets", "skills", "responsibilities", "inputs", "outputs"],
                    "properties": {
                        "profile": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]{0,63}$"},
                        "role": {"type": "string", "minLength": 1},
                        "toolsets": {"type": "array", "items": {"type": "string"}},
                        "skills": {"type": "array", "items": {"type": "string"}},
                        "responsibilities": {"type": "string", "minLength": 1},
                        "inputs": {"type": "string", "minLength": 1},
                        "outputs": {"type": "string", "minLength": 1},
                        "role_rules": {"type": "string"},
                        "external_tools": {"type": "string"}
                    },
                    "additionalProperties": True
                }
            },
            "scenes": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["n", "time", "content", "tool", "audio", "notes"],
                    "properties": {
                        "n": {"type": "integer", "minimum": 1},
                        "time": {"type": "string", "minLength": 1},
                        "content": {"type": "string", "minLength": 1},
                        "tool": {"type": "string", "minLength": 1},
                        "audio": {"type": "string", "minLength": 1},
                        "notes": {"type": "string", "minLength": 1}
                    },
                    "additionalProperties": True
                }
            },
            "audio": {
                "type": "object",
                "required": ["approach", "vo", "music", "sfx"],
                "properties": {
                    "approach": {"type": "string"},
                    "vo": {"type": "string"},
                    "music": {"type": "string"},
                    "sfx": {"type": "string"}
                },
                "additionalProperties": True
            },
            "deliverables": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["format", "resolution", "notes"],
                    "properties": {
                        "format": {"type": "string", "minLength": 1},
                        "resolution": {"type": "string", "minLength": 1},
                        "notes": {"type": "string", "minLength": 1}
                    },
                    "additionalProperties": True
                }
            },
            "brief_extra": {
                "type": "object",
                "required": ["concept_one_liner", "emotional_north_star", "platforms", "tone", "aesthetic_rules"],
                "properties": {
                    "concept_one_liner": {"type": "string", "minLength": 1},
                    "emotional_north_star": {"type": "string", "minLength": 1},
                    "platforms": {"type": "string", "minLength": 1},
                    "deadline": {"type": "string"},
                    "quality_bar": {"type": "string"},
                    "visual_refs": {"type": "string"},
                    "tone": {"type": "string", "minLength": 1},
                    "brand_constraints": {"type": "string"},
                    "aesthetic_rules": {"type": "string", "minLength": 1},
                    "ext_deps": {"type": "string"},
                    "source_assets": {"type": "string"}
                },
                "additionalProperties": True
            },
            "taste": {
                "type": "object",
                "required": ["brand_guide", "emotional_dna"],
                "properties": {
                    "brand_guide": {"type": "string", "minLength": 1},
                    "emotional_dna": {"type": "string", "minLength": 1}
                },
                "additionalProperties": True
            },
            "assets": {
                "type": "object",
                "properties": {
                    "audio_track": {"type": "string"},
                    "logos": {"type": "array", "items": {"type": "string"}},
                    "fonts": {"type": "array", "items": {"type": "string"}},
                    "existing_footage": {"type": "array", "items": {"type": "string"}},
                    "style_frames": {"type": "array", "items": {"type": "string"}},
                    "voiceover_files": {"type": "array", "items": {"type": "string"}},
                    "sfx_files": {"type": "array", "items": {"type": "string"}}
                },
                "additionalProperties": True
            }
        },
        "additionalProperties": True
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan_json", nargs="?", help="Path to plan.json")
    ap.add_argument("--out", default="setup.sh", help="Output path for setup.sh")
    ap.add_argument("--brief-out", default=None, help="Optional output path for brief.md")
    ap.add_argument("--team-out", default=None, help="Optional output path for TEAM.md")
    ap.add_argument("--validate-only", action="store_true", help="Validate the plan and exit without rendering files")
    ap.add_argument("--schema-out", default=None, help="Optional path to write plan.schema.json")
    args = ap.parse_args()

    if args.schema_out:
        Path(args.schema_out).write_text(json.dumps(build_plan_schema(), indent=2), encoding="utf-8")
        print(f"Wrote {args.schema_out}")

    if not args.plan_json:
        if args.schema_out:
            return
        ap.error("plan_json is required unless you only use --schema-out")

    plan = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))
    errors = validate_plan(plan)
    if errors:
        print("Plan validation failed:", file=sys.stderr)
        for e in errors:
            print(f" - {e}", file=sys.stderr)
        sys.exit(2)

    if args.validate_only:
        print(f"Validated {args.plan_json}")
        return

    brief = render_brief(plan)
    team = render_team_md(plan)
    setup = render_setup_sh(plan, brief, team)

    Path(args.out).write_text(setup, encoding="utf-8")
    os.chmod(args.out, 0o755)
    print(f"Wrote {args.out}")
    if args.brief_out:
        Path(args.brief_out).write_text(brief, encoding="utf-8")
        print(f"Wrote {args.brief_out}")
    if args.team_out:
        Path(args.team_out).write_text(team, encoding="utf-8")
        print(f"Wrote {args.team_out}")


if __name__ == "__main__":
    main()
