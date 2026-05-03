#!/usr/bin/env python3
"""
Bootstrap a video production kanban from a structured plan JSON.

Reads a plan.json describing the team + brief, expands templates from
../assets/, and writes a setup.sh that creates Hermes profiles and fires the
initial kanban task.

Profile-config patching, SOUL.md-per-profile, TEAM.md task-graph convention,
and the `hermes kanban create --workspace dir:` initial-task pattern are
adapted from alt-glitch's NousResearch/kanban-video-pipeline.

Usage:
    bootstrap_pipeline.py plan.json [--out setup.sh]

The plan.json schema is documented inline below — see the `validate_plan`
function. A minimal example:

    {
      "title": "Q3 Product Teaser",
      "slug": "q3-product-teaser",
      "tenant": "q3-product-teaser",
      "duration_s": 30,
      "aspect": "1:1",
      "resolution": "1080x1080",
      "fps": 30,
      "team": [
        {
          "profile": "director",
          "role": "director",
          "toolsets": ["kanban", "terminal", "file"],
          "skills": [],
          "responsibilities": "...",
          "inputs": "brief.md, TEAM.md, taste/",
          "outputs": "kanban tasks for the team"
        },
        ...
      ],
      "scenes": [
        {"n": 1, "time": "0:00-0:08", "content": "...", "tool": "renderer-ascii"},
        ...
      ],
      "audio": {"approach": "voiceover + music bed", "vo": "ElevenLabs Lily",
                "music": "license-free", "sfx": "n/a"},
      "deliverables": [
        {"format": "mp4", "resolution": "1080x1080", "notes": "primary"}
      ],
      "api_keys_required": ["ELEVENLABS_API_KEY", "OPENROUTER_API_KEY"],
      "brief_extra": {
        "concept_one_liner": "...",
        "emotional_north_star": "...",
        "visual_refs": "...",
        "tone": "...",
        "brand_constraints": "..."
      }
    }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def load_template(name: str) -> str:
    return (ASSETS_DIR / name).read_text()


PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]+$")


def validate_plan(plan: dict) -> list[str]:
    """Return a list of validation error strings; empty list = valid."""
    errors = []
    required_top = ["title", "slug", "tenant", "duration_s", "aspect",
                    "resolution", "fps", "team", "scenes", "audio",
                    "deliverables"]
    for k in required_top:
        if k not in plan:
            errors.append(f"missing required key: {k}")

    if "team" in plan:
        if not isinstance(plan["team"], list) or not plan["team"]:
            errors.append("team must be a non-empty list")
        else:
            roles = [t.get("role") for t in plan["team"]]
            if "director" not in roles:
                errors.append("team must include a director role")
            seen_profiles = set()
            for i, t in enumerate(plan["team"]):
                for k in ["profile", "role", "toolsets", "skills",
                          "responsibilities"]:
                    if k not in t:
                        errors.append(f"team[{i}] missing {k}")
                # Profile name must match Hermes's regex (lowercase
                # alphanumeric + hyphens + underscores, up to 64 chars).
                if "profile" in t:
                    if not PROFILE_NAME_RE.match(t["profile"]):
                        errors.append(
                            f"team[{i}].profile {t['profile']!r} must match "
                            f"[a-z0-9][a-z0-9_-]{{0,63}} per Hermes profile rules"
                        )
                    if t["profile"] in seen_profiles:
                        errors.append(
                            f"team[{i}].profile {t['profile']!r} is duplicated"
                        )
                    seen_profiles.add(t["profile"])
                # Toolsets / skills must be lists, not strings.
                if "toolsets" in t and not isinstance(t["toolsets"], list):
                    errors.append(
                        f"team[{i}].toolsets must be a list of strings"
                    )
                if "skills" in t and not isinstance(t["skills"], list):
                    errors.append(
                        f"team[{i}].skills must be a list of strings"
                    )

    if "slug" in plan:
        if not SLUG_RE.match(plan["slug"]):
            errors.append("slug must be lowercase, hyphenated, "
                          "starting with [a-z0-9]")

    return errors


def render_brief(plan: dict) -> str:
    """Render brief.md from the plan."""
    tmpl = load_template("brief.md.tmpl")
    extra = plan.get("brief_extra", {})

    # Scene table rows
    scene_rows = []
    for s in plan["scenes"]:
        scene_rows.append(
            f"| {s.get('n', '?')} | {s.get('time', '?')} | "
            f"{s.get('content', '')} | {s.get('tool', '')} | "
            f"{s.get('audio', '')} | {s.get('notes', '')} |"
        )
    scene_table = "\n".join(scene_rows) if scene_rows else "_(none yet)_"

    # Deliverable rows
    deliv_rows = []
    for d in plan["deliverables"]:
        deliv_rows.append(
            f"| {d.get('format', '?')} | {d.get('resolution', '?')} | "
            f"{d.get('notes', '')} |"
        )
    deliv_table = "\n".join(deliv_rows) if deliv_rows else "_(none)_"

    # Replacements (single-pass)
    replacements = {
        "TITLE": plan["title"],
        "SLUG": plan["slug"],
        "TENANT": plan["tenant"],
        "WORKSPACE": f"~/projects/video-pipeline/{plan['slug']}",
        "ONE_LINE_PITCH": extra.get("concept_one_liner", "_(TBD)_"),
        "EMOTIONAL_NORTH_STAR": extra.get("emotional_north_star", "_(TBD)_"),
        "DURATION_S": str(plan["duration_s"]),
        "ASPECT": plan["aspect"],
        "RESOLUTION": plan["resolution"],
        "FPS": str(plan["fps"]),
        "PLATFORMS": extra.get("platforms", "_(TBD)_"),
        "DEADLINE": extra.get("deadline", "_(none)_"),
        "QUALITY_BAR": extra.get("quality_bar", "polished"),
        "VISUAL_REFS": extra.get("visual_refs", "_(none)_"),
        "TONE": extra.get("tone", "_(TBD)_"),
        "BRAND_CONSTRAINTS": extra.get("brand_constraints", "_(none)_"),
        "AESTHETIC_RULES": extra.get("aesthetic_rules", "_(TBD)_"),
        "AUDIO_APPROACH": plan["audio"].get("approach", "_(TBD)_"),
        "VO_DETAILS": plan["audio"].get("vo", "_(n/a)_"),
        "MUSIC_DETAILS": plan["audio"].get("music", "_(n/a)_"),
        "SFX_DETAILS": plan["audio"].get("sfx", "_(n/a)_"),
        "PRIMARY_FORMAT": plan["deliverables"][0]["format"],
        "PRIMARY_RES": plan["deliverables"][0]["resolution"],
        "ALT_FORMAT_1": (plan["deliverables"][1]["format"]
                          if len(plan["deliverables"]) > 1 else "_(none)_"),
        "ALT_RES_1": (plan["deliverables"][1]["resolution"]
                       if len(plan["deliverables"]) > 1 else ""),
        "ALT_NOTES_1": (plan["deliverables"][1].get("notes", "")
                         if len(plan["deliverables"]) > 1 else ""),
        "API_KEYS_REQUIRED": ", ".join(plan.get("api_keys_required", [])) or "none",
        "EXT_DEPS": extra.get("ext_deps", "ffmpeg, Python 3.11+"),
        "SOURCE_ASSETS": extra.get("source_assets", "_(none)_"),
    }
    out = tmpl
    for k, v in replacements.items():
        out = out.replace("{{" + k + "}}", str(v))

    # Scene + deliv tables: replace the placeholder row in the template
    out = re.sub(
        r"\|\s*1\s*\|\s*0:00–0:0X.+?\n\|\s*2\s*\|.+?\n",
        scene_table + "\n",
        out, flags=re.DOTALL,
    )
    return out


def render_team_md(plan: dict) -> str:
    """Render TEAM.md from the team list + scene → tool mapping."""
    lines = [f"# Team & Task Graph — {plan['title']}", "", "## Team", ""]
    for t in plan["team"]:
        skills = (
            f"loads `{', '.join(t['skills'])}`"
            if t["skills"] else "no skills required"
        )
        lines.append(
            f"- `{t['profile']}` — {t['responsibilities']} ({skills})"
        )
    lines.extend(["", "## Task Graph", "", "```"])

    # Build a simple task graph based on conventions
    profiles_by_role = {t["role"]: t["profile"] for t in plan["team"]}
    director = profiles_by_role.get("director", "director")
    lines.append(f"T0  {director} — decompose")

    next_id = 1
    parents_for_renderer: list[str] = ["T0"]

    if "cinematographer" in profiles_by_role:
        cid = f"T{next_id}"
        lines.append(
            f"{cid:5} {profiles_by_role['cinematographer']} — visual spec for all scenes (parent: T0)"
        )
        parents_for_renderer = [cid]
        next_id += 1

    if "music-supervisor" in profiles_by_role:
        cid = f"T{next_id}"
        lines.append(
            f"{cid:5} {profiles_by_role['music-supervisor']} — track analysis + beats.json (parent: T0)"
        )
        next_id += 1
        ms_id = cid
    else:
        ms_id = None

    # Scenes
    scene_ids = []
    for s in plan["scenes"]:
        cid = f"T{next_id}"
        renderer_profile = s.get("tool") or "renderer"
        # Lookup the actual profile name
        for t in plan["team"]:
            if t["role"] == renderer_profile or t["profile"] == renderer_profile:
                renderer_profile = t["profile"]
                break
        parents = parents_for_renderer + ([ms_id] if ms_id else [])
        parent_str = ", ".join(parents)
        lines.append(
            f"{cid:5} {renderer_profile} — scene {s.get('n', '?')}: "
            f"{s.get('content', '')[:50]} (parents: {parent_str})"
        )
        scene_ids.append(cid)
        next_id += 1

    # VO + audio mix
    if "voice-talent" in profiles_by_role:
        vo_id = f"T{next_id}"
        lines.append(f"{vo_id:5} {profiles_by_role['voice-talent']} — narration (parent: T0)")
        next_id += 1
    else:
        vo_id = None

    if "audio-mixer" in profiles_by_role:
        am_id = f"T{next_id}"
        am_parents = [p for p in [ms_id, vo_id] if p]
        lines.append(
            f"{am_id:5} {profiles_by_role['audio-mixer']} — mix audio (parents: {', '.join(am_parents)})"
        )
        next_id += 1
    else:
        am_id = None

    # Editor
    if "editor" in profiles_by_role:
        ed_id = f"T{next_id}"
        ed_parents = scene_ids + [p for p in [am_id, vo_id, ms_id] if p and p not in scene_ids]
        lines.append(
            f"{ed_id:5} {profiles_by_role['editor']} — assemble + mux (parents: {', '.join(ed_parents)})"
        )
        next_id += 1
    else:
        ed_id = None

    # Captioner
    if "captioner" in profiles_by_role and ed_id:
        cap_id = f"T{next_id}"
        lines.append(
            f"{cap_id:5} {profiles_by_role['captioner']} — SRT + burn (parent: {ed_id})"
        )
        next_id += 1
        last = cap_id
    else:
        last = ed_id

    # Reviewer
    if "reviewer" in profiles_by_role and last:
        rv_id = f"T{next_id}"
        lines.append(
            f"{rv_id:5} {profiles_by_role['reviewer']} — final QA (parent: {last})"
        )

    lines.append("```")
    lines.extend([
        "",
        "## Per-task workspace requirement",
        "",
        f"All `kanban_create` calls MUST pass:",
        f"```",
        f'workspace_kind="dir"',
        f'workspace_path="$HOME/projects/video-pipeline/{plan["slug"]}"',
        f'tenant="{plan["tenant"]}"',
        f"```",
    ])
    return "\n".join(lines)


def render_setup_sh(plan: dict, brief_md: str, team_md: str) -> str:
    """Render setup.sh from the plan."""
    tmpl = load_template("setup.sh.tmpl")

    # API key checks
    key_checks = []
    for key in plan.get("api_keys_required", []):
        key_checks.append(f'check_key {key} hermes {key} || exit 1')
    key_checks_str = "\n".join(key_checks) if key_checks else "# (no API keys required)"

    # Scene dirs
    scene_dir_lines = []
    for s in plan["scenes"]:
        n = s.get("n", "?")
        scene_dir_lines.append(f'mkdir -p "$WORKSPACE/scenes/scene-{n:02d}"/checkpoints')
    scene_dirs = "\n".join(scene_dir_lines) if scene_dir_lines else ""

    # Profile create
    profile_creates = []
    for t in plan["team"]:
        profile_creates.append(
            f'hermes profile create {t["profile"]} --clone 2>/dev/null || true'
        )

    # Profile config — emit JSON arrays so the bash function can pass them
    # safely through to the Python YAML patcher.
    profile_configs = []
    for t in plan["team"]:
        ts_json = json.dumps(t["toolsets"])
        sk_json = json.dumps(t["skills"])
        # Use single-quoted bash strings; JSON only contains "/[/], no single
        # quotes, so this is safe.
        profile_configs.append(
            f"configure_profile {t['profile']!r} {ts_json!r} {sk_json!r}"
        )

    # SOUL writes — uses heredocs per profile
    soul_writes = []
    for t in plan["team"]:
        soul_writes.append(
            f'cat > "$HOME/.hermes/profiles/{t["profile"]}/SOUL.md" <<\'SOUL_EOF\'\n'
            f"{render_soul_md(t, plan)}\n"
            f"SOUL_EOF\n"
            f'echo "  ✓ SOUL.md for {t["profile"]}"'
        )

    # Taste writes (placeholder; real content optional)
    taste_writes = (
        'cat > "$WORKSPACE/taste/brand-guide.md" <<\'TASTE_EOF\'\n'
        '# Brand Guide\n\n'
        '_(Populate with project-specific colors, typography, motion rules)_\n'
        'TASTE_EOF\n'
        'cat > "$WORKSPACE/taste/emotional-dna.md" <<\'DNA_EOF\'\n'
        '# Emotional DNA\n\n'
        '_(What this piece should FEEL like — populate from the brief.)_\n'
        'DNA_EOF'
    )

    # Asset copies — leave empty by default; user fills in
    asset_copies = "# Add cp/rsync commands here for any provided assets"

    out = tmpl
    out = out.replace("{{TITLE}}", plan["title"])
    out = out.replace("{{SLUG}}", plan["slug"])
    out = out.replace("{{TENANT}}", plan["tenant"])
    out = out.replace("{{WORKSPACE}}", f"~/projects/video-pipeline/{plan['slug']}")
    out = out.replace("{{KEY_CHECKS}}", key_checks_str)
    out = out.replace("{{SCENE_DIRS}}", scene_dirs)
    out = out.replace("{{PROFILE_CREATE_COMMANDS}}", "\n".join(profile_creates))
    out = out.replace("{{PROFILE_CONFIG_COMMANDS}}", "\n".join(profile_configs))
    out = out.replace("{{SOUL_WRITES}}", "\n".join(soul_writes))
    out = out.replace("{{BRIEF_CONTENTS}}", brief_md)
    out = out.replace("{{TEAM_CONTENTS}}", team_md)
    out = out.replace("{{TASTE_WRITES}}", taste_writes)
    out = out.replace("{{ASSET_COPIES}}", asset_copies)

    return out


def render_soul_md(team_member: dict, plan: dict) -> str:
    """Render a profile's SOUL.md from a team member dict + plan context."""
    tmpl = load_template("soul.md.tmpl")
    role = team_member["role"]

    common_rules = (
        "- **Read the brief and team graph** before doing anything else.\n"
        "- **Pass `workspace_kind=\"dir\"` and `workspace_path` on every "
        "`kanban_create` call.** This keeps the team in one shared workspace.\n"
        f"- **Use tenant `{plan['tenant']}`** on every kanban call.\n"
        "- **Write outputs to predictable paths.** Other profiles depend on "
        "your filename conventions.\n"
        "- **Emit heartbeats** during long-running work. Renderers should "
        "report frame counts; editors should report assembly progress.\n"
    )

    if role == "director":
        common_rules += (
            "- **Do not execute the work yourself.** For every concrete task, "
            "create a kanban task and assign it to the appropriate profile.\n"
            "- **Decompose, route, comment, approve — that's the whole job.**\n"
            "- **Read TEAM.md** for the canonical task graph. Do not invent "
            "new roles unless the brief truly demands it.\n"
            "- **Load the `kanban-orchestrator` skill** for the deeper "
            "decomposition playbook beyond the auto-injected baseline.\n"
        )

    common_commands = (
        "```bash\n"
        "# Inspect a clip\n"
        "ffprobe -v quiet -show_entries format=duration -show_entries "
        "stream=codec_name,width,height,r_frame_rate <file.mp4>\n"
        "\n"
        "# Extract a frame for QA\n"
        "ffmpeg -y -i <input.mp4> -vf \"select='eq(n,30)'\" -vsync vfr <out.png>\n"
        "```"
    )

    out = tmpl
    out = out.replace("{{ROLE_NAME}}", role)
    out = out.replace("{{ROLE_RESPONSIBILITIES}}", team_member["responsibilities"])
    out = out.replace("{{INPUTS_READ}}", team_member.get("inputs", "_(see brief)_"))
    out = out.replace("{{OUTPUTS_PRODUCED}}", team_member.get("outputs", "_(see brief)_"))
    out = out.replace("{{TOOLSETS}}", ", ".join(team_member["toolsets"]))
    out = out.replace(
        "{{SKILLS}}",
        ", ".join(team_member["skills"]) if team_member["skills"] else "(none)"
    )
    out = out.replace(
        "{{EXTERNAL_TOOLS}}",
        team_member.get("external_tools", "ffmpeg, ffprobe (via terminal)")
    )
    out = out.replace(
        "{{ROLE_RULES}}",
        team_member.get("role_rules", "_(see TEAM.md and brief.md)_")
    )
    out = out.replace("{{COMMON_RULES}}", common_rules)
    out = out.replace("{{COMMON_COMMANDS}}", common_commands)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan_json", help="Path to plan.json")
    ap.add_argument("--out", default="setup.sh",
                    help="Output path for setup.sh (default: ./setup.sh)")
    ap.add_argument("--brief-out", default=None,
                    help="Write brief.md alongside (default: skipped)")
    ap.add_argument("--team-out", default=None,
                    help="Write TEAM.md alongside (default: skipped)")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan_json).read_text())
    errors = validate_plan(plan)
    if errors:
        print("Plan validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(2)

    brief = render_brief(plan)
    team = render_team_md(plan)
    setup = render_setup_sh(plan, brief, team)

    Path(args.out).write_text(setup)
    os.chmod(args.out, 0o755)
    print(f"Wrote {args.out}")

    if args.brief_out:
        Path(args.brief_out).write_text(brief)
        print(f"Wrote {args.brief_out}")
    if args.team_out:
        Path(args.team_out).write_text(team)
        print(f"Wrote {args.team_out}")


if __name__ == "__main__":
    main()
