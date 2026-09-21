from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich import box
from rich.panel import Panel

from ._data import (
    AgentProfileSpec,
    ROLE_OPTIONS,
    PERSONALITY_OPTIONS,
    TONE_OPTIONS,
    TARGET_PLATFORM_OPTIONS,
    RESPONSE_LENGTH_OPTIONS,
    EMOJI_OPTIONS,
    FALLBACK_OPTIONS,
    DEFAULT_BOUNDARY_OPTIONS,
    SKILL_BUNDLE_OPTIONS,
    SKILL_CATEGORIES,
    DOCS_SKILL_CATEGORIES,
)
from ._ui import (
    console,
    ACCENT,
    SUCCESS,
    WARNING,
    BACK_SENTINEL,
    _banner,
    _single,
    _multi,
    _text,
    _confirm,
    _comma_list,
    _section,
    _slugify,
    _select_provider,
    _label_for,
)
from ._scoring import _ensure_review_ready_defaults, spec_from_profile_idea
from ._builder import browse_profile_ideas, collect_profile
from ._guided import guided_onboarding
from ._output import render_preview, print_preview, generate_soul_md, generate_user_md, export_profile, _description
from hermes_cli.profiles import (
    check_alias_collision,
    create_profile,
    create_wrapper_script,
    normalize_profile_name,
    seed_profile_skills,
    validate_profile_name,
    write_profile_meta,
    _get_wrapper_dir,
    _is_wrapper_dir_in_path,
)


def _write_profile_files(
    spec: AgentProfileSpec,
    clone: bool = False,
    no_alias: bool = False,
    no_skills: bool = False,
    clone_from: str | None = None,
) -> Path:
    profile_dir = create_profile(
        name=spec.profile_name,
        clone_from=clone_from,
        clone_config=clone,
        no_alias=True,  # create alias after files are written, so failures do not strand a command.
        no_skills=no_skills,
        description=_description(spec),
    )
    (profile_dir / "SOUL.md").write_text(generate_soul_md(spec), encoding="utf-8")
    (profile_dir / "USER.md").write_text(generate_user_md(spec), encoding="utf-8")
    (profile_dir / "profile.json").write_text(json.dumps(asdict(spec), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_profile_meta(profile_dir, description=_description(spec), description_auto=False)

    result = seed_profile_skills(profile_dir, quiet=True)
    if result and result.get("skipped_opt_out"):
        console.print(f"[{WARNING}]No bundled skills seeded (--no-skills marker present).[/{WARNING}]")

    if not no_alias:
        collision = check_alias_collision(spec.profile_name)
        if collision:
            console.print(f"[{WARNING}]Alias skipped: {collision}[/{WARNING}]")
            console.print(f"Use via: hermes -p {spec.profile_name} chat")
        else:
            wrapper = create_wrapper_script(spec.profile_name)
            if wrapper:
                console.print(f"[{SUCCESS}]Wrapper created:[/{SUCCESS}] {wrapper}")
                if not _is_wrapper_dir_in_path():
                    console.print(f"[{WARNING}]{_get_wrapper_dir()} is not in PATH. Add: export PATH=\"$HOME/.local/bin:$PATH\"[/{WARNING}]")
    return profile_dir


def _persist_profile_updates(spec: AgentProfileSpec, profile_dir: Path) -> None:
    (profile_dir / "SOUL.md").write_text(generate_soul_md(spec), encoding="utf-8")
    (profile_dir / "USER.md").write_text(generate_user_md(spec), encoding="utf-8")
    (profile_dir / "profile.json").write_text(json.dumps(asdict(spec), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_profile_meta(profile_dir, description=_description(spec), description_auto=False)


def _edit_section(spec: AgentProfileSpec, profile_dir: Path | None = None) -> None:
    section = _single(
        "Edit blueprint",
        "Choose the section to revisit",
        [
            ("interview", "Interview context"),
            ("identity", "Identity"),
            ("role", "Role"),
            ("personality", "Personality & tone"),
            ("skills", "Skills"),
            ("style", "Response behaviour"),
            ("boundaries", "Guardrails"),
        ],
    )
    if section == "interview":
        r = _text("User context", "Who is this for?", spec.user_context)
        if r != BACK_SENTINEL: spec.user_context = str(r)
        r = _text("Goals", "What should this agent help accomplish?", spec.intended_outcomes)
        if r != BACK_SENTINEL: spec.intended_outcomes = str(r)
        r = _text("Operating context", "Where will it operate?", spec.operating_context)
        if r != BACK_SENTINEL: spec.operating_context = str(r)
        r = _multi("Target surfaces", "Where should it feel at home?", TARGET_PLATFORM_OPTIONS, spec.target_platforms)
        if isinstance(r, list): spec.target_platforms = r
    elif section == "identity":
        r = _text("Identity", "Agent display name", spec.display_name)
        if r != BACK_SENTINEL: spec.display_name = str(r)
        r = _text("Identity", "Tagline / motto", spec.tagline)
        if r != BACK_SENTINEL: spec.tagline = str(r)
        r = _text("Identity", "Avatar emoji", spec.avatar_emoji)
        if r != BACK_SENTINEL and r.strip(): spec.avatar_emoji = str(r)
    elif section == "role":
        r = _single("Primary role", "Choose the main job", ROLE_OPTIONS, spec.primary_role)
        if r != BACK_SENTINEL: spec.primary_role = str(r)
        r = _multi("Secondary roles", "Optional supporting roles", [item for item in ROLE_OPTIONS if item[0] != spec.primary_role], spec.secondary_roles)
        if isinstance(r, list): spec.secondary_roles = r
    elif section == "personality":
        r = _single("Personality", "Pick the operating archetype", PERSONALITY_OPTIONS, spec.personality)
        if r != BACK_SENTINEL: spec.personality = str(r)
        r = _multi("Tone", "Pick 1–3 tones", TONE_OPTIONS, spec.tones)
        if isinstance(r, list):
            spec.tones = r[:3] or ["professional"]
    elif section == "skills":
        skill_sets: dict[str, list[str]] = {}
        for category, options in SKILL_CATEGORIES.items():
            selected = _multi(category, f"Choose skills for {category}", options, spec.skill_sets.get(category, []))
            if isinstance(selected, list) and selected:
                skill_sets[category] = selected
        spec.skill_sets = skill_sets
    elif section == "style":
        r = _single("Response length", "Default answer depth", RESPONSE_LENGTH_OPTIONS, spec.response_length)
        if r != BACK_SENTINEL: spec.response_length = str(r)
        r = _single("Emoji usage", "How much emoji should this agent use?", EMOJI_OPTIONS, spec.emoji_usage)
        if r != BACK_SENTINEL: spec.emoji_usage = str(r)
        r = _single("Fallback behaviour", "When uncertain, what should it do?", FALLBACK_OPTIONS, spec.fallback_behavior)
        if r != BACK_SENTINEL: spec.fallback_behavior = str(r)
    elif section == "boundaries":
        r = _multi("Guardrails", "Select restrictions", DEFAULT_BOUNDARY_OPTIONS, spec.boundaries)
        if isinstance(r, list): spec.boundaries = r
        r = _text("Blocked topics", "Comma-separated", ", ".join(spec.blocked_topics))
        if r != BACK_SENTINEL: spec.blocked_topics = _comma_list(str(r))

    if profile_dir is not None:
        _persist_profile_updates(spec, profile_dir)
        console.print(f"[{SUCCESS}]Profile files updated.[/{SUCCESS}]")


def _add_custom_skills(spec: AgentProfileSpec, profile_dir: Path | None = None) -> None:
    while True:
        r = _single("Custom skills", "Add another custom skill?", [("yes", "Yes"), ("no", "No — done")], default="yes")
        if r == BACK_SENTINEL or r == "no":
            break
        skill = _text("Custom skill", "Describe the skill in one short phrase")
        if skill == BACK_SENTINEL:
            break
        if skill and str(skill).strip():
            spec.custom_skills.append(str(skill).strip())
    if profile_dir is not None:
        _persist_profile_updates(spec, profile_dir)
        console.print(f"[{SUCCESS}]Profile files updated.[/{SUCCESS}]")


def _action_loop(
    spec: AgentProfileSpec,
    clone: bool = False,
    no_alias: bool = False,
    no_skills: bool = False,
    clone_from: str | None = None,
) -> Path | None:
    profile_dir: Path | None = None
    while True:
        created = profile_dir is not None
        actions: list[tuple[str, str]] = [
            ("preview", "Preview blueprint"),
            ("edit", "Re-run a section"),
            ("custom", "Add more custom skills"),
            ("export_json", "Export blueprint as JSON"),
            ("export_js", "Export blueprint as JS/ES module"),
            ("stdout", "Print blueprint to stdout"),
        ]
        if not created:
            actions.append(("create", "Create Hermes profile from this blueprint"))
        else:
            actions.extend([
                ("setup", "Show setup command"),
                ("chat", "Show chat command"),
            ])
        actions.append(("exit", "Exit"))

        action = _single("Next action", "The interview is complete. Choose what to do with the blueprint.", actions, default="preview")
        if action == "preview":
            print_preview(spec)
        elif action == "edit":
            _edit_section(spec, profile_dir)
            print_preview(spec)
        elif action == "custom":
            _add_custom_skills(spec, profile_dir)
            print_preview(spec)
        elif action == "export_json":
            path = export_profile(spec, "json", profile_dir or Path.cwd())
            console.print(f"[{SUCCESS}]Exported:[/{SUCCESS}] {path}")
        elif action == "export_js":
            path = export_profile(spec, "js", profile_dir or Path.cwd())
            console.print(f"[{SUCCESS}]Exported:[/{SUCCESS}] {path}")
        elif action == "stdout":
            export_profile(spec, "stdout")
        elif action == "create":
            if _confirm("Create profile", f"Create Hermes profile '{spec.profile_name}' now?", default=True):
                profile_dir = _write_profile_files(
                    spec,
                    clone=clone,
                    no_alias=no_alias,
                    no_skills=no_skills,
                    clone_from=clone_from,
                )
                console.print(Panel(
                    f"Profile created at {profile_dir}\n\n"
                    f"Next commands:\n  {spec.profile_name} setup\n  {spec.profile_name} chat\n  {spec.profile_name} gateway start",
                    title="Profile Created",
                    border_style=SUCCESS,
                    box=box.ROUNDED,
                    padding=(1, 2),
                ))
        elif action == "setup":
            console.print(Panel(f"Run:\n\n  {spec.profile_name} setup", title="Setup", border_style=ACCENT, box=box.ROUNDED))
        elif action == "chat":
            console.print(Panel(f"Run:\n\n  {spec.profile_name} chat", title="Start chatting", border_style=ACCENT, box=box.ROUNDED))
        else:
            return profile_dir


def run_profile_wizard(args: Any | None = None) -> Path | None:
    initial_name = getattr(args, "profile_name", None) if args is not None else None
    clone = bool(getattr(args, "clone", False)) if args is not None else False
    no_alias = bool(getattr(args, "no_alias", False)) if args is not None else False
    no_skills = bool(getattr(args, "no_skills", False)) if args is not None else False
    clone_from = getattr(args, "clone_from", None) if args is not None else None

    try:
        _banner()
        while True:
            start = _single(
                "Profile workspace",
                "Choose how to start. Ideas are only blueprints; nothing is created until you approve the final preview.",
                [
                    ("guided", "Start guided setup"),
                    ("browse", "Browse starter profiles"),
                    ("custom", "Create a custom profile"),
                    ("exit", "Exit"),
                ],
                default="guided",
            )
            if start in {"exit", BACK_SENTINEL}:
                return None
            if start == "guided":
                seed = guided_onboarding(initial_name=initial_name)
            elif start == "browse":
                seed = browse_profile_ideas(initial_name=initial_name)
            else:
                seed = None
            if seed == BACK_SENTINEL:
                continue
            if start in {"guided", "browse"} and seed is None:
                return None

            # Normalise seed into a list of specs for multi-agent flows
            specs: list[AgentProfileSpec] = []
            if isinstance(seed, list):
                specs = [_ensure_review_ready_defaults(s) for s in seed]
            elif isinstance(seed, AgentProfileSpec):
                specs = [_ensure_review_ready_defaults(seed)]
            else:
                profile_spec = collect_profile(initial_name=initial_name, seed=None)
                if profile_spec == BACK_SENTINEL:
                    continue
                assert isinstance(profile_spec, AgentProfileSpec)
                specs = [profile_spec]

            result = None
            completed = 0
            total_specs = len(specs)
            for idx, current_spec in enumerate(specs, 1):
                if total_specs > 1:
                    body = f"Profile {idx}/{total_specs}\n\n{current_spec.display_name}\n{current_spec.tagline}"
                    console.print(Panel(body, title="Configuring Profile", border_style=ACCENT, box=box.ROUNDED, padding=(1, 2)))
                mode = _single(
                    "Use pre-configured defaults?",
                    "This profile already has identity, model, skills, surfaces, behaviour, and guardrails populated.",
                    [
                        ("review", "Review and approve — skip manual setup"),
                        ("customise", "Customise profile step by step"),
                        ("skip", "Skip this profile"),
                        ("back", "Back to profile workspace"),
                    ],
                    default="review",
                )
                if mode in {BACK_SENTINEL, "back"}:
                    break
                if mode == "skip":
                    continue
                final_spec = current_spec if mode == "review" else collect_profile(initial_name=initial_name, seed=current_spec)
                if final_spec == BACK_SENTINEL:
                    break
                assert isinstance(final_spec, AgentProfileSpec)
                print_preview(final_spec)
                result = _action_loop(
                    final_spec,
                    clone=clone,
                    no_alias=no_alias,
                    no_skills=no_skills,
                    clone_from=clone_from,
                )
                completed += 1
                if total_specs > 1 and idx < total_specs:
                    r = _single(
                        f"Continue? ({completed}/{total_specs} configured)",
                        "",
                        [("next", f"Configure next profile: {specs[idx].display_name}"), ("stop", "Stop here — remaining profiles saved")],
                        default="next",
                    )
                    if r in {BACK_SENTINEL, "stop"}:
                        return result
            return result if total_specs == 1 else None

    except KeyboardInterrupt:
        console.print(f"\n[{WARNING}]Cancelled. No files are written unless you already chose 'Create Hermes profile'.[/{WARNING}]")
        return None
