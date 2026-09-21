from __future__ import annotations

from typing import Any

from rich import box

from ._data import (
    ProfileIdea,
    AgentProfileSpec,
    IDEA_CATEGORY_OPTIONS,
    SKILL_BUNDLE_OPTIONS,
    SKILL_CATEGORIES,
    DOCS_SKILL_CATEGORIES,
    ROLE_OPTIONS,
    PERSONALITY_OPTIONS,
    TONE_OPTIONS,
    TARGET_PLATFORM_OPTIONS,
    RESPONSE_LENGTH_OPTIONS,
    EMOJI_OPTIONS,
    FALLBACK_OPTIONS,
    DEFAULT_BOUNDARY_OPTIONS,
    profile_ideas,
    profile_idea_by_id,
)
from ._ui import (
    console,
    ACCENT,
    MUTED,
    WARNING,
    BACK_SENTINEL,
    Panel,
    _section,
    _text,
    _multi,
    _single,
    _confirm,
    _comma_list,
    _slugify,
    _advance_step_on_back,
    _select_provider,
    _current_model_config,
)
from ._scoring import spec_from_profile_idea, _merge_skill_sets, _skill_bundle_for_key, _choose_skills
from hermes_cli.profiles import normalize_profile_name, validate_profile_name


def _idea_summary(idea: ProfileIdea) -> str:
    return (
        f"{idea.avatar_emoji} {idea.name}\n\n"
        f"{idea.description}\n\n"
        f"Best for:\n  - " + "\n  - ".join(idea.best_for) + "\n\n"
        f"Defaults:\n"
        f"  Role: {idea.primary_role}\n"
        f"  Personality: {idea.personality}\n"
        f"  Tones: {', '.join(idea.tones)}\n"
        f"  Platforms: {', '.join(idea.suggested_platforms)}\n"
        f"  Skills: {', '.join(idea.suggested_skills)}"
    )


def browse_profile_ideas(initial_name: str | None = None) -> AgentProfileSpec | str | None:
    while True:
        category = _single(
            "Browse profile ideas",
            "Start with a proven blueprint, then customise it before anything is created.",
            IDEA_CATEGORY_OPTIONS,
            default="ootb",
        )
        if category in {"back", BACK_SENTINEL}:
            return BACK_SENTINEL
        ideas = profile_ideas(category)
        if not ideas:
            console.print(f"[{WARNING}]No profile ideas found for this category.[/{WARNING}]")
            continue
        values = [(idea.id, f"{idea.avatar_emoji} {idea.name} — {idea.description}") for idea in ideas] + [("back", "Back to categories")]
        selected_id = _single("Profile ideas", "Pick a starting point. You can customise every default afterwards.", values, default=ideas[0].id)
        if selected_id in {"back", BACK_SENTINEL}:
            continue
        idea = profile_idea_by_id(selected_id)
        if idea is None:
            continue
        console.print(Panel(_idea_summary(idea), title="Profile Idea", border_style=ACCENT, box=box.ROUNDED, padding=(1, 2)))
        action = _single(
            "Use this idea?",
            "Choose what to do with this profile idea.",
            [("use", "Review and approve defaults"), ("customise", "Use as starting point and customise"), ("another", "View another idea"), ("categories", "Back to categories")],
            default="use",
        )
        if action in {"use", "customise"}:
            return spec_from_profile_idea(idea, initial_name=initial_name)
        if action in {"categories", BACK_SENTINEL}:
            continue


def collect_profile(initial_name: str | None = None, seed: AgentProfileSpec | None = None) -> AgentProfileSpec | str:
    """Run the complete interview before returning a profile blueprint.

    No files are written here. This function only gathers answers and returns a
    structured blueprint for preview/export/create actions.

    Backspace returns to the previous major step. State is preserved across
    rewinds so re-answering a step overwrites the previous value.
    """
    total = 9

    # ---- mutable state (pre-seeded) ----
    uc: str = seed.user_context if seed else "A user who wants a focused Hermes agent profile."
    outcomes: str = seed.intended_outcomes if seed else "Answer questions, run useful workflows, and stay within clear boundaries."
    opctx: str = seed.operating_context if seed else "Hermes CLI and messaging gateway."
    platforms: list[str] = seed.target_platforms if seed else ["cli"]

    active_provider, active_model = _current_model_config()
    provider: str = seed.model_provider if seed else active_provider
    mdl: str = seed.model if (seed and provider == seed.model_provider) else (active_model if provider == active_provider else "pending-provider-setup")

    display_name: str = seed.display_name if seed else (initial_name or "").replace("-", " ").title()
    profile_name: str = normalize_profile_name(_slugify(initial_name or (seed.profile_name if seed else display_name)))
    tagline: str = seed.tagline if seed else "Focused help, clear boundaries."
    avatar: str = seed.avatar_emoji if seed else "⚕"

    primary_role: str = seed.primary_role if seed else "utility_bot"
    secondary_roles: list[str] = seed.secondary_roles if seed else []

    personality: str = seed.personality if seed else "analyst"
    tones: list[str] = seed.tones if seed else ["professional"]

    skill_sets: dict[str, list[str]] = dict(seed.skill_sets) if seed else {}
    custom_skills: list[str] = list(seed.custom_skills) if seed else []

    response_length: str = seed.response_length if seed else "balanced"
    emoji_usage: str = seed.emoji_usage if seed else "minimal"
    fallback_behavior: str = seed.fallback_behavior if seed else "acknowledge_uncertainty"

    boundaries: list[str] = seed.boundaries if seed else ["no_sensitive_data_disclosure"]
    blocked_topics: list[str] = seed.blocked_topics if seed else []

    step = 0
    while step < total:
        step += 1

        # ── Step 1: User context ──
        if step == 1:
            _section(1, total, "Understand the user")
            substep = 1
            while substep <= 4:
                if substep == 1:
                    r = _text("Who is this for?", "Team, community, or user type", default=uc)
                    if r == BACK_SENTINEL:
                        return BACK_SENTINEL
                    uc = str(r)
                    substep += 1
                elif substep == 2:
                    r = _text("Goals", "Main outcomes this agent should deliver", default=outcomes)
                    if r == BACK_SENTINEL:
                        substep -= 1; continue
                    outcomes = str(r)
                    substep += 1
                elif substep == 3:
                    r = _text("Environment", "Where will it operate?", default=opctx)
                    if r == BACK_SENTINEL:
                        substep -= 1; continue
                    opctx = str(r)
                    substep += 1
                elif substep == 4:
                    r = _multi("Surfaces", "Where should it feel at home?", TARGET_PLATFORM_OPTIONS, default=platforms)
                    if r == BACK_SENTINEL:
                        substep -= 1; continue
                    platforms = list(r) if isinstance(r, list) else platforms
                    substep += 1

        # ── Step 2: Model provider ──
        elif step == 2:
            _section(2, total, "Model provider")
            r = _select_provider(provider)
            if r == BACK_SENTINEL: step = max(0, step - 2); continue
            provider = str(r)
            if seed and provider == seed.model_provider:
                mdl = seed.model
            else:
                mdl = active_model if provider == active_provider else "pending-provider-setup"

        # ── Step 3: Identity ──
        elif step == 3:
            _section(3, total, "Identity")
            substep = 1
            while substep <= 4:
                if substep == 1:
                    r = _text("Display name", "Agent name", default=display_name)
                    if r == BACK_SENTINEL:
                        step = max(0, step - 2); break
                    display_name = str(r)
                    substep += 1
                elif substep == 2:
                    r = _text("CLI name", "Lowercase command alias", default=profile_name)
                    if r == BACK_SENTINEL:
                        substep -= 1; continue
                    profile_name = normalize_profile_name(_slugify(str(r)))
                    validate_profile_name(profile_name)
                    substep += 1
                elif substep == 3:
                    r = _text("Tagline", "Short motto", default=tagline)
                    if r == BACK_SENTINEL:
                        substep -= 1; continue
                    tagline = str(r)
                    substep += 1
                elif substep == 4:
                    r = _text("Avatar emoji", "Single emoji", default=avatar)
                    if r == BACK_SENTINEL:
                        substep -= 1; continue
                    avatar = str(r) or avatar
                    substep += 1
            if step < 3: continue

        # ── Step 4: Role ──
        elif step == 4:
            _section(4, total, "Role")
            substep = 1
            while substep <= 2:
                if substep == 1:
                    r = _single("Primary role", "Main job", ROLE_OPTIONS, default=primary_role)
                    if r == BACK_SENTINEL:
                        step = max(0, step - 2); break
                    primary_role = str(r)
                    substep += 1
                elif substep == 2:
                    r = _multi("Secondary roles", "Supporting roles", [item for item in ROLE_OPTIONS if item[0] != primary_role], default=secondary_roles)
                    if r == BACK_SENTINEL:
                        substep -= 1; continue
                    secondary_roles = list(r) if isinstance(r, list) else secondary_roles
                    substep += 1
            if step < 4: continue

        # ── Step 5: Personality ──
        elif step == 5:
            _section(5, total, "Personality & tone")
            substep = 1
            while substep <= 2:
                if substep == 1:
                    r = _single("Archetype", "Operating style", PERSONALITY_OPTIONS, default=personality)
                    if r == BACK_SENTINEL:
                        step = max(0, step - 2); break
                    personality = str(r)
                    substep += 1
                elif substep == 2:
                    r = _multi("Tones", "Pick 1–3", TONE_OPTIONS, default=tones)
                    if r == BACK_SENTINEL:
                        substep -= 1; continue
                    if isinstance(r, list):
                        tones = r
                    if len(tones) > 3: tones = tones[:3]
                    if not tones: tones = ["professional"]
                    substep += 1
        # ── Step 6: Skills ──
        elif step == 6:
            if seed and seed.skill_sets:
                skill_sets = _merge_skill_sets(skill_sets, seed.skill_sets)
            r = _multi("Skill bundles", "Start with predefined bundles. You can add individual skills next.", SKILL_BUNDLE_OPTIONS, default=[])
            step, rewound = _advance_step_on_back(r, step)
            if rewound:
                continue
            if isinstance(r, list):
                for bundle_key in r:
                    skill_sets = _merge_skill_sets(skill_sets, _skill_bundle_for_key(bundle_key))
            r = _choose_skills(skill_sets)
            if r == BACK_SENTINEL:
                step, _ = _advance_step_on_back(r, step)
                continue

        # ── Step 7: Custom skills ──
        elif step == 7:
            _section(7, total, "Custom skills")
            custom_skills.clear()
            custom_skills.extend(seed.custom_skills if seed else [])
            while True:
                r = _single("Add a custom skill?", "", [("yes", "Yes"), ("no", "No — done"), ("back", "↩ Back")], default="no")
                if r == BACK_SENTINEL: step = max(1, step - 2); break
                if r == "no": break
                if r == "back": step = max(1, step - 2); break
                s = _text("Skill", "Short phrase describing the ability")
                if s == BACK_SENTINEL: step = max(1, step - 2); break
                if s.strip(): custom_skills.append(str(s).strip())
            if step < 7: continue  # rewind

        # ── Step 8: Response behaviour ──
        elif step == 8:
            _section(8, total, "Response behaviour")
            substep = 1
            while substep <= 3:
                if substep == 1:
                    r = _single("Length", "Default answer depth", RESPONSE_LENGTH_OPTIONS, default=response_length)
                    if r == BACK_SENTINEL:
                        step = max(0, step - 2); break
                    response_length = str(r)
                    substep += 1
                elif substep == 2:
                    r = _single("Emoji", "How much emoji?", EMOJI_OPTIONS, default=emoji_usage)
                    if r == BACK_SENTINEL:
                        substep -= 1; continue
                    emoji_usage = str(r)
                    substep += 1
                elif substep == 3:
                    r = _single("Fallback", "When uncertain?", FALLBACK_OPTIONS, default=fallback_behavior)
                    if r == BACK_SENTINEL:
                        substep -= 1; continue
                    fallback_behavior = str(r)
                    substep += 1
            if step < 8: continue

        # ── Step 9: Guardrails ──
        elif step == 9:
            _section(9, total, "Guardrails")
            substep = 1
            while substep <= 2:
                if substep == 1:
                    r = _multi("Restrictions", "What must this profile obey?", DEFAULT_BOUNDARY_OPTIONS, default=boundaries)
                    if r == BACK_SENTINEL:
                        step = max(0, step - 2); break
                    boundaries = list(r) if isinstance(r, list) else boundaries
                    substep += 1
                elif substep == 2:
                    r = _text("Blocked topics", "Comma-separated", default=", ".join(blocked_topics))
                    if r == BACK_SENTINEL:
                        substep -= 1; continue
                    blocked_topics = _comma_list(str(r))
                    substep += 1
            if step < 9: continue

    return AgentProfileSpec(
        profile_name=profile_name,
        display_name=display_name,
        tagline=tagline,
        avatar_emoji=avatar,
        primary_role=primary_role,
        secondary_roles=secondary_roles,
        personality=personality,
        tones=tones,
        skill_sets=skill_sets,
        custom_skills=custom_skills,
        response_length=response_length,
        emoji_usage=emoji_usage,
        fallback_behavior=fallback_behavior,
        boundaries=boundaries,
        blocked_topics=blocked_topics,
        user_context=uc,
        intended_outcomes=outcomes,
        operating_context=opctx,
        target_platforms=platforms,
        model_provider=provider,
        model=mdl,
        source_idea=(seed.source_idea if seed else None),
        source_idea_name=(seed.source_idea_name if seed else None),
        use_cases=(seed.use_cases if seed else []),
    )
