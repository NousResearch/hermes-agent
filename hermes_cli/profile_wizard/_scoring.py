from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ._data import (
    ProfileIdea,
    AgentProfileSpec,
    OnboardingAnswers,
    SKILL_BUNDLES,
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
    USER_TYPE_OPTIONS,
    USE_CASE_OPTIONS,
    INTEREST_OPTIONS,
    APP_TOOL_OPTIONS,
    HERMES_SETUP_OPTIONS,
    AUTOMATION_COMFORT_OPTIONS,
    SENSITIVE_AREA_OPTIONS,
    profile_idea_by_id,
    profile_ideas,
)
from ._ui import _current_model_config, _slugify, _label_for, _section, _single, _multi, WARNING, BACK_SENTINEL, console
from hermes_cli.profiles import normalize_profile_name


def _merge_skill_sets(*sets: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for skill_set in sets:
        for category, skills in skill_set.items():
            bucket = merged.setdefault(category, [])
            for skill in skills:
                if skill not in bucket:
                    bucket.append(skill)
    return merged


def _skill_bundle_for_key(key: str) -> dict[str, list[str]]:
    return {cat: list(skills) for cat, skills in SKILL_BUNDLES.get(key, {}).items()}


def _local_skill_options(limit: int = 80) -> list[tuple[str, str]]:
    roots: list[Path] = []
    home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    roots.append(home / "skills")
    repo_skills = Path(__file__).resolve().parents[2] / "skills"
    roots.append(repo_skills)
    seen: set[str] = set()
    options: list[tuple[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for skill_md in sorted(root.glob("**/SKILL.md")):
            if len(options) >= limit:
                return options
            try:
                text = skill_md.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            name = skill_md.parent.name
            desc = ""
            for line in text.splitlines()[:40]:
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip().strip('"\'') or name
                elif line.startswith("description:"):
                    desc = line.split(":", 1)[1].strip().strip('"\'')
                    break
            if name in seen:
                continue
            seen.add(name)
            label = f"{name} — {desc}" if desc else name
            options.append((name, label))
    return options


def _skill_defaults_for_idea(idea: ProfileIdea) -> tuple[dict[str, list[str]], list[str]]:
    grouped: dict[str, list[str]] = {}
    matched: set[str] = set()
    catalogues = [SKILL_CATEGORIES, DOCS_SKILL_CATEGORIES]
    for catalogue in catalogues:
        for category, options in catalogue.items():
            option_keys = {key for key, _label in options}
            selected = [key for key in idea.suggested_skills if key in option_keys]
            if selected:
                grouped[category] = list(dict.fromkeys(grouped.get(category, []) + selected))
                matched.update(selected)
    custom = [skill for skill in idea.suggested_skills if skill not in matched]
    return grouped, custom


def _ensure_review_ready_defaults(spec: AgentProfileSpec) -> AgentProfileSpec:
    active_provider, active_model = _current_model_config()
    spec.profile_name = normalize_profile_name(_slugify(spec.profile_name or spec.display_name or "agent-profile"))
    spec.display_name = spec.display_name or spec.profile_name.replace("-", " ").title()
    spec.tagline = spec.tagline or "Focused help, clear boundaries."
    spec.avatar_emoji = spec.avatar_emoji or "⚕"
    spec.primary_role = spec.primary_role or "utility_bot"
    spec.secondary_roles = list(dict.fromkeys(spec.secondary_roles or []))
    spec.personality = spec.personality or "analyst"
    spec.tones = list(dict.fromkeys(spec.tones or ["professional"]))[:3] or ["professional"]
    spec.skill_sets = _merge_skill_sets(spec.skill_sets or {"Utility & Tools": ["file_search"]})
    spec.custom_skills = list(dict.fromkeys(spec.custom_skills or []))
    spec.response_length = spec.response_length or "balanced"
    spec.emoji_usage = spec.emoji_usage or "minimal"
    spec.fallback_behavior = spec.fallback_behavior or "acknowledge_uncertainty"
    spec.boundaries = list(dict.fromkeys(spec.boundaries or ["no_sensitive_data_disclosure"]))
    spec.blocked_topics = list(dict.fromkeys(spec.blocked_topics or []))
    spec.user_context = spec.user_context or f"Started from the {spec.display_name} pre-configured profile."
    spec.intended_outcomes = spec.intended_outcomes or "Use this profile for focused Hermes workflows."
    spec.operating_context = spec.operating_context or "Hermes CLI and configured messaging surfaces."
    spec.target_platforms = list(dict.fromkeys(spec.target_platforms or ["cli"]))
    spec.model_provider = spec.model_provider or active_provider
    spec.model = spec.model or (active_model if spec.model_provider == active_provider else "pending-provider-setup")
    spec.use_cases = list(dict.fromkeys(spec.use_cases or []))
    return spec


def spec_from_profile_idea(idea: ProfileIdea, initial_name: str | None = None) -> AgentProfileSpec:
    active_provider, active_model = _current_model_config()
    provider = idea.suggested_provider or active_provider
    model = idea.suggested_model or (active_model if provider == active_provider else "pending-provider-setup")
    profile_name = normalize_profile_name(_slugify(initial_name or idea.id))
    grouped_skills, custom_skills = _skill_defaults_for_idea(idea)
    return _ensure_review_ready_defaults(AgentProfileSpec(
        profile_name=profile_name,
        display_name=initial_name.replace("-", " ").title() if initial_name else idea.name,
        tagline=idea.tagline,
        avatar_emoji=idea.avatar_emoji,
        primary_role=idea.primary_role,
        secondary_roles=list(idea.secondary_roles),
        personality=idea.personality,
        tones=list(idea.tones),
        skill_sets=grouped_skills,
        custom_skills=custom_skills,
        response_length="concise" if "discord" in idea.suggested_platforms else "balanced",
        emoji_usage="minimal",
        fallback_behavior="research_reason" if idea.primary_role in {"knowledge_base", "analytics", "devops_admin"} else "acknowledge_uncertainty",
        boundaries=list(idea.guardrails) or ["no_sensitive_data_disclosure"],
        blocked_topics=[],
        user_context=f"Started from the {idea.name} profile idea.",
        intended_outcomes="; ".join(idea.best_for),
        operating_context=idea.description,
        target_platforms=list(idea.suggested_platforms),
        model_provider=provider,
        model=model,
        source_idea=idea.id,
        source_idea_name=idea.name,
        use_cases=list(idea.use_cases or idea.best_for),
    ))


def _score_profile_ideas(answers: OnboardingAnswers) -> list[ProfileIdea]:
    scores: dict[str, int] = {idea.id: 0 for idea in profile_ideas()}
    boosts: dict[str, list[str]] = {
        "developer": ["coding_assistant", "coding_lead", "code_worker", "qa_lead"],
        "founder": ["general_assistant", "automation_assistant", "research_assistant", "admin_assistant_starter", "kensei_orchestrator"],
        "creator": ["content_assistant", "content_lead", "content_worker", "research_assistant"],
        "researcher": ["research_assistant", "knowledge_base", "research_lead", "research_worker"],
        "community": ["community_assistant", "knowledge_base"],
        "team": ["general_assistant", "knowledge_base", "admin_assistant_starter", "kensei_orchestrator"],
    }
    for pid in boosts.get(answers.user_type, []):
        scores[pid] = scores.get(pid, 0) + 4
    for use_case in answers.use_cases:
        mapping = {
            "organise": ["general_assistant", "admin_assistant_starter"],
            "research": ["research_assistant", "knowledge_base", "research_lead"],
            "content": ["content_assistant", "content_lead", "content_worker"],
            "coding": ["coding_assistant", "coding_lead", "code_worker", "qa_lead"],
            "admin": ["admin_assistant_starter", "general_assistant"],
            "automation": ["automation_assistant", "security_ops_lead", "kensei_orchestrator"],
            "community": ["community_assistant"],
            "learning": ["teaching_mentor", "knowledge_base"],
            "analysis": ["research_assistant", "research_lead", "research_worker"],
            "multi_agent": ["kensei_orchestrator", "research_lead", "coding_lead", "qa_lead", "content_lead", "admin_assistant"],
        }
        for pid in mapping.get(use_case, []):
            scores[pid] = scores.get(pid, 0) + 5
    app_map = {
        "github": ["coding_assistant", "coding_lead", "qa_lead"],
        "linear": ["coding_lead", "qa_lead"],
        "gmail": ["admin_assistant_starter", "admin_assistant"],
        "outlook": ["admin_assistant_starter", "admin_assistant"],
        "google_calendar": ["admin_assistant_starter", "general_assistant"],
        "notion": ["knowledge_base", "research_lead"],
        "obsidian": ["knowledge_base", "research_lead"],
        "discord": ["community_assistant"],
        "telegram": ["community_assistant"],
        "x_twitter": ["content_assistant", "content_lead", "content_worker"],
        "linkedin": ["content_assistant", "content_lead", "content_worker"],
        "youtube": ["content_assistant", "content_lead", "research_assistant"],
    }
    for app in answers.apps:
        for pid in app_map.get(app, []):
            scores[pid] = scores.get(pid, 0) + 3
    ranked = sorted(profile_ideas(), key=lambda idea: (scores.get(idea.id, 0), 1 if idea.source == "ootb" else 0), reverse=True)
    return ranked[:6]


def _recommended_bundle_keys(answers: OnboardingAnswers) -> list[str]:
    keys = ["starter"]
    if "research" in answers.use_cases or "research" in answers.interests or "analysis" in answers.use_cases:
        keys.append("research")
    if "coding" in answers.use_cases or "software" in answers.interests or "github" in answers.apps:
        keys.append("developer")
    if "admin" in answers.use_cases or any(a in answers.apps for a in ["gmail", "outlook", "google_calendar"]):
        keys.append("admin_productivity")
    if "content" in answers.use_cases or "writing" in answers.interests or "social" in answers.interests:
        keys.append("content_social")
    if "community" in answers.use_cases or any(a in answers.apps for a in ["discord", "telegram"]):
        keys.append("community")
    if "automation" in answers.use_cases or answers.setup_style == "automation":
        keys.append("operations")
    return list(dict.fromkeys(keys))


def _boundaries_from_onboarding(answers: OnboardingAnswers) -> list[str]:
    boundaries = ["no_sensitive_data_disclosure"]
    if "external_messages" in answers.sensitive_areas:
        boundaries.append("no_external_messages_without_approval")
    if "money" in answers.sensitive_areas:
        boundaries.append("no_spending_without_approval")
    if answers.automation_comfort in {"suggest_only", "draft_prepare", "cautious"}:
        boundaries.extend(["no_external_messages_without_approval", "escalate_policy_edge_cases"])
    return list(dict.fromkeys(boundaries))


def _spec_from_onboarding(answers: OnboardingAnswers, idea: ProfileIdea, initial_name: str | None = None) -> AgentProfileSpec:
    bundles = [_skill_bundle_for_key(key) for key in _recommended_bundle_keys(answers)]
    skill_sets = _merge_skill_sets(_skill_defaults_for_idea(idea)[0], *bundles)
    context_bits = [
        f"User type: {_label_for(answers.user_type, USER_TYPE_OPTIONS)}.",
        f"Context: {_label_for(answers.working_context, USE_CASE_OPTIONS) if answers.working_context in dict(USE_CASE_OPTIONS) else answers.working_context}.",
        answers.user_context,
    ]
    if answers.interests:
        context_bits.append("Interests: " + ", ".join(answers.interests) + ".")
    if answers.apps:
        context_bits.append("Apps/tools: " + ", ".join(answers.apps) + ".")
    if answers.other_apps:
        context_bits.append("Other tools: " + answers.other_apps + ".")
    outcomes = "Use cases: " + ", ".join(answers.use_cases)
    if answers.ideas:
        outcomes += f". Ideas: {answers.ideas}"
    spec = spec_from_profile_idea(idea, initial_name=initial_name)
    spec.user_context = " ".join(bit for bit in context_bits if bit)
    spec.intended_outcomes = outcomes
    spec.operating_context = f"Hermes setup preference: {answers.setup_style}; automation comfort: {answers.automation_comfort}."
    spec.target_platforms = answers.platforms or idea.suggested_platforms or ["cli"]
    spec.skill_sets = skill_sets
    spec.boundaries = _boundaries_from_onboarding(answers)
    spec.use_cases = answers.use_cases
    return spec


def _choose_skills(
    skill_sets: dict[str, list[str]],
) -> str:
    """Interactive loop for adding skills from built-in, docs, or local sources.
    Returns BACK_SENTINEL if the user wants to rewind, or "done" to continue.
    """
    r = _single(
        "Add more skills?",
        "Choose a source, or continue.",
        [
            ("builtin", "Add Hermes built-in categories"),
            ("docs", "Add skills from hermes-agent.nousresearch.com/docs/skills"),
            ("local", "Add from installed local skills"),
            ("done", "Done — continue"),
        ],
        default="done",
    )
    while r not in {"done", BACK_SENTINEL}:
        if r == "builtin":
            for cat, opts in SKILL_CATEGORIES.items():
                selected_defaults = skill_sets.get(cat, [])
                picked = _multi(cat, f"Skills for {cat}", opts, default=selected_defaults)
                if picked == BACK_SENTINEL:
                    break
                if isinstance(picked, list):
                    skill_sets = _merge_skill_sets(skill_sets, {cat: picked})
        elif r == "docs":
            for cat, opts in DOCS_SKILL_CATEGORIES.items():
                picked = _multi(cat, "Skills Hub selections", opts, default=skill_sets.get(cat, []))
                if picked == BACK_SENTINEL:
                    break
                if isinstance(picked, list) and picked:
                    skill_sets = _merge_skill_sets(skill_sets, {cat: picked})
        elif r == "local":
            opts = _local_skill_options()
            if not opts:
                console.print(f"[{WARNING}]No local skills found.[/{WARNING}]")
            else:
                picked = _multi(
                    "Installed local skills",
                    "Select skills already available on this machine",
                    opts,
                    default=skill_sets.get("Installed local skills", []),
                )
                if picked == BACK_SENTINEL:
                    break
                if isinstance(picked, list) and picked:
                    skill_sets["Installed local skills"] = list(
                        dict.fromkeys(skill_sets.get("Installed local skills", []) + picked)
                    )
        r = _single(
            "Add more skills?",
            "Choose another source, or continue.",
            [
                ("builtin", "Add Hermes built-in categories"),
                ("docs", "Add skills from hermes-agent.nousresearch.com/docs/skills"),
                ("local", "Add from installed local skills"),
                ("done", "Done — continue"),
            ],
            default="done",
        )
    return str(r)
