from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SubagentProfile:
    id: str
    description: str
    default_toolsets: tuple[str, ...] = ()
    preferred_skill_names: tuple[str, ...] = ()
    preferred_skill_categories: tuple[str, ...] = ()
    preferred_tags: tuple[str, ...] = ()
    excluded_skill_names: tuple[str, ...] = ()
    excluded_skill_categories: tuple[str, ...] = ()
    gstack_skill_hints: tuple[str, ...] = ()
    prompt_preamble: str = ""
    runtime_hints: dict[str, Any] = field(default_factory=dict)
    review_mesh_specialists: tuple[str, ...] = ()


_DEFAULT_PROFILE_ID = "builder"

_PROFILE_ALIASES = {
    "default": _DEFAULT_PROFILE_ID,
    "general": _DEFAULT_PROFILE_ID,
    "generalist": _DEFAULT_PROFILE_ID,
    "research": "researcher",
    "implementer": "builder",
    "coder": "builder",
    "engineer": "builder",
    "review": "reviewer",
    "qa": "reviewer",
    "testing": "reviewer",
    "browser": "browser_scout",
    "scout": "browser_scout",
    "archive": "archivist",
    "ops": "operator",
    "security": "security_reviewer",
    "security-review": "security_reviewer",
    "performance": "performance_reviewer",
    "perf": "performance_reviewer",
    "maintainability": "maintainability_reviewer",
    "maint": "maintainability_reviewer",
    "redteam": "red_team",
}


def _profile(
    profile_id: str,
    description: str,
    *,
    default_toolsets: Sequence[str] = (),
    preferred_skill_names: Sequence[str] = (),
    preferred_skill_categories: Sequence[str] = (),
    preferred_tags: Sequence[str] = (),
    excluded_skill_names: Sequence[str] = (),
    excluded_skill_categories: Sequence[str] = (),
    gstack_skill_hints: Sequence[str] = (),
    prompt_preamble: str = "",
    runtime_hints: Optional[Mapping[str, Any]] = None,
    review_mesh_specialists: Sequence[str] = (),
) -> SubagentProfile:
    return SubagentProfile(
        id=profile_id,
        description=description,
        default_toolsets=tuple(default_toolsets),
        preferred_skill_names=tuple(preferred_skill_names),
        preferred_skill_categories=tuple(preferred_skill_categories),
        preferred_tags=tuple(preferred_tags),
        excluded_skill_names=tuple(excluded_skill_names),
        excluded_skill_categories=tuple(excluded_skill_categories),
        gstack_skill_hints=tuple(gstack_skill_hints),
        prompt_preamble=prompt_preamble,
        runtime_hints=dict(runtime_hints or {}),
        review_mesh_specialists=tuple(review_mesh_specialists),
    )


_SUBAGENT_PROFILES: dict[str, SubagentProfile] = {
    "researcher": _profile(
        "researcher",
        "Investigation-focused subagent for synthesis, source gathering, and comparative analysis.",
        default_toolsets=("web", "file", "terminal"),
        preferred_skill_categories=("research", "analysis"),
        preferred_tags=("research", "synthesis", "evidence"),
        gstack_skill_hints=("gstack-investigate", "gstack-browse", "gstack-design-consultation"),
        prompt_preamble="Bias toward evidence-backed findings, concise synthesis, and explicit uncertainty.",
        runtime_hints={"style": "investigative", "gstack_affinity": "medium"},
    ),
    "builder": _profile(
        "builder",
        "Implementation-focused subagent for coding, debugging, and shipping concrete changes.",
        default_toolsets=("terminal", "file"),
        preferred_skill_categories=("coding", "debugging", "testing"),
        preferred_tags=("implementation", "debugging", "patching"),
        gstack_skill_hints=("gstack-careful", "gstack-review", "gstack-qa"),
        prompt_preamble="Prefer direct execution, scoped edits, and verification with targeted tests.",
        runtime_hints={"style": "implementation", "gstack_affinity": "medium"},
    ),
    "reviewer": _profile(
        "reviewer",
        "Review-focused subagent for audits, code review, regression spotting, and risk identification.",
        default_toolsets=("terminal", "file"),
        preferred_skill_categories=("review", "testing", "quality"),
        preferred_tags=("review", "audit", "regression"),
        gstack_skill_hints=("gstack-review", "gstack-plan-eng-review", "gstack-qa"),
        prompt_preamble="Act like a specialist reviewer: inspect claims, surface risks, and rank issues clearly.",
        runtime_hints={"style": "review", "gstack_affinity": "high"},
        review_mesh_specialists=("testing",),
    ),
    "operator": _profile(
        "operator",
        "Operations-focused subagent for deploy, release, incident, and procedural execution tasks.",
        default_toolsets=("terminal", "file", "web"),
        preferred_skill_categories=("operations", "release"),
        preferred_tags=("deploy", "runbook", "incident"),
        gstack_skill_hints=("gstack-ship", "gstack-land-and-deploy", "gstack-document-release"),
        prompt_preamble="Think operationally: protect state, verify commands, and leave a crisp execution summary.",
        runtime_hints={"style": "operations", "gstack_affinity": "high"},
    ),
    "browser_scout": _profile(
        "browser_scout",
        "Browser-first subagent for interactive web discovery, reconnaissance, and UI checks.",
        default_toolsets=("browser", "web", "file"),
        preferred_skill_categories=("browser", "research"),
        preferred_tags=("browser", "navigation", "recon"),
        gstack_skill_hints=("gstack-browse", "gstack-canary", "gstack-setup-browser-cookies"),
        prompt_preamble="Prefer browser-native evidence gathering and capture concrete URLs, states, and screenshots when relevant.",
        runtime_hints={"style": "browser", "gstack_affinity": "high"},
    ),
    "archivist": _profile(
        "archivist",
        "Knowledge-curation subagent for organizing notes, extracting durable facts, and preserving provenance.",
        default_toolsets=("file", "web"),
        preferred_skill_categories=("documentation", "knowledge"),
        preferred_tags=("notes", "curation", "provenance"),
        gstack_skill_hints=("gstack-document-release", "gstack-write-prd"),
        prompt_preamble="Optimize for structured capture, provenance, and clean handoff artifacts.",
        runtime_hints={"style": "archival", "gstack_affinity": "low"},
    ),
    "security_reviewer": _profile(
        "security_reviewer",
        "Security-focused reviewer for auth, secrets, trust boundaries, and exploit paths.",
        default_toolsets=("terminal", "file"),
        preferred_skill_categories=("security", "review"),
        preferred_tags=("security", "auth", "threat-model"),
        gstack_skill_hints=("gstack-review", "gstack-qa"),
        prompt_preamble="Adopt an adversarial but evidence-based security review posture.",
        runtime_hints={"style": "security_review", "gstack_affinity": "high"},
        review_mesh_specialists=("security",),
    ),
    "performance_reviewer": _profile(
        "performance_reviewer",
        "Performance-focused reviewer for latency, throughput, hot paths, and resource efficiency.",
        default_toolsets=("terminal", "file"),
        preferred_skill_categories=("performance", "review"),
        preferred_tags=("performance", "latency", "profiling"),
        gstack_skill_hints=("gstack-review", "gstack-qa"),
        prompt_preamble="Focus on measurable bottlenecks, scaling risks, and verification plans.",
        runtime_hints={"style": "performance_review", "gstack_affinity": "medium"},
        review_mesh_specialists=("performance",),
    ),
    "maintainability_reviewer": _profile(
        "maintainability_reviewer",
        "Maintainability-focused reviewer for code health, clarity, architecture, and long-term operability.",
        default_toolsets=("terminal", "file"),
        preferred_skill_categories=("maintainability", "review"),
        preferred_tags=("maintainability", "readability", "architecture"),
        gstack_skill_hints=("gstack-review", "gstack-plan-eng-review"),
        prompt_preamble="Prioritize clarity, cohesion, and future change safety over cleverness.",
        runtime_hints={"style": "maintainability_review", "gstack_affinity": "medium"},
        review_mesh_specialists=("maintainability",),
    ),
    "red_team": _profile(
        "red_team",
        "Adversarial subagent for abuse-case discovery, exploit thinking, and failure-mode probing.",
        default_toolsets=("terminal", "file", "web"),
        preferred_skill_categories=("security", "adversarial"),
        preferred_tags=("red-team", "abuse-case", "failure-mode"),
        gstack_skill_hints=("gstack-review", "gstack-canary"),
        prompt_preamble="Think like an attacker and enumerate realistic exploit or misuse paths.",
        runtime_hints={"style": "adversarial", "gstack_affinity": "medium"},
        review_mesh_specialists=("red_team",),
    ),
}

DEFAULT_SUBAGENT_PROFILE = _SUBAGENT_PROFILES[_DEFAULT_PROFILE_ID]


def canonicalize_subagent_profile_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    return _PROFILE_ALIASES.get(normalized, normalized)


def list_subagent_profiles() -> list[SubagentProfile]:
    return [
        _SUBAGENT_PROFILES[name]
        for name in sorted(_SUBAGENT_PROFILES)
    ]


def get_subagent_profile(name: Optional[str]) -> SubagentProfile:
    canonical = canonicalize_subagent_profile_name(name)
    if canonical and canonical in _SUBAGENT_PROFILES:
        return _SUBAGENT_PROFILES[canonical]
    return DEFAULT_SUBAGENT_PROFILE


def get_review_mesh_specialist_profile(specialist: Optional[str]) -> SubagentProfile:
    canonical_specialist = canonicalize_subagent_profile_name(specialist)
    if canonical_specialist and canonical_specialist in _SUBAGENT_PROFILES:
        return _SUBAGENT_PROFILES[canonical_specialist]
    for profile in _SUBAGENT_PROFILES.values():
        if specialist in profile.review_mesh_specialists:
            return profile
    if specialist == "testing":
        return _SUBAGENT_PROFILES["reviewer"]
    return DEFAULT_SUBAGENT_PROFILE


def apply_subagent_profile_overrides(
    profile: SubagentProfile,
    overrides: Optional[Mapping[str, Any]] = None,
) -> SubagentProfile:
    if not overrides:
        return profile

    def _coerce_seq(field_name: str) -> tuple[str, ...]:
        value = overrides.get(field_name)
        if value is None:
            return getattr(profile, field_name)
        if isinstance(value, str):
            items = [value]
        else:
            items = list(value)
        return tuple(str(item).strip() for item in items if str(item).strip())

    runtime_hints = dict(profile.runtime_hints)
    raw_runtime_hints = overrides.get("runtime_hints")
    if isinstance(raw_runtime_hints, Mapping):
        runtime_hints.update(raw_runtime_hints)

    return replace(
        profile,
        description=str(overrides.get("description", profile.description)).strip() or profile.description,
        default_toolsets=_coerce_seq("default_toolsets"),
        preferred_skill_names=_coerce_seq("preferred_skill_names"),
        preferred_skill_categories=_coerce_seq("preferred_skill_categories"),
        preferred_tags=_coerce_seq("preferred_tags"),
        excluded_skill_names=_coerce_seq("excluded_skill_names"),
        excluded_skill_categories=_coerce_seq("excluded_skill_categories"),
        gstack_skill_hints=_coerce_seq("gstack_skill_hints"),
        prompt_preamble=str(overrides.get("prompt_preamble", profile.prompt_preamble)).strip() or profile.prompt_preamble,
        runtime_hints=runtime_hints,
        review_mesh_specialists=_coerce_seq("review_mesh_specialists"),
    )


def resolve_subagent_profile(
    role_hint: Optional[str] = None,
    toolsets: Optional[Sequence[str]] = None,
    goal: Optional[str] = None,
    context: Optional[str] = None,
) -> SubagentProfile:
    canonical = canonicalize_subagent_profile_name(role_hint)
    if canonical and canonical in _SUBAGENT_PROFILES:
        return _SUBAGENT_PROFILES[canonical]

    combined_text = "\n".join(part for part in (goal, context) if part).lower()
    toolset_set = {str(toolset).strip().lower() for toolset in (toolsets or []) if str(toolset).strip()}

    if {"browser"} & toolset_set:
        return _SUBAGENT_PROFILES["browser_scout"]
    if any(token in combined_text for token in ("red team", "red-team", "attack", "abuse case", "exploit")):
        return _SUBAGENT_PROFILES["red_team"]
    if any(token in combined_text for token in ("security", "auth", "permission", "secret", "token", "vulnerability")):
        return _SUBAGENT_PROFILES["security_reviewer"]
    if any(token in combined_text for token in ("performance", "latency", "throughput", "hot path", "optimiz", "slow")):
        return _SUBAGENT_PROFILES["performance_reviewer"]
    if any(token in combined_text for token in ("maintainability", "readability", "refactor", "architecture review", "code health")):
        return _SUBAGENT_PROFILES["maintainability_reviewer"]
    if any(token in combined_text for token in ("review", "audit", "regression", "qa", "test coverage")):
        return _SUBAGENT_PROFILES["reviewer"]
    if any(token in combined_text for token in ("research", "investigate", "compare", "analyze", "analysis")):
        return _SUBAGENT_PROFILES["researcher"]
    if any(token in combined_text for token in ("deploy", "release", "ship", "incident", "runbook", "operate")):
        return _SUBAGENT_PROFILES["operator"]
    if any(token in combined_text for token in ("archive", "curate", "document", "summarize notes", "catalog")):
        return _SUBAGENT_PROFILES["archivist"]
    return DEFAULT_SUBAGENT_PROFILE
