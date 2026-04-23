"""Wave 2 intent preclassifier for top-level runtime/specialist activation.

This module is intentionally additive on top of the Wave 1 schema and overlay
work. It does not mutate prompt assembly or delegation plumbing. Instead it
provides a deterministic, compatibility-safe preclassification step that can be
consumed by higher-level runtime activation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agent.archetypes import resolve_archetype, resolve_specialist_defaults, resolve_specialist_mapping
from agent.route_categories import (
    DEFAULT_ROUTE_CATEGORY,
    BUILTIN_ROUTE_CATEGORIES,
    resolve_literal_category,
    resolve_literal_category_from_route_category,
    resolve_literal_category_to_route_category,
)
from agent.runtime_modes import get_default_runtime_mode, resolve_runtime_mode
from agent.task_contracts import TaskContract, validate_task_contract

_DEFAULT_ARCHETYPE = resolve_archetype(None)
_DEFAULT_RUNTIME_MODE = get_default_runtime_mode()


@dataclass(frozen=True)
class IntentPreclassification:
    inferred_archetype: str
    inferred_specialist: str | None
    inferred_category: str
    inferred_route_category: str
    inferred_runtime_mode: str
    inferred_delegation_profile: str | None
    activation_reason: str
    task_contract: TaskContract | None = None
    inference_source: str = "wave2_intent_preclassifier"


@dataclass(frozen=True)
class _KeywordRule:
    label: str
    keywords: tuple[str, ...]
    score: int


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _extract_text_and_contract(
    payload: str | Mapping[str, Any] | None,
    *,
    task_contract: dict[str, Any] | TaskContract | None = None,
) -> tuple[str, TaskContract | None]:
    resolved_contract = validate_task_contract(task_contract) if task_contract is not None else None
    if payload is None:
        return "", resolved_contract
    if isinstance(payload, str):
        return _normalize_text(payload), resolved_contract
    if not isinstance(payload, Mapping):
        return _normalize_text(payload), resolved_contract

    contract_payload = payload.get("task_contract")
    if resolved_contract is None and contract_payload is not None:
        resolved_contract = validate_task_contract(contract_payload)

    text_parts: list[str] = []
    for key in ("message", "user_message", "prompt", "task", "summary", "content"):
        value = payload.get(key)
        if value:
            text_parts.append(str(value))

    if resolved_contract is not None:
        text_parts.extend(
            [
                resolved_contract.task,
                resolved_contract.expected_outcome,
                " ".join(resolved_contract.required_skills),
                " ".join(resolved_contract.required_tools),
            ]
        )

    return _normalize_text(" ".join(text_parts)), resolved_contract


_SPECIALIST_RULES: tuple[_KeywordRule, ...] = (
    _KeywordRule(
        label="multimodal_specialist",
        keywords=(
            "image",
            "images",
            "screenshot",
            "screenshots",
            "diagram",
            "diagrams",
            "chart",
            "charts",
            "figure",
            "figures",
            "pdf",
            "slide deck",
            "slides",
            "visual",
        ),
        score=9,
    ),
    _KeywordRule(
        label="code_reviewer",
        keywords=(
            "code review",
            "review this patch",
            "review the patch",
            "review this diff",
            "review the diff",
            "review the code",
            "pr review",
            "pull request",
            "audit the patch",
        ),
        score=8,
    ),
    _KeywordRule(
        label="qa_guard",
        keywords=(
            "qa",
            "regression",
            "acceptance",
            "smoke test",
            "release confidence",
            "validate the fix",
        ),
        score=7,
    ),
    _KeywordRule(
        label="bug_hunter",
        keywords=(
            "debug",
            "bug",
            "reproduce",
            "root cause",
            "trace the failure",
        ),
        score=7,
    ),
    _KeywordRule(
        label="builder",
        keywords=(
            "implement",
            "implementation",
            "code",
            "patch",
            "refactor",
            "build the feature",
        ),
        score=6,
    ),
    _KeywordRule(
        label="investigator",
        keywords=(
            "research",
            "investigate",
            "investigation",
            "triage",
            "gather evidence",
            "trace the issue",
        ),
        score=6,
    ),
    _KeywordRule(
        label="analyst",
        keywords=(
            "analyze",
            "analysis",
            "compare",
            "synthesize",
            "findings",
        ),
        score=5,
    ),
    _KeywordRule(
        label="writer",
        keywords=(
            "write",
            "draft",
            "edit",
            "revise",
            "rewrite",
        ),
        score=5,
    ),
    _KeywordRule(
        label="planner",
        keywords=(
            "plan",
            "planning",
            "decompose",
            "sequence the work",
            "execution plan",
        ),
        score=5,
    ),
)


_ARCHETYPE_RULES: tuple[_KeywordRule, ...] = (
    _KeywordRule(
        label="implementer",
        keywords=(
            "implement",
            "implementation",
            "code",
            "patch",
            "fix",
            "refactor",
            "bug",
            "repo",
            "repository",
            "pytest",
            "test",
            "tests",
        ),
        score=3,
    ),
    _KeywordRule(
        label="researcher",
        keywords=(
            "research",
            "investigate",
            "analyze",
            "analysis",
            "compare",
            "sources",
            "findings",
            "synthesize",
            "synthesis",
        ),
        score=3,
    ),
    _KeywordRule(
        label="verifier",
        keywords=(
            "verify",
            "verification",
            "validate",
            "review",
            "audit",
            "confirm",
            "check",
            "smoke test",
        ),
        score=2,
    ),
)

_ROUTE_RULES: tuple[_KeywordRule, ...] = (
    _KeywordRule(
        "visual",
        (
            "image",
            "images",
            "screenshot",
            "screenshots",
            "visual",
            "diagram",
            "diagrams",
            "chart",
            "charts",
            "figure",
            "figures",
            "pdf",
            "slides",
            "slide deck",
        ),
        5,
    ),
    _KeywordRule("writing", ("write", "draft", "edit", "copy", "email", "blog", "documentation"), 4),
    _KeywordRule(
        "ultrabrain",
        ("ultrabrain", "deep reasoning", "hardest", "prove", "proof", "formal", "rigorous"),
        6,
    ),
    _KeywordRule(
        "deep",
        (
            "research",
            "investigate",
            "analyze",
            "analysis",
            "implement",
            "implementation",
            "architecture",
            "multi-step",
            "complex",
            "careful",
            "deep lane",
        ),
        3,
    ),
    _KeywordRule("quick", ("quick", "fast", "brief", "simple", "small", "lightweight"), 2),
)

_RUNTIME_RULES: tuple[_KeywordRule, ...] = (
    _KeywordRule("ultrawork", ("ultrawork", "ulw"), 8),
    _KeywordRule("ralph", ("ralph", "ralph-loop"), 7),
    _KeywordRule(
        "interview_planning",
        ("interview", "interview prep", "interview preparation", "mock interview"),
        6,
    ),
    _KeywordRule(
        "execution_supervisor",
        ("supervise", "supervisor", "oversee", "oversight", "coordinate", "orchestrate"),
        5,
    ),
)


def _score_rules(text: str, rules: tuple[_KeywordRule, ...]) -> dict[str, int]:
    scores: dict[str, int] = {rule.label: 0 for rule in rules}
    for rule in rules:
        for keyword in rule.keywords:
            if keyword in text:
                scores[rule.label] += rule.score
    return scores


def _pick_label(text: str, rules: tuple[_KeywordRule, ...], default: str) -> str:
    scores = _score_rules(text, rules)
    best_label = default
    best_score = 0
    for rule in rules:
        score = scores[rule.label]
        if score > best_score:
            best_label = rule.label
            best_score = score
    return best_label if best_score > 0 else default


def _infer_specialist(text: str) -> str | None:
    specialist = _pick_label(text, _SPECIALIST_RULES, "")
    if not specialist:
        return None
    resolved_specialist = resolve_specialist_mapping(specialist)
    return resolved_specialist.name if resolved_specialist is not None else None


def _infer_archetype(text: str, inferred_specialist: str | None = None) -> str:
    if inferred_specialist:
        specialist_mapping = resolve_specialist_mapping(inferred_specialist)
        if specialist_mapping is not None:
            return specialist_mapping.archetype_name
    return _pick_label(text, _ARCHETYPE_RULES, _DEFAULT_ARCHETYPE.name)


def _infer_route_category(text: str, inferred_archetype: str) -> str:
    route = _pick_label(text, _ROUTE_RULES, "")
    if route:
        return route
    if inferred_archetype in {"implementer", "researcher"}:
        return "deep"
    if inferred_archetype == "verifier":
        return "quick"
    return DEFAULT_ROUTE_CATEGORY


def _infer_category(
    *,
    inferred_route_category: str,
    explicit_category: str | None = None,
) -> str:
    if explicit_category:
        resolved_literal_category = resolve_literal_category(explicit_category)
        resolved_route_from_category = resolve_literal_category_to_route_category(resolved_literal_category.name)
        if resolved_route_from_category.name == inferred_route_category:
            return resolved_literal_category.name
    return resolve_literal_category_from_route_category(inferred_route_category).name


def _apply_specialist_overlay_defaults(
    inferred_specialist: str | None,
    *,
    inferred_route_category: str,
    inferred_delegation_profile: str | None,
) -> tuple[str, str | None]:
    if not inferred_specialist:
        return inferred_route_category, inferred_delegation_profile

    specialist_defaults = resolve_specialist_defaults(inferred_specialist)
    if not specialist_defaults:
        return inferred_route_category, inferred_delegation_profile

    route_category = inferred_route_category
    if route_category == DEFAULT_ROUTE_CATEGORY:
        overlay_route_category = str(specialist_defaults.get("default_route_category") or "").strip()
        if overlay_route_category:
            route_category = overlay_route_category

    delegation_profile = inferred_delegation_profile
    if not delegation_profile:
        overlay_profile = str(specialist_defaults.get("default_delegation_profile") or "").strip()
        if overlay_profile:
            delegation_profile = overlay_profile

    return route_category, delegation_profile


def _infer_runtime_mode(text: str, task_contract: TaskContract | None = None) -> str:
    if task_contract is not None and isinstance(task_contract.context, Mapping):
        command_runtime = task_contract.context.get("command_runtime")
        if isinstance(command_runtime, Mapping):
            runtime_mode = resolve_runtime_mode(command_runtime.get("runtime_mode")).name
            if runtime_mode != _DEFAULT_RUNTIME_MODE.name:
                return runtime_mode

        command_name = _normalize_text(task_contract.context.get("command"))
        if command_name == "ralph-loop":
            return "ralph"
        if command_name == "ulw-loop":
            return "ultrawork"

    explicit = _pick_label(text, _RUNTIME_RULES, "")
    if explicit:
        return resolve_runtime_mode(explicit).name
    return _DEFAULT_RUNTIME_MODE.name


def _infer_delegation_profile(inferred_archetype: str) -> str | None:
    archetype = resolve_archetype(inferred_archetype)
    return archetype.default_delegation_profile or None


def _build_activation_reason(
    *,
    raw_text: str,
    inferred_archetype: str,
    inferred_specialist: str | None,
    inferred_category: str,
    inferred_route_category: str,
    inferred_runtime_mode: str,
    inferred_delegation_profile: str | None,
) -> str:
    reasons: list[str] = []
    if not raw_text.strip() or raw_text.strip() in {"help", "hi", "hello"}:
        reasons.append("fallback: underspecified top-level intent")
    if "ultrawork" in raw_text:
        reasons.append("explicit runtime keyword: ultrawork")
    elif "ulw" in raw_text:
        reasons.append("explicit runtime keyword: ulw -> ultrawork")
    if "ralph-loop" in raw_text or "ralph" in raw_text:
        reasons.append("explicit runtime keyword: ralph -> ralph")
    if inferred_runtime_mode == "interview_planning" and "interview" in raw_text:
        reasons.append("explicit runtime keyword: interview")
    if inferred_runtime_mode == "execution_supervisor" and any(
        marker in raw_text for marker in ("supervise", "supervisor", "oversee", "oversight", "coordinate", "orchestrate")
    ):
        reasons.append("explicit runtime keyword: supervision/orchestration")
    if not reasons and inferred_specialist is not None:
        reasons.append(f"keyword-derived specialist: {inferred_specialist}")
    if not reasons and inferred_archetype != _DEFAULT_ARCHETYPE.name:
        reasons.append(f"keyword-derived archetype: {inferred_archetype}")
    if not reasons and inferred_route_category != DEFAULT_ROUTE_CATEGORY:
        reasons.append(f"keyword-derived route_category: {inferred_route_category}")
    if not reasons:
        reasons.append("fallback: compatibility-safe default activation")

    reasons.append(
        "resolved outputs: "
        f"archetype={inferred_archetype}, "
        f"specialist={inferred_specialist or 'none'}, "
        f"category={inferred_category}, "
        f"route_category={inferred_route_category}, "
        f"delegation_profile={inferred_delegation_profile or 'none'}, "
        f"runtime_mode={inferred_runtime_mode}"
    )
    return "; ".join(reasons)


def preclassify_intent(
    payload: str | Mapping[str, Any] | None,
    *,
    task_contract: dict[str, Any] | TaskContract | None = None,
) -> IntentPreclassification:
    """Deterministically classify top-level work into Wave 2 activation outputs.

    The classifier is intentionally lightweight and compatibility-safe:
    - unknown or underspecified input falls back to Wave 1-compatible defaults
    - runtime mode, route category, delegation profile, and archetype remain
      distinct output layers
    - any provided task contract is preserved in structured form
    """

    text, resolved_task_contract = _extract_text_and_contract(payload, task_contract=task_contract)
    explicit_category = None
    if isinstance(payload, Mapping):
        explicit_category = str(payload.get("category") or "").strip() or None

    inferred_specialist = _infer_specialist(text)
    inferred_archetype = _infer_archetype(text, inferred_specialist)
    inferred_route_category = _infer_route_category(text, inferred_archetype)
    inferred_runtime_mode = _infer_runtime_mode(text, resolved_task_contract)
    inferred_delegation_profile = _infer_delegation_profile(inferred_archetype)
    inferred_route_category, inferred_delegation_profile = _apply_specialist_overlay_defaults(
        inferred_specialist,
        inferred_route_category=inferred_route_category,
        inferred_delegation_profile=inferred_delegation_profile,
    )
    inferred_category = _infer_category(
        inferred_route_category=inferred_route_category,
        explicit_category=explicit_category,
    )

    if inferred_route_category not in BUILTIN_ROUTE_CATEGORIES:
        inferred_route_category = DEFAULT_ROUTE_CATEGORY
    inferred_runtime_mode = resolve_runtime_mode(inferred_runtime_mode).name

    activation_reason = _build_activation_reason(
        raw_text=text,
        inferred_archetype=inferred_archetype,
        inferred_specialist=inferred_specialist,
        inferred_category=inferred_category,
        inferred_route_category=inferred_route_category,
        inferred_runtime_mode=inferred_runtime_mode,
        inferred_delegation_profile=inferred_delegation_profile,
    )

    return IntentPreclassification(
        inferred_archetype=inferred_archetype,
        inferred_specialist=inferred_specialist,
        inferred_category=inferred_category,
        inferred_route_category=inferred_route_category,
        inferred_runtime_mode=inferred_runtime_mode,
        inferred_delegation_profile=inferred_delegation_profile,
        activation_reason=activation_reason,
        task_contract=resolved_task_contract,
    )


__all__ = ["IntentPreclassification", "preclassify_intent"]
