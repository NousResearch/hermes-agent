"""Wave 1 archetype presets and canonical resolution helpers."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from types import MappingProxyType
from typing import Any, Final, Mapping

from agent.route_categories import BUILTIN_ROUTE_CATEGORIES, DEFAULT_ROUTE_CATEGORY
from agent.runtime_modes import RUNTIME_MODES_BY_NAME


@dataclass(frozen=True)
class Archetype:
    """Reusable Wave 1 task blueprint."""

    name: str
    summary: str
    default_route_category: str
    default_delegation_profile: str
    default_skills: tuple[str, ...]
    default_required_tools: tuple[str, ...]
    permission_preset: str
    fallback_policy: str
    kind: str = "archetype"


@dataclass(frozen=True)
class SpecialistMapping:
    """Compatibility-safe specialist overlay resolved onto a base archetype."""

    name: str
    archetype_name: str
    default_route_category: str | None = None
    default_delegation_profile: str | None = None
    kind: str = "specialist"


@dataclass(frozen=True)
class NamedWorkflow:
    """Canonical named-workflow taxonomy entry."""

    name: str
    mode: str
    summary: str
    plan: tuple[str, ...]
    acceptance: tuple[str, ...]
    consumption: Mapping[str, Any] | None = None
    default_task_contract: Mapping[str, Any] | None = None
    kind: str = "named_workflow"


DEFAULT_ARCHETYPE_NAME: Final[str] = "generalist"
_SPECIALIST_ALIASES: Final[Mapping[str, str]] = MappingProxyType({"reviewer": "code_reviewer"})


def _normalize_name(value: str | None) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_names(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(value.strip() for value in values if value and value.strip()))
    if not normalized:
        raise ValueError("Archetype defaults must include at least one normalized value")
    return normalized


def _required_archetype_fields() -> tuple[str, ...]:
    return tuple(
        field.name
        for field in fields(Archetype)
        if field.name not in {"name", "summary", "kind"}
        and field.default is MISSING
        and field.default_factory is MISSING
    )


def _make_named_workflow(
    *,
    name: str,
    mode: str,
    summary: str,
    plan: tuple[str, ...],
    acceptance: tuple[str, ...],
    consumption: Mapping[str, Any] | None = None,
    default_task_contract: Mapping[str, Any] | None = None,
) -> NamedWorkflow:
    normalized_name = _require_non_empty_normalized_string(name, "name")
    normalized_mode = _require_non_empty_normalized_string(mode, "mode")
    return NamedWorkflow(
        name=normalized_name,
        mode=normalized_mode,
        summary=summary.strip(),
        plan=_normalize_names(plan),
        acceptance=_normalize_names(acceptance),
        consumption=MappingProxyType(dict(consumption)) if consumption is not None else None,
        default_task_contract=MappingProxyType(dict(default_task_contract)) if default_task_contract is not None else None,
    )


def _require_non_empty_normalized_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must be non-empty")
    if normalized_value != value:
        raise ValueError(f"{field_name} must use canonical normalized form")
    return normalized_value


def _make_archetype(
    *,
    name: str,
    summary: str,
    default_route_category: str,
    default_delegation_profile: str,
    default_skills: tuple[str, ...],
    default_required_tools: tuple[str, ...],
    permission_preset: str,
    fallback_policy: str,
) -> Archetype:
    if default_route_category not in BUILTIN_ROUTE_CATEGORIES:
        raise ValueError(f"Unknown default route category for archetype {name}: {default_route_category}")
    if not default_delegation_profile.strip():
        raise ValueError(f"Archetype {name} requires a default delegation profile")
    return Archetype(
        name=name,
        summary=summary.strip(),
        default_route_category=default_route_category,
        default_delegation_profile=default_delegation_profile.strip(),
        default_skills=_normalize_names(default_skills),
        default_required_tools=_normalize_names(default_required_tools),
        permission_preset=permission_preset.strip(),
        fallback_policy=fallback_policy.strip(),
    )


BUILTIN_ARCHETYPES: Final[tuple[Archetype, ...]] = (
    _make_archetype(
        name="generalist",
        summary="Balanced starter preset for legacy-compatible delegated work without specialist bias.",
        default_route_category=DEFAULT_ROUTE_CATEGORY,
        default_delegation_profile="general",
        default_skills=("general_reasoning", "task_execution"),
        default_required_tools=("read_file", "search_files"),
        permission_preset="inherit",
        fallback_policy="legacy_default_mapping",
    ),
    _make_archetype(
        name="researcher",
        summary="Research-oriented preset for evidence gathering and synthesis without changing route semantics.",
        default_route_category="deep",
        default_delegation_profile="research",
        default_skills=("research", "analysis", "synthesis"),
        default_required_tools=("read_file", "search_files", "web_search", "web_extract"),
        permission_preset="inherit",
        fallback_policy="degrade_to_generalist",
    ),
    _make_archetype(
        name="implementer",
        summary="Implementation-oriented preset for repository changes and local verification.",
        default_route_category="deep",
        default_delegation_profile="implementation",
        default_skills=("implementation", "python", "testing"),
        default_required_tools=("read_file", "search_files", "patch", "terminal"),
        permission_preset="workspace_write",
        fallback_policy="degrade_to_generalist",
    ),
    _make_archetype(
        name="verifier",
        summary="Verification-oriented preset for targeted test runs, review, and evidence capture.",
        default_route_category="quick",
        default_delegation_profile="verification",
        default_skills=("verification", "testing", "review"),
        default_required_tools=("read_file", "search_files", "terminal"),
        permission_preset="workspace_write",
        fallback_policy="degrade_to_generalist",
    ),
)

ARCHETYPES_BY_NAME: Final[Mapping[str, Archetype]] = MappingProxyType(
    {archetype.name: archetype for archetype in BUILTIN_ARCHETYPES}
)

ARCHETYPE_SCHEMA_FIELDS: Final[tuple[str, ...]] = tuple(field.name for field in fields(Archetype))
REQUIRED_ARCHETYPE_FIELDS: Final[tuple[str, ...]] = _required_archetype_fields()

def validate_specialist_mapping(mapping: SpecialistMapping) -> SpecialistMapping:
    if not isinstance(mapping, SpecialistMapping):
        raise ValueError("specialist mapping must be a SpecialistMapping instance")

    _require_non_empty_normalized_string(mapping.name, "name")
    _require_non_empty_normalized_string(mapping.archetype_name, "archetype_name")

    if mapping.kind != "specialist":
        raise ValueError("kind must equal specialist")
    if mapping.name in ARCHETYPES_BY_NAME:
        raise ValueError(f"specialist name collides with archetype: {mapping.name}")
    if mapping.name in BUILTIN_ROUTE_CATEGORIES:
        raise ValueError(f"specialist name collides with route category: {mapping.name}")
    if mapping.name in RUNTIME_MODES_BY_NAME:
        raise ValueError(f"specialist name collides with runtime mode: {mapping.name}")
    if mapping.archetype_name not in ARCHETYPES_BY_NAME:
        raise ValueError(f"Unknown archetype for specialist {mapping.name}: {mapping.archetype_name}")
    if mapping.default_route_category is not None and mapping.default_route_category not in BUILTIN_ROUTE_CATEGORIES:
        raise ValueError(
            f"Unknown default route category for specialist {mapping.name}: {mapping.default_route_category}"
        )
    if mapping.default_delegation_profile is not None and not mapping.default_delegation_profile.strip():
        raise ValueError(f"Specialist {mapping.name} requires a non-empty default delegation profile")
    return mapping


def validate_specialist_mappings(
    mappings: Mapping[str, SpecialistMapping],
) -> Mapping[str, SpecialistMapping]:
    seen_names: set[str] = set()
    for mapping_name, mapping in mappings.items():
        normalized_mapping_name = _require_non_empty_normalized_string(mapping_name, "mapping_name")
        validate_specialist_mapping(mapping)
        if normalized_mapping_name != mapping.name:
            raise ValueError(
                f"specialist mapping registry key must match canonical name: {mapping_name} != {mapping.name}"
            )
        if mapping.name in seen_names:
            raise ValueError(f"duplicate specialist mapping name: {mapping.name}")
        seen_names.add(mapping.name)
    return mappings


SPECIALIST_MAPPINGS_BY_NAME: Final[Mapping[str, SpecialistMapping]] = MappingProxyType(
    validate_specialist_mappings(
        {
            "analyst": SpecialistMapping(
                name="analyst",
                archetype_name="researcher",
                default_route_category="deep",
                default_delegation_profile="research",
            ),
            "investigator": SpecialistMapping(
                name="investigator",
                archetype_name="researcher",
                default_route_category="deep",
                default_delegation_profile="research",
            ),
            "builder": SpecialistMapping(
                name="builder",
                archetype_name="implementer",
                default_route_category="deep",
                default_delegation_profile="implementation",
            ),
            "code_reviewer": SpecialistMapping(
                name="code_reviewer",
                archetype_name="verifier",
                default_route_category="quick",
                default_delegation_profile="verification",
            ),
            "qa_guard": SpecialistMapping(
                name="qa_guard",
                archetype_name="verifier",
                default_route_category="quick",
                default_delegation_profile="verification",
            ),
            "bug_hunter": SpecialistMapping(
                name="bug_hunter",
                archetype_name="implementer",
                default_route_category="deep",
                default_delegation_profile="implementation",
            ),
            "planner": SpecialistMapping(
                name="planner",
                archetype_name="generalist",
                default_route_category="deep",
                default_delegation_profile="general",
            ),
        }
    )
)

BUILTIN_NAMED_WORKFLOWS: Final[tuple[NamedWorkflow, ...]] = (
    _make_named_workflow(
        name="planner",
        mode="plan",
        summary="Structured planning workflow that emits an execution-ready handoff.",
        plan=(
            "capture objective, constraints, and success criteria before execution",
            "decompose work into ordered executable steps with dependencies",
            "emit a structured handoff contract that a deep worker can execute",
        ),
        acceptance=(
            "structured workflow artifact is present",
            "execution handoff remains machine-readable",
        ),
        consumption={
            "downstream_role": "deep_worker",
            "consumes": "execution_task_contract",
        },
        default_task_contract={
            "expected_outcome": "Execution-ready handoff produced from planner workflow activation.",
            "required_skills": ["general_reasoning", "task_execution"],
            "required_tools": ["read_file", "search_files"],
            "must_do": [
                "decompose the work into ordered steps",
                "preserve constraints and verification requirements in structured form",
            ],
            "must_not_do": [
                "do not collapse the workflow into prose-only instructions",
                "do not treat planner as an execution-only label",
            ],
        },
    ),
    _make_named_workflow(
        name="deep_worker",
        mode="execute",
        summary="Execution workflow that consumes a structured handoff contract.",
        plan=(
            "read the structured handoff before acting",
            "execute the contracted work in order",
            "report verification evidence against the contract",
        ),
        acceptance=(
            "structured handoff contract is consumed before execution",
            "execution result reports contract-driven verification evidence",
        ),
        consumption={
            "consumes": "execution_task_contract",
            "task_contract_present": True,
        },
    ),
)

NAMED_WORKFLOWS_BY_NAME: Final[Mapping[str, NamedWorkflow]] = MappingProxyType(
    {workflow.name: workflow for workflow in BUILTIN_NAMED_WORKFLOWS}
)


def list_archetypes() -> tuple[Archetype, ...]:
    """Return built-in archetypes in declaration order."""

    return BUILTIN_ARCHETYPES


def get_default_archetype() -> Archetype:
    """Return the canonical compatibility-safe default archetype."""

    return ARCHETYPES_BY_NAME[DEFAULT_ARCHETYPE_NAME]


def resolve_archetype(name: str | None) -> Archetype:
    """Resolve a canonical archetype name, falling back to the default archetype."""

    normalized_name = _normalize_name(name)
    if not normalized_name:
        return get_default_archetype()
    return ARCHETYPES_BY_NAME.get(normalized_name, get_default_archetype())


def resolve_archetype_defaults(
    archetype_name: str | None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an archetype into the supported default-bearing fields."""

    archetype = resolve_archetype(archetype_name)
    resolved: dict[str, Any] = {
        field_name: list(getattr(archetype, field_name))
        if field_name in {"default_skills", "default_required_tools"}
        else getattr(archetype, field_name)
        for field_name in REQUIRED_ARCHETYPE_FIELDS
    }
    if not overrides:
        return resolved

    for field_name in REQUIRED_ARCHETYPE_FIELDS:
        if field_name not in overrides or overrides[field_name] is None:
            continue
        value = overrides[field_name]
        if field_name in {"default_skills", "default_required_tools"}:
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{field_name} overrides must be a list or tuple")
            resolved[field_name] = [str(item).strip() for item in value if str(item).strip()]
            continue
        resolved[field_name] = str(value).strip()
    return resolved


def resolve_specialist_mapping(name: str | None) -> SpecialistMapping | None:
    """Resolve a canonical specialist overlay name."""

    normalized_name = _normalize_name(name)
    if not normalized_name:
        return None
    canonical_name = _SPECIALIST_ALIASES.get(normalized_name, normalized_name)
    return SPECIALIST_MAPPINGS_BY_NAME.get(canonical_name)


def resolve_specialist_defaults(name: str | None) -> dict[str, Any]:
    """Return the default-bearing overlay fields for a specialist."""

    specialist = resolve_specialist_mapping(name)
    if specialist is None:
        return {}
    resolved: dict[str, Any] = {}
    if specialist.default_route_category:
        resolved["default_route_category"] = specialist.default_route_category
    if specialist.default_delegation_profile:
        resolved["default_delegation_profile"] = specialist.default_delegation_profile
    return resolved


def resolve_named_workflow(name: str | None) -> NamedWorkflow | None:
    """Resolve a canonical named-workflow taxonomy entry."""

    normalized_name = _normalize_name(name)
    if not normalized_name:
        return None
    return NAMED_WORKFLOWS_BY_NAME.get(normalized_name)


__all__ = [
    "ARCHETYPES_BY_NAME",
    "ARCHETYPE_SCHEMA_FIELDS",
    "BUILTIN_ARCHETYPES",
    "BUILTIN_NAMED_WORKFLOWS",
    "DEFAULT_ARCHETYPE_NAME",
    "NAMED_WORKFLOWS_BY_NAME",
    "NamedWorkflow",
    "REQUIRED_ARCHETYPE_FIELDS",
    "SPECIALIST_MAPPINGS_BY_NAME",
    "Archetype",
    "SpecialistMapping",
    "get_default_archetype",
    "list_archetypes",
    "resolve_archetype",
    "resolve_archetype_defaults",
    "resolve_named_workflow",
    "resolve_specialist_defaults",
    "resolve_specialist_mapping",
    "validate_specialist_mapping",
    "validate_specialist_mappings",
]
