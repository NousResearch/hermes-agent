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
    blocked_tools: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    kind: str = "archetype"


@dataclass(frozen=True)
class SpecialistMapping:
    """Compatibility-safe specialist overlay resolved onto a base archetype."""

    name: str
    archetype_name: str
    default_route_category: str | None = None
    default_delegation_profile: str | None = None
    blocked_tools: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
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


@dataclass(frozen=True)
class NamedAgentContract:
    """Canonical normalized runtime contract for a named agent."""

    name: str
    role: str
    archetype: str
    specialist: str | None
    mode: str
    color: str | None
    category: str
    route_category: str
    provider: str | None
    model: str | None
    fallback_models: tuple[Mapping[str, Any], ...]
    providerOptions: Mapping[str, Any] | None
    ultrawork: Mapping[str, str] | None
    allowed_tools: tuple[str, ...]
    blocked_tools: tuple[str, ...]
    permissions: Mapping[str, Any]
    description: str
    safe_claim_text: str
    aliases: tuple[str, ...] = ()
    kind: str = "named_agent"


DEFAULT_ARCHETYPE_NAME: Final[str] = "generalist"
NAMED_AGENT_PERMISSION_KEYS: Final[tuple[str, ...]] = (
    "edit",
    "bash",
    "webfetch",
    "doom_loop",
    "external_directory",
)
NAMED_AGENT_PERMISSION_VALUES: Final[frozenset[str]] = frozenset({"allow", "ask", "deny"})
NAMED_AGENT_MODES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "primary": "primary",
        "subagent": "subagent-only",
        "subagent_only": "subagent-only",
        "subagent-only": "subagent-only",
        "disabled": "disabled",
    }
)
_SPECIALIST_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "reviewer": "code_reviewer",
        "multimodal": "multimodal_specialist",
        "vision": "multimodal_specialist",
        "visual": "multimodal_specialist",
        "oracle": "consultant",
    }
)


def _normalize_name(value: str | None) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_names(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(value.strip() for value in values if value and value.strip()))
    if not normalized:
        raise ValueError("Archetype defaults must include at least one normalized value")
    return normalized


def _normalize_optional_names(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def _normalize_named_string_list(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return _normalize_optional_names((values,))
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    return _normalize_optional_names(tuple(str(value or "") for value in values))


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


def _normalize_named_agent_name(value: str, field_name: str = "name") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized_value = value.strip().lower()
    if not normalized_value:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized_value


def _normalize_named_agent_mode(value: Any, *, agent_name: str) -> str:
    normalized_value = _normalize_name(str(value or ""))
    mode = NAMED_AGENT_MODES.get(normalized_value)
    if mode is None:
        raise ValueError(
            f"Named agent '{agent_name}' has invalid mode '{value}'. Expected one of: primary | subagent-only | disabled"
        )
    return mode


def _normalize_named_agent_fallback_models(value: Any, *, agent_name: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    entries = value if isinstance(value, list) else [value]
    normalized: list[Mapping[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            model = entry.strip()
            if model:
                normalized.append(MappingProxyType({"model": model}))
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Named agent '{agent_name}' fallback_models entries must be strings or mappings")
        model = str(entry.get("model") or "").strip()
        if not model:
            raise ValueError(f"Named agent '{agent_name}' fallback_models entries must include model")
        normalized_entry: dict[str, Any] = {"model": model}
        for key in ("provider", "variant", "reasoningEffort", "temperature", "top_p", "maxTokens", "thinking"):
            if key not in entry or entry.get(key) in (None, ""):
                continue
            normalized_entry[key] = entry.get(key)
        normalized.append(MappingProxyType(normalized_entry))
    return tuple(normalized)


def _normalize_named_agent_permissions(value: Any, *, agent_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, dict):
        raise ValueError(f"Named agent '{agent_name}' permission surface must be a mapping")

    normalized: dict[str, Any] = {}
    for key, raw in value.items():
        if key not in NAMED_AGENT_PERMISSION_KEYS:
            raise ValueError(
                f"Named agent '{agent_name}' permission.{key} is not supported; expected keys: {', '.join(NAMED_AGENT_PERMISSION_KEYS)}"
            )
        if key == "bash" and isinstance(raw, Mapping):
            per_command: dict[str, str] = {}
            for raw_command, raw_permission in raw.items():
                command = str(raw_command or "").strip()
                permission_value = str(raw_permission or "").strip().lower()
                if permission_value not in NAMED_AGENT_PERMISSION_VALUES:
                    raise ValueError(
                        f"Named agent '{agent_name}' permission.bash[{command}] must be one of allow, ask, deny"
                    )
                if command:
                    per_command[command] = permission_value
            normalized[key] = MappingProxyType(per_command)
            continue
        permission_value = str(raw or "").strip().lower()
        if permission_value not in NAMED_AGENT_PERMISSION_VALUES:
            raise ValueError(
                f"Named agent '{agent_name}' permission.{key} must be one of allow, ask, deny"
            )
        normalized[key] = permission_value
    return MappingProxyType(normalized)


def _normalize_provider_options(value: Any, *, agent_name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Named agent '{agent_name}' providerOptions must be a mapping")
    return MappingProxyType(dict(value))


def _normalize_ultrawork(value: Any, *, agent_name: str) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Named agent '{agent_name}' ultrawork override must be a mapping")
    normalized = {
        key: str(value.get(key) or "").strip()
        for key in ("model", "variant")
        if str(value.get(key) or "").strip()
    }
    if not normalized:
        return None
    return MappingProxyType(normalized)


def _build_named_agent_description(name: str, archetype: str, specialist: str | None, mode: str) -> str:
    if specialist:
        return f"Named agent '{name}' runs as specialist '{specialist}' on archetype '{archetype}' in mode '{mode}'."
    return f"Named agent '{name}' runs on archetype '{archetype}' in mode '{mode}'."


def _build_named_agent_safe_claim_text(name: str, archetype: str, specialist: str | None, mode: str) -> str:
    identity = f"specialist '{specialist}' on archetype '{archetype}'" if specialist else f"archetype '{archetype}'"
    return f"Configured named agent '{name}' is limited to its declared {identity} contract and mode '{mode}'."


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
    blocked_tools: tuple[str, ...] = (),
    allowed_tools: tuple[str, ...] = (),
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
        blocked_tools=_normalize_optional_names(blocked_tools),
        allowed_tools=_normalize_optional_names(allowed_tools),
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
    normalized_blocked_tools = _normalize_optional_names(mapping.blocked_tools)
    normalized_allowed_tools = _normalize_optional_names(mapping.allowed_tools)
    if normalized_blocked_tools != mapping.blocked_tools:
        raise ValueError(f"Specialist {mapping.name} blocked_tools must use canonical normalized form")
    if normalized_allowed_tools != mapping.allowed_tools:
        raise ValueError(f"Specialist {mapping.name} allowed_tools must use canonical normalized form")
    overlapping_tools = set(mapping.blocked_tools).intersection(mapping.allowed_tools)
    if overlapping_tools:
        raise ValueError(
            f"Specialist {mapping.name} cannot both block and allow the same tools: {sorted(overlapping_tools)}"
        )
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
            "consultant": SpecialistMapping(
                name="consultant",
                archetype_name="researcher",
                default_route_category="deep",
                default_delegation_profile="research",
                blocked_tools=("write_file", "patch", "terminal", "execute_code", "delegate_task", "task"),
            ),
            "librarian": SpecialistMapping(
                name="librarian",
                archetype_name="researcher",
                default_route_category="deep",
                default_delegation_profile="research",
                blocked_tools=("write_file", "patch", "terminal", "execute_code", "delegate_task", "task"),
            ),
            "explorer": SpecialistMapping(
                name="explorer",
                archetype_name="researcher",
                default_route_category="deep",
                default_delegation_profile="research",
                blocked_tools=("write_file", "patch", "execute_code", "delegate_task"),
            ),
            "planner": SpecialistMapping(
                name="planner",
                archetype_name="generalist",
                default_route_category="deep",
                default_delegation_profile="general",
            ),
            "looker": SpecialistMapping(
                name="looker",
                archetype_name="researcher",
                default_route_category="visual",
                default_delegation_profile="research",
                allowed_tools=(
                    "read_file",
                    "search_files",
                    "vision_analyze",
                    "browser_vision",
                    "browser_snapshot",
                    "browser_get_images",
                    "browser_console",
                ),
            ),
            "momus": SpecialistMapping(
                name="momus",
                archetype_name="verifier",
                default_route_category="quick",
                default_delegation_profile="verification",
                blocked_tools=("write_file", "patch", "terminal", "execute_code", "delegate_task", "task"),
            ),
            "multimodal_specialist": SpecialistMapping(
                name="multimodal_specialist",
                archetype_name="researcher",
                default_route_category="visual",
                default_delegation_profile="research",
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


def get_tool_restrictions(
    archetype_name: str,
    specialist_name: str | None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Resolve merged blocked/allowed tool restrictions for an archetype+specialist pair."""

    archetype = resolve_archetype(archetype_name)
    specialist = resolve_specialist_mapping(specialist_name)

    blocked_tools = set(archetype.blocked_tools)
    allowed_tools = set(archetype.allowed_tools)

    if specialist is not None:
        blocked_tools.update(specialist.blocked_tools)
        specialist_allowed_tools = set(specialist.allowed_tools)
        if specialist_allowed_tools:
            allowed_tools = allowed_tools.intersection(specialist_allowed_tools) if allowed_tools else specialist_allowed_tools

    if allowed_tools:
        allowed_tools.difference_update(blocked_tools)

    return frozenset(blocked_tools), frozenset(allowed_tools)


def normalize_named_agent_contract(name: str, entry: Mapping[str, Any] | str | None) -> NamedAgentContract:
    """Normalize one named-agent config entry into the canonical contract."""

    agent_name = _normalize_named_agent_name(name)
    if entry is None:
        entry_mapping: dict[str, Any] = {}
    elif isinstance(entry, str):
        entry_mapping = {"model": entry}
    elif isinstance(entry, Mapping):
        entry_mapping = dict(entry)
    else:
        raise ValueError(f"Named agent '{agent_name}' must be configured with a mapping or model string")

    raw_specialist = str(entry_mapping.get("specialist") or "").strip() or None
    specialist_mapping = resolve_specialist_mapping(raw_specialist)
    if raw_specialist and specialist_mapping is None:
        raise ValueError(f"Named agent '{agent_name}' has unknown specialist '{raw_specialist}'")

    raw_archetype = str(entry_mapping.get("archetype") or entry_mapping.get("role") or "").strip() or None
    archetype_name = raw_archetype or (specialist_mapping.archetype_name if specialist_mapping is not None else DEFAULT_ARCHETYPE_NAME)
    archetype = resolve_archetype(archetype_name)
    if raw_archetype and archetype.name != _normalize_name(raw_archetype):
        raise ValueError(f"Named agent '{agent_name}' has unknown archetype '{raw_archetype}'")

    mode = _normalize_named_agent_mode(entry_mapping.get("mode", "primary"), agent_name=agent_name)
    route_category = (
        str(entry_mapping.get("route_category") or entry_mapping.get("category") or "").strip()
        or (specialist_mapping.default_route_category if specialist_mapping is not None else "")
        or archetype.default_route_category
    )
    if route_category not in BUILTIN_ROUTE_CATEGORIES:
        raise ValueError(f"Named agent '{agent_name}' has unknown route_category '{route_category}'")
    category = str(entry_mapping.get("category") or "").strip() or route_category

    blocked_tools, allowed_tools = get_tool_restrictions(archetype.name, raw_specialist)
    named_blocked = set(_normalize_named_string_list(entry_mapping.get("blocked_tools")))
    named_allowed = set(_normalize_named_string_list(entry_mapping.get("allowed_tools")))
    effective_blocked = set(blocked_tools) | named_blocked
    effective_allowed = set(allowed_tools)
    if named_allowed:
        effective_allowed = effective_allowed.intersection(named_allowed) if effective_allowed else set(named_allowed)
    effective_allowed.difference_update(effective_blocked)

    provider_options = _normalize_provider_options(
        entry_mapping.get("providerOptions", entry_mapping.get("provider_options")),
        agent_name=agent_name,
    )
    permissions = _normalize_named_agent_permissions(entry_mapping.get("permission"), agent_name=agent_name)
    ultrawork = _normalize_ultrawork(entry_mapping.get("ultrawork"), agent_name=agent_name)
    fallback_models = _normalize_named_agent_fallback_models(entry_mapping.get("fallback_models"), agent_name=agent_name)

    color = str(entry_mapping.get("color") or "").strip() or None
    provider = str(entry_mapping.get("provider") or "").strip() or None
    model = str(entry_mapping.get("model") or "").strip() or None
    description = str(entry_mapping.get("description") or "").strip() or _build_named_agent_description(
        agent_name,
        archetype.name,
        specialist_mapping.name if specialist_mapping is not None else raw_specialist,
        mode,
    )
    safe_claim_text = str(entry_mapping.get("safe_claim_text") or "").strip() or _build_named_agent_safe_claim_text(
        agent_name,
        archetype.name,
        specialist_mapping.name if specialist_mapping is not None else raw_specialist,
        mode,
    )

    return NamedAgentContract(
        name=agent_name,
        role=archetype.name,
        archetype=archetype.name,
        specialist=specialist_mapping.name if specialist_mapping is not None else raw_specialist,
        mode=mode,
        color=color,
        category=category,
        route_category=route_category,
        provider=provider,
        model=model,
        fallback_models=fallback_models,
        providerOptions=provider_options,
        ultrawork=ultrawork,
        allowed_tools=tuple(sorted(effective_allowed)),
        blocked_tools=tuple(sorted(effective_blocked)),
        permissions=permissions,
        description=description,
        safe_claim_text=safe_claim_text,
        aliases=_normalize_named_string_list(entry_mapping.get("aliases")),
    )


def validate_named_agent_contract(contract: NamedAgentContract) -> NamedAgentContract:
    """Validate an already-normalized named-agent contract instance."""

    if not isinstance(contract, NamedAgentContract):
        raise ValueError("named agent contract must be a NamedAgentContract instance")
    normalize_named_agent_contract(
        contract.name,
        {
            "role": contract.role,
            "archetype": contract.archetype,
            "specialist": contract.specialist,
            "mode": contract.mode,
            "color": contract.color,
            "category": contract.category,
            "route_category": contract.route_category,
            "provider": contract.provider,
            "model": contract.model,
            "fallback_models": list(contract.fallback_models),
            "providerOptions": dict(contract.providerOptions) if contract.providerOptions is not None else None,
            "ultrawork": dict(contract.ultrawork) if contract.ultrawork is not None else None,
            "allowed_tools": list(contract.allowed_tools),
            "blocked_tools": list(contract.blocked_tools),
            "permission": dict(contract.permissions),
            "description": contract.description,
            "safe_claim_text": contract.safe_claim_text,
            "aliases": list(contract.aliases),
        },
    )
    return contract


def normalize_named_agent_registry(bucket: Mapping[str, Any] | None) -> dict[str, NamedAgentContract]:
    """Normalize a registry of named agents into canonical contracts keyed by name."""

    if bucket is None:
        return {}
    if not isinstance(bucket, Mapping):
        raise ValueError("named agent registry must be a mapping")
    normalized: dict[str, NamedAgentContract] = {}
    for raw_name, raw_entry in bucket.items():
        contract = normalize_named_agent_contract(str(raw_name), raw_entry)
        validate_named_agent_contract(contract)
        normalized[contract.name] = contract
    return normalized


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
    "NAMED_AGENT_MODES",
    "NAMED_AGENT_PERMISSION_KEYS",
    "NAMED_WORKFLOWS_BY_NAME",
    "NamedAgentContract",
    "NamedWorkflow",
    "REQUIRED_ARCHETYPE_FIELDS",
    "SPECIALIST_MAPPINGS_BY_NAME",
    "Archetype",
    "SpecialistMapping",
    "get_default_archetype",
    "get_tool_restrictions",
    "list_archetypes",
    "normalize_named_agent_contract",
    "normalize_named_agent_registry",
    "resolve_archetype",
    "resolve_archetype_defaults",
    "resolve_named_workflow",
    "resolve_specialist_defaults",
    "resolve_specialist_mapping",
    "validate_named_agent_contract",
    "validate_specialist_mapping",
    "validate_specialist_mappings",
]
