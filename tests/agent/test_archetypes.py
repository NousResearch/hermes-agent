from dataclasses import MISSING, fields

from agent import archetypes
from agent.route_categories import DEFAULT_ROUTE_CATEGORY


_EXPECTED_BUILTIN_ARCHETYPES = {
    "generalist": {
        "summary": "Balanced starter preset for legacy-compatible delegated work without specialist bias.",
        "default_route_category": DEFAULT_ROUTE_CATEGORY,
        "default_delegation_profile": "general",
        "default_skills": ("general_reasoning", "task_execution"),
        "default_required_tools": ("read_file", "search_files"),
        "permission_preset": "inherit",
        "fallback_policy": "legacy_default_mapping",
    },
    "researcher": {
        "summary": "Research-oriented preset for evidence gathering and synthesis without changing route semantics.",
        "default_route_category": "deep",
        "default_delegation_profile": "research",
        "default_skills": ("research", "analysis", "synthesis"),
        "default_required_tools": ("read_file", "search_files", "web_search", "web_extract"),
        "permission_preset": "inherit",
        "fallback_policy": "degrade_to_generalist",
    },
    "implementer": {
        "summary": "Implementation-oriented preset for repository changes and local verification.",
        "default_route_category": "deep",
        "default_delegation_profile": "implementation",
        "default_skills": ("implementation", "python", "testing"),
        "default_required_tools": ("read_file", "search_files", "patch", "terminal"),
        "permission_preset": "workspace_write",
        "fallback_policy": "degrade_to_generalist",
    },
    "verifier": {
        "summary": "Verification-oriented preset for targeted test runs, review, and evidence capture.",
        "default_route_category": "quick",
        "default_delegation_profile": "verification",
        "default_skills": ("verification", "testing", "review"),
        "default_required_tools": ("read_file", "search_files", "terminal"),
        "permission_preset": "workspace_write",
        "fallback_policy": "degrade_to_generalist",
    },
}


def test_builtin_archetypes_are_wave_1_canonical_only():
    builtins = archetypes.list_archetypes()

    assert tuple(item.name for item in builtins) == (
        "generalist",
        "researcher",
        "implementer",
        "verifier",
    )
    assert archetypes.DEFAULT_ARCHETYPE_NAME == "generalist"
    assert set(archetypes.ARCHETYPES_BY_NAME) == set(_EXPECTED_BUILTIN_ARCHETYPES)

    for item in builtins:
        expected = _EXPECTED_BUILTIN_ARCHETYPES[item.name]
        assert item.summary == expected["summary"]
        assert item.default_route_category == expected["default_route_category"]
        assert item.default_delegation_profile == expected["default_delegation_profile"]
        assert item.default_skills == expected["default_skills"]
        assert item.default_required_tools == expected["default_required_tools"]
        assert item.permission_preset == expected["permission_preset"]
        assert item.fallback_policy == expected["fallback_policy"]
        assert item.kind == "archetype"


def test_resolve_archetype_normalizes_case_space_and_hyphen_only():
    assert archetypes.resolve_archetype("Researcher").name == "researcher"
    assert archetypes.resolve_archetype("researcher").name == "researcher"
    assert archetypes.resolve_archetype("  IMPLEMENTER  ").name == "implementer"
    assert archetypes.resolve_archetype("imple menter").name == "generalist"
    assert archetypes.resolve_archetype("imple-menter").name == "generalist"
    assert archetypes.resolve_archetype("Generalist").name == "generalist"
    assert archetypes.resolve_archetype("generalist").name == "generalist"
    assert archetypes.resolve_archetype("generalist ").name == "generalist"
    assert archetypes.resolve_archetype("re-searcher").name == "generalist"
    assert archetypes.resolve_archetype("re searcher").name == "generalist"


def test_resolve_archetype_falls_back_to_default_for_missing_or_unknown_names():
    default = archetypes.get_default_archetype()

    assert archetypes.resolve_archetype(None) is default
    assert archetypes.resolve_archetype("") is default
    assert archetypes.resolve_archetype("unknown") is default
    assert archetypes.resolve_archetype("planner") is default


def test_resolve_archetype_defaults_returns_required_fields_and_applies_overrides():
    defaults = archetypes.resolve_archetype_defaults("Implementer")

    assert defaults == {
        "default_route_category": "deep",
        "default_delegation_profile": "implementation",
        "default_skills": ["implementation", "python", "testing"],
        "default_required_tools": ["read_file", "search_files", "patch", "terminal"],
        "permission_preset": "workspace_write",
        "fallback_policy": "degrade_to_generalist",
    }

    overridden = archetypes.resolve_archetype_defaults(
        "implementer",
        overrides={
            "default_route_category": "quick",
            "default_skills": [" refactor ", "", "python"],
            "default_required_tools": ("terminal", "read_file"),
            "permission_preset": " inherit ",
            "fallback_policy": " custom ",
        },
    )

    assert overridden == {
        "default_route_category": "quick",
        "default_delegation_profile": "implementation",
        "default_skills": ["refactor", "python"],
        "default_required_tools": ["terminal", "read_file"],
        "permission_preset": "inherit",
        "fallback_policy": "custom",
    }


def test_schema_fields_and_required_fields_align_with_dataclass_shape():
    dataclass_fields = tuple(field.name for field in fields(archetypes.Archetype))
    required_default_fields = tuple(
        field.name
        for field in fields(archetypes.Archetype)
        if field.name not in {"name", "summary", "kind"}
        and field.default is MISSING
        and field.default_factory is MISSING
    )

    assert archetypes.ARCHETYPE_SCHEMA_FIELDS == dataclass_fields
    assert archetypes.REQUIRED_ARCHETYPE_FIELDS == required_default_fields
    assert archetypes.REQUIRED_ARCHETYPE_FIELDS == (
        "default_route_category",
        "default_delegation_profile",
        "default_skills",
        "default_required_tools",
        "permission_preset",
        "fallback_policy",
    )


def test_wave_1_module_exports_include_compatibility_specialist_surface():
    assert archetypes.__all__ == [
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

    for exported_name in (
        "SpecialistMapping",
        "SPECIALIST_MAPPINGS_BY_NAME",
        "resolve_specialist_mapping",
        "resolve_specialist_defaults",
        "NamedWorkflow",
        "NAMED_WORKFLOWS_BY_NAME",
        "resolve_named_workflow",
    ):
        assert hasattr(archetypes, exported_name)

    for removed_name in (
        "SPECIALIST_ALIASES",
        "SPECIALIST_MAPPING_SCHEMA_FIELDS",
        "BUILTIN_SPECIALIST_MAPPINGS",
        "list_specialist_mappings",
    ):
        assert not hasattr(archetypes, removed_name)


def test_named_workflow_registry_exposes_canonical_members_and_resolution():
    assert set(archetypes.NAMED_WORKFLOWS_BY_NAME) == {"planner", "deep_worker"}

    planner = archetypes.resolve_named_workflow("planner")
    assert planner is archetypes.NAMED_WORKFLOWS_BY_NAME["planner"]
    assert planner.name == "planner"
    assert planner.mode == "plan"
    assert planner.kind == "named_workflow"

    deep_worker = archetypes.resolve_named_workflow("deep worker")
    assert deep_worker is archetypes.NAMED_WORKFLOWS_BY_NAME["deep_worker"]
    assert deep_worker.name == "deep_worker"
    assert deep_worker.mode == "execute"
    assert deep_worker.kind == "named_workflow"

    assert archetypes.resolve_named_workflow("unknown") is None
