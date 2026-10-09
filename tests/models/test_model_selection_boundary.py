"""Architecture guards for the Phase 5.7 model-selection domain."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SELECTION_FILES = (
    ROOT / "models" / "selection.py",
    ROOT / "models" / "selection_types.py",
    ROOT / "models" / "selection_explicit.py",
    ROOT / "models" / "selection_detection.py",
    ROOT / "models" / "selection_defaults.py",
    ROOT / "models" / "selection_auxiliary.py",
    ROOT / "models" / "selection_picker.py",
)
METADATA_ROOT = ROOT / "models" / "metadata"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _definitions(path: Path) -> set[str]:
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_selection_domain_exists_and_has_no_upward_imports():
    assert all(path.exists() for path in SELECTION_FILES)
    forbidden = (
        "agent",
        "hermes_cli",
        "gateway",
        "tui_gateway",
        "acp_adapter",
        "runtime",
    )
    violations = []
    for path in SELECTION_FILES:
        for module in sorted(_imports(path)):
            if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden):
                violations.append(f"{path.name} -> {module}")
    assert violations == []


def test_selection_local_imports_stay_in_model_and_provider_domains():
    violations = []
    for path in SELECTION_FILES:
        for module in sorted(_imports(path)):
            root = module.split(".", 1)[0]
            if root in {"models", "providers"}:
                continue
            if (ROOT / root).exists() or (ROOT / f"{root}.py").exists():
                violations.append(f"{path.name} -> {module}")
    assert violations == []


def test_selection_uses_existing_identity_metadata_and_route_owners():
    imports = set().union(*(_imports(path) for path in SELECTION_FILES))
    assert "models.identity" in imports
    assert "models.metadata.types" in imports
    assert "providers.identity" in imports
    assert "providers.routing" in imports

    duplicate_owners = {
        "ModelRef",
        "ModelMetadata",
        "InvocationRequest",
        "InvocationRoute",
        "normalize_provider",
        "resolve_invocation_route",
    }
    definitions = set().union(*(_definitions(path) for path in SELECTION_FILES))
    assert definitions.isdisjoint(duplicate_owners)


def test_metadata_domain_does_not_depend_on_selection():
    offenders = [
        str(path.relative_to(ROOT))
        for path in METADATA_ROOT.rglob("*.py")
        if any(module.startswith("models.selection") for module in _imports(path))
    ]
    assert offenders == []


def test_selection_surface_is_pure_and_credential_free():
    imports = set().union(*(_imports(path) for path in SELECTION_FILES))
    forbidden_fragments = (
        "auth",
        "credential",
        "inventory",
        "model_switch",
        "urllib",
        "requests",
        "httpx",
        "openai",
    )
    offenders = [
        module for module in sorted(imports)
        if any(fragment in module.lower() for fragment in forbidden_fragments)
    ]
    assert offenders == []


def test_selection_public_definitions_are_owned_once():
    expected = {
        "CapabilityRequirements",
        "SelectionConstraints",
        "SelectionPolicy",
        "SelectionCandidate",
        "SelectionRequest",
        "CandidateRejection",
        "SelectionReason",
        "ModelSelection",
        "ExplicitAlias",
        "ExplicitDetectionFacts",
        "ExplicitProviderFacts",
        "ExplicitSelectionError",
        "build_selection_candidate",
        "explicit_provider_hint",
        "select_default_model",
        "select_auxiliary_fallback_model",
        "select_auxiliary_model",
        "select_fast_auxiliary_model",
        "select_vision_auxiliary_model",
        "selected_auxiliary_model_id",
        "auxiliary_task_prefers_fast_model",
        "list_picker_candidates",
        "picker_model_ids",
        "select_nous_default_model",
        "select_detected_model",
        "select_explicit_model",
        "select_model",
    }
    owners: dict[str, list[str]] = {name: [] for name in expected}
    for path in SELECTION_FILES:
        for name in _definitions(path):
            if name in owners:
                owners[name].append(path.name)
    assert all(len(paths) == 1 for paths in owners.values()), owners

def test_model_switch_consumes_selection_and_old_route_owners_are_deleted():
    path = ROOT / "hermes_cli" / "model_switch.py"
    imports = _imports(path)
    assert "models.selection" in imports

    obsolete = {
        "resolve_alias",
        "_resolve_alias_fallback",
        "_route_explicit_provider",
        "_route_configured_provider",
        "_route_from_model_input",
        "_aggregator_catalog_match",
        "_current_provider_match",
        "_resolve_named_custom_model_id",
    }
    assert _definitions(path).isdisjoint(obsolete)

    calls = {
        node.func.id
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "select_explicit_model" in calls
    assert "normalize_model_id" not in calls
    assert "detect_provider_for_model" not in calls


def test_legacy_default_selection_owners_are_deleted():
    path = ROOT / "hermes_cli" / "models.py"
    obsolete = {
        "get_preferred_silent_default_model",
        "pick_silent_default_model",
        "recommended_nous_default_model",
        "get_default_model_for_provider",
    }
    assert _definitions(path).isdisjoint(obsolete)


def test_auxiliary_client_has_no_model_selection_authority():
    path = ROOT / "agent" / "auxiliary_client.py"
    obsolete_defs = {
        "_model_recency_key",
        "_fast_model_from_catalog",
        "_get_aux_model_for_provider",
        "_resolve_provider_vision_default",
        "_task_prefers_fast_model",
    }
    assert _definitions(path).isdisjoint(obsolete_defs)
    source = path.read_text(encoding="utf-8")
    for obsolete in (
        "_FAST_MODEL_FAMILIES",
        "_FAST_MODEL_EXCLUDE",
        "_API_KEY_PROVIDER_AUX_MODELS_FALLBACK",
        "_API_KEY_PROVIDER_AUX_MODELS",
        "_PROVIDER_VISION_MODELS",
        "_PROVIDERS_WITHOUT_VISION",
        "_OPENROUTER_MODEL",
        "_NOUS_MODEL",
        "_FAST_MODEL_TASKS",
    ):
        assert obsolete not in source
    assert "select_provider_auxiliary_model" in source
    assert "select_provider_auxiliary_fallback" in source
    assert "select_vision_auxiliary_model" in source
    assert "resolve_supports_vision" in source
    assert "get_provider_profile" in source
    assert "provider_vision_default" not in source


def test_application_selection_adapters_consume_canonical_domain():
    for relative in (
        "hermes_cli/model_switch.py",
        "hermes_cli/model_selection_facts.py",
        "application_model_selection_defaults.py",
        "hermes_cli/model_selection_picker.py",
        "agent/auxiliary_client.py",
    ):
        imports = _imports(ROOT / relative)
        assert "models.selection" in imports, relative


def test_picker_and_setup_surfaces_consume_selection_candidate_projection():
    for relative in (
        "hermes_cli/auth_model_picker.py",
        "hermes_cli/cli_model_switch_mixin.py",
        "hermes_cli/inventory.py",
        "application_provider_discovery.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "model_selection_picker" in source

    validate = ROOT / "hermes_cli" / "models_validate.py"
    assert _definitions(validate).isdisjoint(
        {"offered_model_ids", "drop_unofferable_model_ids"}
    )

def test_phase_5_9_detection_has_no_second_application_ladder():
    assert {"detect_static_provider_for_model", "_detection_candidates"}.isdisjoint(
        _definitions(ROOT / "hermes_cli/models.py"))
    assert {"_zero_credit_model", "_policy_filtered_ids"}.isdisjoint(
        _definitions(ROOT / "models/selection_defaults.py"))
    assert "models.metadata.pricing" in _imports(ROOT / "models/selection_defaults.py")
    assert "models.catalog_policy" in _imports(ROOT / "models/selection_defaults.py")
    assert "models.selection" in _imports(ROOT / "hermes_cli/model_selection_facts.py")
    assert "models.selection_detection" in _imports(ROOT / "hermes_cli/models.py")
