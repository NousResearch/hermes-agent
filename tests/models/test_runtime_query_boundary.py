"""Architecture guards for the Phase 5.8.2 runtime-query domains."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOWER_QUERY_FILES = (
    ROOT / "models" / "catalog_static.py",
    ROOT / "models" / "catalog_detection.py",
    ROOT / "models" / "catalog_projection.py",
    ROOT / "models" / "catalog_policy.py",
    ROOT / "models" / "catalog_chat.py",
    ROOT / "models" / "metadata" / "pricing.py",
    ROOT / "models" / "catalog_local.py",
    ROOT / "models" / "catalog_probe.py",
    ROOT / "models" / "catalog_github.py",
    ROOT / "models" / "catalog_manifest.py",
    ROOT / "models" / "catalog_runtime.py",
    ROOT / "models" / "codex_catalog.py",
    ROOT / "models" / "models_dev_cache.py",
    ROOT / "models" / "metadata" / "fast_mode.py",
    ROOT / "models" / "metadata" / "github.py",
    ROOT / "models" / "metadata" / "local.py",
    ROOT / "providers" / "opencode.py",
    ROOT / "providers" / "github.py",
    ROOT / "providers" / "configured.py",
    ROOT / "providers" / "route_identity.py",
    ROOT / "providers" / "routing.py",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Call) and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                func = node.func
                if (isinstance(func, ast.Name) and func.id in {"__import__", "import_module"}
                    or isinstance(func, ast.Attribute) and func.attr == "import_module"):
                    found.add(arg.value)
    return found


def _definitions(path: Path) -> set[str]:
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_runtime_query_domains_exist_and_do_not_import_upward():
    assert all(path.exists() for path in LOWER_QUERY_FILES)
    forbidden = (
        "agent",
        "hermes_cli",
        "gateway",
        "tui_gateway",
        "acp_adapter",
        "plugins",
    )
    violations = []
    for path in LOWER_QUERY_FILES:
        for module in sorted(_imports(path)):
            if any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden):
                violations.append(f"{path.relative_to(ROOT)} -> {module}")
    assert violations == []


def test_cli_models_no_longer_owns_moved_runtime_queries():
    definitions = _definitions(ROOT / "hermes_cli" / "models.py")
    moved = {
        "opencode_provider_family",
        "normalize_opencode_model_id",
        "normalize_opencode_base_url",
        "clamp_reasoning_effort_to_supported",
        "clamp_github_reasoning_effort",
        "model_supports_fast_mode",
        "resolve_fast_mode_overrides",
        "github_model_reasoning_efforts",
        "copilot_default_headers",
    }
    assert definitions.isdisjoint(moved)


def test_primary_agent_runtime_has_no_cli_model_semantic_dependencies():
    paths = (
        ROOT / "agent" / "agent_init.py",
        ROOT / "agent" / "agent_runtime_helpers.py",
        ROOT / "agent" / "client_lifecycle.py",
        ROOT / "agent" / "chat_completion_helpers.py",
        ROOT / "agent" / "fast_mode.py",
        ROOT / "agent" / "reasoning_params.py",
        ROOT / "agent" / "model_metadata.py",
        ROOT / "agent" / "credits_tracker.py",
        ROOT / "agent" / "error_surface.py",
        ROOT / "agent" / "turn_failure_copy.py",
        ROOT / "agent" / "auxiliary_model_resolution.py",
        ROOT / "agent" / "models_dev.py",
        ROOT / "agent" / "opencode_affinity.py",
        *(ROOT / "agent" / "transports").glob("*.py"),
    )
    forbidden = (
        "hermes_cli.models",
        "hermes_cli.model_switch",
        "hermes_cli.model_selection",
        "hermes_cli.models_validate",
        "hermes_cli.model_catalog",
    )
    offenders = []
    for path in paths:
        imports = _imports(path)
        for dependency in forbidden:
            if any(module == dependency or module.startswith(dependency + ".") for module in imports):
                offenders.append(f"{path.relative_to(ROOT)} -> {dependency}")
    assert offenders == []


def test_agent_warning_and_copilot_headers_have_final_owners():
    switch_definitions = _definitions(ROOT / "hermes_cli" / "model_switch.py")
    assert "is_nous_hermes_non_agentic" not in switch_definitions
    assert "_check_hermes_model_warning" not in switch_definitions
    assert "copilot_default_headers" not in _definitions(ROOT / "hermes_cli" / "models.py")
    assert "fetch_github_model_catalog" not in _definitions(ROOT / "hermes_cli" / "models.py")
    assert "get_copilot_model_context" not in _definitions(ROOT / "hermes_cli" / "models.py")
    assert "copilot_request_headers" not in _definitions(ROOT / "hermes_cli" / "copilot_auth.py")
    assert "nous_hermes_non_agentic_warning" in _definitions(ROOT / "agent" / "model_warnings.py")
    assert "copilot_request_headers" in _definitions(ROOT / "providers" / "github.py")
    assert "fetch_github_model_catalog" in _definitions(ROOT / "models" / "catalog_github.py")
    assert "github_model_context_length" in _definitions(ROOT / "models" / "metadata" / "github.py")
    local_defs = _definitions(ROOT / "models" / "metadata" / "local.py")
    assert {"lmstudio_model_reasoning_options", "ollama_model_supports_thinking"} <= local_defs
    old_local_defs = _definitions(ROOT / "hermes_cli" / "models_local.py")
    assert {"lmstudio_model_reasoning_options", "ollama_model_supports_thinking"}.isdisjoint(old_local_defs)


def test_route_identity_and_runtime_kind_have_lower_owners():
    assert "normalize_route_base_url" not in _definitions(ROOT / "hermes_cli" / "route_identity.py")
    assert "is_actual_route" not in _definitions(ROOT / "hermes_cli" / "providers.py")
    assert "_is_external_process_provider" not in _definitions(
        ROOT / "hermes_cli" / "runtime_provider_backends.py"
    )
    route_defs = _definitions(ROOT / "providers" / "route_identity.py")
    assert "normalize_route_base_url" in route_defs
    assert "is_actual_route" in route_defs
    assert "is_foreign_provider_endpoint" in route_defs
    assert "is_foreign_provider_endpoint" not in _definitions(
        ROOT / "hermes_cli" / "runtime_provider.py"
    )
    assert "is_external_process_provider" in _definitions(ROOT / "providers" / "routing.py")


def test_static_catalogue_old_owner_is_deleted():
    assert not (ROOT / "hermes_cli" / "models_catalog_static.py").exists()


def test_auxiliary_model_resolution_is_not_cli_owned():
    runtime = ROOT / "agent" / "auxiliary_model_resolution.py"
    assert runtime.exists()
    assert not (ROOT / "hermes_cli" / "model_selection_auxiliary.py").exists()
    assert "hermes_cli.model_selection_auxiliary" not in (
        ROOT / "agent" / "auxiliary_client.py"
    ).read_text(encoding="utf-8")


def test_configured_provider_semantics_have_final_owner():
    lower_defs = _definitions(ROOT / "providers" / "configured.py")
    assert {
        "match_configured_provider",
        "resolves_to_custom_provider",
        "expand_direct_api_alias",
    } <= lower_defs

    assert "_resolves_to_custom" not in _definitions(
        ROOT / "hermes_cli" / "runtime_provider.py"
    )
    old_custom_defs = _definitions(ROOT / "hermes_cli" / "runtime_provider_custom.py")
    assert {
        "_shadowed_by_builtin",
        "_match_new_style_provider",
        "_match_legacy_custom_provider",
        "expand_direct_api_alias",
    }.isdisjoint(old_custom_defs)

    paths = (
        ROOT / "agent" / "auxiliary_client.py",
        ROOT / "agent" / "auxiliary_health.py",
        ROOT / "agent" / "chat_completion_helpers.py",
        ROOT / "agent" / "client_lifecycle.py",
        ROOT / "agent" / "opencode_affinity.py",
    )
    forbidden = (
        "runtime_provider import _get_named_custom_provider",
        "runtime_provider_custom import _get_named_custom_provider",
        "runtime_provider_custom import expand_direct_api_alias",
        "_resolves_to_custom",
    )
    offenders = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for dependency in forbidden:
            if dependency in source:
                offenders.append(f"{path.relative_to(ROOT)} -> {dependency}")
    assert offenders == []


def test_models_dev_persistence_is_lower_owned():
    definitions = _definitions(ROOT / "agent" / "models_dev.py")
    obsolete = {
        "_get_cache_path",
        "_get_etag_path",
        "_load_disk_cache",
        "_load_etag",
        "_save_disk_cache",
        "_save_etag",
        "_clear_etag",
        "_quarantine_corrupt_cache",
    }
    assert definitions.isdisjoint(obsolete)


def test_provider_grouping_is_application_owned():
    presentation = ROOT / "application_provider_groups.py"
    assert presentation.exists()
    assert {"PROVIDER_GROUPS", "group_providers", "provider_group_for_slug"} <= (
        _definitions(presentation)
        | {
            target.id
            for node in _tree(presentation).body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
    )
    static_source = (ROOT / "models" / "catalog_static.py").read_text(encoding="utf-8")
    assert "PROVIDER_GROUPS" not in static_source
    assert "group_providers" not in static_source


def test_production_code_has_no_old_static_catalogue_imports():
    offenders = []
    for root_name in ("agent", "gateway", "tui_gateway", "acp_adapter", "hermes_cli", "plugins"):
        for path in (ROOT / root_name).rglob("*.py"):
            if "hermes_cli.models_catalog_static" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_auxiliary_fallback_routing_has_one_canonical_route_owner():
    fallback = ROOT / "agent" / "fallback_routing.py"
    assert fallback.exists()
    fallback_source = fallback.read_text(encoding="utf-8")
    assert "resolve_invocation_route" in fallback_source
    assert "hermes_cli.runtime_provider" not in fallback_source

    auxiliary = ROOT / "agent" / "auxiliary_client.py"
    tree = _tree(auxiliary)
    complete = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_complete_fallback_destination"
    )
    complete_source = ast.get_source_segment(auxiliary.read_text(encoding="utf-8"), complete) or ""
    assert "resolve_fallback_invocation_route" in complete_source
    assert "hermes_cli.runtime_provider" not in complete_source

    helpers = ROOT / "agent" / "chat_completion_helpers.py"
    helper_defs = _definitions(helpers)
    assert {
        "_fallback_invocation_route",
        "_fallback_api_mode_hint",
        "_fallback_api_mode_resolved",
        "_is_anthropic_wire_url",
    }.isdisjoint(helper_defs)
    assert "agent.fallback_routing" in _imports(helpers)

    health_source = (ROOT / "agent" / "auxiliary_health.py").read_text(encoding="utf-8")
    assert "normalize_provider" in health_source
    assert "_normalize_chain_label" not in health_source


def test_phase_5_8_4_auxiliary_runtime_has_no_cli_semantic_authority():
    paths = (
        ROOT / "agent" / "auxiliary_client.py",
        ROOT / "agent" / "auxiliary_health.py",
        ROOT / "agent" / "auxiliary_model_resolution.py",
        ROOT / "agent" / "configured_provider_resolution.py",
        ROOT / "agent" / "fallback_routing.py",
    )
    forbidden = (
        "hermes_cli.model_selection_auxiliary",
        "hermes_cli.model_switch",
        "hermes_cli.model_selection",
        "hermes_cli.models_validate",
        "hermes_cli.model_catalog",
        "hermes_cli.runtime_provider_custom",
    )
    offenders = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for dependency in forbidden:
            if dependency in source:
                offenders.append(f"{path.relative_to(ROOT)} -> {dependency}")
    assert offenders == []

    # runtime_provider remains application-owned for credential/config acquisition only.
    # These are the only two auxiliary call sites allowed until Phase 6 moves auth ownership.
    auxiliary = ROOT / "agent" / "auxiliary_client.py"
    tree = _tree(auxiliary)
    allowed_runtime_provider_imports = {
        ("_resolve_custom_runtime", "resolve_runtime_provider"),
        ("_try_azure_foundry", "_resolve_azure_foundry_runtime"),
    }
    found = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.ImportFrom)
                and child.module == "hermes_cli.runtime_provider"
            ):
                found.update((node.name, alias.name) for alias in child.names)
    assert found == allowed_runtime_provider_imports


def test_phase_5_8_4_auxiliary_final_owners_are_present():
    assert (ROOT / "agent" / "fallback_routing.py").exists()
    assert (ROOT / "agent" / "configured_provider_resolution.py").exists()
    assert (ROOT / "agent" / "auxiliary_model_resolution.py").exists()

    assert "resolve_invocation_route" in _definitions(ROOT / "providers" / "routing.py")
    assert "match_configured_provider" in _definitions(ROOT / "providers" / "configured.py")
    assert "select_vision_auxiliary_model" in _definitions(
        ROOT / "models" / "selection_auxiliary.py"
    )
    assert "resolve_supports_vision" in _definitions(
        ROOT / "models" / "metadata" / "capabilities.py"
    )


def test_phase_5_8_5_gateway_effective_model_precedence_is_application_owned():
    helper = ROOT / "gateway" / "model_resolution.py"
    assert helper.exists()
    assert "effective_model_candidate" in _definitions(helper)

    offenders = []
    for path in (
        ROOT / "gateway" / "run_config_loaders.py",
        ROOT / "gateway" / "platforms" / "api_server.py",
    ):
        source = path.read_text(encoding="utf-8")
        if "resolve_effective_model" in source:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_phase_5_8_5_session_launch_resolution_uses_lower_domains():
    route = ROOT / "gateway" / "session_local_route.py"
    aliases = ROOT / "application_model_aliases.py"
    route_source = route.read_text(encoding="utf-8")
    alias_source = aliases.read_text(encoding="utf-8")

    assert "hermes_cli.model_switch" not in route_source
    assert "hermes_cli.model_switch" not in alias_source
    assert "select_explicit_model" in route_source
    assert "resolve_invocation_route" in route_source
    assert "model_aliases_from_config" in route_source
    assert "find_static_provider_model_id" in _definitions(
        ROOT / "models" / "catalog_static.py"
    )


def test_phase_5_8_5_session_mutation_is_gateway_owned():
    mutation = ROOT / "gateway" / "session_mutation_model.py"
    resolver = ROOT / "gateway" / "session_model_resolution.py"
    facts = ROOT / "application_model_facts.py"

    for path in (mutation, resolver, facts):
        source = path.read_text(encoding="utf-8")
        assert "hermes_cli.model_switch" not in source
        assert "hermes_cli.model_selection" not in source

    mutation_source = mutation.read_text(encoding="utf-8")
    resolver_source = resolver.read_text(encoding="utf-8")
    assert "resolve_session_model" in mutation_source
    assert "select_explicit_model" in resolver_source
    assert "resolve_invocation_route" in resolver_source

    # Phase 6 exception: session mutation may acquire credentials through the
    # existing runtime provider, but that module cannot choose model identity
    # or invocation semantics for this path.
    tree = _tree(resolver)
    runtime_imports = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "hermes_cli.runtime_provider"
        ):
            runtime_imports.extend(alias.name for alias in node.names)
    assert runtime_imports == ["resolve_runtime_provider"]

def test_phase_5_8_5_model_command_orchestration_is_gateway_owned():
    paths = (
        ROOT / "gateway" / "slash_commands_model.py",
        ROOT / "application_model_command_request.py",
        ROOT / "gateway" / "model_switch_resolution.py",
        ROOT / "application_model_switch_enrichment.py",
        ROOT / "application_model_switch_persistence.py",
        ROOT / "gateway" / "model_switch_display.py",
        ROOT / "application_model_selection_guards.py",
        ROOT / "gateway" / "model_switch_preflight.py",
    )
    forbidden = (
        "from hermes_cli.model_switch import",
        "import hermes_cli.model_switch",
        "hermes_cli.model_selection_guards",
        "hermes_cli.context_switch_guard",
    )
    offenders = []
    for source_path in paths:
        source = source_path.read_text(encoding="utf-8")
        for dependency in forbidden:
            if dependency in source:
                offenders.append(f"{source_path.relative_to(ROOT)} -> {dependency}")
    assert offenders == []

    assert "parse_model_command" in _definitions(
        ROOT / "application_model_command_request.py"
    )
    assert "resolve_model_switch" in _definitions(
        ROOT / "gateway" / "model_switch_resolution.py"
    )
    assert "persist_model_selection" in _definitions(
        ROOT / "application_model_switch_persistence.py"
    )
    assert "combined_selection_warning" in _definitions(
        ROOT / "application_model_selection_guards.py"
    )


def test_phase_5_8_5_turn_runtime_semantics_use_lower_owners():
    paths = (
        ROOT / "gateway" / "run_agent_cache.py",
        ROOT / "gateway" / "run_turn_prepare.py",
        ROOT / "gateway" / "run_turn.py",
        ROOT / "gateway" / "platforms" / "api_server.py",
        ROOT / "gateway" / "model_runtime_facts.py",
        ROOT / "plugins" / "platforms" / "feishu" / "feishu_comment.py",
    )
    forbidden = (
        "hermes_cli.model_selection_defaults",
        "from hermes_cli.runtime_provider import is_foreign_provider_endpoint",
        "hermes_cli.models_catalog_static",
        "hermes_cli.model_switch",
    )
    offenders = []
    for source_path in paths:
        source = source_path.read_text(encoding="utf-8")
        for dependency in forbidden:
            if dependency in source:
                offenders.append(f"{source_path.relative_to(ROOT)} -> {dependency}")
    assert offenders == []

    facts = ROOT / "gateway" / "model_runtime_facts.py"
    assert {"provider_default_model", "normalize_runtime_model"} <= _definitions(facts)
    assert "select_default_model" in facts.read_text(encoding="utf-8")

    prepare_source = (ROOT / "gateway" / "run_turn_prepare.py").read_text(
        encoding="utf-8"
    )
    api_source = (ROOT / "gateway" / "platforms" / "api_server.py").read_text(
        encoding="utf-8"
    )
    turn_source = (ROOT / "gateway" / "run_turn.py").read_text(encoding="utf-8")
    cache_source = (ROOT / "gateway" / "run_agent_cache.py").read_text(
        encoding="utf-8"
    )
    assert "provider_default_model" in prepare_source
    assert "provider_default_model" in api_source
    assert "normalize_runtime_model" in turn_source
    assert "is_foreign_provider_endpoint" in cache_source

    # Credential/fallback mechanics remain the Phase 6 seam.
    assert "hermes_cli.model_catalog" not in facts.read_text(encoding="utf-8")
    assert "cached_default_model" in facts.read_text(encoding="utf-8")
    assert "resolve_runtime_with_fallback" in prepare_source
    assert "resolve_runtime_provider" in api_source

    tui_source = (ROOT / "tui_gateway" / "agent_factory.py").read_text(
        encoding="utf-8"
    )
    assert "hermes_cli.model_selection_defaults" not in tui_source
    assert (
        "from hermes_cli.runtime_provider import is_foreign_provider_endpoint"
        not in tui_source
    )
    assert "provider_default_model" in tui_source
    assert "is_foreign_provider_endpoint" in tui_source


def test_phase_5_8_5_catalogue_picker_runtime_has_lower_semantic_owners():
    paths = (
        ROOT / "gateway" / "run_watchers.py",
        ROOT / "gateway" / "run_turn.py",
        ROOT / "gateway" / "run_turn_prepare.py",
        ROOT / "gateway" / "slash_commands_model.py",
        ROOT / "gateway" / "model_runtime_facts.py",
        ROOT / "gateway" / "model_catalog_runtime.py",
        ROOT / "gateway" / "model_picker_inventory.py",
        ROOT / "gateway" / "platforms" / "api_server.py",
        ROOT / "gateway" / "session_config.py",
    )
    forbidden = (
        "hermes_cli.models",
        "hermes_cli.model_selection",
        "hermes_cli.model_catalog",
        "application_provider_discovery",
    )
    offenders = []
    for source_path in paths:
        source = source_path.read_text(encoding="utf-8")
        for dependency in forbidden:
            if dependency in source:
                offenders.append(f"{source_path.relative_to(ROOT)} -> {dependency}")
    assert offenders == []

    watcher = (ROOT / "gateway" / "run_watchers.py").read_text(encoding="utf-8")
    picker = (ROOT / "gateway" / "slash_commands_model.py").read_text(encoding="utf-8")
    assert "gateway.model_catalog_runtime" in watcher
    assert "model_provider_rows" in picker

    manifest_defs = _definitions(ROOT / "models" / "catalog_manifest.py")
    runtime_defs = _definitions(ROOT / "models" / "catalog_runtime.py")
    assert {"catalog_settings", "validate_manifest", "default_model"} <= manifest_defs
    assert {
        "get_catalog",
        "cached_default_model",
        "refresh_manifest",
        "refresh_interval_seconds",
    } <= runtime_defs

    # Historical pre-handoff updaters import this path after pulling the new
    # tree. Keep only that seed hook; live catalogue semantics must not return.
    compat = ROOT / "hermes_cli" / "model_catalog.py"
    assert _definitions(compat) == {"seed_cache_from_checkout"}
    compat_source = compat.read_text(encoding="utf-8")
    assert "models.catalog_seed" in compat_source
    assert "get_catalog" not in compat_source
    assert "refresh_catalogs" not in compat_source
    assert "get_curated_openrouter_models" not in compat_source


def test_phase_5_8_5_gateway_semantic_boundary_is_closed():
    forbidden_modules = {
        "hermes_cli.model_switch",
        "hermes_cli.model_selection_defaults",
        "hermes_cli.model_selection_guards",
        "application_provider_discovery",
        "hermes_cli.models",
        "hermes_cli.model_catalog",
    }
    offenders = []
    for path in (ROOT / "gateway").rglob("*.py"):
        for module in sorted(_imports(path)):
            if module in forbidden_modules:
                offenders.append(f"{path.relative_to(ROOT)} -> {module}")
    assert offenders == []

    assert "detect_single_openai_model" in _definitions(
        ROOT / "models" / "catalog_probe.py"
    )
    assert "configured_custom_identity" in _definitions(
        ROOT / "providers" / "configured.py"
    )
    assert "configured_model_facts" in _definitions(
        ROOT / "gateway" / "model_resolution.py"
    )


def test_phase_5_8_5_gateway_runtime_provider_exceptions_are_exact():
    # Phase 6 debt only: credential/runtime acquisition and its error projection.
    allowed = {
        (
            "gateway/platforms/api_server.py",
            "_resolve_request_runtime_agent_kwargs",
            "hermes_cli.runtime_provider",
            "resolve_runtime_provider",
        ),
        (
            "gateway/platforms/api_server.py",
            "_resolve_request_runtime_agent_kwargs",
            "hermes_cli.runtime_provider",
            "format_runtime_provider_error",
        ),
        (
            "gateway/run.py",
            "_resolve_runtime_agent_kwargs",
            "hermes_cli.runtime_provider",
            "resolve_runtime_with_fallback",
        ),
        (
            "gateway/run.py",
            "_resolve_runtime_agent_kwargs",
            "hermes_cli.runtime_provider",
            "format_runtime_provider_error",
        ),
        (
            "gateway/run.py",
            "_resolve_runtime_agent_kwargs_for_provider",
            "hermes_cli.runtime_provider",
            "resolve_runtime_provider",
        ),
        (
            "gateway/run.py",
            "_resolve_runtime_agent_kwargs_for_provider",
            "hermes_cli.runtime_provider",
            "format_runtime_provider_error",
        ),
        (
            "gateway/run_turn_prepare.py",
            "_resolve_session_agent_runtime",
            "hermes_cli.runtime_provider_custom",
            "_resolve_named_custom_runtime",
        ),
        (
            "gateway/run_turn_prepare.py",
            "_resolve_session_agent_runtime",
            "hermes_cli.runtime_provider",
            "resolve_runtime_with_fallback",
        ),
        (
            "gateway/session_model_resolution.py",
            "_resolve_runtime_credentials",
            "hermes_cli.runtime_provider",
            "resolve_runtime_provider",
        ),
    }

    found = set()

    def collect(path: Path, node: ast.AST, scope: str = "<module>") -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = node.name
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("hermes_cli.runtime_provider")
        ):
            rel = path.relative_to(ROOT).as_posix()
            found.update((rel, scope, node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            rel = path.relative_to(ROOT).as_posix()
            for alias in node.names:
                if alias.name.startswith("hermes_cli.runtime_provider"):
                    found.add((rel, scope, alias.name, alias.name))
        for child in ast.iter_child_nodes(node):
            collect(path, child, scope)

    for path in (ROOT / "gateway").rglob("*.py"):
        collect(path, _tree(path))

    assert found == allowed

def test_phase_5_8_6_2_tui_switch_owns_application_mutation():
    paths = (
        ROOT / "tui_gateway" / "model_switch.py",
        ROOT / "tui_gateway" / "model_switch_resolution.py",
        ROOT / "tui_gateway" / "methods_config_set.py",
    )
    forbidden = {
        "hermes_cli.model_switch",
        "application_provider_discovery",
        "hermes_cli.model_selection_guards",
        "hermes_cli.context_switch_guard",
    }
    offenders = [
        f"{path.relative_to(ROOT)} -> {module}"
        for path in paths
        for module in _imports(path)
        if module in forbidden
    ]
    assert offenders == []
    assert "resolve_tui_model_switch" in _definitions(paths[1])
    assert "_apply_model_switch" in _definitions(paths[0])

    shared = (
        ROOT / "application_model_command_request.py",
        ROOT / "application_model_selection_guards.py",
        ROOT / "application_model_switch_enrichment.py",
        ROOT / "application_model_switch_persistence.py",
        ROOT / "application_model_switch_preflight.py",
    )
    for path in shared:
        assert path.exists()
        assert not any(
            module == "gateway" or module.startswith("gateway.")
            or module == "tui_gateway" or module.startswith("tui_gateway.")
            or module == "hermes_cli.model_switch"
            for module in _imports(path)
        )


def test_phase_5_8_6_2_tui_credential_acquisition_exceptions_are_exact():
    expected = {
        ("tui_gateway/model_switch.py", "_current_model_runtime",
         "hermes_cli.runtime_provider", "resolve_runtime_provider"),
        ("tui_gateway/model_switch_resolution.py", "resolve_tui_model_switch",
         "hermes_cli.runtime_provider", "resolve_runtime_provider"),
    }
    found = set()

    def visit(path, node, scope="<module>"):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = node.name
        if (isinstance(node, ast.ImportFrom)
                and node.module and node.module.startswith("hermes_cli.runtime_provider")):
            found.update(
                (path.relative_to(ROOT).as_posix(), scope, node.module, alias.name)
                for alias in node.names
            )
        for child in ast.iter_child_nodes(node):
            visit(path, child, scope)

    for rel in ("tui_gateway/model_switch.py", "tui_gateway/model_switch_resolution.py"):
        path = ROOT / rel
        visit(path, _tree(path))
    assert found == expected


# Phase 5.9 scans every production root, including root application modules,
# provider/media plugins and literal lazy imports. Test fixtures are excluded.
from functools import lru_cache
import os

@lru_cache(maxsize=1)
def _production_sources():
    excluded = {".git", ".venv", "venv", "node_modules", "__pycache__", "tests",
                "evals", "scripts", "skills", "optional-skills", "website", "build"}
    sources = {}
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in excluded and not name.startswith(".")]
        for name in files:
            if name.endswith(".py"):
                path = Path(directory) / name
                sources[path] = path.read_text(encoding="utf-8-sig")
    return sources


def test_phase_5_9_deleted_modules_have_no_production_imports():
    deleted = {
        "hermes_cli.model_normalize", "hermes_cli.models_catalog_static",
        "hermes_cli.model_selection_auxiliary", "hermes_cli.model_selection_guards",
        "hermes_cli.provider_groups", "hermes_cli.model_switch_providers",
        "hermes_cli.chat_catalog",
    }
    for module in deleted:
        assert not (ROOT / (module.replace(".", "/") + ".py")).exists()
    offenders = []
    for path, source in _production_sources().items():
        if not any(module in source for module in deleted):
            continue
        for module in _imports(path):
            if any(module == old or module.startswith(old + ".") for old in deleted):
                offenders.append((str(path.relative_to(ROOT)), module))
    assert offenders == []


def test_phase_5_9_catalogue_billing_and_matching_have_single_owners():
    expected = {
        "_resolve_static_model_alias": "models/catalog_detection.py",
        "_static_catalog_matches": "models/catalog_detection.py",
        "current_provider_owns_vendor": "models/catalog_detection.py",
        "resolve_declared_provider_prefix": "models/catalog_detection.py",
        "model_alias_canonical": "models/catalog_projection.py",
        "merge_profile_models": "models/catalog_projection.py",
        "_drop_delisted_opencode_models": "models/catalog_projection.py",
        "_is_model_free": "models/metadata/pricing.py",
        "partition_nous_models_by_tier": "models/metadata/pricing.py",
        "compute_sale_discount": "models/metadata/pricing.py",
        "restrict_to_nous_policy": "models/catalog_policy.py",
        "allows_model_whitespace": "models/catalog_policy.py",
        "chat_catalog_ids": "models/catalog_chat.py",
        "is_official_openai_host": "providers/routing.py",
    }
    owners = {name: [] for name in expected}
    for path, source in _production_sources().items():
        if not any(name in source for name in expected):
            continue
        for name in _definitions(path):
            if name in owners:
                owners[name].append(path.relative_to(ROOT).as_posix())
    assert owners == {name: [owner] for name, owner in expected.items()}
    assert {"detect_static_provider_for_model", "_detection_candidates"}.isdisjoint(
        _definitions(ROOT / "hermes_cli/models.py"))
    assert _definitions(ROOT / "hermes_cli/models_pricing.py") == {"_format_price_per_mtok"}


def test_phase_5_9_external_compat_targets_follow_final_owners():
    import json
    entries = json.loads((ROOT / "compat_manifest.json").read_text(encoding="utf-8"))["entries"]
    for entry in entries:
        if entry["facade"] != "hermes_cli.models":
            continue
        if entry["name"] == "compute_sale_discount":
            assert entry["target"] == "models.metadata.pricing"
        elif entry["name"] == "restrict_to_nous_policy":
            assert entry["target"] == "models.catalog_policy"
        elif entry["name"] in {"fetch_models_with_pricing", "get_pricing_for_provider",
                               "peek_cached_pricing", "pricing_cache_scope"}:
            assert entry["target"] == "application_model_pricing"
