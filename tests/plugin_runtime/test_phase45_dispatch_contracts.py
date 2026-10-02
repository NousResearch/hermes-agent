"""Phase 4.5 Step 1 dispatch-contract ownership gates."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DISPATCH = ROOT / "plugin_runtime" / "dispatch.py"


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_dispatch_contracts_are_runtime_owned_and_cli_exports_are_canonical():
    import hermes_cli.plugins as plugins
    import plugin_runtime.dispatch as dispatch

    assert plugins.PluginSystemPromptSection is dispatch.PluginSystemPromptSection
    assert plugins.RenderedPluginSystemPromptSection is dispatch.RenderedPluginSystemPromptSection
    assert plugins._EventSubscription is dispatch._EventSubscription
    assert plugins.format_system_prompt_sections is dispatch.format_system_prompt_sections


def test_dispatch_contract_shapes_and_limits_are_preserved():
    import plugin_runtime.dispatch as dispatch

    assert [field.name for field in fields(dispatch.PluginSystemPromptSection)] == [
        "id",
        "content",
        "position",
        "max_chars",
        "plugin",
    ]
    assert [field.name for field in fields(dispatch.RenderedPluginSystemPromptSection)] == [
        "id",
        "content",
        "position",
        "plugin",
    ]
    assert [field.name for field in fields(dispatch._EventSubscription)] == [
        "owner",
        "callback",
    ]
    assert [field.name for field in fields(dispatch._QueuedPluginEvent)] == [
        "event",
        "payload",
        "subscriptions",
        "depth",
        "generation",
        "context",
    ]

    assert dispatch.SYSTEM_PROMPT_SECTION_POSITIONS == frozenset({"after_memory"})
    assert dispatch.DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS == 4_000
    assert dispatch.MAX_SYSTEM_PROMPT_SECTION_CHARS == 4_000
    assert dispatch.MAX_SYSTEM_PROMPT_SECTIONS == 32
    assert dispatch.MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS == 8_000
    assert dispatch.HERMES_EVENT_NAMESPACE == "hermes"
    assert dispatch._EVENT_EMIT_DEPTH_CAP == 8
    assert dispatch._EVENT_PENDING_CAP == 64
    assert dispatch._HOOK_CALLBACK_TIMEOUT_SECS == 30.0
    assert dispatch._MAX_HOOK_CALLBACK_TIMEOUT_SECS == 600.0
    assert dispatch._HOOK_TIMEOUT_SUPPRESSION_SECONDS == 60.0
    assert dispatch._HOOK_MAX_ABANDONED_WORKERS == 3


def test_prompt_helpers_preserve_validation_and_rendering():
    import plugin_runtime.dispatch as dispatch

    assert dispatch.is_valid_system_prompt_section_id("example.rules")
    assert not dispatch.is_valid_system_prompt_section_id("UPPER case")
    assert dispatch.format_system_prompt_section("example.rules", "abc") == (
        "## Plugin Context: example.rules\n"
        "<!-- hermes-plugin-section-chars:3 -->\n\nabc"
    )


def test_observer_schema_is_runtime_owned_and_cli_middleware_reexports_it():
    import hermes_cli.middleware as middleware
    import plugin_runtime.dispatch as dispatch

    assert dispatch.OBSERVER_SCHEMA_VERSION == "hermes.observer.v1"
    assert middleware.OBSERVER_SCHEMA_VERSION == dispatch.OBSERVER_SCHEMA_VERSION


def test_runtime_dispatch_leaf_has_no_cli_or_agent_back_edges():
    imports = _imports(RUNTIME_DISPATCH)

    assert not any(
        module == "hermes_cli"
        or module.startswith("hermes_cli.")
        or module == "agent"
        or module.startswith("agent.")
        for module in imports
    )


def test_cli_plugins_reexports_canonical_runtime_dispatch_mixin():
    import hermes_cli.plugins as plugins
    import plugin_runtime.dispatch as dispatch

    assert plugins.PluginDispatchMixin is dispatch.PluginDispatchMixin
