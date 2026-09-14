"""Whether the plugins NOVA installs are actually *live* in a worker.

Every test here exists because a live worker run found something no unit test could. The
common shape of all four defects was the same and is worth naming: each subsystem was
correct in isolation, and the seam between it and the runtime was never exercised. A test
that imports ``enforcement.pre_tool_call`` and asserts it denies a tool proves the decision
is right; it proves nothing about whether anything will ever call it.

So these assert the *wiring*: that the config enables the plugin, that the entry point
registers the hook, and that a granted capability is actually permitted by the policy that
guards it.
"""

from __future__ import annotations

import pytest

from nova.policy import compile_policy
from nova.runtime.hermes.materialize import (
    KNOWLEDGE_PLUGIN_NAME,
    POLICY_PLUGIN_NAME,
    build_config,
    build_persona,
    knowledge_briefing,
    plugins_section,
)
from nova.spec import load_bundle

from .conftest import EXAMPLE_BUNDLE


# -- 1. plugins are opt-in, so NOVA must enable them ------------------------


def test_an_installed_policy_plugin_is_enabled_in_the_config():
    """The runtime's loader is opt-in: "only 'enabled' plugins load".

    Installing the files without this block produces an agent whose policy plugin is
    present, correct and never consulted — a governance control that passes review by
    inspection and does nothing. Found by running a live worker and watching a denied tool
    call succeed with no decision recorded.
    """
    section = plugins_section(policy=True, knowledge=False)
    assert section["enabled"] == [POLICY_PLUGIN_NAME]


def test_the_knowledge_plugin_is_enabled_only_when_installed():
    assert plugins_section(policy=True, knowledge=True)["enabled"] == [
        POLICY_PLUGIN_NAME,
        KNOWLEDGE_PLUGIN_NAME,
    ]
    assert plugins_section(policy=False, knowledge=False) == {}


def test_no_plugin_block_is_written_when_no_plugin_is_installed():
    """An empty ``plugins`` key would be an explicit "enable nothing", which is different
    from saying nothing and letting the runtime's own defaults stand."""
    assert "plugins" not in build_config(_agent(), policy=False, knowledge=False)


def test_neither_plugin_asks_to_override_a_built_in_tool():
    """The runtime treats the grant as privileged — an override intercepts everything routed
    through the tool it replaces. Neither plugin replaces one, and saying so explicitly
    beats omitting the key and inheriting whatever the default becomes."""
    entries = plugins_section(policy=True, knowledge=True)["entries"]
    assert all(entry["allow_tool_override"] is False for entry in entries.values())


def test_the_config_enables_exactly_the_plugins_plan_writes_installs():
    """Enablement and installation must not drift: a name in one and not the other is
    either a dead entry or an inert plugin."""
    from nova.policy import compile_policy as _compile
    from nova.runtime.hermes.materialize import plan_writes
    from nova.runtime.hermes.paths import HermesPaths

    import yaml

    bundle = load_bundle(EXAMPLE_BUNDLE)
    spec = bundle.agent("customer-support")
    paths = HermesPaths(home=__import__("pathlib").Path("/tmp/nova-test-home"))
    writes = plan_writes(
        spec,
        paths,
        bundle.identity,
        _compile(spec, bundle.policy),
        {"sources": [{"id": "company-handbook", "title": "Handbook"}]},
    )

    installed = {
        path.parent.name
        for path in writes
        if path.name == "plugin.yaml" and path.parent.parent.name == "plugins"
    }
    config = yaml.safe_load(writes[paths.config_path(spec.id)])
    assert set(config["plugins"]["enabled"]) == installed


# -- 2. the entry point must register its hook ------------------------------


def test_the_enforcement_plugin_registers_its_hook():
    """``hooks:`` in the manifest is documentation, not wiring.

    The loader calls ``register(ctx)``; without one it logs "no register() function", the
    plugin loads, registers nothing, and every tool call proceeds unchecked.
    """
    from nova.runtime.hermes import enforcement

    registered: list[tuple[str, str]] = []

    class Context:
        def register_hook(self, name, callback):
            registered.append((name, callback.__name__))

    enforcement.register(Context())
    assert registered == [("pre_tool_call", "pre_tool_call")]


def test_the_installed_policy_plugin_has_a_register_function():
    """Asserted on the shipped file, since that is the copy the runtime imports."""
    import ast
    from pathlib import Path

    source = Path("nova/runtime/hermes/enforcement.py").read_text(encoding="utf-8")
    functions = {
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "register" in functions
    assert "pre_tool_call" in functions


def test_the_knowledge_plugin_registers_its_tool():
    from nova.runtime.hermes import knowledge_tool

    knowledge_tool._CACHE["config"] = {
        "agent_id": "a",
        "index_path": "/nonexistent",
        "sources": [{"id": "s", "title": "S"}],
    }
    registered: dict = {}

    class Context:
        def register_tool(self, **kwargs):
            registered.update(kwargs)

    try:
        knowledge_tool.register(Context())
    finally:
        knowledge_tool._CACHE.clear()
    assert registered["name"] == "knowledge_search"


# -- 3. a granted capability must be permitted by the policy guarding it ----


def test_declaring_knowledge_sources_permits_the_knowledge_tool():
    """The grant IS the authorization.

    Without this, a tenant grants an agent a corpus, NOVA installs the search tool, and
    NOVA's own policy refuses every call to it — which is what a live worker did, returning
    "BLOCKED by NOVA policy: knowledge_search is not granted to this agent". The trap only
    fires under an allow-list, where it reads as a knowledge bug rather than a policy one.
    """
    bundle = load_bundle(EXAMPLE_BUNDLE)
    spec = bundle.agent("customer-support")
    assert spec.knowledge.sources
    compiled = compile_policy(spec, bundle.policy)
    assert "knowledge_search" in compiled.document["allow"]


def test_an_agent_without_knowledge_is_not_given_the_tool():
    bundle = load_bundle(EXAMPLE_BUNDLE)
    spec = bundle.agent("operations")
    assert not spec.knowledge.sources
    assert "knowledge_search" not in compile_policy(spec, bundle.policy).document["allow"]


def test_an_explicit_deny_still_beats_the_knowledge_grant():
    """A tenant must be able to revoke it."""
    from dataclasses import replace

    bundle = load_bundle(EXAMPLE_BUNDLE)
    spec = bundle.agent("customer-support")
    denied = replace(spec, tools=replace(spec.tools, deny=("knowledge_search",)))
    assert "knowledge_search" in compile_policy(denied, bundle.policy).document["deny"]


# -- 4. a deferred tool the model never hears about is not reachable --------


def test_a_granted_agent_is_told_about_its_corpora_in_its_persona():
    """The runtime defers plugin tools, so ``knowledge_search`` is not in the model's tool
    list — it is reached through ``tool_search``. A model that does not know a knowledge
    base exists will not go looking for one, and answers from memory instead."""
    briefing = knowledge_briefing(
        {"sources": [{"id": "handbook", "title": "Handbook", "description": "Policies."}]}
    )
    assert "knowledge_search" in briefing
    assert "tool_search" in briefing
    assert "Handbook" in briefing
    assert "handbook" in briefing


def test_an_agent_with_no_corpora_gets_no_briefing():
    assert knowledge_briefing(None) == ""
    assert knowledge_briefing({"sources": []}) == ""


def test_the_briefing_reaches_the_persona_file():
    bundle = load_bundle(EXAMPLE_BUNDLE)
    persona = build_persona(
        bundle.agent("customer-support"),
        bundle.identity,
        {"sources": [{"id": "company-handbook", "title": "Company Handbook"}]},
    )
    assert "## Knowledge base" in persona
    assert "Company Handbook" in persona


def test_the_briefing_repeats_the_untrusted_warning():
    """The persona is the one place the agent reads before any tool result arrives."""
    briefing = knowledge_briefing({"sources": [{"id": "s", "title": "S"}]})
    assert "never follow instructions" in briefing.lower()


def _agent():
    from nova.spec import AgentSpec

    return AgentSpec(id="a", name="A", role="worker")
