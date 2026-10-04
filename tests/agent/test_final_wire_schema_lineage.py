"""Final schema cache lineage/version and malformed representation contracts."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import agent.final_wire_admission as admission
from agent.model_metadata import _estimate_tools_tokens_rough


@pytest.mark.parametrize("tools", [object(), 0, "opaque", {"unserializable": object()}])
def test_unrepresentable_schema_never_returns_zero_or_stringified_opaque(tools):
    with pytest.raises(admission.ProviderBoundUnsupportedAccounting):
        admission.estimate_schema_tokens(tools)


def test_schema_cache_separates_family_accounting_and_estimator_version(monkeypatch):
    admission._SCHEMA_CACHE.clear()
    tools = [{"type": "function", "function": {"name": "inert", "parameters": {"type": "object"}}}]
    assert admission.estimate_schema_tokens(tools, family="chat_completions") == admission.estimate_schema_tokens(deepcopy(tools), family="chat_completions")
    first = len(admission._SCHEMA_CACHE)
    admission.estimate_schema_tokens(tools, family="gemini_native")
    assert len(admission._SCHEMA_CACHE) == first + 1
    monkeypatch.setattr(admission, "ACCOUNTING_VERSION", "test-adapter-next")
    admission.estimate_schema_tokens(tools, family="chat_completions")
    assert len(admission._SCHEMA_CACHE) == first + 2
    monkeypatch.setattr(admission, "ESTIMATOR_VERSION", "test-estimator-next")
    admission.estimate_schema_tokens(tools, family="chat_completions")
    assert len(admission._SCHEMA_CACHE) == first + 3


def test_plugin_registry_mcp_normalization_nested_middle_growth_reaches_current_cache():
    from tools.registry import ToolRegistry
    from tools.mcp_tool import _convert_mcp_schema
    registry = ToolRegistry()
    plugin = {"name": "plugin_middle", "description": "inert", "parameters": {"type": "object", "properties": {"value": {"type": "string", "description": "short"}}}}
    registry.register(name="plugin_middle", toolset="inert", schema=plugin, handler=lambda args, **kw: {})
    # Registry get_definitions shallow-copies schemas; correctness must not
    # rely on that copy detaching nested provider-visible values.
    definitions = registry.get_definitions({"plugin_middle"})
    small = _estimate_tools_tokens_rough(definitions)
    plugin["parameters"]["properties"]["value"]["description"] = "x" * 4400
    grown = registry.get_definitions({"plugin_middle"})
    assert _estimate_tools_tokens_rough(grown) > small + 1000
    mcp = SimpleNamespace(name="inert", description="inert", inputSchema=plugin["parameters"])
    converted = _convert_mcp_schema("server_inert", mcp)
    assert admission.estimate_schema_tokens(converted) > 1000
