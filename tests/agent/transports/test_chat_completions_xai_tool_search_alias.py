"""Tests for the xAI ``tool_search`` reserved-name alias (#95003).

xAI's chat-completions API reserves the function name ``tool_search`` for
its native server-side tool and rejects the whole request when the client
Tool Search bridge declares it (HTTP 400 "The function name tool_search is
reserved for the tool_search tool"). The fix mirrors the web_search treatment
in ``transports/codex.py``: rename the bridge's wire declaration to
``hermes_tool_search`` for xAI targets and map the alias back to
``tool_search`` in ``normalize_response`` so dispatch is unchanged.

The reverse map is request-local: ``normalize_response`` only rewrites
aliases the paired request actually emitted (stashed on the transport as
``_last_wire_aliases``), so a real user/plugin/MCP tool that happens to be
named ``hermes_tool_search`` is never silently dispatched as the bridge.
"""

from types import SimpleNamespace

import pytest

from agent.transports import get_transport
from agent.transports.chat_completions import (
    _XAI_TOOL_SEARCH_ALIAS,
    _rename_tool_search_bridge_for_xai,
)


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


class TestRenameToolSearchBridgeForXai:
    def test_tool_search_renamed_alias_value(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "tool_search",
                "description": "Search deferred tools",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        out, alias_map = _rename_tool_search_bridge_for_xai(tools)
        assert out[0]["function"]["name"] == _XAI_TOOL_SEARCH_ALIAS
        assert out[0]["function"]["name"] == "hermes_tool_search"
        assert alias_map == {"hermes_tool_search": "tool_search"}

    def test_schema_and_description_untouched(self):
        fn = {
            "name": "tool_search",
            "description": "Search the deferred tool catalog",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
        out, _ = _rename_tool_search_bridge_for_xai([{"type": "function", "function": fn}])
        assert out[0]["function"]["description"] == fn["description"]
        assert out[0]["function"]["parameters"] == fn["parameters"]

    def test_sibling_bridge_names_not_reserved(self):
        # xAI's error names only tool_search; tool_describe / tool_call stay
        # on the wire unchanged so the model keeps calling them directly.
        tools = [
            {"type": "function", "function": {"name": "tool_describe"}},
            {"type": "function", "function": {"name": "tool_call"}},
        ]
        out, alias_map = _rename_tool_search_bridge_for_xai(tools)
        assert [t["function"]["name"] for t in out] == ["tool_describe", "tool_call"]
        assert alias_map == {}

    def test_ordinary_tools_untouched(self):
        tools = [{"type": "function", "function": {"name": "web_search"}}]
        out, alias_map = _rename_tool_search_bridge_for_xai(tools)
        assert out[0]["function"]["name"] == "web_search"
        assert alias_map == {}

    def test_input_not_mutated(self):
        # The helper feeds a deep-copied list on the helper-layer path, but
        # pin copy semantics anyway: the shared per-agent tool registry must
        # never see the alias (#27907 lesson).
        tools = [{"type": "function", "function": {"name": "tool_search"}}]
        _rename_tool_search_bridge_for_xai(tools)
        assert tools[0]["function"]["name"] == "tool_search"

    def test_collision_with_real_hermes_tool_search_takes_suffix(self):
        # A legitimate tool already using the alias name must NOT be
        # shadowed and no duplicate wire names may be produced: the bridge
        # takes hermes_tool_search_2 instead.
        tools = [
            {"type": "function", "function": {"name": "hermes_tool_search"}},
            {"type": "function", "function": {"name": "tool_search"}},
        ]
        out, alias_map = _rename_tool_search_bridge_for_xai(tools)
        names = [t["function"]["name"] for t in out]
        assert names == ["hermes_tool_search", "hermes_tool_search_2"]
        assert len(names) == len(set(names))
        assert alias_map == {"hermes_tool_search_2": "tool_search"}


def _fake_response(tool_name):
    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name=tool_name, arguments='{"query": "x"}'),
    )
    msg = SimpleNamespace(tool_calls=[tc], reasoning=None, reasoning_content=None)
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None)


class TestNormalizeResponseMapsAliasBack:
    def test_alias_call_maps_back_to_bridge_name(self, transport):
        transport._last_wire_aliases = {"hermes_tool_search": "tool_search"}
        resp = transport.normalize_response(_fake_response(_XAI_TOOL_SEARCH_ALIAS))
        assert resp.tool_calls[0].name == "tool_search"

    def test_ordinary_call_name_preserved(self, transport):
        transport._last_wire_aliases = {"hermes_tool_search": "tool_search"}
        resp = transport.normalize_response(_fake_response("tool_describe"))
        assert resp.tool_calls[0].name == "tool_describe"

    def test_no_alias_emitted_means_no_reverse_rewrite(self, transport):
        # Provenance contract: if THIS request emitted no aliases, a tool
        # call named hermes_tool_search is a REAL tool (user/plugin/MCP)
        # and must dispatch under its own name.
        transport._last_wire_aliases = {}
        resp = transport.normalize_response(_fake_response("hermes_tool_search"))
        assert resp.tool_calls[0].name == "hermes_tool_search"

    def test_suffixed_alias_maps_back(self, transport):
        transport._last_wire_aliases = {"hermes_tool_search_2": "tool_search"}
        resp = transport.normalize_response(_fake_response("hermes_tool_search_2"))
        assert resp.tool_calls[0].name == "tool_search"
        # And the real tool occupying the plain alias name is untouched.
        resp2 = transport.normalize_response(_fake_response("hermes_tool_search"))
        assert resp2.tool_calls[0].name == "hermes_tool_search"

    def test_legacy_fallback_without_provenance(self, transport):
        # Normalize-only call sites (no request built on this instance)
        # keep the historical unconditional mapping.
        transport._last_wire_aliases = None
        resp = transport.normalize_response(_fake_response(_XAI_TOOL_SEARCH_ALIAS))
        assert resp.tool_calls[0].name == "tool_search"
