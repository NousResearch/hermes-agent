"""Context-engine selection and subtraction agree at startup and reload."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

class TestContextEngineToolsetGate:
    """Issue #5544 (sibling): context engine tools follow the same gate.

    `agent.context_compressor.get_tool_schemas()` (e.g. lcm_grep, lcm_describe,
    lcm_expand) was appended to AIAgent.tools unconditionally. Same blind
    injection class as the memory bug; same local-model penalty. Gate name:
    "context_engine" (matches the existing plugin-system convention).
    """

    @staticmethod
    def _run_context_engine_injection(
        enabled_toolsets,
        compressor,
        *,
        disabled_toolsets=None,
    ):
        """Run the production startup injector against a minimal agent."""
        from agent.context_engine import inject_context_engine_tools

        agent = SimpleNamespace(
            context_compressor=compressor,
            tools=[],
            valid_tool_names=set(),
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
        )
        inject_context_engine_tools(agent)
        return (
            agent.tools,
            agent.valid_tool_names,
            agent._context_engine_tool_names,
        )

    class _FakeCompressor:
        def __init__(self, schemas):
            self._schemas = schemas

        def get_tool_schemas(self):
            return list(self._schemas)

    def _compressor_with(self, *tool_names):
        return self._FakeCompressor(
            [{"name": n, "description": n, "parameters": {}} for n in tool_names]
        )

    def test_hidden_registry_route_is_not_claimed_by_context_engine(self):
        from agent.context_engine import inject_context_engine_tools

        agent = SimpleNamespace(
            context_compressor=self._compressor_with("shared_tool"),
            _tool_registry_routes={"shared_tool": object()},
            tools=[],
            valid_tool_names=set(),
            enabled_toolsets=None,
            disabled_toolsets=None,
        )

        inject_context_engine_tools(agent)

        assert agent.tools == []
        assert agent.valid_tool_names == set()
        assert agent._context_engine_tool_names == set()

    def test_none_toolsets_injects(self):
        """enabled_toolsets=None injects context-engine tools — backward compat."""
        c = self._compressor_with("lcm_grep", "lcm_describe", "lcm_expand")
        tools, names, engine_names = self._run_context_engine_injection(None, c)
        assert engine_names == {"lcm_grep", "lcm_describe", "lcm_expand"}

    def test_context_engine_in_toolsets_injects(self):
        """enabled_toolsets including 'context_engine' injects the tools."""
        c = self._compressor_with("lcm_grep")
        tools, names, engine_names = self._run_context_engine_injection(
            ["terminal", "context_engine"], c
        )
        assert "lcm_grep" in engine_names

    @pytest.mark.parametrize("enabled_toolsets", [["all"], "*"])
    def test_global_enabled_aliases_inject_context_engine(
        self,
        enabled_toolsets,
    ):
        c = self._compressor_with("lcm_grep")

        _tools, _names, engine_names = self._run_context_engine_injection(
            enabled_toolsets,
            c,
        )

        assert engine_names == {"lcm_grep"}

    def test_composite_enabled_toolset_injects_context_engine(self, monkeypatch):
        import toolsets

        monkeypatch.setitem(
            toolsets.TOOLSETS,
            "context-bundle",
            {
                "description": "test",
                "tools": [],
                "includes": ["context_engine"],
            },
        )
        c = self._compressor_with("lcm_grep")

        _tools, _names, engine_names = self._run_context_engine_injection(
            ["context-bundle"],
            c,
        )

        assert engine_names == {"lcm_grep"}

    def test_empty_toolsets_blocks_injection(self):
        """`platform_toolsets: telegram: []` must suppress context-engine tools."""
        c = self._compressor_with("lcm_grep")
        tools, names, engine_names = self._run_context_engine_injection([], c)
        assert tools == []
        assert engine_names == set()

    def test_toolsets_without_context_engine_blocks_injection(self):
        """A toolset list that doesn't name 'context_engine' suppresses injection."""
        c = self._compressor_with("lcm_grep", "lcm_describe")
        tools, names, engine_names = self._run_context_engine_injection(
            ["terminal", "memory"], c
        )
        assert tools == []
        assert engine_names == set()

    @pytest.mark.parametrize(
        ("enabled_toolsets", "disabled_toolsets"),
        [
            ([], None),
            (["terminal"], None),
            (None, ["all"]),
            (None, ["*"]),
            (None, ["context_engine"]),
        ],
    )
    def test_family_policy_denial_does_not_enumerate_schemas(
        self,
        enabled_toolsets,
        disabled_toolsets,
    ):
        """A denied dynamic family must not require disabled plugin code."""
        calls = []

        def _disabled_schema_callback():
            calls.append("called")
            raise RuntimeError("disabled engine touched")

        compressor = SimpleNamespace(get_tool_schemas=_disabled_schema_callback)

        tools, names, engine_names = self._run_context_engine_injection(
            enabled_toolsets,
            compressor,
            disabled_toolsets=disabled_toolsets,
        )

        assert calls == []
        assert tools == []
        assert names == set()
        assert engine_names == set()

    @pytest.mark.parametrize(
        "disabled_toolsets",
        [["all"], ["*"], ["context_engine"]],
    )
    def test_final_disabled_subtraction_blocks_dynamic_family(
        self,
        disabled_toolsets,
    ):
        c = self._compressor_with("lcm_grep")

        tools, names, engine_names = self._run_context_engine_injection(
            None,
            c,
            disabled_toolsets=disabled_toolsets,
        )

        assert tools == []
        assert names == set()
        assert engine_names == set()

    def test_custom_disabled_toolset_subtracts_exact_dynamic_name(
        self,
        monkeypatch,
    ):
        import toolsets

        monkeypatch.setitem(
            toolsets.TOOLSETS,
            "deny-lcm-grep",
            {"description": "test", "tools": ["lcm_grep"], "includes": []},
        )
        c = self._compressor_with("lcm_grep", "lcm_describe")

        tools, names, engine_names = self._run_context_engine_injection(
            None,
            c,
            disabled_toolsets=["deny-lcm-grep"],
        )

        assert [tool["function"]["name"] for tool in tools] == ["lcm_describe"]
        assert names == {"lcm_describe"}
        assert engine_names == {"lcm_describe"}

    def test_disabled_toolset_resolution_failure_fails_closed(
        self,
        monkeypatch,
    ):
        import toolsets

        monkeypatch.setattr(
            toolsets,
            "validate_toolset",
            lambda _name: (_ for _ in ()).throw(RuntimeError("resolution failed")),
        )
        c = self._compressor_with("lcm_grep")

        tools, names, engine_names = self._run_context_engine_injection(
            None,
            c,
            disabled_toolsets=["broken-policy"],
        )

        assert tools == []
        assert names == set()
        assert engine_names == set()

    def test_no_compressor_no_injection(self):
        """Gate is moot without a context_compressor."""
        tools, names, engine_names = self._run_context_engine_injection(None, None)
        assert tools == []


def _engine_agent(name="engine_lookup"):
    schema = {"name": name, "description": "engine", "parameters": {}}
    return SimpleNamespace(tools=[{"type": "function", "function": schema}], valid_tool_names={name},
                           enabled_toolsets=None, disabled_toolsets=None, _context_engine_tool_names={name},
                           context_compressor=SimpleNamespace(get_tool_schemas=MagicMock(return_value=[schema])))


def test_reload_applies_exact_engine_subtraction(monkeypatch):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    import toolsets
    a = _engine_agent()
    a.disabled_toolsets = ["custom-deny"]
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [])
    monkeypatch.setattr(toolsets, "validate_toolset", lambda name: name == "custom-deny")
    monkeypatch.setattr(toolsets, "get_toolset", lambda _name: {"tools": ["engine_lookup"]})
    monkeypatch.setattr(toolsets, "resolve_toolset", lambda _name: {"engine_lookup"})
    refresh_agent_mcp_tools(a)
    assert a.tools == []
    assert a.valid_tool_names == set()
    assert a._context_engine_tool_names == set()


def test_same_name_reload_transfers_engine_schema_and_ownership(monkeypatch):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    a = _engine_agent()
    replacement = {"type": "function", "function": {"name": "engine_lookup", "description": "registry", "parameters": {}}}
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [replacement])
    refresh_agent_mcp_tools(a)
    assert a.tools == [replacement]
    assert a._context_engine_tool_names == set()


def test_preserved_prefix_does_not_restore_revoked_engine_schema(monkeypatch):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    from tools.registry import registry
    a = _engine_agent()
    registry.register(name="engine_lookup", toolset="policy-test", schema={"name": "engine_lookup"}, handler=lambda _args, **_kw: "registry")
    try:
        a.disabled_toolsets = ["context_engine"]
        monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [])
        refresh_agent_mcp_tools(a, preserve_prefix=True)
        assert a.tools == []
        assert a._context_engine_tool_names == set()
        a.context_compressor.get_tool_schemas.assert_not_called()
    finally:
        registry.deregister("engine_lookup")
