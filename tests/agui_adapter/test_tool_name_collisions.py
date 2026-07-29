"""Client-declared tool names must never shadow server tool names.

``registry.dispatch`` resolves a call purely by name (``tools/registry.py``),
so a frontend or state-writer declaration reusing a name like ``terminal`` or
``write_file`` would execute the SERVER tool instead of handing off to the
client.  ``_reject_name_collisions`` fails the run instead; these tests pin
that behaviour, including the two ways a naive check gets it wrong (rejecting
the adapter's own re-registered names, and missing intra-run duplicates).
"""
import pytest

from agui_adapter.session import (
    ToolNameCollisionError,
    _ADAPTER_TOOLSETS,
    _FRONTEND_TOOLSET,
    _STATE_WRITER_TOOLSET,
    _reject_name_collisions,
)


@pytest.fixture
def registry():
    """The populated global tool registry.

    Importing ``run_agent`` is what fills it (``model_tools`` runs
    ``discover_builtin_tools()`` at import time) — the same ordering
    ``build_run_agent`` relies on.  Without this the registry is empty and
    every collision assertion below would pass vacuously.
    """
    import run_agent  # noqa: F401  - imported for its registration side effect

    from tools.registry import registry as _registry

    return _registry


@pytest.fixture
def adapter_registration(registry):
    """Register a name under one of the adapter's own toolsets, as a previous
    run would have, and clean it up afterwards."""
    registered = []

    def _register(toolset, name="ui_previously_declared_tool"):
        registry.register(
            name=name,
            toolset=toolset,
            schema={"name": name, "description": "x",
                    "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kwargs: "",
            check_fn=lambda: True,
        )
        registered.append(name)
        return name

    yield _register
    for name in registered:
        registry.deregister(name)


@pytest.fixture
def server_tool_name(registry):
    """A real, registered server tool name to collide against."""
    for candidate in ("terminal", "write_file", "execute_code"):
        if registry.get_entry(candidate) is not None:
            return candidate
    pytest.skip("no known server tool registered to test collisions against")


class TestRejectsServerToolNames:
    def test_frontend_declaration_colliding_is_rejected(self, server_tool_name):
        with pytest.raises(ToolNameCollisionError) as exc:
            _reject_name_collisions({server_tool_name}, set())
        assert server_tool_name in exc.value.names

    def test_state_writer_declaration_colliding_is_rejected(self, server_tool_name):
        with pytest.raises(ToolNameCollisionError) as exc:
            _reject_name_collisions(set(), {server_tool_name})
        assert server_tool_name in exc.value.names

    @pytest.mark.parametrize("name", ["terminal", "write_file"])
    def test_named_default_hermes_acp_tools_are_reserved(self, registry, name):
        """The specific names called out in review, when actually registered."""
        if registry.get_entry(name) is None:
            pytest.skip(f"{name} is not registered in this configuration")
        with pytest.raises(ToolNameCollisionError):
            _reject_name_collisions({name}, set())

    def test_error_names_the_offending_tool_only(self, registry, server_tool_name):
        """The message echoes the client's own bad name, and does not enumerate
        the server's tool inventory back to the client."""
        with pytest.raises(ToolNameCollisionError) as exc:
            _reject_name_collisions({server_tool_name, "safe_client_tool"}, set())
        message = str(exc.value)
        assert server_tool_name in message
        assert "safe_client_tool" not in message


class TestAllowsLegitimateDeclarations:
    def test_non_colliding_names_pass(self, registry):
        _reject_name_collisions({"ui_confirm", "ui_pick_date"}, {"set_theme"})

    def test_empty_declarations_pass(self, registry):
        _reject_name_collisions(set(), set())

    @pytest.mark.parametrize(
        "toolset, declare_as",
        [(_FRONTEND_TOOLSET, "frontend"), (_STATE_WRITER_TOOLSET, "state_writer")],
    )
    def test_adapter_owned_names_are_not_collisions(
        self, registry, adapter_registration, toolset, declare_as
    ):
        """Re-declaring the same client tool on a LATER run must still work.

        Adapter registration is process-global and idempotent while
        declarations are per-run, so by run two the adapter's own handler is
        sitting in the registry under that name.  A check that only asked
        "is this name registered?" would reject every repeat run.
        """
        name = adapter_registration(toolset)
        assert registry.get_entry(name) is not None
        if declare_as == "frontend":
            _reject_name_collisions({name}, set())  # must not raise
        else:
            _reject_name_collisions(set(), {name})  # must not raise


class TestRejectsAmbiguousDeclarations:
    def test_same_name_as_both_frontend_and_state_writer(self, registry):
        """One name cannot be both client-executed and server-executed."""
        with pytest.raises(ToolNameCollisionError) as exc:
            _reject_name_collisions({"ui_ambiguous"}, {"ui_ambiguous"})
        assert "ui_ambiguous" in exc.value.names

    def test_state_writer_name_redeclared_as_frontend_is_rejected(
        self, registry, adapter_registration
    ):
        """Cross-kind reuse across runs is the same shadowing bug one layer in.

        Registration is skipped when the name is already in the registry, so
        without this the name would stay bound to the state-writer handler
        while being advertised to the model as client-executed.
        """
        name = adapter_registration(_STATE_WRITER_TOOLSET)
        with pytest.raises(ToolNameCollisionError) as exc:
            _reject_name_collisions({name}, set())
        assert name in exc.value.names

    def test_frontend_name_redeclared_as_state_writer_is_rejected(
        self, registry, adapter_registration
    ):
        """The mirror case: a client-handoff name must not become server-executed."""
        name = adapter_registration(_FRONTEND_TOOLSET)
        with pytest.raises(ToolNameCollisionError) as exc:
            _reject_name_collisions(set(), {name})
        assert name in exc.value.names


class TestGuardIsWiredIntoBuildRunAgent:
    """The guard has to fire on the real construction path, not just when the
    helper is called directly.  ``build_run_agent`` rejects before it resolves
    settings or constructs an ``AIAgent``, so these need no model or network.
    """

    def test_frontend_collision_rejected_by_build_run_agent(self, server_tool_name):
        from agui_adapter.session import AgentConfig, build_run_agent

        with pytest.raises(ToolNameCollisionError):
            build_run_agent(AgentConfig(), frontend_tool_names={server_tool_name})

    def test_state_writer_collision_rejected_by_build_run_agent(self, server_tool_name):
        from agui_adapter.session import AgentConfig, StateWriterSpec, build_run_agent

        with pytest.raises(ToolNameCollisionError):
            build_run_agent(
                AgentConfig(),
                state_writer_specs={server_tool_name: StateWriterSpec()},
            )

    def test_collision_is_rejected_before_the_name_becomes_callable(
        self, registry, server_tool_name
    ):
        """The actual vulnerability: the declared name must never reach
        ``agent.valid_tool_names``, since ``registry.dispatch`` would then route
        the model's call to the server tool of that name."""
        from agui_adapter.session import AgentConfig, build_run_agent

        entry_before = registry.get_entry(server_tool_name)
        with pytest.raises(ToolNameCollisionError):
            build_run_agent(AgentConfig(), frontend_tool_names={server_tool_name})
        # The server tool is untouched: still owned by its original toolset,
        # never rebound to the adapter's client-handoff handler.
        entry_after = registry.get_entry(server_tool_name)
        assert entry_after is entry_before
        assert entry_after.toolset not in _ADAPTER_TOOLSETS
