"""Memory-provider tool map recorded beside the session's sidecar state (#157197).

The achievements scan sees stored tool names only, so ``record_memory_provider_tools``
persists the session's ``{tool name: provider name}`` map into its ``model_config``
sidecar and the scanner stops judging provider calls by whether the name contains
"memory".
"""

from types import SimpleNamespace

from agent.agent_init_memory_provider import record_memory_provider_tools


def _agent(writes, *, tools=None, persistence=None):
    """An agent-shaped namespace whose ``db`` path records its ``model_config`` patches."""
    return SimpleNamespace(
        _memory_manager=SimpleNamespace(
            get_provider_tool_map=lambda: dict(tools if tools is not None else {"hindsight_retain": "hindsight"})),
        session_id="s1",
        _session_db=SimpleNamespace(
            patch_session_model_config=lambda session_id, patch: writes.append((session_id, patch))),
        persistence=persistence,
    )


def test_records_the_map_once_for_the_session_sidecar():
    writes = []
    record_memory_provider_tools(_agent(writes))
    assert writes == [("s1", {"memory_tools": {"hindsight_retain": "hindsight"}})]


def test_writer_interface_is_preferred_over_the_bare_db():
    store_writes = []
    persistence = SimpleNamespace(
        read_only=False,
        sessions=SimpleNamespace(
            patch_model_config=lambda session_id, patch: store_writes.append((session_id, patch))),
    )
    writes = []
    record_memory_provider_tools(_agent(writes, persistence=persistence))
    assert writes == []
    assert store_writes == [("s1", {"memory_tools": {"hindsight_retain": "hindsight"}})]


def test_no_provider_tools_writes_nothing():
    writes = []
    record_memory_provider_tools(_agent(writes, tools={}))
    assert writes == []


def test_no_memory_manager_or_no_session_store_is_a_no_op():
    writes = []
    agent = _agent(writes)
    agent._memory_manager = None
    record_memory_provider_tools(agent)
    assert writes == []

    agent = SimpleNamespace(_memory_manager=None, session_id="s1",
                            _session_db=SimpleNamespace(patch_session_model_config=None))
    record_memory_provider_tools(agent)  # must not raise


def test_a_failing_sidecar_patch_is_swallowed_and_logged(caplog):
    import logging

    agent = SimpleNamespace(
        _memory_manager=SimpleNamespace(get_provider_tool_map=lambda: {"hindsight_retain": "hindsight"}),
        session_id="s1",
        _session_db=SimpleNamespace(
            patch_session_model_config=lambda session_id, patch: (_ for _ in ()).throw(RuntimeError("locked"))),
        persistence=None,
    )
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        record_memory_provider_tools(agent)  # agent init must not break over a busy store
    assert any("not recorded" in record.message for record in caplog.records)


def test_injection_records_the_map_for_the_session():
    """The injection of the provider schemas is what records the map: the surface and its
    tool-ownership sidecar stay one consistent pair."""
    from agent.memory_manager_tools import inject_memory_provider_tools

    writes = []
    agent = _agent(writes)
    agent.tools = [{"type": "function", "function": {"name": "memory"}}]
    agent.enabled_toolsets = ["memory"]
    agent.disabled_toolsets = []
    agent._memory_manager.get_all_tool_schemas = lambda: [
        {"name": "hindsight_retain", "description": "", "parameters": {}}]

    assert inject_memory_provider_tools(agent) == 1
    assert writes == [("s1", {"memory_tools": {"hindsight_retain": "hindsight"}})]


def test_a_gated_off_toolset_records_nothing():
    from agent.memory_manager_tools import inject_memory_provider_tools

    writes = []
    agent = _agent(writes)
    agent.tools = [{"type": "function", "function": {"name": "memory"}}]
    agent.enabled_toolsets = ["memory"]
    agent.disabled_toolsets = ["memory"]  # the memory toolset is denylisted: no provider surface

    assert inject_memory_provider_tools(agent) == 0
    assert writes == []
