"""The agent's memory manager reaches the context engine at every ``on_session_start`` site.

An engine that compacts outside ``compress()`` offers the span to ``MemoryManager.on_pre_compress``
through this handle; the compression-complete notification is pinned in
``test_compression_boundary_hook.py`` and the gateway ``/compress`` boundary in
``tests/gateway/test_session_mutation_compress_plugin_engine.py``.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_state import SessionDB


def test_agent_init_hands_the_engine_its_memory_manager():
    from agent.agent_init import _inject_context_engine_tools

    engine, manager = MagicMock(), object()
    agent = SimpleNamespace(context_compressor=engine, tools=None, enabled_toolsets=None, session_id="s1",
                            platform="cli", model="m", _gateway_session_key="agent:main:cli:dm:1",
                            _memory_manager=manager)
    _inject_context_engine_tools(agent)

    assert engine.on_session_start.call_args.kwargs["memory_manager"] is manager


def test_compression_child_adoption_hands_the_engine_its_memory_manager(tmp_path):
    from agent.conversation_compression import _adopt_live_compression_child

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="cli", model="m")
    db.end_session("parent", "compression")
    db.create_session("child", source="cli", model="m", parent_session_id="parent")
    db.append_message("child", "user", "hi")
    agent = MagicMock()
    agent.session_id, agent.model, agent.provider = "parent", "m", ""
    agent.pass_session_id = False
    agent._cached_system_prompt = None

    assert _adopt_live_compression_child(agent, db, "parent") is not None
    call = agent.context_compressor.on_session_start.call_args
    assert call.args == ("child",) and call.kwargs["boundary_reason"] == "compression"
    assert call.kwargs["memory_manager"] is agent._memory_manager
