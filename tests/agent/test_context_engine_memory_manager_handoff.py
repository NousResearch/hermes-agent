"""The agent's memory manager reaches the context engine at every ``on_session_start`` site.

An engine that compacts outside ``compress()`` offers the span to ``MemoryManager.on_pre_compress``
through this handle; the compression-complete notification is pinned in
``test_compression_boundary_hook.py`` and the gateway ``/compress`` boundary in
``tests/gateway/test_session_mutation_compress_plugin_engine.py``.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_state import SessionDB


class _NarrowEngine:
    """An engine whose ``on_session_start`` names only the keywords the host passed before ``memory_manager``."""

    def __init__(self):
        self.started = []

    def on_session_start(self, session_id, hermes_home=None, platform=None, model=None, context_length=None,
                         conversation_id=None, boundary_reason=None, old_session_id=None, session_db=None):
        self.started.append((session_id, boundary_reason))


def test_agent_init_hands_the_engine_its_memory_manager():
    from agent.agent_init import _inject_context_engine_tools

    engine, manager = MagicMock(), object()
    agent = SimpleNamespace(context_compressor=engine, tools=None, enabled_toolsets=None, session_id="s1",
                            platform="cli", model="m", _gateway_session_key="agent:main:cli:dm:1",
                            _memory_manager=manager)
    _inject_context_engine_tools(agent)

    assert engine.on_session_start.call_args.kwargs["memory_manager"] is manager


def _compression_child(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent", source="cli", model="m")
    db.end_session("parent", "compression")
    db.create_session("child", source="cli", model="m", parent_session_id="parent")
    db.append_message("child", "user", "hi")
    agent = MagicMock()
    agent.session_id, agent.model, agent.provider = "parent", "m", ""
    agent.pass_session_id = False
    agent._cached_system_prompt = None
    return agent, db


def test_compression_child_adoption_hands_the_engine_its_memory_manager(tmp_path):
    from agent.conversation_compression import _adopt_live_compression_child

    agent, db = _compression_child(tmp_path)

    assert _adopt_live_compression_child(agent, db, "parent") is not None
    call = agent.context_compressor.on_session_start.call_args
    assert call.args == ("child",) and call.kwargs["boundary_reason"] == "compression"
    assert call.kwargs["memory_manager"] is agent._memory_manager


def test_an_engine_with_a_narrow_signature_still_gets_every_session_start(tmp_path):
    from agent.agent_init import _inject_context_engine_tools
    from agent.conversation_compression import (
        _adopt_live_compression_child, _notify_context_engine_compression_complete,
    )

    init_engine = _NarrowEngine()
    _inject_context_engine_tools(SimpleNamespace(context_compressor=init_engine, tools=None, enabled_toolsets=None,
                                                 session_id="s1", platform="cli", model="m",
                                                 _gateway_session_key=None, _memory_manager=object()))
    adopt_agent, db = _compression_child(tmp_path)
    adopt_agent.context_compressor = adopt_engine = _NarrowEngine()
    _adopt_live_compression_child(adopt_agent, db, "parent")
    notify_engine = _NarrowEngine()
    notified = _notify_context_engine_compression_complete(
        SimpleNamespace(context_compressor=notify_engine, platform="cli", _gateway_session_key=None,
                        _memory_manager=object()), new_session_id="s2", old_session_id="s1")

    assert init_engine.started == [("s1", None)]
    assert adopt_engine.started == [("child", "compression")]
    assert notified is True and notify_engine.started == [("s2", "compression")]
