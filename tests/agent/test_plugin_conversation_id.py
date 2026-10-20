"""Plugin tool handlers and tool hooks get a ``conversation_id`` that compression rotation keeps and a
delegated child shares with its delegator; ``session_id`` stays the physical segment id."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.prompt_cache_scope import conversation_id
from agent.runtime_session_store import RuntimeSessionStore
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, mutate_worker_execution, register_worker_execution
from run_agent import AIAgent

PLUGIN = '''
import json
events = []
def record(kind, kw):
    events.append((kind, kw["session_id"], kw.get("conversation_id")))
def handle(args, **kw):
    record("handler", kw)
    return json.dumps({"ok": True})
# No **kwargs: a new payload field must not knock a narrow middleware out of the chain.
def narrow_request(tool_name, args, session_id):
    record("tool_request", {"session_id": session_id})
def narrow_execution(args, next_call, session_id):
    record("tool_execution", {"session_id": session_id})
    return next_call(args)
def register(ctx):
    ctx.register_tool(name="conv_probe", toolset="conv_probe", handler=handle, schema={
        "name": "conv_probe", "description": "probe", "parameters": {"type": "object", "properties": {}}})
    ctx.register_hook("pre_tool_call", lambda **kw: record("pre_tool_call", kw))
    ctx.register_hook("post_tool_call", lambda **kw: record("post_tool_call", kw))
    ctx.register_middleware("tool_request", narrow_request)
    ctx.register_middleware("tool_execution", narrow_execution)
'''
_GETS_CONVERSATION_ID = {"handler", "pre_tool_call", "post_tool_call"}  # the narrow middleware doesn't ask


@pytest.fixture
def probe(tmp_path, monkeypatch):
    from hermes_cli import plugins
    from tools.registry import registry

    home = tmp_path / "home"
    plugin = home / "plugins" / "conv_probe"
    plugin.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("plugins:\n  enabled: [conv_probe]\n")
    (plugin / "plugin.yaml").write_text("name: conv_probe\nversion: '0.1.0'\n")
    (plugin / "__init__.py").write_text(PLUGIN)
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    manager.discover_and_load()
    yield manager._hooks["pre_tool_call"][0].__globals__["events"]
    registry.deregister("conv_probe")


def _agent(db, **kw):
    return AIAgent(api_key="k", base_url="https://example.invalid/v1", model="fixture/model", quiet_mode=True,
                   session_db=db, enabled_toolsets=["conv_probe"], skip_context_files=True, skip_memory=True, **kw)


def _call(agent, path):
    call = SimpleNamespace(id=f"call-{len(agent.session_id)}", type="function",
                           function=SimpleNamespace(name="conv_probe", arguments="{}"))
    getattr(agent, f"_execute_tool_calls_{path}")(SimpleNamespace(content="", tool_calls=[call]), [], "task")


# ``tool``: rotation copies a ``--source tool`` run's source, which the lineage walk reads as a fork.
@pytest.mark.parametrize("path,source", [("sequential", "cli"), ("concurrent", "tool")])
def test_conversation_id_survives_compression_rotation(path, source, tmp_path, probe):
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = _agent(db)
    try:
        root = agent.session_id
        db.create_session(session_id=root, source=source)
        _call(agent, path)
        # Only the summary model is stubbed; the rotation and the child row are the real runtime's.
        compressor = MagicMock(compression_count=1, last_prompt_tokens=0, last_completion_tokens=0,
                               _last_summary_error=None, _last_compress_aborted=False)
        compressor.compress.return_value = [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}]
        agent.context_compressor, agent._compression_feasibility_checked = compressor, True
        agent.compression_in_place = False
        agent._compress_context([{"role": "user", "content": "history " * 200}], "sys", approx_tokens=10000)
        assert agent.session_id != root
        _call(agent, path)
        assert {sid for _, sid, _ in probe} == {root, agent.session_id}
        assert sorted({kind for kind, _, _ in probe}) == [
            "handler", "post_tool_call", "pre_tool_call", "tool_execution", "tool_request"]
        assert {cid for kind, _, cid in probe if kind in _GETS_CONVERSATION_ID} == {root}
        # A managed worker reads the same rows through its RuntimeSessionStore and must agree.
        scope = dict(epoch=begin_runtime_epoch(db, instance_id="fixture"), execution_id="w",
                     session_id=agent.session_id, generation=0)
        register_worker_execution(db, **scope, kind="compute", adoption_secret="secret")
        store = RuntimeSessionStore(lambda method, **p: mutate_worker_execution(db, **p), scope, tmp_path / "outbox")
        try:
            assert conversation_id(SimpleNamespace(session_id=agent.session_id, _session_db=store)) == root
        finally:
            store.close()
        # A tool-source fork of another source's compacted session is its own conversation.
        db.create_session("cli-parent", source="cli")
        db.end_session("cli-parent", "compression")
        db.create_session("tool-fork", source="tool", parent_session_id="cli-parent")
        assert conversation_id(SimpleNamespace(session_id="tool-fork", _session_db=db)) == "tool-fork"
    finally:
        agent.close()
        db.close()


def test_delegated_child_reports_its_delegators_conversation(tmp_path, probe, monkeypatch):
    from tools import delegate_tool
    import tools.delegate_tool_config as delegate_config

    monkeypatch.setattr(delegate_tool, "_load_config", dict)
    monkeypatch.setattr(delegate_config, "_load_config", dict)
    parent = _agent(None, session_id="parent-root")
    child = delegate_tool._build_child_agent(task_index=0, goal="goal", context=None, toolsets=["conv_probe"],
                                             model=None, max_iterations=2, task_count=1, parent_agent=parent)
    try:
        _call(child, "sequential")
        assert child.session_id != "parent-root"
        assert [(sid, cid) for kind, sid, cid in probe if kind in _GETS_CONVERSATION_ID] == [
            (child.session_id, "parent-root")] * 3
    finally:
        child.close()
        parent.close()
